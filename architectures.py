from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from dino import DinoEncoder
from vit import ViTPredictor


def init_module_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Projector(nn.Module):
    """
    spec like "512-2048-2048"
    """
    def __init__(self, mlp_spec: str):
        super().__init__()
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        self.net = nn.Sequential(*layers)
        self.out_dim = f[-1]
        self.apply(init_module_weights)

    def forward(self, x):
        return self.net(x)


class ResnetBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + identity)


class ResnetStack(nn.Module):
    def __init__(self, input_channels, num_features, num_blocks, max_pooling=True):
        super().__init__()
        self.initial_conv = nn.Conv2d(
            input_channels, num_features, kernel_size=3, padding=1
        )
        self.blocks = nn.ModuleList(
            [ResnetBlock(num_features) for _ in range(num_blocks)]
        )
        self.max_pool = (
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if max_pooling else nn.Identity()
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.max_pool(x)
        for block in self.blocks:
            x = block(x)
        return x


class ImpalaEncoder(nn.Module):
    """
    Input:  [B, C, T, H, W]
    Output: [B, D, T, 1, 1]

    Also supports:
      - forward_features(x) -> [B, C_f, T, H_f, W_f]
      - project_features(feats) -> [B, D, T, 1, 1]
    """
    def __init__(
        self,
        width=1,
        stack_sizes=(16, 32, 32),
        num_blocks=2,
        dropout_rate=None,
        layer_norm=False,
        input_channels=3,
        final_ln=True,
        mlp_output_dim=512,
        input_shape=(3, 224, 224),
    ):
        super().__init__()
        self.width = width
        self.stack_sizes = stack_sizes
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        self.input_shape = input_shape
        self.mlp_output_dim = mlp_output_dim

        channels = [input_channels] + list(stack_sizes)

        self.stack_blocks = nn.ModuleList(
            [
                ResnetStack(
                    input_channels=channels[i],
                    num_features=stack_size * width,
                    num_blocks=num_blocks,
                )
                for i, stack_size in enumerate(stack_sizes)
            ]
        )

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate else nn.Identity()

        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            out = dummy
            for block in self.stack_blocks:
                out = block(out)
            self.feature_channels = out.shape[1]
            self.feature_hw = (out.shape[2], out.shape[3])
            flattened_dim = out.reshape(out.size(0), -1).shape[1]

        self.mlp = nn.Linear(flattened_dim, self.mlp_output_dim)
        self.final_ln = nn.LayerNorm(self.mlp_output_dim) if final_ln else nn.Identity()

        self.apply(init_module_weights)

    def forward_features(self, x):
        """
        x: [B, C, T, H, W]
        return: [B, C_f, T, H_f, W_f]
        """
        b, c, t, h, w = x.shape
        x = x.permute(2, 0, 1, 3, 4)  # [T, B, C, H, W]

        feats = []
        for i in range(t):
            conv_out = x[i]
            for block in self.stack_blocks:
                conv_out = block(conv_out)
                if self.dropout_rate is not None:
                    conv_out = self.dropout(conv_out)

            conv_out = F.relu(conv_out)
            feats.append(conv_out)

        feats = torch.stack(feats, dim=2)  # [B, C_f, T, H_f, W_f]
        return feats

    def project_features(self, feats):
        """
        feats: [B, C_f, T, H_f, W_f]
        return: [B, D, T, 1, 1]
        """
        b, c, t, h, w = feats.shape
        x = feats.permute(0, 2, 1, 3, 4).contiguous().reshape(b * t, c * h * w)
        x = self.mlp(x)
        x = self.final_ln(x)
        x = x.view(b, t, self.mlp_output_dim).transpose(1, 2).unsqueeze(-1).unsqueeze(-1)
        return x.contiguous()

    def forward(self, x):
        feats = self.forward_features(x)
        return self.project_features(feats)


class RNNPredictor(nn.Module):
    """
    Input:
      state:  [B, D, T, 1, 1]
      action: [B, A, T]
    Output:
      pred:   [B, D, T, 1, 1]
    """
    def __init__(
        self,
        hidden_size: int = 512,
        action_dim: int = 7,
        num_layers: int = 1,
        final_ln: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=action_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.final_ln = final_ln if final_ln is not None else nn.Identity()
        self.is_rnn = True
        self.context_length = 0

    def forward(self, state, action):
        b, d, t, _, _ = state.shape
        _, a, ta = action.shape
        assert t == ta

        outs = []
        h = None
        for i in range(t):
            s_i = state[:, :, i].reshape(1, b, d).contiguous()
            a_i = action[:, :, i].reshape(1, b, a).contiguous()
            h0 = s_i if h is None else h
            out_i, h = self.rnn(a_i, h0)
            out_i = self.final_ln(out_i)
            outs.append(out_i[0])

        outs = torch.stack(outs, dim=2).unsqueeze(-1).unsqueeze(-1)
        return outs


class InverseDynamicsModel(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.apply(init_module_weights)

    def forward(self, state_t, state_tp1):
        x = torch.cat([state_t, state_tp1], dim=-1)
        return self.model(x)



class DinoGridEncoder(nn.Module):
    def __init__(
        self,
        name="dinov2_vits14",
        feature_key="x_norm_patchtokens",
        freeze=True,
        adapter_dim=384,
        use_adapter=True,
    ):
        super().__init__()
        self.dino = DinoEncoder(name=name, feature_key=feature_key)
        self.dino_dim = self.dino.emb_dim
        self.patch_size = self.dino.patch_size
        self.freeze = freeze
        self.use_adapter = use_adapter
        self.emb_dim = adapter_dim if use_adapter else self.dino_dim

        if freeze:
            for p in self.dino.parameters():
                p.requires_grad = False
            self.dino.eval()

        if use_adapter:
            self.adapter = nn.Sequential(
                nn.LayerNorm(self.dino_dim),
                nn.Linear(self.dino_dim, adapter_dim),
                nn.GELU(),
                nn.Linear(adapter_dim, adapter_dim),
                nn.LayerNorm(adapter_dim),
            )
        else:
            self.adapter = nn.Identity()

    def forward(self, x):
        # x: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        if self.freeze:
            with torch.no_grad():
                z = self.dino(x)
        else:
            z = self.dino(x)

        # z: [B*T, P, dino_dim]
        z = self.adapter(z)  # [B*T, P, adapter_dim]
        z = z.reshape(b, t, z.shape[1], z.shape[2])
        return z.contiguous()



class DINOViTPredictor(nn.Module):
    """
    Input:
        z:      [B, T, P, D]
        action: [B, T, A]
    Output:
        pred:   [B, T, P, D]
    """
    def __init__(
        self,
        num_patches: int,
        num_frames: int,
        dim: int,
        action_dim: int,
        depth: int = 6,
        heads: int = 6,
        mlp_dim: int = 1536,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.num_frames = num_frames
        self.dim = dim

        self.action_proj = nn.Linear(action_dim, dim)

        self.predictor = ViTPredictor(
            num_patches=num_patches,
            num_frames=num_frames,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=dropout,
            use_sdpa=True,
        )

    def forward(self, z, action):
        # z: [B,T,P,D]
        # action: [B,T,A]
        B, T, P, D = z.shape

        a = self.action_proj(action).unsqueeze(2)  # [B,T,1,D]
        x = z + a                                  # [B,T,P,D]

        x = x.reshape(B, T * P, D)
        y = self.predictor(x)
        y = y.reshape(B, T, P, D)
        return y