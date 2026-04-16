from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from architectures import (
    ImpalaEncoder,
    InverseDynamicsModel,
    PointCloudTemporalEncoder,
    PointNetEncoderXYZRGB,
    Projector,
    RNNPredictor,
    build_vision_tactile_encoder,
)
from jepa import WorldModel
from losses import (
    SquareLossSeq,
    VCRegularizer,
    SIGReg,
    SIGRegRegularizer,
    FusedDynamicsRegularizer,
)


@dataclass
class PlannerConfig:
    history_size: int = 1
    horizon: int = 6
    candidates: int = 64
    topk: int = 8
    iterations: int = 4
    action_low: float = -1.0
    action_high: float = 1.0
    action_dim: int = 6

    vision_key: str = "front"
    vision_type: str = "image"   # image | pc
    image_size: int = 224

    pc_in_channels: int = 3
    pc_use_layernorm: bool = False
    pc_final_norm: str = "none"

    sum_all_diffs: bool = False
    discount: float = 1.0

    vision_dim: int = 512
    tactile_dim: int = 512

    use_tactile: bool = False
    tactile_key: str = "left_tactile_camera_taxim"
    tactile_in_channels: int = 3
    tactile_height: int = 10
    tactile_width: int = 14

    fusion_type: str = "concat"
    fusion_latent_dim: Optional[int] = None
    fusion_hidden_dim: Optional[int] = None

    attn_d_model: int = 256
    attn_heads: int = 4
    attn_layers: int = 2
    attn_mlp_ratio: float = 4.0
    attn_dropout: float = 0.0

    reg_loss_type: str = "vc"    # vc | sigreg
    use_proj: bool = False

    # branch VC
    cov_coeff: float = 1.0
    std_coeff: float = 1.0

    # branch SIGReg
    sigreg_coeff: float = 1.0
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024

    # fused dynamics reg
    sim_coeff_t: float = 0.1
    idm_coeff: float = 0.1
    idm_after_proj: bool = False
    sim_t_after_proj: bool = False

    # branch switches
    reg_vision: bool = True
    reg_tactile: bool = False


def build_vision_encoder(cfg: PlannerConfig) -> nn.Module:
    if cfg.vision_type == "image":
        return ImpalaEncoder(
            input_channels=3,
            input_shape=(3, cfg.image_size, cfg.image_size),
            mlp_output_dim=cfg.vision_dim,
            final_ln=True,
        )

    if cfg.vision_type == "pc":
        point_encoder = PointNetEncoderXYZRGB(
            in_channels=cfg.pc_in_channels,
            out_channels=cfg.vision_dim,
        )
        return PointCloudTemporalEncoder(
            point_encoder=point_encoder,
            out_dim=cfg.vision_dim,
            final_ln=True,
        )

    raise ValueError(f"Unknown vision_type: {cfg.vision_type}")


def build_tactile_encoder(cfg: PlannerConfig) -> nn.Module:
    return ImpalaEncoder(
        input_channels=cfg.tactile_in_channels,
        input_shape=(cfg.tactile_in_channels, cfg.tactile_height, cfg.tactile_width),
        mlp_output_dim=cfg.tactile_dim,
        final_ln=True,
    )


def build_branch_regularizer(branch_dim: int, cfg: PlannerConfig):
    projector = None
    if cfg.use_proj:
        projector = Projector(f"{branch_dim}-{branch_dim*4}-{branch_dim*4}")

    if cfg.reg_loss_type == "vc":
        return VCRegularizer(
            cov_coeff=cfg.cov_coeff,
            std_coeff=cfg.std_coeff,
            projector=projector,
        )

    if cfg.reg_loss_type == "sigreg":
        sigreg = SIGReg(
            knots=cfg.sigreg_knots,
            num_proj=cfg.sigreg_num_proj,
        )
        return SIGRegRegularizer(
            sigreg_coeff=cfg.sigreg_coeff,
            sigreg=sigreg,
            projector=projector,
        )

    raise ValueError(f"Unknown reg_loss_type: {cfg.reg_loss_type}")


def build_fused_regularizer(fused_dim: int, action_dim: int, cfg: PlannerConfig):
    projector = None
    if cfg.use_proj:
        projector = Projector(f"{fused_dim}-{fused_dim*4}-{fused_dim*4}")

    idm_in_dim = projector.out_dim if projector is not None and cfg.idm_after_proj else fused_dim
    idm = InverseDynamicsModel(
        state_dim=idm_in_dim,
        hidden_dim=256,
        action_dim=action_dim,
    )

    return FusedDynamicsRegularizer(
        sim_coeff_t=cfg.sim_coeff_t,
        idm_coeff=cfg.idm_coeff,
        idm=idm,
        projector=projector,
        idm_after_proj=cfg.idm_after_proj,
        sim_t_after_proj=cfg.sim_t_after_proj,
    )


def build_model(cfg: PlannerConfig) -> WorldModel:
    vision_encoder = build_vision_encoder(cfg)

    if cfg.use_tactile:
        tactile_encoder = build_tactile_encoder(cfg)
        encoder, predictor_hidden = build_vision_tactile_encoder(
            fusion_type=cfg.fusion_type,
            vision_encoder=vision_encoder,
            tactile_encoder=tactile_encoder,
            vision_dim=cfg.vision_dim,
            tactile_dim=cfg.tactile_dim,
            fusion_latent_dim=cfg.fusion_latent_dim,
            fusion_hidden_dim=cfg.fusion_hidden_dim,
            attn_d_model=cfg.attn_d_model,
            attn_heads=cfg.attn_heads,
            attn_layers=cfg.attn_layers,
            attn_mlp_ratio=cfg.attn_mlp_ratio,
            attn_dropout=cfg.attn_dropout,
        )
    else:
        encoder = vision_encoder
        predictor_hidden = cfg.vision_dim

    predictor = RNNPredictor(
        hidden_size=predictor_hidden,
        action_dim=cfg.action_dim,
        num_layers=1,
        final_ln=nn.LayerNorm(predictor_hidden),
    )

    vision_regularizer = build_branch_regularizer(cfg.vision_dim, cfg) if cfg.reg_vision else None
    tactile_regularizer = build_branch_regularizer(cfg.tactile_dim, cfg) if (cfg.use_tactile and cfg.reg_tactile) else None
    fused_regularizer = build_fused_regularizer(predictor_hidden, cfg.action_dim, cfg)

    predcost = SquareLossSeq()

    return WorldModel(
        encoder=encoder,
        predictor=predictor,
        vision_regularizer=vision_regularizer,
        tactile_regularizer=tactile_regularizer,
        fused_regularizer=fused_regularizer,
        predcost=predcost,
        action_dim=cfg.action_dim,
        vision_key=cfg.vision_key,
        vision_type=cfg.vision_type,
        image_size=cfg.image_size,
        use_tactile=cfg.use_tactile,
        tactile_key=cfg.tactile_key,
        tactile_size=(cfg.tactile_height, cfg.tactile_width),
        vision_dim=cfg.vision_dim,
        tactile_dim=cfg.tactile_dim,
        fusion_type=cfg.fusion_type,
        reg_vision=cfg.reg_vision,
        reg_tactile=cfg.reg_tactile,
    )


def load_lightning_ckpt(model: torch.nn.Module, ckpt_path: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            cleaned[k[len("model."):]] = v
        else:
            cleaned[k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model


def _numpy_obs_to_tensor(x: Union[np.ndarray, torch.Tensor], device: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.to(device=device)
    else:
        t = torch.from_numpy(x).to(device=device)

    if t.ndim == 5 and t.shape[-1] in (1, 3):
        t = t.permute(0, 1, 4, 2, 3).contiguous()
    elif t.ndim == 4 and t.shape[-1] in (1, 3):
        t = t.permute(0, 3, 1, 2).contiguous()

    return t.float()


def init_history_buffers(
    obs: Dict[str, np.ndarray],
    history_size: int,
    vision_key: str,
    use_tactile: bool = False,
    tactile_key: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    out = {}
    v = obs[vision_key]
    out[vision_key] = np.repeat(v[:, None], history_size, axis=1)

    if use_tactile:
        t = obs[tactile_key]
        out[tactile_key] = np.repeat(t[:, None], history_size, axis=1)

    return out


def update_history_buffers(
    history: Dict[str, np.ndarray],
    obs: Dict[str, np.ndarray],
    vision_key: str,
    use_tactile: bool = False,
    tactile_key: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    history[vision_key] = np.concatenate(
        [history[vision_key][:, 1:], obs[vision_key][:, None]],
        axis=1,
    )
    if use_tactile:
        history[tactile_key] = np.concatenate(
            [history[tactile_key][:, 1:], obs[tactile_key][:, None]],
            axis=1,
        )
    return history


class BatchedCEMPlanner:
    def __init__(self, model: WorldModel, cfg: PlannerConfig, device: str = "cuda") -> None:
        self.model = model
        self.cfg = cfg
        self.device = device

    @torch.no_grad()
    def plan(self, current_info: Dict[str, np.ndarray], goal_info: Dict[str, np.ndarray]):
        vision_hist = _numpy_obs_to_tensor(current_info[self.cfg.vision_key], self.device)
        goal_vision = _numpy_obs_to_tensor(goal_info[self.cfg.vision_key], self.device)

        current_t = {self.cfg.vision_key: vision_hist}
        goal_t = {self.cfg.vision_key: goal_vision}

        if self.cfg.use_tactile:
            current_t[self.cfg.tactile_key] = _numpy_obs_to_tensor(
                current_info[self.cfg.tactile_key], self.device
            )
            goal_t[self.cfg.tactile_key] = _numpy_obs_to_tensor(
                goal_info[self.cfg.tactile_key], self.device
            )

        b = vision_hist.shape[0]
        h = self.cfg.horizon
        a = self.cfg.action_dim
        s = self.cfg.candidates

        mean = torch.zeros(b, h, a, device=self.device)
        std = torch.ones(b, h, a, device=self.device)

        low = self.cfg.action_low
        high = self.cfg.action_high

        final_cost = None
        final_topk_idx = None
        final_topk_actions = None

        for _ in range(self.cfg.iterations):
            noise = torch.randn(b, s, h, a, device=self.device)
            samples = mean[:, None] + std[:, None] * noise
            samples = torch.clamp(samples, low, high)

            cost = self.model.get_cost(
                current_info=current_t,
                action_candidates=samples,
                goal_info=goal_t,
                sum_all_diffs=self.cfg.sum_all_diffs,
                discount=self.cfg.discount,
            )  # [B, S]

            topk_idx = torch.topk(cost, k=self.cfg.topk, dim=1, largest=False).indices
            topk_actions = torch.gather(
                samples,
                dim=1,
                index=topk_idx[:, :, None, None].expand(-1, -1, h, a),
            )

            mean = topk_actions.mean(dim=1)
            std = topk_actions.std(dim=1, unbiased=False) + 1e-4

            final_cost = cost
            final_topk_idx = topk_idx
            final_topk_actions = topk_actions

        best_action_seq = final_topk_actions[:, 0]
        best_action = best_action_seq[:, 0]
        best_cost = torch.gather(final_cost, 1, final_topk_idx[:, :1]).squeeze(1)

        return (
            best_action.detach().cpu().numpy().astype(np.float32),
            best_action_seq.detach().cpu().numpy().astype(np.float32),
            best_cost.detach().cpu().numpy().astype(np.float32),
        )