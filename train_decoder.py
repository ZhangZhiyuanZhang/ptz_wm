from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from dataset.zarr_dataset import ZarrDataset
from architectures import DinoGridEncoder


torch.set_float32_matmul_precision("high")


class DINOGridImageDecoder(nn.Module):
    """
    Decode DINO grid latent back to image.

    Input:
        z: [B,T,P,D]
    Output:
        img: [B,T,3,H,W], range [0,1]
    """
    def __init__(
        self,
        in_dim: int = 384,
        image_size: int = 224,
        out_channels: int = 3,
    ):
        super().__init__()
        self.image_size = image_size

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 512, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 16 -> 32
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 32 -> 64
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 64 -> 128
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Upsample(size=(image_size, image_size), mode="bilinear", align_corners=False),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, T, P, D = z.shape
        H = W = int(P ** 0.5)
        assert H * W == P, f"P must be square, got P={P}"

        z = z.reshape(B * T, H, W, D)
        z = z.permute(0, 3, 1, 2).contiguous()  # [B*T,D,H,W]

        img = self.net(z)  # [B*T,3,image_size,image_size]
        img = img.reshape(B, T, 3, self.image_size, self.image_size)
        return img


class DINOImageDecoderModule(pl.LightningModule):
    def __init__(
        self,
        dino_name: str,
        vision_key: str,
        image_size: int = 224,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vision_key = vision_key
        self.image_size = image_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.encoder = DinoGridEncoder(
            name=dino_name,
            feature_key="x_norm_patchtokens",
            freeze=True,
            use_adapter=False,
        )
        self.encoder.eval()
        self.encoder.requires_grad_(False)

        self.decoder = DINOGridImageDecoder(
            in_dim=self.encoder.emb_dim,
            image_size=image_size,
            out_channels=3,
        )

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("vision_mean", mean, persistent=False)
        self.register_buffer("vision_std", std, persistent=False)

    def preprocess_image(self, x: torch.Tensor):
        """
        Input supports:
            [B,T,H,W,C]
            [B,T,C,H,W]

        Returns:
            dino_input: [B,C,T,H,W], normalized
            target:     [B,T,3,H,W], range [0,1]
        """
        x = x.float()

        if x.max() > 1.5:
            x = x / 255.0

        if x.ndim != 5:
            raise ValueError(f"Expected image tensor with ndim=5, got {tuple(x.shape)}")

        # [B,T,H,W,C] -> [B,T,C,H,W]
        if x.shape[-1] in (1, 3):
            x = x.permute(0, 1, 4, 2, 3).contiguous()

        B, T, C, H, W = x.shape
        if C == 1:
            x = x.repeat(1, 1, 3, 1, 1)

        x_flat = x.reshape(B * T, 3, H, W)
        if H != self.image_size or W != self.image_size:
            x_flat = F.interpolate(
                x_flat,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        target = x_flat.reshape(B, T, 3, self.image_size, self.image_size)

        dino_x = (x_flat - self.vision_mean) / self.vision_std
        dino_x = dino_x.reshape(B, T, 3, self.image_size, self.image_size)
        dino_x = dino_x.permute(0, 2, 1, 3, 4).contiguous()  # [B,C,T,H,W]

        return dino_x, target

    def forward(self, batch):
        dino_x, target = self.preprocess_image(batch[self.vision_key])

        with torch.no_grad():
            z = self.encoder(dino_x)  # [B,T,P,D]

        recon = self.decoder(z)       # [B,T,3,H,W]
        return recon, target

    def _step(self, batch, stage: str):
        recon, target = self.forward(batch)

        l1 = F.l1_loss(recon, target)
        mse = F.mse_loss(recon, target)
        loss = l1 + 0.1 * mse

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/l1", l1, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/mse", mse, prog_bar=False, on_step=(stage == "train"), on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.decoder.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


@torch.no_grad()
def save_reconstruction_preview(model, loader, device, save_path: Path, max_items: int = 4):
    model.eval()
    batch = next(iter(loader))
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    recon, target = model(batch)

    recon = recon[:, 0].detach().cpu().clamp(0, 1)   # [B,3,H,W]
    target = target[:, 0].detach().cpu().clamp(0, 1)

    n = min(max_items, recon.shape[0])
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))

    if n == 1:
        axes = axes[None]

    for i in range(n):
        axes[i, 0].imshow(target[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Target")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(recon[i].permute(1, 2, 0).numpy())
        axes[i, 1].set_title("Reconstruction")
        axes[i, 1].axis("off")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved preview to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--vision-key", type=str, default="image")

    parser.add_argument("--dino-name", type=str, default="dinov2_vits14")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-steps", type=int, default=5)

    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--save-dir", type=str, default="logs/decoder_ckpts")
    parser.add_argument("--ckpt-every-n-steps", type=int, default=5000)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="ptz-wm-decoder")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-save-dir", type=str, default="logs/wandb")
    parser.add_argument("--wandb-name", type=str, default=None)

    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--preview-path", type=str, default="decoder_preview/recon.png")

    return parser.parse_args()


def build_dataloaders(args):
    dataset = ZarrDataset(
        root=args.data_root,
        frameskip=1,
        num_steps=args.num_steps,
        keys_to_load=[args.vision_key],
        keys_to_cache=[],
    )

    train_len = int(args.train_ratio * len(dataset))
    val_len = len(dataset) - train_len

    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.split_seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    return train_loader, val_loader


def main():
    args = parse_args()

    train_loader, val_loader = build_dataloaders(args)

    model = DINOImageDecoderModule(
        dino_name=args.dino_name,
        vision_key=args.vision_key,
        image_size=args.image_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded decoder checkpoint: {args.ckpt_path}")

    if args.preview_only:
        model = model.to(device)
        save_reconstruction_preview(
            model=model,
            loader=val_loader,
            device=device,
            save_path=Path(args.preview_path),
        )
        return

    exp_name = f"{args.vision_key}_{args.dino_name}_decoder"

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.save_dir}/{exp_name}",
        filename="{step:06d}",
        save_top_k=-1,
        save_last=True,
        every_n_train_steps=args.ckpt_every_n_steps,
        auto_insert_metric_name=False,
    )

    logger = None
    if args.wandb:
        run_name = args.wandb_name or exp_name
        logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            save_dir=args.wandb_save_dir,
            log_model=False,
        )
        logger.log_hyperparams(vars(args))

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
    )

    ckpt_dir = f"{args.save_dir}/{exp_name}"
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")

    if os.path.exists(last_ckpt):
        print(f"[INFO] Resuming from {last_ckpt}")
        ckpt_path = last_ckpt
    else:
        print("[INFO] No checkpoint found, training from scratch")
        ckpt_path = None

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()