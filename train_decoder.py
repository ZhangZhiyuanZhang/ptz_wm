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
from train import build_model, JointTrainingModule


torch.set_float32_matmul_precision("high")


class DINOGridImageDecoder(nn.Module):
    """
    Input:
        z: [B,T,P,D]
    Output:
        img: [B,T,3,H,W]
    """
    def __init__(self, in_dim: int = 384, image_size: int = 224, out_channels: int = 3):
        super().__init__()
        self.image_size = image_size

        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 512, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
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
        z = z.permute(0, 3, 1, 2).contiguous()

        img = self.net(z)
        img = img.reshape(B, T, 3, self.image_size, self.image_size)
        return img


class WMDecoderModule(pl.LightningModule):
    def __init__(
        self,
        wm_args,
        action_dim: int,
        wm_ckpt_path: str,
        vision_key: str,
        image_size: int = 224,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pred_loss_weight: float = 1.0,
        enc_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["wm_args"])

        self.vision_key = vision_key
        self.image_size = image_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.pred_loss_weight = pred_loss_weight
        self.enc_loss_weight = enc_loss_weight

        # Build same WM architecture as train.py
        wm = build_model(wm_args, action_dim)

        # Load Lightning checkpoint from JointTrainingModule
        lit_wm = JointTrainingModule(
            model=wm,
            lr=wm_args.lr,
            weight_decay=wm_args.weight_decay,
            nsteps=wm_args.nsteps,
        )

        ckpt = torch.load(wm_ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        missing, unexpected = lit_wm.load_state_dict(state_dict, strict=False)

        print(f"[INFO] Loaded WM checkpoint: {wm_ckpt_path}")
        print(f"[INFO] Missing keys: {len(missing)}")
        print(f"[INFO] Unexpected keys: {len(unexpected)}")

        self.wm = lit_wm.model
        self.wm.eval()
        self.wm.requires_grad_(False)

        assert self.wm.latent_type == "grid", \
            "This decoder script is for DINO grid latent only. For Impala vector latent, use a different decoder."

        self.decoder = DINOGridImageDecoder(
            in_dim=self.wm.vision_dim,
            image_size=image_size,
            out_channels=3,
        )

    def preprocess_target_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Supports:
            [B,T,H,W,C]
            [B,T,C,H,W]
            [B,T,H,W]

        Returns:
            target: [B,T,3,H,W], range [0,1]
        """
        x = x.float()

        if x.max() > 1.5:
            x = x / 255.0

        if x.ndim == 4:
            x = x.unsqueeze(2)  # [B,T,1,H,W]

        if x.ndim != 5:
            raise ValueError(f"Expected ndim=5 image tensor, got {tuple(x.shape)}")

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
        return target

    def forward(self, batch):
        target = self.preprocess_target_image(batch[self.vision_key])

        wm_batch = {
            self.vision_key: batch[self.vision_key],
            "action": batch["action"],
        }

        with torch.no_grad():
            # z_enc: [B,T,P,D]
            z_enc = self.wm.encode(wm_batch)

            # Teacher-forcing 1-step prediction:
            # z_in:    [B,T-1,P,D]
            # a_in:    [B,T-1,A]
            # z_pred:  [B,T-1,P,D]
            z_in = z_enc[:, :-1]
            a_in = batch["action"].float()[:, :-1]
            z_pred = self.wm.predict_sequence(z_in, a_in)

        # Decode encoder latent for all frames
        recon_enc = self.decoder(z_enc)

        # Decode predicted latent for frames 1:T
        recon_pred = self.decoder(z_pred)

        return recon_enc, recon_pred, target

    def _step(self, batch, stage: str):
        recon_enc, recon_pred, target = self.forward(batch)

        # encoder latent reconstructs image 0:T
        loss_enc_l1 = F.l1_loss(recon_enc, target)
        loss_enc_mse = F.mse_loss(recon_enc, target)
        loss_enc = loss_enc_l1 + 0.1 * loss_enc_mse

        # predictor latent predicts image 1:T
        target_future = target[:, 1:]
        loss_pred_l1 = F.l1_loss(recon_pred, target_future)
        loss_pred_mse = F.mse_loss(recon_pred, target_future)
        loss_pred = loss_pred_l1 + 0.1 * loss_pred_mse

        loss = self.enc_loss_weight * loss_enc + self.pred_loss_weight * loss_pred

        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/loss_enc", loss_enc, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/loss_pred", loss_pred, prog_bar=True, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/enc_l1", loss_enc_l1, prog_bar=False, on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/pred_l1", loss_pred_l1, prog_bar=False, on_step=(stage == "train"), on_epoch=True)

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

    recon_enc, recon_pred, target = model(batch)

    target0 = target[:, 0].detach().cpu().clamp(0, 1)
    recon0 = recon_enc[:, 0].detach().cpu().clamp(0, 1)

    target1 = target[:, 1].detach().cpu().clamp(0, 1)
    pred1 = recon_pred[:, 0].detach().cpu().clamp(0, 1)

    n = min(max_items, target.shape[0])
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))

    if n == 1:
        axes = axes[None]

    for i in range(n):
        axes[i, 0].imshow(target0[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Target t")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(recon0[i].permute(1, 2, 0).numpy())
        axes[i, 1].set_title("Recon enc t")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(target1[i].permute(1, 2, 0).numpy())
        axes[i, 2].set_title("Target t+1")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(pred1[i].permute(1, 2, 0).numpy())
        axes[i, 3].set_title("Recon pred t+1")
        axes[i, 3].axis("off")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[INFO] Saved preview to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--vision-key", type=str, default="wrist")
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    # Decoder
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--enc-loss-weight", type=float, default=1.0)
    parser.add_argument("--pred-loss-weight", type=float, default=1.0)

    # WM checkpoint
    parser.add_argument("--wm-ckpt-path", type=str, required=True)

    # These args must match your WM training args
    parser.add_argument("--dino-name", type=str, default="dinov2_vits14")
    parser.add_argument("--encoder-type", type=str, default="dino", choices=["dino"])
    parser.add_argument("--predictor-type", type=str, default="vit", choices=["vit"])
    parser.add_argument("--vision-dim", type=int, default=512)

    parser.add_argument("--nsteps", type=int, default=2)
    parser.add_argument("--pred-depth", type=int, default=6)
    parser.add_argument("--pred-heads", type=int, default=6)
    parser.add_argument("--pred-embed-dim", type=int, default=384)
    parser.add_argument("--pred-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--eq-weight", type=float, default=0.0)
    parser.add_argument("--backward-weight", type=float, default=0.0)

    # Dummy regularizer args required by build_model / JointTrainingModule compatibility
    parser.add_argument("--reg-loss-type", type=str, default="vc", choices=["vc", "sigreg"])
    parser.add_argument("--use-proj", action="store_true")
    parser.add_argument("--cov-coeff", type=float, default=1.0)
    parser.add_argument("--std-coeff", type=float, default=1.0)
    parser.add_argument("--sigreg-coeff", type=float, default=0.1)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--sim-coeff-t", type=float, default=0.1)
    parser.add_argument("--idm-coeff", type=float, default=0.1)
    parser.add_argument("--idm-after-proj", action="store_true")
    parser.add_argument("--sim-t-after-proj", action="store_true")

    # Saving
    parser.add_argument("--save-dir", type=str, default="logs/decoder_ckpts")
    parser.add_argument("--ckpt-every-n-steps", type=int, default=5000)

    # Preview
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--decoder-ckpt-path", type=str, default=None)
    parser.add_argument("--preview-path", type=str, default="decoder_preview/recon.png")

    # Wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="ptz-wm-decoder")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-save-dir", type=str, default="logs/wandb")
    parser.add_argument("--wandb-name", type=str, default=None)

    return parser.parse_args()


def build_dataloaders(args):
    dataset = ZarrDataset(
        root=args.data_root,
        frameskip=args.frameskip,
        num_steps=args.num_steps,
        keys_to_load=[args.vision_key, "action"],
        keys_to_cache=["action"],
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

    return dataset, train_loader, val_loader


def main():
    args = parse_args()

    assert args.num_steps >= 2, "num_steps must be >= 2 because predictor reconstruction uses t -> t+1."

    dataset, train_loader, val_loader = build_dataloaders(args)
    action_dim = dataset.get_dim("action")

    model = WMDecoderModule(
        wm_args=args,
        action_dim=action_dim,
        wm_ckpt_path=args.wm_ckpt_path,
        vision_key=args.vision_key,
        image_size=args.image_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pred_loss_weight=args.pred_loss_weight,
        enc_loss_weight=args.enc_loss_weight,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.decoder_ckpt_path is not None:
        ckpt = torch.load(args.decoder_ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded decoder checkpoint: {args.decoder_ckpt_path}")

    if args.preview_only:
        model = model.to(device)
        save_reconstruction_preview(
            model=model,
            loader=val_loader,
            device=device,
            save_path=Path(args.preview_path),
        )
        return

    exp_name = f"{args.vision_key}_wm_decoder"

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
        print(f"[INFO] Resuming decoder from {last_ckpt}")
        ckpt_path = last_ckpt
    else:
        print("[INFO] No decoder checkpoint found, training from scratch")
        ckpt_path = None

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()