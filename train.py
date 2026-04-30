import argparse

import lightning as pl
import torch
torch.set_float32_matmul_precision("high")
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
import os

from architectures import (
    ImpalaEncoder,
    DinoGridEncoder,
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
    DINOViTPredictor,
)

from dataset.zarr_dataset import ZarrDataset
from jepa import WorldModel
from losses import (
    SquareLossSeq,
    VC_IDM_Sim_Regularizer,
    SIGReg,
    SIGReg_IDM_Sim_Regularizer,
)


class JointTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model: WorldModel,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        nsteps: int = 2,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.nsteps = nsteps

    def _forward_loss(self, batch):
        return self.model.training_losses(batch, nsteps=self.nsteps)

    def training_step(self, batch, batch_idx):
        out = self._forward_loss(batch)
        self.log("train/loss", out["loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/pred_loss", out["pred_loss"], prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/reg_loss", out["reg_loss"], prog_bar=True, on_step=True, on_epoch=True)
        for k, v in out["reg_dict"].items():
            self.log(f"train/{k}", v, prog_bar=False, on_step=True, on_epoch=True)
        
        if "eq_loss" in out:
            self.log("train/eq_loss", out["eq_loss"], prog_bar=True, on_step=True, on_epoch=True)
        
        if "backward_loss" in out:
            self.log("train/backward_loss", out["backward_loss"], prog_bar=True, on_step=True, on_epoch=True)

        return out["loss"]

    def validation_step(self, batch, batch_idx):
        out = self._forward_loss(batch)
        self.log("val/loss", out["loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/pred_loss", out["pred_loss"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/reg_loss", out["reg_loss"], prog_bar=True, on_step=False, on_epoch=True)
        for k, v in out["reg_dict"].items():
            self.log(f"val/{k}", v, prog_bar=False, on_step=False, on_epoch=True)
        
        if "eq_loss" in out:
            self.log("val/eq_loss", out["eq_loss"], prog_bar=True, on_step=False, on_epoch=True)

        if "backward_loss" in out:
            self.log("val/backward_loss", out["backward_loss"], prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)


def build_dataloaders(args):
    keys_to_load = ["action", args.vision_key]

    dataset = ZarrDataset(
        root=args.data_root,
        frameskip=args.frameskip,
        num_steps=args.num_steps,
        keys_to_load=keys_to_load,
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--vision-key", type=str, default="wrist")

    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=5)

    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-seed", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=560)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--vision-dim", type=int, default=512)

    parser.add_argument("--nsteps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Reg choice
    parser.add_argument("--reg-loss-type", type=str, default="vc", choices=["vc", "sigreg"])
    parser.add_argument("--use-proj", action="store_true")

    # VC loss
    parser.add_argument("--cov-coeff", type=float, default=1.0)
    parser.add_argument("--std-coeff", type=float, default=1.0)

    # SIGReg loss
    parser.add_argument("--sigreg-coeff", type=float, default=0.1)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)

    # Shared extras
    parser.add_argument("--sim-coeff-t", type=float, default=0.1)
    parser.add_argument("--idm-coeff", type=float, default=0.1)
    parser.add_argument("--idm-after-proj", action="store_true")
    parser.add_argument("--sim-t-after-proj", action="store_true")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="ptz-wm")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-save-dir", type=str, default="logs/wandb")
    parser.add_argument("--wandb-name", type=str, default=None)

    parser.add_argument("--save-dir", type=str, default="logs/ckpts")
    parser.add_argument("--ckpt-every-n-steps", type=int, default=10000)

    parser.add_argument("--dino-name", type=str, default="dinov2_vits14")
    parser.add_argument("--pred-depth", type=int, default=6)
    parser.add_argument("--pred-heads", type=int, default=6)
    parser.add_argument("--pred-embed-dim", type=int, default=384)
    parser.add_argument("--pred-mlp-ratio", type=float, default=4.0)

    parser.add_argument("--encoder-type", type=str, default="dino", choices=["impala", "dino"])
    parser.add_argument("--predictor-type", type=str, default="vit", choices=["rnn", "vit"])
    parser.add_argument("--eq-weight", type=float, default=0.0)
    parser.add_argument("--backward-weight", type=float, default=0.0)
    
    return parser.parse_args()

def build_regularizer(reg_hidden_dim: int, action_dim: int, args):
    projector = None
    if args.use_proj:
        projector = Projector(f"{reg_hidden_dim}-{reg_hidden_dim*4}-{reg_hidden_dim*4}")

    idm_in_dim = projector.out_dim if projector is not None and args.idm_after_proj else reg_hidden_dim
    idm = InverseDynamicsModel(
        state_dim=idm_in_dim,
        hidden_dim=256,
        action_dim=action_dim,
    )

    if args.reg_loss_type == "vc":
        return VC_IDM_Sim_Regularizer(
            cov_coeff=args.cov_coeff,
            std_coeff=args.std_coeff,
            sim_coeff_t=args.sim_coeff_t,
            idm_coeff=args.idm_coeff,
            idm=idm,
            projector=projector,
            spatial_as_samples=False,
            idm_after_proj=args.idm_after_proj,
            sim_t_after_proj=args.sim_t_after_proj,
        )

    if args.reg_loss_type == "sigreg":
        sigreg = SIGReg(
            knots=args.sigreg_knots,
            num_proj=args.sigreg_num_proj,
        )
        return SIGReg_IDM_Sim_Regularizer(
            sigreg_coeff=args.sigreg_coeff,
            sim_coeff_t=args.sim_coeff_t,
            idm_coeff=args.idm_coeff,
            sigreg=sigreg,
            idm=idm,
            projector=projector,
            idm_after_proj=args.idm_after_proj,
            sim_t_after_proj=args.sim_t_after_proj,
        )

    raise ValueError(f"Unknown reg_loss_type: {args.reg_loss_type}")


def build_model(args, action_dim: int) -> WorldModel:
    if args.encoder_type == "impala":
        encoder = ImpalaEncoder(
            input_channels=3,
            input_shape=(3, args.image_size, args.image_size),
            mlp_output_dim=args.vision_dim,
            final_ln=True,
        )

        predictor = RNNPredictor(
            hidden_size=args.vision_dim,
            action_dim=action_dim,
            num_layers=1,
            final_ln=nn.LayerNorm(args.vision_dim),
        )

        return WorldModel(
            encoder=encoder,
            predictor=predictor,
            regularizer=build_regularizer(args.vision_dim, action_dim, args),
            predcost=SquareLossSeq(),
            action_dim=action_dim,
            vision_key=args.vision_key,
            image_size=args.image_size,
            vision_dim=args.vision_dim,
            latent_type="vector",
            grid_size=None,
            eq_weight=0.0,
        )

    if args.encoder_type == "dino":
        encoder = DinoGridEncoder(
            name=args.dino_name,
            feature_key="x_norm_patchtokens",
            freeze=True,
            use_adapter=False,
        )

        grid_size = args.image_size // encoder.patch_size
        num_patches = grid_size * grid_size
        vision_dim = encoder.emb_dim

        if args.predictor_type == "vit":
            predictor = DINOViTPredictor(
                num_patches=num_patches,
                num_frames=args.num_steps,
                dim=vision_dim,
                action_dim=action_dim,
                depth=args.pred_depth,
                heads=args.pred_heads,
                mlp_dim=int(vision_dim * args.pred_mlp_ratio),
                dim_head=64,
                dropout=0.0,
            )

        return WorldModel(
            encoder=encoder,
            predictor=predictor,
            regularizer=None,
            predcost=SquareLossSeq(),
            action_dim=action_dim,
            vision_key=args.vision_key,
            image_size=args.image_size,
            vision_dim=vision_dim,
            latent_type="grid",
            grid_size=grid_size,
            eq_weight=args.eq_weight,
            backward_weight=args.backward_weight,
        )


def main():
    args = parse_args()

    if args.encoder_type == "impala":
        assert args.predictor_type == "rnn"

    if args.encoder_type == "dino":
        assert args.predictor_type == "vit"

    dataset, train_loader, val_loader = build_dataloaders(args)
    action_dim = dataset.get_dim("action")
    model = build_model(args, action_dim)

    lit_model = JointTrainingModule(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        nsteps=args.nsteps,
    )

    if args.eq_weight > 0.0:
        exp_name = f"{args.vision_key}_{args.encoder_type}_{args.predictor_type}_eq"
    else:
        exp_name = f"{args.vision_key}_{args.encoder_type}_{args.predictor_type}"

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

    trainer.fit(lit_model, train_loader, val_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()