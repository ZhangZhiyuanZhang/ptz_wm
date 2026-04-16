from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, random_split

from dataset.zarr_dataset import ZarrDataset
from planner_utils import PlannerConfig, build_model, load_lightning_ckpt


@torch.no_grad()
def compute_rollout_errors(
    model,
    loader,
    history_size=1,
    max_rollout=6,
    device="cuda",
    action_mode="gt",
):
    model.eval()
    step_errors = {k: [] for k in range(1, max_rollout + 1)}

    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        emb = model.encode(batch)
        act = batch["action"].float()  # [B, T, A]

        if action_mode == "gt":
            act_used = act
        elif action_mode == "random_uniform":
            act_used = 2.0 * torch.rand_like(act) - 1.0
        elif action_mode == "zero":
            act_used = torch.zeros_like(act)
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

        B, T, A = act_used.shape
        assert T >= history_size + max_rollout, (
            f"T={T} too short for history_size={history_size}, max_rollout={max_rollout}"
        )

        z_hist = emb[:, :history_size].clone()

        preds = []
        for step in range(max_rollout):
            a_t = act_used[:, step: step + 1, :]
            z_pred_seq = model.predict_sequence(
                z_hist[:, -1:],
                a_t,
            )
            z_next = z_pred_seq[:, -1:]
            preds.append(z_next)
            z_hist = torch.cat([z_hist, z_next], dim=1)

        pred_rollout = torch.cat(preds, dim=1)
        target_rollout = emb[:, history_size:history_size + max_rollout]

        for k in range(1, max_rollout + 1):
            mse_k = ((pred_rollout[:, :k] - target_rollout[:, :k]) ** 2).mean().item()
            step_errors[k].append(mse_k)

    return {f"{k}_step": sum(v) / len(v) for k, v in step_errors.items()}


@torch.no_grad()
def compute_rollout_errors_for_modes(
    model,
    loader,
    history_size=1,
    max_rollout=6,
    device="cuda",
    modes=None,
):
    if modes is None:
        modes = ["gt", "random_uniform", "zero"]

    out = {}
    for mode in modes:
        out[mode] = compute_rollout_errors(
            model=model,
            loader=loader,
            history_size=history_size,
            max_rollout=max_rollout,
            device=device,
            action_mode=mode,
        )
    return out


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)

    parser.add_argument("--vision-key", type=str, default="wrist")
    parser.add_argument("--vision-type", type=str, default="image", choices=["image", "pc"])

    parser.add_argument("--use-tactile", action="store_true")
    parser.add_argument("--tactile-key", type=str, default="left_tactile_camera_taxim")

    parser.add_argument("--history-size", type=int, default=1)
    parser.add_argument("--max-rollout", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--num-workers", type=int, default=6)

    parser.add_argument(
        "--action-mode",
        type=str,
        default="all",
        choices=["gt", "random_uniform", "zero", "all"],
    )

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--vision-dim", type=int, default=512)

    parser.add_argument("--pc-in-channels", type=int, default=3)

    parser.add_argument("--tactile-dim", type=int, default=512)
    parser.add_argument("--tactile-in-channels", type=int, default=3)
    parser.add_argument("--tactile-height", type=int, default=10)
    parser.add_argument("--tactile-width", type=int, default=14)

    parser.add_argument("--fusion-type", type=str, default="concat",
                        choices=["concat", "gate", "film", "attn"])
    parser.add_argument("--fusion-latent-dim", type=int, default=None)
    parser.add_argument("--fusion-hidden-dim", type=int, default=None)
    parser.add_argument("--attn-d-model", type=int, default=256)
    parser.add_argument("--attn-heads", type=int, default=4)
    parser.add_argument("--attn-layers", type=int, default=2)
    parser.add_argument("--attn-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--attn-dropout", type=float, default=0.0)

    parser.add_argument("--reg-loss-type", type=str, default="vc", choices=["vc", "sigreg"])
    parser.add_argument("--use-proj", action="store_true")

    parser.add_argument("--cov-coeff", type=float, default=1.0)
    parser.add_argument("--std-coeff", type=float, default=1.0)

    parser.add_argument("--sigreg-coeff", type=float, default=1.0)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)

    parser.add_argument("--sim-coeff-t", type=float, default=0.1)
    parser.add_argument("--idm-coeff", type=float, default=0.1)
    parser.add_argument("--idm-after-proj", action="store_true")
    parser.add_argument("--sim-t-after-proj", action="store_true")

    parser.add_argument("--reg-vision", action="store_true")
    parser.add_argument("--reg-tactile", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    keys_to_load = ["action", args.vision_key]
    if args.use_tactile:
        keys_to_load.append(args.tactile_key)

    dataset = ZarrDataset(
        root=args.data_root,
        frameskip=1,
        num_steps=max(args.history_size + args.max_rollout + 1, 8),
        keys_to_load=keys_to_load,
        keys_to_cache=["action"],
    )

    val_len = min(2048, len(dataset))
    train_len = len(dataset) - val_len
    _, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )

    loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    action_dim = dataset.get_dim("action")
    cfg = PlannerConfig(
        action_dim=action_dim,
        vision_key=args.vision_key,
        vision_type=args.vision_type,
        image_size=args.image_size,
        pc_in_channels=args.pc_in_channels,
        vision_dim=args.vision_dim,
        tactile_dim=args.tactile_dim,
        use_tactile=args.use_tactile,
        tactile_key=args.tactile_key,
        tactile_in_channels=args.tactile_in_channels,
        tactile_height=args.tactile_height,
        tactile_width=args.tactile_width,
        fusion_type=args.fusion_type,
        fusion_latent_dim=args.fusion_latent_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        attn_d_model=args.attn_d_model,
        attn_heads=args.attn_heads,
        attn_layers=args.attn_layers,
        attn_mlp_ratio=args.attn_mlp_ratio,
        attn_dropout=args.attn_dropout,
        reg_loss_type=args.reg_loss_type,
        use_proj=args.use_proj,
        cov_coeff=args.cov_coeff,
        std_coeff=args.std_coeff,
        sigreg_coeff=args.sigreg_coeff,
        sigreg_knots=args.sigreg_knots,
        sigreg_num_proj=args.sigreg_num_proj,
        sim_coeff_t=args.sim_coeff_t,
        idm_coeff=args.idm_coeff,
        idm_after_proj=args.idm_after_proj,
        sim_t_after_proj=args.sim_t_after_proj,
        reg_vision=args.reg_vision,
        reg_tactile=args.reg_tactile,
    )

    model = build_model(cfg)
    model = load_lightning_ckpt(model, args.ckpt_path)
    model = model.to(device).eval()
    model.requires_grad_(False)

    if args.action_mode == "all":
        metrics = compute_rollout_errors_for_modes(
            model=model,
            loader=loader,
            history_size=args.history_size,
            max_rollout=args.max_rollout,
            device=device,
            modes=["gt", "random_uniform", "zero"],
        )
    else:
        metrics = {
            args.action_mode: compute_rollout_errors(
                model=model,
                loader=loader,
                history_size=args.history_size,
                max_rollout=args.max_rollout,
                device=device,
                action_mode=args.action_mode,
            )
        }

    print(metrics)


if __name__ == "__main__":
    main()