from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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
    discrete_action=True,
):
    model.eval()
    step_errors = {k: [] for k in range(1, max_rollout + 1)}

    for batch in loader:
        batch = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        # DINO grid latent:
        # emb: [B, T, P, D]
        emb = model.encode(batch)
        act = batch["action"].float()  # [B, T, A]

        if action_mode == "gt":
            act_used = act
        elif action_mode == "random_uniform":
            if discrete_action:
                values = torch.tensor([-1.0, 0.0, 1.0], device=act.device, dtype=act.dtype)
                idx = torch.randint(0, values.numel(), act.shape, device=act.device)
                act_used = values[idx]
            else:
                act_used = 2.0 * torch.rand_like(act) - 1.0
        elif action_mode == "zero":
            act_used = torch.zeros_like(act)
        else:
            raise ValueError(f"Unknown action_mode: {action_mode}")

        B, T, A = act_used.shape
        assert T >= history_size + max_rollout, (
            f"T={T} too short for history_size={history_size}, "
            f"max_rollout={max_rollout}"
        )

        # z_hist: [B, history_size, P, D]
        z_hist = emb[:, :history_size].clone()

        preds = []
        for step in range(max_rollout):
            action_idx = history_size - 1 + step
            a_t = act_used[:, action_idx: action_idx + 1, :]  # [B,1,A]

            # predict next latent from last predicted latent
            # z_next: [B,1,P,D]
            z_pred_seq = model.predict_sequence(z_hist[:, -1:], a_t)
            z_next = z_pred_seq[:, -1:]

            preds.append(z_next)
            z_hist = torch.cat([z_hist, z_next], dim=1)

        pred_rollout = torch.cat(preds, dim=1)  # [B,max_rollout,P,D]
        target_rollout = emb[:, history_size:history_size + max_rollout]  # [B,max_rollout,P,D]

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
    discrete_action=True,
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
            discrete_action=discrete_action,
        )
    return out


def save_metrics_json(metrics: dict, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Saved metrics json to: {save_path}")


def plot_rollout_metrics(
    metrics: dict,
    save_path: Path,
    title: str = "Multi-step Rollout Error",
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    preferred_order = ["gt", "random_uniform", "zero"]
    mode_names = [m for m in preferred_order if m in metrics]
    mode_names += [m for m in metrics if m not in preferred_order]

    for mode in mode_names:
        step_dict = metrics[mode]
        steps = sorted(int(k.split("_")[0]) for k in step_dict.keys())
        values = [step_dict[f"{s}_step"] for s in steps]
        plt.plot(steps, values, marker="o", linewidth=2, label=mode)

    all_steps = sorted(
        {
            int(k.split("_")[0])
            for mode_dict in metrics.values()
            for k in mode_dict.keys()
        }
    )

    plt.xlabel("Rollout step")
    plt.ylabel("Grid latent MSE")
    plt.title(title)
    plt.xticks(all_steps)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved rollout plot to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)

    parser.add_argument("--vision-key", type=str, default="wrist")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--vision-dim", type=int, default=512)

    # DINO / AdaLN predictor args
    parser.add_argument("--dino-name", type=str, default="dinov2_vits14")
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--pred-depth", type=int, default=6)
    parser.add_argument("--pred-heads", type=int, default=6)
    parser.add_argument("--pred-embed-dim", type=int, default=384)
    parser.add_argument("--pred-mlp-ratio", type=float, default=4.0)

    # rollout eval args
    parser.add_argument("--history-size", type=int, default=1)
    parser.add_argument("--max-rollout", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument(
        "--action-mode",
        type=str,
        default="all",
        choices=["gt", "random_uniform", "zero", "all"],
    )

    # save / plot args
    parser.add_argument("--output-dir", type=str, default="eval_rollout_outputs")
    parser.add_argument("--plot-filename", type=str, default="rollout_plot.png")
    parser.add_argument("--json-filename", type=str, default="rollout_metrics.json")
    parser.add_argument("--plot-title", type=str, default="DINO-AdaLN Multi-step Rollout Error")

    parser.add_argument("--encoder-type", type=str, default="dino", choices=["impala", "dino"])
    parser.add_argument("--predictor-type", type=str, default="vit", choices=["rnn", "vit"])
    parser.add_argument("--discrete-action", action="store_true", default=True,
                        help="Use discrete random actions in {-1,0,1} for random_uniform rollout.")
    parser.add_argument("--continuous-action", dest="discrete_action", action="store_false",
                        help="Use continuous random actions in [-1,1] for random_uniform rollout.")

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_num_steps = max(args.history_size + args.max_rollout + 1, 8)

    if args.num_steps is None:
        model_num_steps = 5
    else:
        model_num_steps = args.num_steps
    
    keys_to_load = ["action", args.vision_key]

    dataset = ZarrDataset(
        root=args.data_root,
        frameskip=1,
        num_steps=data_num_steps,
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
        image_size=args.image_size,
        vision_dim=args.vision_dim,
        dino_name=args.dino_name,
        num_steps=model_num_steps,
        pred_depth=args.pred_depth,
        pred_heads=args.pred_heads,
        pred_embed_dim=args.pred_embed_dim,
        pred_mlp_ratio=args.pred_mlp_ratio,
        encoder_type=args.encoder_type,
        predictor_type=args.predictor_type,
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
            discrete_action=args.discrete_action,
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
                discrete_action=args.discrete_action,
            )
        }

    print(json.dumps(metrics, indent=2))

    save_metrics_json(metrics, output_dir / args.json_filename)
    plot_rollout_metrics(
        metrics,
        output_dir / args.plot_filename,
        title=args.plot_title,
    )


if __name__ == "__main__":
    main()