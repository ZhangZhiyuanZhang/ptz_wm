from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import hydra

# IMPORTANT: Isaac Gym must be imported before torch.
from envs.vistac_isaacgym_multiple_env_wrapper import MultipleIsaacEnvWrapper
from envs.video_recording_wrapper import VideoRecordingWrapper

import numpy as np
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from dataset.zarr_dataset import ZarrDataset
from planner_utils import (
    BatchedCEMPlanner,
    PlannerConfig,
    build_model,
    init_history_buffers,
    load_lightning_ckpt,
    update_history_buffers,
)


def is_rgb_like_key(key: str) -> bool:
    key = key.lower()
    rgb_tokens = ["image", "camera", "rgb", "taxim", "vision"]
    return any(tok in key for tok in rgb_tokens)


def sample_init_goal_segments(
    dataset: ZarrDataset,
    num_goals: int,
    goal_offset_steps: int,
    seed: int,
    vision_key: str,
    tactile_key: Optional[str] = None,
):
    rng = np.random.default_rng(seed)

    valid = []
    for ep_idx, ep_len in enumerate(dataset.lengths):
        max_start = int(ep_len - goal_offset_steps - 1)
        if max_start >= 0:
            for s in range(max_start + 1):
                valid.append((ep_idx, s, s + goal_offset_steps))

    if len(valid) == 0:
        raise RuntimeError("No valid (init, goal) segments found in dataset.")

    choice_idx = rng.choice(len(valid), size=num_goals, replace=len(valid) < num_goals)
    chosen = [valid[i] for i in choice_idx]

    ep_ids = np.array([x[0] for x in chosen], dtype=np.int64)
    start_steps = np.array([x[1] for x in chosen], dtype=np.int64)
    goal_steps = np.array([x[2] for x in chosen], dtype=np.int64)

    init_rows = dataset.offsets[ep_ids] + start_steps
    goal_rows = dataset.offsets[ep_ids] + goal_steps

    result = {
        "episode_idx": ep_ids,
        "start_step": start_steps,
        "goal_step": goal_steps,
        "init_row_idx": init_rows.astype(np.int64),
        "goal_row_idx": goal_rows.astype(np.int64),
        "init_dof_pos": dataset.get_col_data("dof_pos")[init_rows],
        "init_dof_vel": dataset.get_col_data("dof_vel")[init_rows],
        "init_plug_pos": dataset.get_col_data("plug_pos")[init_rows],
        "init_plug_quat": dataset.get_col_data("plug_quat")[init_rows],
        "init_socket_pos": dataset.get_col_data("socket_pos_gt")[init_rows],
        "init_socket_quat": dataset.get_col_data("socket_quat")[init_rows],
        "goal_plug_pos": dataset.get_col_data("plug_pos")[goal_rows],
        "goal_plug_quat": dataset.get_col_data("plug_quat")[goal_rows],
        "goal_socket_pos": dataset.get_col_data("socket_pos_gt")[goal_rows],
        "goal_socket_quat": dataset.get_col_data("socket_quat")[goal_rows],
    }

    result[vision_key] = dataset.get_col_data(vision_key)[goal_rows]

    if tactile_key is not None and tactile_key in dataset.column_names:
        result[tactile_key] = dataset.get_col_data(tactile_key)[goal_rows]

    return result


def make_env(args) -> VideoRecordingWrapper:
    if not GlobalHydra.instance().is_initialized():
        initialize(config_path="config", version_base="1.1")

    cfg = compose(config_name=args.isaacgym_cfg_name)
    cfg.num_envs = args.num_envs
    cfg.capture_video = True
    cfg.force_render = True

    obs_meta = {
        "plug_pos": {"type": "low_dim"},
        "plug_quat": {"type": "low_dim"},
        "socket_pos_gt": {"type": "low_dim"},
        "socket_quat": {"type": "low_dim"},
        "dof_pos": {"type": "low_dim"},
        "dof_vel": {"type": "low_dim"},
    }

    obs_meta["front"] = {"type": "rgb"}

    if args.vision_type == "image":
        obs_meta[args.vision_key] = {"type": "rgb"}
    else:
        obs_meta[args.vision_key] = {"type": "low_dim"}

    if args.use_tactile:
        if is_rgb_like_key(args.tactile_key):
            obs_meta[args.tactile_key] = {"type": "rgb"}
        else:
            obs_meta[args.tactile_key] = {"type": "low_dim"}

    cfg["shape_meta"] = OmegaConf.create({"obs": obs_meta})

    steps_per_render = max(10 // args.fps, 1)
    env = VideoRecordingWrapper(
        MultipleIsaacEnvWrapper(cfg),
        output_dir=str(args.output_dir),
        n_records=args.num_record,
        fps=args.fps,
        crf=args.crf,
        file_paths=None,
        steps_per_render=steps_per_render,
    )
    return env


def safe_env_seed(seed: int) -> int:
    if seed == 95:
        return 96
    return seed


def np_quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    dot = np.sum(q1 * q2, axis=-1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    ang_rad = 2.0 * np.arccos(dot)
    return np.rad2deg(ang_rad)


def end_pose_metrics_object(
    current_plug_pos: np.ndarray,
    current_plug_quat: np.ndarray,
    goal_plug_pos: np.ndarray,
    goal_plug_quat: np.ndarray,
    pos_thresh: float,
    quat_thresh_deg: float,
):
    pos_err = np.linalg.norm(current_plug_pos - goal_plug_pos, axis=-1)
    quat_err_deg = np_quat_angle_deg(current_plug_quat, goal_plug_quat)
    success = ((pos_err < pos_thresh) & (quat_err_deg < quat_thresh_deg)).astype(np.int32)

    return {
        "success": success,
        "pos_err": pos_err,
        "quat_err_deg": quat_err_deg,
    }


def reset_env_to_dataset_state(env: VideoRecordingWrapper, batch: dict):
    kwargs = dict(
        dof_pos=batch["init_dof_pos"],
        dof_vel=batch["init_dof_vel"],
        plug_pos=batch["init_plug_pos"],
        plug_quat=batch["init_plug_quat"],
        socket_pos=batch["init_socket_pos"],
        socket_quat=batch["init_socket_quat"],
    )

    if hasattr(env, "reset_to_dataset_state"):
        env.reset_to_dataset_state(**kwargs)
    elif hasattr(env, "env") and hasattr(env.env, "reset_to_dataset_state"):
        env.env.reset_to_dataset_state(**kwargs)
    else:
        raise AttributeError(
            "reset_to_dataset_state(...) not found on env/wrapper. "
            "Please add it to MultipleIsaacEnvWrapper or the underlying task env."
        )


def build_goal_info_numpy(batch: dict, args) -> dict:
    goal_info = {}
    goal_info[args.vision_key] = batch[args.vision_key][:, None]

    if args.use_tactile:
        goal_info[args.tactile_key] = batch[args.tactile_key][:, None]

    return goal_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--isaacgym-cfg-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vision-key", type=str, default="front")
    parser.add_argument("--vision-type", type=str, default="image", choices=["image", "pc"])

    parser.add_argument("--use-tactile", action="store_true")
    parser.add_argument("--tactile-key", type=str, default="tactile_force_field_right")

    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--num-record", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=30)

    parser.add_argument("--history-size", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--goal-offset-steps", type=int, default=20)

    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--crf", type=int, default=22)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--vision-dim", type=int, default=512)

    parser.add_argument("--pc-in-channels", type=int, default=3)

    parser.add_argument("--sum-all-diffs", action="store_true")
    parser.add_argument("--discount", type=float, default=1.0)

    parser.add_argument("--tactile-dim", type=int, default=512)

    parser.add_argument("--pos-thresh", type=float, default=0.01)
    parser.add_argument("--quat-thresh-deg", type=float, default=15.0)

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

    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    keys_to_load = [
        "action",
        args.vision_key,
        "dof_pos",
        "dof_vel",
        "plug_pos",
        "plug_quat",
        "socket_pos_gt",
        "socket_quat",
    ]
    if args.use_tactile:
        keys_to_load.append(args.tactile_key)

    dataset = ZarrDataset(
        root=args.data_root,
        frameskip=1,
        num_steps=1,
        keys_to_load=keys_to_load,
        keys_to_cache=[
            "action",
            "dof_pos",
            "dof_vel",
            "plug_pos",
            "plug_quat",
            "socket_pos_gt",
            "socket_quat",
        ],
    )
    action_dim = dataset.get_dim("action")

    planner_cfg = PlannerConfig(
        history_size=args.history_size,
        horizon=args.horizon,
        candidates=args.candidates,
        topk=args.topk,
        iterations=args.iterations,
        action_dim=action_dim,
        vision_key=args.vision_key,
        vision_type=args.vision_type,
        image_size=args.image_size,
        pc_in_channels=args.pc_in_channels,
        sum_all_diffs=args.sum_all_diffs,
        discount=args.discount,
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

    model = build_model(planner_cfg)
    model = load_lightning_ckpt(model, args.ckpt_path)
    model = model.to(device).eval()
    model.requires_grad_(False)

    planner = BatchedCEMPlanner(model=model, cfg=planner_cfg, device=device)

    batch = sample_init_goal_segments(
        dataset=dataset,
        num_goals=args.num_envs,
        goal_offset_steps=args.goal_offset_steps,
        seed=args.seed,
        vision_key=args.vision_key,
        tactile_key=args.tactile_key if args.use_tactile else None,
    )

    goal_info = build_goal_info_numpy(batch, args)

    goal_plug_pos = np.asarray(batch["goal_plug_pos"], dtype=np.float32)
    goal_plug_quat = np.asarray(batch["goal_plug_quat"], dtype=np.float32)

    env = make_env(args)
    if hasattr(env, "seed"):
        env.seed(safe_env_seed(args.seed))

    _ = env.reset()
    reset_env_to_dataset_state(env, batch)

    zero_action = np.zeros((args.num_envs, action_dim), dtype=np.float32)
    obs, _, _, _ = env.step(zero_action)

    history = init_history_buffers(
        obs=obs,
        history_size=args.history_size,
        vision_key=args.vision_key,
        use_tactile=args.use_tactile,
        tactile_key=args.tactile_key,
    )

    first_success_step = np.full(args.num_envs, fill_value=-1, dtype=np.int32)
    done_mask = np.zeros(args.num_envs, dtype=bool)
    metrics_over_time = []
    last_cost = np.zeros(args.num_envs, dtype=np.float32)

    for step_idx in range(args.max_steps):
        current_info = history

        plan_out = planner.plan(current_info=current_info, goal_info=goal_info)

        if isinstance(plan_out, tuple):
            action_np = np.asarray(plan_out[0], dtype=np.float32)
            if len(plan_out) > 2 and plan_out[2] is not None:
                last_cost = np.asarray(plan_out[2], dtype=np.float32)
        else:
            action_np = np.asarray(plan_out, dtype=np.float32)

        obs, _, _, _ = env.step(action_np)

        history = update_history_buffers(
            history=history,
            obs=obs,
            vision_key=args.vision_key,
            use_tactile=args.use_tactile,
            tactile_key=args.tactile_key,
        )

        m = end_pose_metrics_object(
            current_plug_pos=np.asarray(obs["plug_pos"], dtype=np.float32),
            current_plug_quat=np.asarray(obs["plug_quat"], dtype=np.float32),
            goal_plug_pos=goal_plug_pos,
            goal_plug_quat=goal_plug_quat,
            pos_thresh=args.pos_thresh,
            quat_thresh_deg=args.quat_thresh_deg,
        )
        metrics_over_time.append(m)

        success_now = m["success"].astype(bool)
        newly_success = (~done_mask) & success_now
        first_success_step[newly_success] = step_idx + 1
        done_mask = done_mask | success_now

        print(
            f"[step {step_idx + 1:03d}/{args.max_steps}] "
            f"success={success_now.mean():.4f} "
            f"done={done_mask.mean():.4f} "
            f"mean_pos_err={m['pos_err'].mean():.6f} "
            f"mean_quat_err_deg={m['quat_err_deg'].mean():.6f}"
        )

        if done_mask.all():
            break

    final_metrics = metrics_over_time[-1] if len(metrics_over_time) > 0 else {
        "success": np.zeros(args.num_envs, dtype=np.int32),
        "pos_err": np.full(args.num_envs, np.nan, dtype=np.float32),
        "quat_err_deg": np.full(args.num_envs, np.nan, dtype=np.float32),
    }

    success_float = final_metrics["success"].astype(np.float32)
    successful_steps = first_success_step[first_success_step > 0]

    summary = {
        "num_envs": int(args.num_envs),
        "max_steps": int(args.max_steps),
        "num_executed_steps": int(len(metrics_over_time)),
        "num_success": int(final_metrics["success"].sum()),
        "success_rate": float(np.mean(success_float)),
        "mean_pos_err": float(np.mean(final_metrics["pos_err"])),
        "std_pos_err": float(np.std(final_metrics["pos_err"])),
        "mean_quat_err_deg": float(np.mean(final_metrics["quat_err_deg"])),
        "std_quat_err_deg": float(np.std(final_metrics["quat_err_deg"])),
        "mean_first_success_step": float(np.mean(successful_steps)) if successful_steps.size > 0 else -1.0,
        "std_first_success_step": float(np.std(successful_steps)) if successful_steps.size > 0 else -1.0,
        "last_cost_mean": float(np.mean(last_cost)) if last_cost is not None else None,
        "last_cost_std": float(np.std(last_cost)) if last_cost is not None else None,
    }

    print("\n===== Final Summary =====")
    print(json.dumps(summary, indent=2))

    with open(args.output_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()