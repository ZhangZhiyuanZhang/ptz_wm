from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

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
        "init_ee_pos": dataset.get_col_data("ee_pos")[init_rows],
        "init_ee_quat": dataset.get_col_data("ee_quat")[init_rows],
        "goal_plug_pos": dataset.get_col_data("plug_pos")[goal_rows],
        "goal_plug_quat": dataset.get_col_data("plug_quat")[goal_rows],
        "goal_socket_pos": dataset.get_col_data("socket_pos_gt")[goal_rows],
        "goal_socket_quat": dataset.get_col_data("socket_quat")[goal_rows],
        "goal_ee_pos": dataset.get_col_data("ee_pos")[goal_rows],
        "goal_ee_quat": dataset.get_col_data("ee_quat")[goal_rows],
    }

    result[vision_key] = dataset.get_col_data(vision_key)[goal_rows]

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
        "ee_pos": {"type": "low_dim"},
        "ee_quat": {"type": "low_dim"},
    }

    obs_meta["front"] = {"type": "rgb"}

    obs_meta[args.vision_key] = {"type": "rgb"}

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


def end_pose_metrics_joint(
    current_plug_pos: np.ndarray,
    current_plug_quat: np.ndarray,
    goal_plug_pos: np.ndarray,
    goal_plug_quat: np.ndarray,
    current_ee_pos: np.ndarray,
    current_ee_quat: np.ndarray,
    goal_ee_pos: np.ndarray,
    goal_ee_quat: np.ndarray,
    plug_pos_thresh: float,
    plug_quat_thresh_deg: float,
    ee_pos_thresh: float,
    ee_quat_thresh_deg: float,
):
    plug_pos_err = np.linalg.norm(current_plug_pos - goal_plug_pos, axis=-1)
    plug_quat_err_deg = np_quat_angle_deg(current_plug_quat, goal_plug_quat)

    ee_pos_err = np.linalg.norm(current_ee_pos - goal_ee_pos, axis=-1)
    ee_quat_err_deg = np_quat_angle_deg(current_ee_quat, goal_ee_quat)

    plug_success = (
        (plug_pos_err < plug_pos_thresh) &
        (plug_quat_err_deg < plug_quat_thresh_deg)
    )

    ee_success = (
        (ee_pos_err < ee_pos_thresh) &
        (ee_quat_err_deg < ee_quat_thresh_deg)
    )

    joint_success = plug_success & ee_success

    return {
        "success": joint_success.astype(np.int32),  # joint success as primary
        "plug_success": plug_success.astype(np.int32),
        "ee_success": ee_success.astype(np.int32),
        "plug_pos_err": plug_pos_err,
        "plug_quat_err_deg": plug_quat_err_deg,
        "ee_pos_err": ee_pos_err,
        "ee_quat_err_deg": ee_quat_err_deg,
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
        raise AttributeError("reset_to_dataset_state(...) not found on env/wrapper.")


def build_goal_info_numpy(batch: dict, args) -> dict:
    goal_info = {}
    goal_info[args.vision_key] = batch[args.vision_key][:, None]

    return goal_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--isaacgym-cfg-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vision-key", type=str, default="front")

    parser.add_argument("--num-envs", type=int, default=50)
    parser.add_argument("--num-record", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=30)

    parser.add_argument("--history-size", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--goal-offset-steps", type=int, default=20)

    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=4)

    # Action sampling prior for CEM.
    parser.add_argument(
        "--use-action-prior",
        action="store_true",
    )
    parser.set_defaults(use_action_prior=True)

    parser.add_argument("--action-prior-std-scale", type=float, default=1.0)
    parser.add_argument("--action-prior-min-std", type=float, default=0.05)
    parser.add_argument("--warm-start-mode", type=str, default="prev_action", choices=["none", "prev_action"])
    parser.add_argument("--warm-start-std", type=float, default=0.15)
    parser.add_argument("--warm-start-mix", type=float, default=0.5)
    parser.add_argument("--cem-min-std", type=float, default=0.03)
    parser.add_argument("--action-smooth-weight", type=float, default=0.05)
    parser.add_argument("--action-magnitude-weight", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--crf", type=int, default=22)

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--vision-dim", type=int, default=512)

    parser.add_argument("--sum-all-diffs", action="store_true")
    parser.add_argument("--discount", type=float, default=1.0)

    parser.add_argument("--pos-thresh", type=float, default=0.01)
    parser.add_argument("--quat-thresh-deg", type=float, default=15.0)
    parser.add_argument("--ee-pos-thresh", type=float, default=0.01)
    parser.add_argument("--ee-quat-thresh-deg", type=float, default=15.0)

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

    parser.add_argument("--encoder-type", type=str, default="dino", choices=["impala", "dino"])
    parser.add_argument("--predictor-type", type=str, default="vit", choices=["rnn", "vit"])

    parser.add_argument("--stop-on-success", action="store_true")

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
        "ee_pos",
        "ee_quat",
    ]
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
            "ee_pos",
            "ee_quat",
        ],
    )
    action_dim = dataset.get_dim("action")

    # Dataset-level action prior: makes CEM samples closer to the action distribution
    # seen by the predictor during GT-action training. This is not an oracle future
    # action sequence; it only uses marginal action statistics.
    all_actions = np.asarray(dataset.get_col_data("action"), dtype=np.float32)
    action_mean = all_actions.mean(axis=0).astype(np.float32)
    action_std = (all_actions.std(axis=0) + 1e-6).astype(np.float32)
    print("Dataset action mean:", action_mean)
    print("Dataset action std:", action_std)

    planner_cfg = PlannerConfig(
        history_size=args.history_size,
        horizon=args.horizon,
        candidates=args.candidates,
        topk=args.topk,
        iterations=args.iterations,
        action_dim=action_dim,
        use_action_prior=args.use_action_prior,
        action_prior_std_scale=args.action_prior_std_scale,
        action_prior_min_std=args.action_prior_min_std,
        warm_start_mode=args.warm_start_mode,
        warm_start_std=args.warm_start_std,
        warm_start_mix=args.warm_start_mix,
        cem_min_std=args.cem_min_std,
        action_smooth_weight=args.action_smooth_weight,
        action_magnitude_weight=args.action_magnitude_weight,
        vision_key=args.vision_key,
        image_size=args.image_size,
        sum_all_diffs=args.sum_all_diffs,
        discount=args.discount,
        vision_dim=args.vision_dim,
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
        encoder_type=args.encoder_type,
        predictor_type=args.predictor_type,
    )

    model = build_model(planner_cfg)
    model = load_lightning_ckpt(model, args.ckpt_path)
    model = model.to(device).eval()
    model.requires_grad_(False)

    planner = BatchedCEMPlanner(
        model=model,
        cfg=planner_cfg,
        device=device,
        action_mean=action_mean if args.use_action_prior else None,
        action_std=action_std if args.use_action_prior else None,
    )

    batch = sample_init_goal_segments(
        dataset=dataset,
        num_goals=args.num_envs,
        goal_offset_steps=args.goal_offset_steps,
        seed=args.seed,
        vision_key=args.vision_key,
    )

    goal_info = build_goal_info_numpy(batch, args)

    goal_plug_pos = np.asarray(batch["goal_plug_pos"], dtype=np.float32)
    goal_plug_quat = np.asarray(batch["goal_plug_quat"], dtype=np.float32)
    goal_ee_pos = np.asarray(batch["goal_ee_pos"], dtype=np.float32)
    goal_ee_quat = np.asarray(batch["goal_ee_quat"], dtype=np.float32)

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
    )

    first_success_step = np.full(args.num_envs, fill_value=-1, dtype=np.int32)
    done_mask = np.zeros(args.num_envs, dtype=bool)

    plug_done_mask = np.zeros(args.num_envs, dtype=bool)
    ee_done_mask = np.zeros(args.num_envs, dtype=bool)

    metrics_over_time = []
    last_cost = np.zeros(args.num_envs, dtype=np.float32)

    # Previous action is known in real closed-loop deployment because it is the
    # action we just executed. First step has no previous action, so use zero.
    prev_action = np.zeros((args.num_envs, action_dim), dtype=np.float32)

    for step_idx in range(args.max_steps):
        current_info = history

        plan_out = planner.plan(
            current_info=current_info,
            goal_info=goal_info,
            prev_action=prev_action,
        )

        if isinstance(plan_out, tuple):
            action_np = np.asarray(plan_out[0], dtype=np.float32)
            if len(plan_out) > 2 and plan_out[2] is not None:
                last_cost = np.asarray(plan_out[2], dtype=np.float32)
        else:
            action_np = np.asarray(plan_out, dtype=np.float32)

        if args.stop_on_success:
            action_np = action_np.copy()
            action_np[done_mask] = 0.0

        obs, _, _, _ = env.step(action_np)
        prev_action = action_np.copy()

        history = update_history_buffers(
            history=history,
            obs=obs,
            vision_key=args.vision_key,
        )

        m = end_pose_metrics_joint(
            current_plug_pos=np.asarray(obs["plug_pos"], dtype=np.float32),
            current_plug_quat=np.asarray(obs["plug_quat"], dtype=np.float32),
            goal_plug_pos=goal_plug_pos,
            goal_plug_quat=goal_plug_quat,
            current_ee_pos=np.asarray(obs["ee_pos"], dtype=np.float32),
            current_ee_quat=np.asarray(obs["ee_quat"], dtype=np.float32),
            goal_ee_pos=goal_ee_pos,
            goal_ee_quat=goal_ee_quat,
            plug_pos_thresh=args.pos_thresh,
            plug_quat_thresh_deg=args.quat_thresh_deg,
            ee_pos_thresh=args.ee_pos_thresh,
            ee_quat_thresh_deg=args.ee_quat_thresh_deg,
        )
        metrics_over_time.append(m)

        success_now = m["success"].astype(bool)
        plug_success_now = m["plug_success"].astype(bool)
        ee_success_now = m["ee_success"].astype(bool)

        newly_success = (~done_mask) & success_now
        first_success_step[newly_success] = step_idx + 1

        done_mask = done_mask | success_now
        plug_done_mask = plug_done_mask | plug_success_now
        ee_done_mask = ee_done_mask | ee_success_now

        print(
            f"[step {step_idx + 1:03d}/{args.max_steps}] "
            f"joint_success_now={success_now.mean():.4f} "
            f"joint_ever_success={done_mask.mean():.4f} "
            f"plug_ever_success={plug_done_mask.mean():.4f} "
            f"ee_ever_success={ee_done_mask.mean():.4f} "
            f"mean_plug_pos_err={m['plug_pos_err'].mean():.6f} "
            f"mean_plug_quat_err_deg={m['plug_quat_err_deg'].mean():.6f} "
            f"mean_ee_pos_err={m['ee_pos_err'].mean():.6f} "
            f"mean_ee_quat_err_deg={m['ee_quat_err_deg'].mean():.6f}"
        )

        if done_mask.all():
            break

    final_metrics = metrics_over_time[-1] if len(metrics_over_time) > 0 else {
        "success": np.zeros(args.num_envs, dtype=np.int32),
        "plug_success": np.zeros(args.num_envs, dtype=np.int32),
        "ee_success": np.zeros(args.num_envs, dtype=np.int32),
        "plug_pos_err": np.full(args.num_envs, np.nan, dtype=np.float32),
        "plug_quat_err_deg": np.full(args.num_envs, np.nan, dtype=np.float32),
        "ee_pos_err": np.full(args.num_envs, np.nan, dtype=np.float32),
        "ee_quat_err_deg": np.full(args.num_envs, np.nan, dtype=np.float32),
    }

    summary = {
        "num_envs": int(args.num_envs),
        "max_steps": int(args.max_steps),
        "num_executed_steps": int(len(metrics_over_time)),

        "num_final_plug_success": int(final_metrics["plug_success"].sum()),
        "num_final_ee_success": int(final_metrics["ee_success"].sum()),
        "num_final_joint_success": int(final_metrics["success"].sum()),

        "mean_plug_pos_err": float(np.mean(final_metrics["plug_pos_err"])),
        "mean_plug_quat_err_deg": float(np.mean(final_metrics["plug_quat_err_deg"])),

        "mean_ee_pos_err": float(np.mean(final_metrics["ee_pos_err"])),
        "mean_ee_quat_err_deg": float(np.mean(final_metrics["ee_quat_err_deg"])),

        "last_cost_mean": float(np.mean(last_cost)) if last_cost is not None else None,
        "last_cost_std": float(np.std(last_cost)) if last_cost is not None else None,
    }

    print("\n===== Final Summary =====")
    print(json.dumps(summary, indent=2))

    with open(args.output_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()