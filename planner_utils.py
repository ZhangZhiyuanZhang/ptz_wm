from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from architectures import (
    ImpalaEncoder,
    DinoGridEncoder,
    InverseDynamicsModel,
    Projector,
    RNNPredictor,
    DINOViTPredictor,
)
from jepa import WorldModel
from losses import (
    SquareLossSeq,
    VC_IDM_Sim_Regularizer,
    SIGReg,
    SIGReg_IDM_Sim_Regularizer,
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

    # Make eval-time CEM action samples closer to the training/demo action distribution.
    # None of these use future GT action chunks.
    use_action_prior: bool = True
    action_prior_std_scale: float = 1.0
    action_prior_min_std: float = 0.05
    warm_start_mode: str = "prev_action"  # none | prev_action
    warm_start_std: float = 0.15
    warm_start_mix: float = 0.5  # 0: pure dataset mean, 1: pure previous action
    cem_min_std: float = 0.03
    action_smooth_weight: float = 0.05
    action_magnitude_weight: float = 0.01

    vision_key: str = "front"
    image_size: int = 224

    sum_all_diffs: bool = False
    discount: float = 1.0

    vision_dim: int = 512

    reg_loss_type: str = "vc"    # vc | sigreg
    use_proj: bool = False

    cov_coeff: float = 1.0
    std_coeff: float = 1.0

    sigreg_coeff: float = 0.1
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024

    sim_coeff_t: float = 0.1
    idm_coeff: float = 0.1
    idm_after_proj: bool = False
    sim_t_after_proj: bool = False

    dino_name: str = "dinov2_vits14"
    num_steps: int = 5
    pred_depth: int = 6
    pred_heads: int = 6
    pred_embed_dim: int = 384
    pred_mlp_ratio: float = 4.0
    eq_weight: float = 0.0

    encoder_type: str = "dino"
    predictor_type: str = "vit"

    # PTZ actions are discrete commands, usually each dim in {-1, 0, 1}.
    discrete_action: bool = True


def build_vision_encoder(cfg):
    return ImpalaEncoder(
        input_channels=3,
        input_shape=(3, cfg.image_size, cfg.image_size),
        mlp_output_dim=cfg.vision_dim,
        final_ln=True,
    )


def build_regularizer(reg_hidden_dim: int, action_dim: int, cfg: PlannerConfig):
    projector = None
    if cfg.use_proj:
        projector = Projector(f"{reg_hidden_dim}-{reg_hidden_dim*4}-{reg_hidden_dim*4}")

    idm_in_dim = projector.out_dim if projector is not None and cfg.idm_after_proj else reg_hidden_dim
    idm = InverseDynamicsModel(
        state_dim=idm_in_dim,
        hidden_dim=256,
        action_dim=action_dim,
    )

    if cfg.reg_loss_type == "vc":
        return VC_IDM_Sim_Regularizer(
            cov_coeff=cfg.cov_coeff,
            std_coeff=cfg.std_coeff,
            sim_coeff_t=cfg.sim_coeff_t,
            idm_coeff=cfg.idm_coeff,
            idm=idm,
            projector=projector,
            spatial_as_samples=False,
            idm_after_proj=cfg.idm_after_proj,
            sim_t_after_proj=cfg.sim_t_after_proj,
        )

    if cfg.reg_loss_type == "sigreg":
        sigreg = SIGReg(
            knots=cfg.sigreg_knots,
            num_proj=cfg.sigreg_num_proj,
        )
        return SIGReg_IDM_Sim_Regularizer(
            sigreg_coeff=cfg.sigreg_coeff,
            sim_coeff_t=cfg.sim_coeff_t,
            idm_coeff=cfg.idm_coeff,
            sigreg=sigreg,
            idm=idm,
            projector=projector,
            idm_after_proj=cfg.idm_after_proj,
            sim_t_after_proj=cfg.sim_t_after_proj,
        )

    raise ValueError(f"Unknown reg_loss_type: {cfg.reg_loss_type}")


def build_model(cfg: PlannerConfig) -> WorldModel:
    if cfg.encoder_type == "impala":
        encoder = ImpalaEncoder(
            input_channels=3,
            input_shape=(3, cfg.image_size, cfg.image_size),
            mlp_output_dim=cfg.vision_dim,
            final_ln=True,
        )

        predictor = RNNPredictor(
            hidden_size=cfg.vision_dim,
            action_dim=cfg.action_dim,
            num_layers=1,
            final_ln=nn.LayerNorm(cfg.vision_dim),
        )

        return WorldModel(
            encoder=encoder,
            predictor=predictor,
            regularizer=build_regularizer(cfg.vision_dim, cfg.action_dim, cfg),
            predcost=SquareLossSeq(),
            action_dim=cfg.action_dim,
            vision_key=cfg.vision_key,
            image_size=cfg.image_size,
            vision_dim=cfg.vision_dim,
            latent_type="vector",
            grid_size=None,
            eq_weight=0.0,
        )

    if cfg.encoder_type == "dino":
        encoder = DinoGridEncoder(
            name=cfg.dino_name,
            feature_key="x_norm_patchtokens",
            freeze=True,
            use_adapter=False,
        )

        grid_size = cfg.image_size // encoder.patch_size
        num_patches = grid_size * grid_size
        vision_dim = encoder.emb_dim

        if cfg.predictor_type == "vit":
            predictor = DINOViTPredictor(
                num_patches=num_patches,
                num_frames=cfg.num_steps,
                dim=vision_dim,
                action_dim=cfg.action_dim,
                depth=cfg.pred_depth,
                heads=cfg.pred_heads,
                mlp_dim=int(vision_dim * cfg.pred_mlp_ratio),
                dim_head=64,
                dropout=0.0,
            )

        return WorldModel(
            encoder=encoder,
            predictor=predictor,
            regularizer=None,
            predcost=SquareLossSeq(),
            action_dim=cfg.action_dim,
            vision_key=cfg.vision_key,
            image_size=cfg.image_size,
            vision_dim=vision_dim,
            latent_type="grid",
            grid_size=grid_size,
            eq_weight=cfg.eq_weight,
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
) -> Dict[str, np.ndarray]:
    out = {}
    v = obs[vision_key]
    out[vision_key] = np.repeat(v[:, None], history_size, axis=1)

    return out


def update_history_buffers(
    history: Dict[str, np.ndarray],
    obs: Dict[str, np.ndarray],
    vision_key: str,
) -> Dict[str, np.ndarray]:
    history[vision_key] = np.concatenate(
        [history[vision_key][:, 1:], obs[vision_key][:, None]],
        axis=1,
    )
    return history


class BatchedCEMPlanner:
    def __init__(
        self,
        model: WorldModel,
        cfg: PlannerConfig,
        device: str = "cuda",
        action_mean: Optional[np.ndarray] = None,
        action_std: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.device = device

        self.action_mean = None
        self.action_std = None

        if action_mean is not None:
            self.action_mean = torch.as_tensor(
                action_mean,
                dtype=torch.float32,
                device=device,
            )

        if action_std is not None:
            self.action_std = torch.as_tensor(
                action_std,
                dtype=torch.float32,
                device=device,
            )

    def _init_cem_distribution(
        self,
        batch_size: int,
        horizon: int,
        action_dim: int,
        prev_action: Optional[np.ndarray],
        low: float,
        high: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize CEM from demo action stats and optionally previous executed action.

        This avoids using any future/GT action sequence. It only uses:
          1) dataset-level action mean/std from the training/eval data distribution,
          2) previous action actually executed by the planner in the closed loop.
        """
        b, h, a = batch_size, horizon, action_dim

        if self.cfg.use_action_prior and self.action_mean is not None and self.action_std is not None:
            mean = self.action_mean.view(1, 1, a).expand(b, h, a).clone()
            std = self.action_std.view(1, 1, a).expand(b, h, a).clone()
            std = std * float(self.cfg.action_prior_std_scale)
            std = torch.clamp(std, min=float(self.cfg.action_prior_min_std))
        else:
            mean = torch.zeros(b, h, a, device=self.device)
            std = torch.ones(b, h, a, device=self.device)

        if self.cfg.warm_start_mode == "prev_action" and prev_action is not None:
            prev_action_t = torch.as_tensor(
                prev_action,
                dtype=torch.float32,
                device=self.device,
            )
            prev_action_t = torch.clamp(prev_action_t, low, high)
            prev_mean = prev_action_t[:, None, :].expand(b, h, a).clone()

            mix = float(self.cfg.warm_start_mix)
            mix = max(0.0, min(1.0, mix))
            mean = mix * prev_mean + (1.0 - mix) * mean

            warm_std = torch.full_like(std, float(self.cfg.warm_start_std))
            std = torch.minimum(std, warm_std)
            std = torch.clamp(std, min=float(self.cfg.cem_min_std))

        mean = torch.clamp(mean, low, high)
        return mean, std

    @torch.no_grad()
    def plan(
        self,
        current_info: Dict[str, np.ndarray],
        goal_info: Dict[str, np.ndarray],
        prev_action: Optional[np.ndarray] = None,
    ):
        vision_hist = _numpy_obs_to_tensor(current_info[self.cfg.vision_key], self.device)
        goal_vision = _numpy_obs_to_tensor(goal_info[self.cfg.vision_key], self.device)

        current_t = {self.cfg.vision_key: vision_hist}
        goal_t = {self.cfg.vision_key: goal_vision}

        b = vision_hist.shape[0]
        h = self.cfg.horizon
        a = self.cfg.action_dim
        s = self.cfg.candidates

        low = self.cfg.action_low
        high = self.cfg.action_high

        mean, std = self._init_cem_distribution(
            batch_size=b,
            horizon=h,
            action_dim=a,
            prev_action=prev_action,
            low=low,
            high=high,
        )

        final_cost = None
        final_topk_idx = None
        final_topk_actions = None

        values = torch.tensor([-1.0, 0.0, 1.0], device=self.device)

        for _ in range(self.cfg.iterations):
            if self.cfg.discrete_action:
                # Sample valid PTZ commands from {-1, 0, 1} instead of fractional Gaussian actions.
                idx = torch.randint(
                    low=0,
                    high=values.numel(),
                    size=(b, s, h, a),
                    device=self.device,
                )
                samples = values[idx]

                # Optionally warm-start a subset of candidates with the previous executed action.
                if self.cfg.warm_start_mode == "prev_action" and prev_action is not None and s > 1:
                    prev_action_t = torch.as_tensor(prev_action, dtype=torch.float32, device=self.device)
                    prev_action_t = torch.round(torch.clamp(prev_action_t, low, high))
                    n_warm = max(1, int(s * float(self.cfg.warm_start_mix)))
                    samples[:, :n_warm] = prev_action_t[:, None, None, :].expand(b, n_warm, h, a)
            else:
                noise = torch.randn(b, s, h, a, device=self.device)
                samples = mean[:, None] + std[:, None] * noise
                samples = torch.clamp(samples, low, high)

            cost = self.model.get_cost(
                current_info=current_t,
                action_candidates=samples,
                goal_info=goal_t,
                sum_all_diffs=self.cfg.sum_all_diffs,
                discount=self.cfg.discount,
            )

            # Regularize candidate sequences toward demo-like actions.
            # This helps when the predictor was trained only on GT/demo actions.
            if self.cfg.action_magnitude_weight > 0:
                mag_cost = (samples ** 2).mean(dim=(2, 3))
                cost = cost + float(self.cfg.action_magnitude_weight) * mag_cost

            if self.cfg.action_smooth_weight > 0 and h > 1:
                smooth_cost = ((samples[:, :, 1:] - samples[:, :, :-1]) ** 2).mean(dim=(2, 3))
                cost = cost + float(self.cfg.action_smooth_weight) * smooth_cost

            topk_idx = torch.topk(cost, k=self.cfg.topk, dim=1, largest=False).indices
            topk_actions = torch.gather(
                samples,
                dim=1,
                index=topk_idx[:, :, None, None].expand(-1, -1, h, a),
            )

            mean = topk_actions.mean(dim=1)
            std = topk_actions.std(dim=1, unbiased=False)
            std = torch.clamp(std, min=float(self.cfg.cem_min_std))

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
