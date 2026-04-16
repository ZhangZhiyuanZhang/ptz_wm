from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAbase(nn.Module):
    def __init__(self, encoder, aencoder, predictor):
        super().__init__()
        self.encoder = encoder
        self.action_encoder = aencoder
        self.predictor = predictor
        self.single_unroll = getattr(self.predictor, "is_rnn", False)

    @torch.no_grad()
    def encode_raw(self, observations):
        return self.encoder(observations)


class JEPA(JEPAbase):
    def __init__(
        self,
        encoder,
        aencoder,
        predictor,
        vision_regularizer,
        tactile_regularizer,
        fused_regularizer,
        predcost,
        vision_dim: int,
        tactile_dim: int,
        fusion_type: str = "concat",
        reg_vision: bool = True,
        reg_tactile: bool = False,
    ):
        super().__init__(encoder, aencoder, predictor)
        self.vision_regularizer = vision_regularizer
        self.tactile_regularizer = tactile_regularizer
        self.fused_regularizer = fused_regularizer
        self.predcost = predcost

        self.vision_dim = vision_dim
        self.tactile_dim = tactile_dim
        self.fusion_type = fusion_type.lower()

        self.reg_vision = reg_vision
        self.reg_tactile = reg_tactile

    def _encode_modalities_and_joint(self, observations):
        """
        returns:
          state: fused latent [B, Df, T, 1, 1]
          z_v:   vision latent [B, Dv, T, 1, 1] or None
          z_t:   tactile latent [B, Dt, T, 1, 1] or None
        """
        if (
            isinstance(observations, dict)
            and hasattr(self.encoder, "encode_modalities")
            and hasattr(self.encoder, "fuse_latents")
        ):
            z_v, z_t = self.encoder.encode_modalities(observations)
            state = self.encoder.fuse_latents(z_v, z_t)
            return state, z_v, z_t

        state = self.encoder(observations)
        return state, None, None

    def _compute_reg(self, z_v, z_t, state, actions_encoded):
        total = state.new_tensor(0.0)
        total_unweighted = state.new_tensor(0.0)
        reg_dict = {}

        if self.reg_vision and z_v is not None and self.vision_regularizer is not None:
            rv, uv, dv = self.vision_regularizer(z_v)
            total = total + rv
            total_unweighted = total_unweighted + uv
            for k, v in dv.items():
                reg_dict[f"vision_{k}"] = v

        if self.reg_tactile and z_t is not None and self.tactile_regularizer is not None:
            rt, ut, dt = self.tactile_regularizer(z_t)
            total = total + rt
            total_unweighted = total_unweighted + ut
            for k, v in dt.items():
                reg_dict[f"tactile_{k}"] = v

        if self.fused_regularizer is not None:
            rf, uf, df = self.fused_regularizer(state, actions_encoded)
            total = total + rf
            total_unweighted = total_unweighted + uf
            for k, v in df.items():
                reg_dict[f"fused_{k}"] = v

        reg_dict["reg_vision"] = float(self.reg_vision)
        reg_dict["reg_tactile"] = float(self.reg_tactile)
        return total, total_unweighted, reg_dict

    def unroll(
        self,
        observations,
        actions,
        nsteps=1,
        unroll_mode="autoregressive",
        ctxt_window_time=1,
        compute_loss=True,
        return_all_steps=False,
    ):
        state, z_v, z_t = self._encode_modalities_and_joint(observations)
        actions_encoded = self.action_encoder(actions) if actions is not None else None

        if compute_loss:
            rloss, rloss_unweight, rloss_dict = self._compute_reg(
                z_v=z_v,
                z_t=z_t,
                state=state,
                actions_encoded=actions_encoded,
            )
            ploss = 0.0
        else:
            rloss = rloss_unweight = rloss_dict = ploss = None

        all_steps = [] if return_all_steps else None

        if unroll_mode != "autoregressive":
            raise ValueError("This TVB EB wrapper only supports autoregressive mode.")

        effective_ctxt_window = 1 if self.single_unroll else ctxt_window_time
        predicted_states = state[:, :, :effective_ctxt_window]

        for i in range(nsteps):
            context_states = predicted_states[:, :, -effective_ctxt_window:]
            context_actions = actions_encoded[
                :, :, max(0, i + 1 - effective_ctxt_window): i + 1
            ]

            pred_joint_step = self.predictor(context_states, context_actions)[:, :, -1:]
            predicted_states = torch.cat([predicted_states, pred_joint_step], dim=2)

            if return_all_steps:
                all_steps.append(predicted_states.clone())

            if compute_loss:
                target = state[:, :, i + 1: i + 2]
                ploss += self.predcost(pred_joint_step, target) / nsteps

        if compute_loss:
            loss = rloss + ploss
            losses = (loss, rloss, rloss_unweight, rloss_dict, ploss)
        else:
            losses = None

        if return_all_steps:
            return all_steps, losses
        return predicted_states, losses


class WorldModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        vision_regularizer: nn.Module,
        tactile_regularizer: nn.Module,
        fused_regularizer: nn.Module,
        predcost: nn.Module,
        action_dim: int,
        vision_key: str,
        vision_type: str = "image",
        image_size: Union[int, Tuple[int, int]] = 224,
        use_tactile: bool = False,
        tactile_key: str = None,
        tactile_size=None,
        vision_dim: int = 512,
        tactile_dim: int = 512,
        fusion_type: str = "concat",
        reg_vision: bool = True,
        reg_tactile: bool = False,
    ):
        super().__init__()
        self.vision_key = vision_key
        self.vision_type = vision_type.lower()
        self.image_size = image_size
        self.action_dim = action_dim

        self.use_tactile = use_tactile
        self.tactile_key = tactile_key
        self.tactile_size = tactile_size

        self.vision_dim = vision_dim
        self.tactile_dim = tactile_dim
        self.fusion_type = fusion_type.lower()

        self.reg_vision = reg_vision
        self.reg_tactile = reg_tactile

        self.jepa = JEPA(
            encoder=encoder,
            aencoder=nn.Identity(),
            predictor=predictor,
            vision_regularizer=vision_regularizer,
            tactile_regularizer=tactile_regularizer,
            fused_regularizer=fused_regularizer,
            predcost=predcost,
            vision_dim=vision_dim,
            tactile_dim=tactile_dim,
            fusion_type=self.fusion_type,
            reg_vision=reg_vision,
            reg_tactile=reg_tactile,
        )

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("vision_mean", mean, persistent=False)
        self.register_buffer("vision_std", std, persistent=False)

    def _resolve_image_hw(self) -> Tuple[int, int]:
        if isinstance(self.image_size, int):
            return self.image_size, self.image_size
        if isinstance(self.image_size, (tuple, list)) and len(self.image_size) == 2:
            return int(self.image_size[0]), int(self.image_size[1])
        raise ValueError(f"Unsupported image_size: {self.image_size}")

    def _preprocess_image_btchw_to_bcthw(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0

        if x.ndim == 4:
            x = x.unsqueeze(2)

        if x.ndim != 5:
            raise ValueError(f"Image input must be [B,T,C,H,W] or [B,T,H,W], got {tuple(x.shape)}")

        out_h, out_w = self._resolve_image_hw()

        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        if h != out_h or w != out_w:
            x = F.interpolate(
                x,
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )

        if c == 3:
            x = (x - self.vision_mean) / self.vision_std

        x = x.reshape(b, t, c, out_h, out_w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x

    def _preprocess_pc_btnc(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.ndim != 4:
            raise ValueError(f"Point cloud input must be [B,T,N,C], got {tuple(x.shape)}")
        return x.contiguous()

    def _preprocess_tactile_btchw_to_bcthw(self, x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        x = x.float()
        if x.ndim == 4:
            x = x.unsqueeze(2)

        if x.ndim != 5:
            raise ValueError(f"Unsupported tactile shape: {tuple(x.shape)}")

        if x.max() > 1.5:
            x = x / 255.0

        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        if h != out_h or w != out_w:
            x = F.interpolate(
                x,
                size=(out_h, out_w),
                mode="bilinear",
                align_corners=False,
            )

        x = x.reshape(b, t, c, out_h, out_w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x

    def preprocess_vision(self, x: torch.Tensor) -> torch.Tensor:
        if self.vision_type == "image":
            return self._preprocess_image_btchw_to_bcthw(x)
        if self.vision_type == "pc":
            return self._preprocess_pc_btnc(x)
        raise ValueError(f"Unknown vision_type: {self.vision_type}")

    def _prepare_batch_obs(self, batch: dict):
        vision_obs = self.preprocess_vision(batch[self.vision_key])

        if self.use_tactile:
            tactile_obs = self._preprocess_tactile_btchw_to_bcthw(
                batch[self.tactile_key],
                out_h=self.tactile_size[0],
                out_w=self.tactile_size[1],
            )
            return {
                "vision": vision_obs,
                "tactile": tactile_obs,
            }
        return vision_obs

    def encode_modalities(self, batch: dict):
        obs = self._prepare_batch_obs(batch)

        if not (
            self.use_tactile
            and hasattr(self.jepa.encoder, "encode_modalities")
            and hasattr(self.jepa.encoder, "fuse_latents")
        ):
            emb = self.encode(batch)
            return {"joint_emb": emb}

        z_v, z_t = self.jepa.encoder.encode_modalities(obs)
        z_f = self.jepa.encoder.fuse_latents(z_v, z_t)

        return {
            "vision_emb": z_v.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous(),
            "tactile_emb": z_t.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous(),
            "joint_emb": z_f.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous(),
        }

    def training_losses(self, batch: dict, nsteps: int):
        obs = self._prepare_batch_obs(batch)
        act = batch["action"].float().transpose(1, 2)

        _, losses = self.jepa.unroll(
            obs,
            act,
            nsteps=nsteps,
            unroll_mode="autoregressive",
            ctxt_window_time=1,
            compute_loss=True,
            return_all_steps=False,
        )
        total_loss, reg_loss, _, reg_dict, pred_loss = losses
        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "reg_loss": reg_loss,
            "reg_dict": reg_dict,
        }

    def encode(self, batch: dict):
        obs = self._prepare_batch_obs(batch)
        enc_out = self.jepa.encoder(obs)
        emb = enc_out.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous()
        return emb

    def predict_sequence(self, z_hist: torch.Tensor, action_seq: torch.Tensor) -> torch.Tensor:
        z5 = z_hist.transpose(1, 2).unsqueeze(-1).unsqueeze(-1).contiguous()
        a = action_seq.transpose(1, 2).contiguous()
        pred5 = self.jepa.predictor(z5, a)
        pred = pred5.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous()
        return pred

    def rollout_latent(self, z0: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        B, S, T, A = action_sequence.shape
        D = z0.size(-1)

        z = z0.unsqueeze(1).expand(B, S, 1, D).reshape(B * S, 1, D)
        act = action_sequence.reshape(B * S, T, A)

        preds = []
        for t in range(T):
            a_t = act[:, t:t + 1, :]
            pred = self.predict_sequence(z[:, -1:, :], a_t)
            z_next = pred[:, -1:, :]
            preds.append(z_next)
            z = torch.cat([z, z_next], dim=1)

        pred_rollout = torch.cat(preds, dim=1)
        pred_rollout = pred_rollout.reshape(B, S, T, D).contiguous()
        return pred_rollout

    def _expand_goal_to_rollout(self, goal_emb: torch.Tensor, pred_emb: torch.Tensor) -> torch.Tensor:
        if goal_emb.ndim == 2:
            goal_emb = goal_emb[:, None, None, :].expand(
                pred_emb.size(0), pred_emb.size(1), pred_emb.size(2), -1
            )
        elif goal_emb.ndim == 3:
            goal_emb = goal_emb[:, :, None, :].expand_as(pred_emb)
        else:
            raise ValueError(f"Unsupported goal_emb shape: {tuple(goal_emb.shape)}")
        return goal_emb

    def criterion(self, pred_rollout, goal_emb, sum_all_diffs=False, discount=1.0):
        goal_emb = self._expand_goal_to_rollout(goal_emb, pred_rollout)
        total_step_cost = F.mse_loss(pred_rollout, goal_emb.detach(), reduction="none").sum(dim=-1)

        if sum_all_diffs:
            if discount != 1.0:
                T = total_step_cost.size(-1)
                w = total_step_cost.new_tensor([discount ** t for t in range(T)]).view(1, 1, T)
                total_step_cost = total_step_cost * w
            return total_step_cost.sum(dim=-1)

        return total_step_cost[:, :, -1]

    def get_cost(self, current_info, action_candidates, goal_info, sum_all_diffs=False, discount=1.0):
        device = action_candidates.device

        current_batch = {self.vision_key: current_info[self.vision_key].to(device)}
        goal_batch = {self.vision_key: goal_info[self.vision_key].to(device)}

        if self.use_tactile:
            current_batch[self.tactile_key] = current_info[self.tactile_key].to(device)
            goal_batch[self.tactile_key] = goal_info[self.tactile_key].to(device)

        emb = self.encode(current_batch)
        goal_emb = self.encode(goal_batch)

        z0 = emb[:, -1:, :]
        pred_rollout = self.rollout_latent(z0, action_candidates)

        return self.criterion(
            pred_rollout=pred_rollout,
            goal_emb=goal_emb[:, -1, :],
            sum_all_diffs=sum_all_diffs,
            discount=discount,
        )