from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

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
        regularizer,
        predcost,
        vision_dim: int,
    ):
        super().__init__(encoder, aencoder, predictor)
        self.regularizer = regularizer
        self.predcost = predcost
        self.vision_dim = vision_dim

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
        state = self.encoder(observations)
        actions_encoded = self.action_encoder(actions) if actions is not None else None

        if compute_loss:
            rloss, rloss_unweight, rloss_dict = self.regularizer(state, actions_encoded)
            ploss = 0.0
        else:
            rloss = rloss_unweight = rloss_dict = ploss = None

        all_steps = [] if return_all_steps else None

        if unroll_mode != "autoregressive":
            raise ValueError("This wrapper only supports autoregressive mode.")

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
        regularizer: nn.Module,
        predcost: nn.Module,
        action_dim: int,
        vision_key: str,
        image_size: Union[int, Tuple[int, int]] = 224,
        vision_dim: int = 512,
        grid_size: int = 16,
        eq_weight: float = 0.0,
        backward_weight: float = 0.0,
        latent_type: str = "grid",
    ):
        super().__init__()
        self.vision_key = vision_key
        self.image_size = image_size
        self.action_dim = action_dim
        self.vision_dim = vision_dim
        self.encoder = encoder
        self.predictor = predictor
        self.grid_size = grid_size
        self.eq_weight = eq_weight
        self.backward_weight = backward_weight

        self.latent_type = latent_type

        self.jepa = JEPA(
            encoder=encoder,
            aencoder=nn.Identity(),
            predictor=predictor,
            regularizer=regularizer,
            predcost=predcost,
            vision_dim=vision_dim,
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
        """
        image-like input supports:
          [B, T, H, W]      -> treated as single-channel
          [B, T, C, H, W]

        output:
          [B, C, T, H, W]
        """
        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0

        if x.ndim == 4:
            x = x.unsqueeze(2)  # [B,T,1,H,W]

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


    def transform_grid_tokens(self, z, flip=None, angle=0):
        """
        z: [B,T,P,D]
        flip: None | "h" | "v"
        angle: 0 | 1 | 2 | 3, rot90 k times
        """
        B, T, P, D = z.shape
        H = W = self.grid_size
        assert P == H * W, f"Expected P={H*W}, got P={P}"

        z = z.view(B, T, H, W, D)

        if flip == "h":
            z = torch.flip(z, dims=[3])  # W direction
        elif flip == "v":
            z = torch.flip(z, dims=[2])  # H direction

        if angle is not None and angle != 0:
            z = torch.rot90(z, k=angle, dims=[2, 3])

        return z.reshape(B, T, P, D).contiguous()


    def transform_ptz_action(self, a, flip=None, angle=0):
        """
        a: [B,T,A]

        Assumption:
        a[..., 0] = pan delta
        a[..., 1] = tilt delta
        a[..., 2] = zoom delta, unchanged by flip
        a[..., 3] = focus delta, unchanged by flip

        Only pan/tilt are transformed.
        """
        if a.size(-1) < 2:
            return a

        out = a.clone()

        xy = out[..., :2]

        if angle == 0:
            R = xy.new_tensor([[1.0, 0.0], [0.0, 1.0]])
        elif angle == 1:
            R = xy.new_tensor([[0.0, -1.0], [1.0, 0.0]])
        elif angle == 2:
            R = xy.new_tensor([[-1.0, 0.0], [0.0, -1.0]])
        elif angle == 3:
            R = xy.new_tensor([[0.0, 1.0], [-1.0, 0.0]])
        else:
            raise ValueError(f"angle must be 0,1,2,3, got {angle}")

        if flip == "h":
            Fmat = xy.new_tensor([[-1.0, 0.0], [0.0, 1.0]])
            R = Fmat @ R
        elif flip == "v":
            Fmat = xy.new_tensor([[1.0, 0.0], [0.0, -1.0]])
            R = Fmat @ R

        xy_new = torch.einsum("ij,btj->bti", R, xy)
        out[..., :2] = xy_new

        return out


    def training_losses(self, batch: dict, nsteps: int):
        if self.latent_type == "vector":
            obs = self._preprocess_image_btchw_to_bcthw(batch[self.vision_key])
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
                "eq_loss": obs.new_tensor(0.0),
                "reg_dict": reg_dict,
            }

        elif self.latent_type == "grid":
            obs = self._preprocess_image_btchw_to_bcthw(batch[self.vision_key])
            act = batch["action"].float()  # [B,T,A]

            z = self.encoder(obs)          # [B,T,P,D]

            # 1-step teacher-forcing prediction over all adjacent pairs
            z_in = z[:, :-1]               # [B,T-1,P,D]
            a_in = act[:, :-1]             # [B,T-1,A]
            target = z[:, 1:].detach()     # [B,T-1,P,D]

            pred = self.predictor(z_in, a_in)
            pred_loss = F.mse_loss(pred, target)

            eq_loss = z.new_tensor(0.0)
            backward_loss = z.new_tensor(0.0)

            if self.eq_weight > 0 or self.backward_weight > 0:
                if random.random() < 0.5 and self.eq_weight > 0:
                    # only run eq this batch
                    flip = random.choice([None, "h", "v"])
                    angle = 0
                    z_tau = self.transform_grid_tokens(z_in, flip=flip, angle=angle)
                    a_tau = self.transform_ptz_action(a_in, flip=flip, angle=angle)
                    pred_tau = self.predictor(z_tau, a_tau)
                    pred_expected = self.transform_grid_tokens(pred.detach(), flip=flip, angle=angle)
                    eq_loss = F.mse_loss(pred_tau, pred_expected)
                elif self.backward_weight > 0:
                    # only run backward this batch
                    inv_a = -a_in
                    pred_backward = self.predictor(target.detach(), inv_a)
                    backward_loss = F.mse_loss(pred_backward, z_in.detach())

            total_loss = pred_loss + self.eq_weight * eq_loss + self.backward_weight * backward_loss

            reg_loss = z.new_tensor(0.0)

            return {
                "loss": total_loss,
                "pred_loss": pred_loss,
                "reg_loss": reg_loss,
                "eq_loss": eq_loss,
                "backward_loss": backward_loss,
                "reg_dict": {},
            }
        else:
            raise ValueError(f"Unsupported latent_type: {self.latent_type}")

    def encode(self, batch):
        obs = self._preprocess_image_btchw_to_bcthw(batch[self.vision_key])

        if self.latent_type == "vector":
            enc_out = self.encoder(obs)  # [B,D,T,1,1]
            return enc_out.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous()

        if self.latent_type == "grid":
            return self.encoder(obs)     # [B,T,P,D]

    def predict_sequence(self, z_hist, action_seq):
        if self.latent_type == "vector":
            z5 = z_hist.transpose(1, 2).unsqueeze(-1).unsqueeze(-1).contiguous()
            a = action_seq.transpose(1, 2).contiguous()
            pred5 = self.predictor(z5, a)
            return pred5.squeeze(-1).squeeze(-1).transpose(1, 2).contiguous()

        if self.latent_type == "grid":
            return self.predictor(z_hist, action_seq)

    def rollout_latent(self, z0: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        # z0: [B,P,D]
        # action_sequence: [B,S,T,A]
        B, S, T, A = action_sequence.shape
        P, D = z0.shape[1], z0.shape[2]

        z = z0[:, None, None].expand(B, S, 1, P, D).reshape(B * S, 1, P, D)
        act = action_sequence.reshape(B * S, T, A)

        preds = []
        for t in range(T):
            pred = self.predict_sequence(z[:, -1:], act[:, t:t+1])
            z_next = pred[:, -1:]
            preds.append(z_next)
            z = torch.cat([z, z_next], dim=1)

        pred_rollout = torch.cat(preds, dim=1)
        return pred_rollout.reshape(B, S, T, P, D).contiguous()

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
        # pred_rollout: [B,S,T,P,D]
        # goal_emb: [B,P,D] or [B,1,P,D]
        if goal_emb.ndim == 4:
            goal_emb = goal_emb[:, -1]   # [B,P,D]

        goal = goal_emb[:, None, None, :, :]  # [B,1,1,P,D]
        step_cost = ((pred_rollout - goal.detach()) ** 2).mean(dim=(-1, -2))  # [B,S,T]

        if sum_all_diffs:
            if discount != 1.0:
                T = step_cost.size(-1)
                w = step_cost.new_tensor([discount ** t for t in range(T)]).view(1, 1, T)
                step_cost = step_cost * w
            return step_cost.sum(dim=-1)

        return step_cost[:, :, -1]

    def get_cost(
        self,
        current_info: dict,
        action_candidates: torch.Tensor,
        goal_info: dict,
        sum_all_diffs: bool = False,
        discount: float = 1.0,
    ) -> torch.Tensor:
        device = action_candidates.device

        current_batch = {
            self.vision_key: current_info[self.vision_key].to(device),
        }
        goal_batch = {
            self.vision_key: goal_info[self.vision_key].to(device),
        }

        emb = self.encode(current_batch)
        goal_emb = self.encode(goal_batch)

        z0 = emb[:, -1:, :]  # [B,1,D]
        pred_rollout = self.rollout_latent(z0, action_candidates)  # [B,S,T,D]

        return self.criterion(
            pred_rollout=pred_rollout,
            goal_emb=goal_emb[:, -1, :],
            sum_all_diffs=sum_all_diffs,
            discount=discount,
        )