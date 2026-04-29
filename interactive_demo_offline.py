from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import zarr
from PIL import Image

from planner_utils import PlannerConfig, build_model, load_lightning_ckpt
from train_decoder import DINOImageDecoderModule
from PIL import Image, ImageDraw, ImageFont

ACTION_MAP = {
    "left":  [-1,  0,  0,  0],
    "right": [ 1,  0,  0,  0],
    "up":    [ 0, -1,  0,  0],
    "down":  [ 0, 1,  0,  0],
    "zero":  [ 0,  0,  0,  0],
}

def action_to_label(a):
    pan, tilt, zoom, focus = [int(x) for x in a]

    parts = []
    if pan < 0:
        parts.append("LEFT")
    elif pan > 0:
        parts.append("RIGHT")

    if tilt < 0:
        parts.append("UP")
    elif tilt > 0:
        parts.append("DOWN")

    if zoom > 0:
        parts.append("ZOOM+")
    elif zoom < 0:
        parts.append("ZOOM-")

    if focus > 0:
        parts.append("FOCUS+")
    elif focus < 0:
        parts.append("FOCUS-")

    if len(parts) == 0:
        return "ZERO"

    return " + ".join(parts)


def draw_action_overlay(frame_uint8, action=None, step_idx=0):
    """
    frame_uint8: [H,W,3], uint8
    action: [4] or None
    """
    img = Image.fromarray(frame_uint8)
    draw = ImageDraw.Draw(img)

    w, h = img.size

    if action is None:
        label = "Initial"
    else:
        label = f"Step {step_idx}: {action_to_label(action)}  action={list(map(int, action))}"

    # background box
    box_h = 34
    draw.rectangle([0, 0, w, box_h], fill=(0, 0, 0))

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    draw.text((10, 7), label, fill=(255, 255, 255), font=font)

    # draw arrow
    if action is not None:
        pan, tilt, _, _ = [int(x) for x in action]
        cx, cy = w // 2, box_h + 35
        scale = 35

        dx = pan * scale
        dy = tilt * scale

        if dx != 0 or dy != 0:
            x2, y2 = cx + dx, cy + dy
            draw.line([cx, cy, x2, y2], fill=(255, 0, 0), width=5)

            # arrow head
            r = 8
            draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill=(255, 0, 0))

    return np.asarray(img)

def parse_action_sequence(action_str: str):
    """
    Examples:
      "right right up left"
      "right:5 up:3 left:2 down:1"
    """
    actions = []
    tokens = action_str.replace(",", " ").split()

    for tok in tokens:
        if ":" in tok:
            name, rep = tok.split(":")
            rep = int(rep)
        else:
            name, rep = tok, 1

        name = name.lower()
        if name not in ACTION_MAP:
            raise ValueError(f"Unknown action {name}. Choices: {list(ACTION_MAP.keys())}")

        for _ in range(rep):
            actions.append(ACTION_MAP[name])

    return np.asarray(actions, dtype=np.float32)


def load_image_from_zarr(data_root: str, vision_key: str, init_index: int):
    root = zarr.open(data_root, mode="r")
    img = np.asarray(root["data"][vision_key][init_index])

    if img.max() > 1.5:
        img_float = img.astype(np.float32) / 255.0
    else:
        img_float = img.astype(np.float32)

    # HWC -> CHW
    if img_float.ndim == 3 and img_float.shape[-1] in (1, 3):
        img_float_chw = np.transpose(img_float, (2, 0, 1))
    elif img_float.ndim == 3 and img_float.shape[0] in (1, 3):
        img_float_chw = img_float
    else:
        raise ValueError(f"Unsupported image shape: {img_float.shape}")

    # [C,H,W] -> [B,T,C,H,W]
    img_batch = torch.from_numpy(img_float_chw)[None, None].float()

    return img_float, img_batch


def tensor_to_uint8(img):
    """
    img: [3,H,W], range [0,1]
    """
    img = img.detach().cpu().clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return img


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--wm-ckpt", type=str, default="logs/ckpts/image_dino_vit/last.ckpt")
    parser.add_argument("--decoder-ckpt", type=str, default="logs/decoder_ckpts/image_dinov2_vits14_decoder/last.ckpt")
    parser.add_argument("--data-root", type=str, default="data/replay_buffer.zarr")
    parser.add_argument("--vision-key", type=str, default="image")
    parser.add_argument("--init-index", type=int, default=0)

    parser.add_argument("--actions", type=str, default="right:10 up:5 left:5 down:5")

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dino-name", type=str, default="dinov2_vits14")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--pred-depth", type=int, default=6)
    parser.add_argument("--pred-heads", type=int, default=6)
    parser.add_argument("--pred-embed-dim", type=int, default=384)
    parser.add_argument("--pred-mlp-ratio", type=float, default=4.0)

    parser.add_argument("--output-dir", type=str, default="interactive_rollout_outputs")
    parser.add_argument("--fps", type=int, default=5)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    actions_np = parse_action_sequence(args.actions)
    action_dim = actions_np.shape[-1]

    cfg = PlannerConfig(
        action_dim=action_dim,
        vision_key=args.vision_key,
        image_size=args.image_size,
        dino_name=args.dino_name,
        num_steps=args.num_steps,
        pred_depth=args.pred_depth,
        pred_heads=args.pred_heads,
        pred_embed_dim=args.pred_embed_dim,
        pred_mlp_ratio=args.pred_mlp_ratio,
        encoder_type="dino",
        predictor_type="vit",
    )

    wm = build_model(cfg)
    wm = load_lightning_ckpt(wm, args.wm_ckpt)
    wm = wm.to(device).eval()
    wm.requires_grad_(False)

    decoder_module = DINOImageDecoderModule(
        dino_name=args.dino_name,
        vision_key=args.vision_key,
        image_size=args.image_size,
    )

    ckpt = torch.load(args.decoder_ckpt, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    decoder_module.load_state_dict(state_dict, strict=False)
    decoder_module = decoder_module.to(device).eval()
    decoder_module.requires_grad_(False)

    raw_img, img_batch = load_image_from_zarr(
        data_root=args.data_root,
        vision_key=args.vision_key,
        init_index=args.init_index,
    )

    batch = {
        args.vision_key: img_batch.to(device)
    }

    z = wm.encode(batch)          # [1,1,256,384]
    z_hist = z.clone()

    decoded_frames = []

    # frame 0: decoder reconstruction from initial image latent
    init_recon = decoder_module.decoder(z_hist)[:, 0]  # [1,3,H,W]
    frame0 = tensor_to_uint8(init_recon[0])
    frame0 = draw_action_overlay(frame0, action=None, step_idx=0)
    decoded_frames.append(frame0)

    for t, a_np in enumerate(actions_np):
        a = torch.from_numpy(a_np)[None, None].to(device)  # [1,1,4]
        z_next = wm.predict_sequence(z_hist[:, -1:], a)[:, -1:]  # [1,1,P,D]
        z_hist = torch.cat([z_hist, z_next], dim=1)

        pred_img = decoder_module.decoder(z_next)[:, 0]
        frame = tensor_to_uint8(pred_img[0])
        frame = draw_action_overlay(frame, action=a_np, step_idx=t + 1)
        decoded_frames.append(frame)

    # save frames
    for i, frame in enumerate(decoded_frames):
        Image.fromarray(frame).save(frames_dir / f"frame_{i:04d}.png")

    # save mp4 if imageio is available
    try:
        import imageio.v2 as imageio
        video_path = out_dir / "rollout.mp4"
        imageio.mimsave(video_path, decoded_frames, fps=args.fps)
        print(f"[INFO] Saved video to: {video_path}")
    except Exception as e:
        print(f"[WARN] Could not save mp4: {e}")

    print(f"[INFO] Saved frames to: {frames_dir}")
    print(f"[INFO] Actions used: {actions_np.tolist()}")


if __name__ == "__main__":
    main()