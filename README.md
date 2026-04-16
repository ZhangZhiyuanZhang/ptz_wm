# PTZ-WM: Vision-Only World Model for Real PTZ Camera Data

This repository contains a vision-only JEPA/world-model pipeline adapted from the VT-WM benchmark codebase for **real-world PTZ camera experiments**.

The current setup focuses on:
- vision-only training
- real recorded trajectories
- Zarr dataset support
- multi-step latent rollout evaluation
- action sensitivity evaluation (GT / zero / random actions)

---

## 1. Overview

We train a JEPA-style world model consisting of:
- a **vision encoder**
- an **action-conditioned predictor**
- regularization to prevent collapse

Supported modes:
- vision-only (used for PTZ)
- vision + tactile (not used here)

Regularization types:
- `vc` (variance/covariance)
- `sigreg` (sketch isotropic Gaussian)

---

## 2. Installation

### Create environment

```bash
conda env create -f environment.yml
conda activate ptz_wm
```

## 3. Dataset Format

Expected Zarr structure:

```bash
replay_buffer.zarr/
├── data
│   ├── action
│   ├── image
│   ├── state
│   └── timestamp
└── meta
    └── episode_ends
```

## 4. Training

```bash
python train.py \
  --data-root /path/to/replay_buffer.zarr \
  --vision-key image \
  --vision-type image \
  --reg-vision \
  --reg-loss-type vc \
  --epochs 100 \
  --image-size 224
```

### Important Flags
```bash
--vision-key: dataset key (e.g., image)
--reg-vision: enable regularization
--reg-loss-type: vc or sigreg
--image-size: resize resolution
```

Checkpoints Saved at: `outputs/ckpts/<exp_name>/xxx`. Auto-resume is enabled if last.ckpt exists.

## 5. Multi-Step Rollout Evaluation
```bash
python eval_rollout.py \
  --ckpt-path outputs/ckpts/.../xxxx.ckpt \
  --data-root /path/to/replay_buffer.zarr \
  --vision-key image \
  --vision-type image \
  --image-size 224 \
  --batch-size 128 \
  --action-mode all \
  --reg-vision \
  --reg-loss-type vc \
  --max-rollout 6
```
### Metrics
Evaluates 3 modes:
```bash
gt
zero
random_uniform
```

### Expected:
```bash
gt < zero < random
```

### Indicates:
```bash
model uses action
dynamics learned
no collapse
```

## 8. Real-World PTZ Evaluation
### Problem
Naive evaluation:

manually record init
manually record goal
manually reset
run 1 trial

→ inefficient

### Recommended Protocol
#### Step 1: Collect demos similiar to data collection

    Record multiple PTZ trajectories into Zarr

#### Step 2: Sample chunks

    For each demo:

    init = frame t
    goal = frame t + horizon

#### Step 3: Add reset capability
    reset_to_state(ptz_state)

#### Step 4: Automatic evaluation
    For each sampled chunk:

    reset to init
    set goal image
    run planner
    execute actions
    compare final vs goal

#### Step 5: Possible evaluation metrics
    Latent space:
        - MSE between embeddings
    Image space
        - MSE
        - SSIM
        - PSNR
    Task-level
        - target center error
        - camera pose error

## 11. Notes

    1. Current eval_planner.py is based on Simulation Env, change accordingly to readworld.
    2. Train longer to get 050000.ckpts, 1000000.ckpts, 1500000.ckpts, 2000000.ckpts. And eval/planning seperately.