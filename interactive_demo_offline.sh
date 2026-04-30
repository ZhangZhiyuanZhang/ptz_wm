#!/bin/bash

set -e
set -o pipefail

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT="data/replay_buffer.zarr"
VISION_KEY="image"

WM_CKPT="logs/ckpts/image_dino_vit/100000.ckpt"
DECODER_CKPT="logs/decoder_ckpts/image_wm_decoder/005000.ckpt"

OUTPUT_DIR="logs/interactive_outputs/image_wm_decoder"

python interactive_demo_offline.py \
  --data-root "${DATA_ROOT}" \
  --vision-key "${VISION_KEY}" \
  --wm-ckpt "${WM_CKPT}" \
  --decoder-ckpt "${DECODER_CKPT}" \
  --init-index 850 \
  --actions "right:5 up:5 left:5 down:5" \
  --image-size 224 \
  --dino-name dinov2_vits14 \
  --num-steps 5 \
  --nsteps 2 \
  --pred-depth 6 \
  --pred-heads 6 \
  --pred-embed-dim 384 \
  --pred-mlp-ratio 4.0 \
  --eq-weight 0.0 \
  --output-dir "${OUTPUT_DIR}" \
  --fps 2