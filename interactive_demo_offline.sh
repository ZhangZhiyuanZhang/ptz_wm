python interactive_demo_offline.py \
  --wm-ckpt logs/ckpts/image_dino_vit/090000.ckpt \
  --decoder-ckpt logs/decoder_ckpts/image_dinov2_vits14_decoder/015000.ckpt \
  --data-root data/replay_buffer.zarr \
  --vision-key image \
  --init-index 682 \
  --actions "up:5 left:5 down:5 right:5" \
  --output-dir interactive_rollout_outputs/demo_90k \
  --fps 2