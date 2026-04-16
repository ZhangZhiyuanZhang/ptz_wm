python train.py \
  --data-root /home/zhiyuan/Project/ptz_wm/data/replay_buffer.zarr \
  --vision-key image \
  --vision-type image \
  --reg-vision \
  --reg-loss-type vc \
  --epochs 100 \
  --image-size 224
