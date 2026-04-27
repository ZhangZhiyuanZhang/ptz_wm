import torch

ckpt = torch.load("/home/zhiyuan/Project/ptz_wm/logs/ckpts/image_dinov2_vits14_frozen_eq/last.ckpt", map_location="cpu")

print("epoch:", ckpt["epoch"])
print("global_step:", ckpt["global_step"])