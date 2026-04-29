# import torch

# ckpt = torch.load("/home/zhiyuan/Project/ptz_wm/logs/ckpts/image_dinov2_vits14_frozen_eq/last.ckpt", map_location="cpu")

# print("epoch:", ckpt["epoch"])
# print("global_step:", ckpt["global_step"])

import zarr
import numpy as np

root = zarr.open("data/replay_buffer.zarr", mode="r")

action = root["data"]["action"][:100]
state = root["data"]["state"][:100]

print("action:")
print(action)

print("\nstate:")
print(state)