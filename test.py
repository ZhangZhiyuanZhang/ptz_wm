# import torch

# ckpt = torch.load("logs/decoder_ckpts/image_dinov2_vits14_decoder/last.ckpt", map_location="cpu")

# print("epoch:", ckpt["epoch"])
# print("global_step:", ckpt["global_step"])

import zarr
import numpy as np

root = zarr.open("data/replay_buffer.zarr", mode="r")
a = np.asarray(root["data"]["action"])

for d, name in enumerate(["pan", "tilt", "zoom", "focus"]):
    vals, counts = np.unique(a[:, d], return_counts=True)
    print(name, dict(zip(vals.tolist(), counts.tolist())))