import argparse
import zarr
import numpy as np
import imageio.v2 as imageio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", type=str, default="/home/zhiyuan/Project/ptz_wm/data/replay_buffer.zarr")
    parser.add_argument("--key", type=str, default="image")
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--output", type=str, default="/home/zhiyuan/Project/ptz_wm/data/output.mp4")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    print("[INFO] Opening zarr...")
    root = zarr.open_group(args.zarr_path, mode="r")

    if "data" in root:
        arr = root["data"][args.key]
    else:
        arr = root[args.key]

    print(f"[INFO] shape = {arr.shape}, dtype = {arr.dtype}")

    T = min(args.num_frames, arr.shape[0])

    print(f"[INFO] Writing {T} frames to {args.output} ...")

    writer = imageio.get_writer(args.output, fps=args.fps)

    for i in range(T):
        frame = arr[i]  # [H, W, 3]

        # 安全检查
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        writer.append_data(frame)

        if i % 50 == 0:
            print(f"[INFO] frame {i}/{T}")

    writer.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()