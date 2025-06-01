import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from modules.depth_estimator import DepthAnythingV2Estimator

INPUT_DIR = 'datasets/lab_scene_kinect_xyz/rgb'
OUTPUT_DIR = 'datasets/lab_scene_kinect_xyz/depth_mono'
ENCODER = 'vits'
CHECKPOINT_DIR = 'checkpoints'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Running DepthAnythingV2 on device: {device}")

    estimator = DepthAnythingV2Estimator(
        encoder=ENCODER,
        checkpoint_dir=CHECKPOINT_DIR,
        device=device
    )

    rgb_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith('.png'))

    for fname in tqdm(rgb_files, desc="Inferring depth"):
        rgb_path = os.path.join(INPUT_DIR, fname)
        rgb = cv2.imread(rgb_path)

        # DepthAnything expects RGB in [0, 255] uint8
        depth = estimator.infer_depth(rgb)
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()

        # Save as npy (float32 in meters, normalized by DepthAnything)
        output_path = os.path.join(OUTPUT_DIR, fname.replace('.png', '.npy'))

        np.save(output_path, depth)


if __name__ == '__main__':
    main()
