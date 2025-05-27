"""
Monocular depth inference using the DepthAnythingV2 model.

Supports static testing, batch inference from image directories, and
real-time webcam input. The estimated depth maps are visualized and
can be optionally saved to disk.
"""

import os
from typing import Optional

import cv2
import numpy as np

from modules.depth_estimator import DepthAnythingV2Estimator


def visualize_depth_map(rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """
    Generates a side-by-side visualization of RGB and depth map.

    Args:
        rgb (np.ndarray): Original RGB image.
        depth (np.ndarray): Depth map to visualize.

    Returns:
        np.ndarray: Concatenated RGB + depth image.
    """
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(
        depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO
    )
    return cv2.hconcat([rgb, depth_colored])


def run_static_test(estimator: DepthAnythingV2Estimator) -> None:
    """
    Tests depth inference using a single static image.

    Args:
        estimator (DepthAnythingV2Estimator): Inference engine.
    """
    directory = 'datasets/rgbd_dataset_freiburg1_xyz/rgb'
    image_path = f'{directory}/1305031102.175304.png'
    rgb = cv2.imread(image_path)
    if rgb is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    depth = estimator.infer_depth(rgb)
    display = visualize_depth_map(rgb, depth)

    cv2.imshow('RGB + Depth Map', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_directory_mode(
    estimator: DepthAnythingV2Estimator,
    image_dir: str,
    output_dir: Optional[str] = None,
    save_output: bool = False
) -> None:
    """
    Processes and visualizes all images in a directory.

    Args:
        estimator (DepthAnythingV2Estimator): Inference engine.
        image_dir (str): Path to input image directory.
        output_dir (Optional[str]): Path to save visualizations.
        save_output (bool): If True, saves output to disk.
    """
    files = sorted(f for f in os.listdir(image_dir) if f.endswith('.png'))

    for fname in files:
        path = os.path.join(image_dir, fname)
        image = cv2.imread(path)
        if image is None:
            print(f"[Warning] Could not load image: {path}")
            continue

        depth = estimator.infer_depth(image)
        display = visualize_depth_map(image, depth)

        if save_output and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f'depth_{fname}')
            cv2.imwrite(save_path, display)

        cv2.imshow('RGB + Depth Map', display)
        if cv2.waitKey(1) == 27:
            break


def run_webcam_mode(
    estimator: DepthAnythingV2Estimator,
    camera_index: int = 0,
    output_dir: Optional[str] = None,
    save_output: bool = False
) -> None:
    """
    Runs real-time depth inference using webcam input.

    Args:
        estimator (DepthAnythingV2Estimator): Inference engine.
        camera_index (int): Index of the webcam device.
        output_dir (Optional[str]): Where to save frames.
        save_output (bool): Whether to save output images.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[Error] Unable to open webcam.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to capture frame.")
            break

        depth = estimator.infer_depth(frame)
        display = visualize_depth_map(frame, depth)

        if save_output and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fname = f'depth_frame_{frame_count:04d}.png'
            cv2.imwrite(os.path.join(output_dir, fname), display)

        cv2.imshow('RGB + Depth Map', display)
        frame_count += 1

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    """
    Entry point for real-time or batch inference testing.
    """
    encoder = 'vits'
    checkpoint_dir = 'checkpoints'
    image_dir = 'datasets/rgbd_dataset_freiburg1_xyz/rgb'
    output_dir = 'results/depth_maps'
    use_webcam = False
    save_output = True

    estimator = DepthAnythingV2Estimator(encoder, checkpoint_dir)
    print(f"[INFO] Running on device: {estimator.device}")

    if use_webcam:
        run_webcam_mode(estimator, camera_index=0,
                        output_dir=output_dir,
                        save_output=save_output)
    else:
        run_directory_mode(estimator, image_dir=image_dir,
                           output_dir=output_dir,
                           save_output=save_output)


if __name__ == '__main__':
    # main()
    run_static_test()
