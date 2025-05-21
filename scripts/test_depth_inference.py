"""
Real-time depth inference using the DepthAnythingV2 model.

This script supports both webcam-based live inference and batch processing
from an image directory. The resulting depth maps are visualized and optionally
saved to disk. The model is configured via a checkpoint and supports different
encoder variants.
"""

import os
from typing import Optional

import cv2
import torch
import numpy as np

from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2


def get_device() -> str:
    """
    Returns the best available device: CUDA, MPS or CPU.

    :return: str: Device type ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_model(encoder: str,
               checkpoint_dir: str,
               device: str) -> DepthAnythingV2:
    """
    Loads the DepthAnythingV2 model with the specified encoder and checkpoint.

    :param encoder: str: Encoder type ('vits', 'vitb', 'vitl', 'vitg').
    :param checkpoint_dir: str: Directory containing the model checkpoint.
    :param device: str: Device type ('cuda', 'mps', or 'cpu').
    :return: DepthAnythingV2: Loaded model.
    :raises ValueError: If the encoder is not supported.
    """
    model_configs = {
        'vits': {
            'encoder': 'vits',
            'features': 64,
            'out_channels': [48, 96, 192, 384]
        },
        'vitb': {
            'encoder': 'vitb',
            'features': 128,
            'out_channels': [96, 192, 384, 768]
        },
        'vitl': {
            'encoder': 'vitl',
            'features': 256,
            'out_channels': [256, 512, 1024, 1024]
        },
        'vitg': {
            'encoder': 'vitg',
            'features': 384,
            'out_channels': [1536, 1536, 1536, 1536]
        }
    }

    if encoder not in model_configs:
        raise ValueError(f"Unsupported encoder: {encoder}")

    model = DepthAnythingV2(**model_configs[encoder])
    ckpt_path = os.path.join(checkpoint_dir,
                             f'depth_anything_v2_{encoder}.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()


def process_and_display(model: DepthAnythingV2,
                        image: np.ndarray,
                        output_dir: Optional[str] = None,
                        name: Optional[str] = None,
                        save_output: bool = False) -> None:
    """
    Performs depth estimation on the input image and displays the result.

    :param model (DepthAnythingV2): DepthAnythingV2 model.
    :param image (np.ndarray): Input RGB image.
    :param output_dir (str, optional): Directory to save output images.
    :param name (str, optional): Name of the image file for saving output.
    :param save_output (bool): Whether to save the output image.
    """
    depth = model.infer_image(image)
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_color = cv2.applyColorMap(depth_norm.astype(np.uint8),
                                    cv2.COLORMAP_INFERNO)
    concat_image = cv2.hconcat([image, depth_color])

    if save_output and output_dir and name:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'depth_{name}')
        cv2.imwrite(path, concat_image)

    cv2.imshow('RGB + Depth Map', concat_image)


def run_image_directory_mode(model: DepthAnythingV2,
                             image_dir: str,
                             output_dir: Optional[str],
                             save_output: bool) -> None:
    """
    Runs inference on all images in a specified directory.

    :param model (DepthAnythingV2): DepthAnythingV2 model.
    :param image_dir (str): Directory containing input images.
    :param output_dir (str, optional): Directory to save output images.
    :param save_output (bool): Whether to save the output images.
    """
    files = sorted(f for f in os.listdir(image_dir) if f.endswith('.png'))
    for filename in files:
        path = os.path.join(image_dir, filename)
        image = cv2.imread(path)
        if image is None:
            print(f"[Warning] Failed to load image: {path}")
            continue
        process_and_display(model, image, output_dir, filename, save_output)
        if cv2.waitKey(1) == 27:
            break  # ESC


def run_webcam_mode(model: DepthAnythingV2,
                    cam_index: int,
                    output_dir: Optional[str],
                    save_output: bool) -> None:
    """
    Runs inference in real-time using a webcam feed.

    :param model (DepthAnythingV2): DepthAnythingV2 model.
    :param cam_index (int): Index of the webcam.
    :param output_dir (str, optional): Directory to save output images.
    :param save_output (bool): Whether to save the output images.
    """
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("[Error] Unable to open webcam.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to capture frame.")
            break
        filename = f'frame_{frame_count:04d}.png'
        process_and_display(model, frame, output_dir, filename, save_output)
        frame_count += 1
        if cv2.waitKey(1) == 27:
            break  # ESC
    cap.release()


def depth_inference_static():
    """
    Runs monocular depth inference on a single image and displays results.
    """
    encoder = 'vits'
    image_dir = 'datasets/rgbd_dataset_freiburg1_xyz/rgb'
    image_path = f'{image_dir}/1305031102.175304.png'
    checkpoint_dir = 'checkpoints'

    device = get_device()
    print(f"Using device: {device} and encoder: {encoder}")

    model = load_model(encoder, checkpoint_dir, device)

    rgb_image = cv2.imread(image_path)
    if rgb_image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    depth = model.infer_image(rgb_image)

    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_colored = cv2.applyColorMap(
        depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO
    )

    combined = cv2.hconcat([rgb_image, depth_colored])
    cv2.imshow('RGB + Depth Map', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def depth_inference_real_time():
    """
    Entry point of the application.
    """
    mode = 'images'  # 'images' or 'webcam'
    encoder = 'vits'
    image_dir = 'datasets/rgbd_dataset_freiburg1_xyz/rgb'
    output_dir = 'results/depth_maps'
    checkpoint_path = 'checkpoints'
    save_output = True
    camera_index = 0

    device = get_device()
    print(f"Using device: {device} and encoder: {encoder}")

    model = load_model(encoder, checkpoint_path, device)

    if mode == 'images':
        run_image_directory_mode(model, image_dir, output_dir, save_output)
    elif mode == 'webcam':
        run_webcam_mode(model, camera_index, output_dir, save_output)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    depth_inference_static()
