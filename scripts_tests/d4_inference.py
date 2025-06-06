"""
Depth estimation module using the DepthAnythingV2 model.

This class handles model loading, encoder configuration,
device management, and RGB-to-depth inference.
"""

import sys

import os
import cv2
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2


class DepthAnythingV2Estimator:
    """
    Wrapper for the DepthAnythingV2 model to estimate depth from RGB input.
    """

    def __init__(
        self,
        encoder: str = 'vits',
        checkpoint_dir: str = 'checkpoints',
        device: Optional[str] = None
    ) -> None:
        """
        Initializes the model with selected encoder and device.

        Args:
            encoder (str): Encoder type ('vits', 'vitb', 'vitl', 'vitg').
            checkpoint_dir (str): Directory containing model checkpoint.
            device (Optional[str]): Device to use ('cuda' or 'cpu').
        """
        self.encoder = encoder.lower()
        self.device = device or self._select_device()
        self.model = self._load_model(checkpoint_dir)

    def _select_device(self) -> str:
        """
        Automatically selects available compute device.

        Returns:
            str: 'cuda' if GPU is available, otherwise 'cpu'.
        """
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def _get_model_config(self) -> dict:
        """
        Returns the encoder-specific configuration.

        Returns:
            dict: Model initialization config.

        Raises:
            ValueError: If encoder is not supported.
        """
        configs = {
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
        if self.encoder not in configs:
            raise ValueError(f"Unsupported encoder: {self.encoder}")
        return configs[self.encoder]

    def _load_model(self, checkpoint_dir: str) -> DepthAnythingV2:
        """
        Loads the model weights from the checkpoint.

        Args:
            checkpoint_dir (str): Directory where weights are stored.

        Returns:
            DepthAnythingV2: Model ready for inference.
        """
        config = self._get_model_config()
        model = DepthAnythingV2(**config)
        ckpt_path = os.path.join(
            checkpoint_dir,
            f'depth_anything_v2_{self.encoder}.pth'
        )
        model.load_state_dict(
            torch.load(ckpt_path, map_location=self.device)
        )
        return model.to(self.device).eval()

    def infer_depth(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Estimates depth from an RGB input image.

        Args:
            rgb_image (np.ndarray): RGB image (shape: H x W x 3).

        Returns:
            torch.Tensor: Estimated depth map as tensor.
        """
        depth = self.model.infer_image(rgb_image)
        return depth.cpu().numpy() if isinstance(depth,
                                                 torch.Tensor) else depth


"""
Batch monocular depth estimation tool using DepthAnythingV2.

Supports standalone CLI or ROS 2 usage to process image folders and export
depth maps in .npy format.
"""


class DepthBatchInferencer:
    """
    Performs depth estimation over a directory of RGB images.
    """

    def __init__(
        self,
        input_dir: Path,
        output_path: Path,
        encoder: str = 'vits',
        checkpoint_dir: Path = Path('checkpoints'),
        device: str = 'cuda'
    ) -> None:
        """
        Initializes estimator and creates output directory.

        Args:
            input_dir (Path): Folder with RGB input images (.png).
            output_dir (Path): Folder to save depth maps as .npy.
            encoder (str): Encoder type ('vits', 'vitb', etc.).
            checkpoint_dir (Path): Directory with model checkpoints.
            device (Optional[str]): Inference device ('cuda' or 'cpu').
        """
        self.input_dir = input_dir
        self.output_npy_dir = output_path / "depth_npy"
        self.output_png_dir = output_path / "depth_png"
        self.device = device
        self.estimator = DepthAnythingV2Estimator(
            encoder=encoder,
            checkpoint_dir=str(checkpoint_dir),
            device=self.device
        )

        self.output_npy_dir.mkdir(parents=True, exist_ok=True)
        self.output_png_dir.mkdir(parents=True, exist_ok=True)

    def _load_images(self) -> List[Path]:
        """
        Loads PNG images from the input directory.

        Returns:
            List[Path]: Sorted list of input image paths.
        """
        image_paths = sorted(self.input_dir.glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(f"No .png images in: {self.input_dir}")
        return image_paths

    def _infer_and_save(self, img_path: Path) -> None:
        """
        Runs inference and saves the depth map.

        Args:
            img_path (Path): Path to the input image.
        """
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[Warning] Could not read image: {img_path}")
            return

        depth = self.estimator.infer_depth(image)
        npy_path = self.output_npy_dir / img_path.with_suffix('.npy').name
        np.save(npy_path, depth)

        # Save depth visualization
        depth_scaled = (depth * 255 / np.max(depth)).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
        png_path = self.output_png_dir / img_path.with_suffix('.png').name
        cv2.imwrite(str(png_path), depth_colored)

    def run(self) -> None:
        """
        Runs batch inference for all input images.
        """
        image_paths = self._load_images()
        for img_path in tqdm(image_paths, desc="Estimating depth"):
            self._infer_and_save(img_path)


def main() -> None:
    scene = "lab_scene_f"
    input_dir = Path(f"datasets/{scene}/rgb")
    output_dir = Path(f"results/{scene}/estimated")
    checkpoint_dir = Path("checkpoints")
    encoder = "vits"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input not found: {input_dir}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    inferencer = DepthBatchInferencer(
        input_dir=input_dir,
        output_path=output_dir,
        encoder=encoder,
        checkpoint_dir=checkpoint_dir
    )

    print("[INFO] Running depth inference...")
    inferencer.run()
    print("[✓] Inference complete.")


if __name__ == "__main__":
    main()
