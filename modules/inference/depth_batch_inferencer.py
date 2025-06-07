"""
Batch monocular depth estimation tool using DepthAnythingV2.

Supports standalone CLI or ROS 2 usage to process image folders and export
depth maps in .npy format.
"""

import json
import cv2
from tqdm import tqdm
from pathlib import Path
from typing import List, Optional
import numpy as np

from modules.inference.depth_estimator import DepthAnythingV2Estimator


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
        device: str = 'cuda',
        scaling_factor: Optional[float] = None
    ) -> None:
        """
        Initializes estimator and creates output directory.

        Args:
            input_dir (Path): Folder with RGB input images (.png).
            output_dir (Path): Folder to save depth maps as .npy.
            encoder (str): Encoder type ('vits', 'vitb', etc.).
            checkpoint_dir (Path): Directory with model checkpoints.
            device (Optional[str]): Inference device ('cuda' or 'cpu').
            scaling_factor (Optional[float]): Optional scaling factor for
                depth.
        """
        self.input_dir = input_dir
        self.output_npy_dir = output_path / "depth_npy"
        self.output_png_dir = output_path / "depth_png"
        self.device = device
        self.scaling_factor = scaling_factor
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
        if self.scaling_factor:
            depth *= self.scaling_factor
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

        # Validate matching outputs
        num_input = len(image_paths)
        num_npy = len(list(self.output_npy_dir.glob("*.npy")))
        num_png = len(list(self.output_png_dir.glob("*.png")))

        if num_input != num_npy or num_input != num_png:
            print("[WARNING] Mismatch detected between RGB and depth outputs!")
        else:
            print("[✓] All depth outputs match RGB inputs.")

        summary = {
            "num_input_rgb": num_input,
            "num_output_npy": num_npy,
            "num_output_png": num_png,
            "match_npy": num_input == num_npy,
            "match_png": num_input == num_png,
            "all_match": (num_input == num_npy) and (num_input == num_png)
        }

        summary_path = self.output_npy_dir.parent / "inference_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        print(f"[✓] Summary saved to: {summary_path}")
