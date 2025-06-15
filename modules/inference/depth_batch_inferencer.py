"""
Batch monocular depth estimation tool using DepthAnythingV2.

Supports standalone CLI or ROS 2 usage to process image folders and export
depth maps in .npy format.
"""

import json
import cv2
from tqdm import tqdm
from pathlib import Path
from typing import List
import numpy as np

from modules.inference.depth_estimator import DepthAnythingV2Estimator


class DepthBatchInferencer:
    """
    Performs batch monocular depth estimation using DepthAnythingV2.

    Loads RGB images from a folder, estimates their depth, and saves results
    as .npy and visualized .png files. A summary of the process is also saved.
    """

    def __init__(
        self,
        rgb_dir: Path,
        output_path: Path,
        encoder: str = "vits",
        checkpoint_dir: Path = Path("checkpoints"),
        device: str = "cuda",
    ) -> None:
        """
        Initializes the depth estimator and output directories.

        Args:
            rgb_dir (Path): Directory containing RGB .png images.
            output_path (Path): Directory where output files will be saved.
            encoder (str): Model encoder type (e.g., 'vits', 'vitb').
            checkpoint_dir (Path): Directory containing model weights.
            device (str): Inference device ('cuda' or 'cpu').
        """
        self.rgb_dir = rgb_dir
        self.output_npy_dir = output_path / "depth_npy"
        self.output_png_dir = output_path / "depth_png"
        self.checkpoint_dir = checkpoint_dir
        self.device = device

        self._validate_paths()

        self.estimator = DepthAnythingV2Estimator(
            encoder=encoder,
            checkpoint_dir=str(self.checkpoint_dir),
            device=self.device
        )

    def _validate_paths(self) -> None:
        """Checks for required directories and files."""
        for path in [
            self.rgb_dir,
            self.checkpoint_dir
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Missing path: {path}")
        self.output_npy_dir.mkdir(parents=True, exist_ok=True)
        self.output_png_dir.mkdir(parents=True, exist_ok=True)

    def _load_image_paths(self) -> List[Path]:
        """
        Retrieves all .png images from the input directory.

        Returns:
            List[Path]: Sorted list of image paths.
        """
        image_paths = sorted(self.rgb_dir.glob("*.png"))
        return image_paths

    def _save_depth_outputs(
        self,
        img_path: Path,
        depth_m: np.ndarray,
        depth_norm_png: np.ndarray
    ) -> None:
        """
        Saves the raw and visualized depth maps.

        Args:
            img_path (Path): Path of the original RGB image.
            depth_m (np.ndarray): Raw depth map in meters.
            depth_norm_png (np.ndarray): Normalized depth map for
                visualization.
        """

        npy_path = self.output_npy_dir / img_path.with_suffix(".npy").name
        np.save(npy_path, depth_m)

        png_path = self.output_png_dir / img_path.with_suffix(".png").name
        cv2.imwrite(str(png_path), depth_norm_png)

    def _process_single_image(self, img_path: Path) -> None:
        """
        Processes a single image: inference and saving depth maps.

        Args:
            img_path (Path): Path to the input image.
        """
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[Warning] Failed to load image: {img_path}")
            return

        depth_m = self.estimator.infer_depth(image)
        depth_norm_png = self.estimator.normalize_png_depth(depth_m)
        self._save_depth_outputs(img_path, depth_m, depth_norm_png)

    def _generate_summary(self, num_input: int) -> None:
        """
        Verifies output consistency and saves an inference summary.

        Args:
            num_input (int): Total number of input images processed.
        """
        num_npy = len(list(self.output_npy_dir.glob("*.npy")))
        num_png = len(list(self.output_png_dir.glob("*.png")))

        summary = {
            "num_input_rgb": num_input,
            "num_output_npy": num_npy,
            "num_output_png": num_png,
            "match_npy": num_input == num_npy,
            "match_png": num_input == num_png,
            "all_match": (num_input == num_npy) and (num_input == num_png)
        }

        summary_path = self.output_npy_dir.parent / "inference_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        print(f"[✓] Summary saved to: {summary_path}")

        if not summary["all_match"]:
            print("[Warning] Mismatch between RGB and depth outputs!")
        else:
            print("[✓] All depth outputs match the input images.")

    def run(self) -> None:
        """
        Runs the batch depth estimation process for all input images.
        """
        image_paths = self._load_image_paths()
        for img_path in tqdm(image_paths, desc="Estimating depth"):
            self._process_single_image(img_path)

        self._generate_summary(len(image_paths))
