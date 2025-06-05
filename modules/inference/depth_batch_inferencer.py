"""
Batch monocular depth estimation tool using DepthAnythingV2.

Supports standalone CLI or ROS 2 usage to process image folders and export
depth maps in .npy format.
"""

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm
import rclpy
from rclpy.node import Node

from modules.inference.depth_estimator import DepthAnythingV2Estimator


class DepthBatchInferencer:
    """
    Runs monocular depth inference over a folder of RGB images.
    """

    def __init__(
        self,
        input_dir: Path,
        output_path: Path,
        encoder: str = 'vits',
        checkpoint_dir: Path = Path('checkpoints'),
        device: Optional[str] = None
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
        self.output_mono_dir = output_path / "depth_mono"
        self.output_png_dir = output_path / "depth_png"
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.estimator = DepthAnythingV2Estimator(
            encoder=encoder,
            checkpoint_dir=str(checkpoint_dir),
            device=self.device
        )

        self.output_mono_dir.mkdir(parents=True, exist_ok=True)
        self.output_png_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Inference device: {self.device}")
        print(f"[INFO] Output directory: {output_path}")

    def _load_images(self) -> List[Path]:
        """
        Loads PNG images from the input directory.

        Returns:
            List[Path]: Sorted list of input image paths.
        """
        image_paths = sorted(self.input_dir.glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(
                f"No PNG images found in: {self.input_dir}"
            )
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
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()

        # Show real image and depth image concatenated
        depth_scaled = (depth * 255 / np.max(depth)).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
        combined_image = cv2.hconcat([image, depth_colored])
        cv2.imshow("Image and Depth Map", combined_image)
        cv2.waitKey(1)

        # Save depth map as .npy file
        out_npy_file = self.output_mono_dir / img_path.with_suffix('.npy').name
        np.save(out_npy_file, depth)

        # Save depth map as .png for visualization
        out_png_file = self.output_png_dir / img_path.with_suffix('.png').name
        depth_scaled = (depth * 255 / np.max(depth)).astype(np.uint8)
        cv2.imwrite(str(out_png_file), depth_scaled)

    def run(self) -> None:
        """
        Runs batch inference for all input images.
        """
        image_paths = self._load_images()
        for img_path in tqdm(image_paths, desc="Estimating depth"):
            self._infer_and_save(img_path)
        cv2.destroyAllWindows()


class DepthBatchInferencerNode(Node):
    """
    ROS 2 node to execute depth inference on a folder of images.
    """

    def __init__(
        self,
        input_dir: str = "datasets/sample_rgb",
        output_dir: str = "datasets/sample_depth_mono",
        encoder: str = "vits",
        checkpoint_dir: str = "checkpoints"
    ) -> None:
        super().__init__('depth_batch_inferencer_node')

        self.get_logger().info("Initializing batch depth inference...")

        inferencer = DepthBatchInferencer(
            input_dir=Path(input_dir),
            output_dir=Path(output_dir),
            encoder=encoder,
            checkpoint_dir=Path(checkpoint_dir)
        )
        inferencer.run()

        self.get_logger().info("Depth batch inference completed.")


def main() -> None:
    """
    Entry point for ROS 2 execution.
    """
    rclpy.init()
    try:
        node = DepthBatchInferencerNode()
        rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("[Shutdown] Interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
