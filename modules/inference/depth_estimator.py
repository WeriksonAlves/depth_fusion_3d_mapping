"""
Depth estimation module using the DepthAnythingV2 model.

This class handles model loading, encoder configuration,
device management, and RGB-to-depth inference.
"""

import os
from typing import Optional
import numpy as np
import torch

from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2


class DepthAnythingV2Estimator:
    """
    Wrapper for the DepthAnythingV2 model.

    Responsible for loading model weights, setting device, and running
    depth inference from RGB images.
    """

    def __init__(
        self,
        encoder: str = "vits",
        checkpoint_dir: str = "checkpoints",
        device: Optional[str] = None
    ) -> None:
        """
        Initializes the estimator with model configuration and device.

        Args:
            encoder (str): Encoder type ('vits', 'vitb', 'vitl', 'vitg').
            checkpoint_dir (str): Path to the folder containing checkpoints.
            device (str, optional): Inference device ('cuda' or 'cpu').
        """
        self.encoder = encoder.lower()
        self.device = device or self._select_device()
        self.model = self._initialize_model(checkpoint_dir)

    def _select_device(self) -> str:
        """
        Selects the best available compute device.

        Returns:
            str: 'cuda' if available, otherwise 'cpu'.
        """
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_encoder_config(self) -> dict:
        """
        Retrieves configuration parameters based on encoder type.

        Returns:
            dict: Encoder-specific model configuration.

        Raises:
            ValueError: If encoder type is unsupported.
        """
        configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384]
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768]
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024]
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536]
            }
        }

        if self.encoder not in configs:
            raise ValueError(f"Unsupported encoder: {self.encoder}")

        return configs[self.encoder]

    def _initialize_model(self, checkpoint_dir: str) -> DepthAnythingV2:
        """
        Loads model weights and prepares model for inference.

        Args:
            checkpoint_dir (str): Directory containing model weights.

        Returns:
            DepthAnythingV2: Initialized and ready-to-use model.
        """
        config = self._get_encoder_config()
        model = DepthAnythingV2(**config)

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"depth_anything_v2_{self.encoder}.pth"
        )

        model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )

        return model.to(self.device).eval()

    def infer_depth(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Runs depth inference from a single RGB image.

        Args:
            rgb_image (np.ndarray): RGB image with shape (H, W, 3).

        Returns:
            np.ndarray: Depth map as a 2D NumPy array.
        """
        depth = self.model.infer_image(rgb_image)

        if isinstance(depth, torch.Tensor):
            return depth.cpu().numpy()
        return depth
