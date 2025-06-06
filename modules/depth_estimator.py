"""
Depth estimation module using the DepthAnythingV2 model.

This class handles model loading, configuration management,
and depth inference from RGB images.
"""

import os
from typing import Optional

import numpy as np
import torch

from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2


class DepthAnythingV2Estimator:
    """
    A wrapper class for the DepthAnythingV2 model to estimate depth
    from RGB images using different encoder configurations.
    """

    def __init__(
        self,
        encoder: str = 'vits',
        checkpoint_dir: str = 'checkpoints',
        device: Optional[str] = None
    ) -> None:
        """
        Initializes the estimator with the selected encoder and device.

        Args:
            encoder (str): Encoder type ('vits', 'vitb', 'vitl', 'vitg').
            checkpoint_dir (str): Directory containing model checkpoints.
            device (Optional[str]): Device to run inference ('cuda' or 'cpu').
        """
        self.encoder = encoder.lower()
        self.device = device or self._select_device()
        self.model = self._load_model(checkpoint_dir)

    def _select_device(self) -> str:
        """
        Automatically selects the best available device.

        Returns:
            str: 'cuda' if available, else 'cpu'.
        """
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def _load_model(self, checkpoint_dir: str) -> DepthAnythingV2:
        """
        Loads the model from a checkpoint file.

        Args:
            checkpoint_dir (str): Path to model checkpoints.

        Returns:
            DepthAnythingV2: Loaded model ready for inference.

        Raises:
            ValueError: If the encoder type is not supported.
        """
        config = self._get_model_config()
        model = DepthAnythingV2(**config)

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'depth_anything_v2_{self.encoder}.pth'
        )
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        return model.to(self.device).eval()

    def _get_model_config(self) -> dict:
        """
        Returns the configuration dictionary for the selected encoder.

        Returns:
            dict: Model configuration.

        Raises:
            ValueError: If encoder is not one of the supported types.
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

    def infer_depth(self, rgb_image: np.ndarray) -> torch.Tensor:
        """
        Performs depth estimation on a given RGB image.

        Args:
            rgb_image (np.ndarray): Input RGB image (H x W x 3).

        Returns:
            torch.Tensor: Estimated depth map.
        """
        return self.model.infer_image(rgb_image)
