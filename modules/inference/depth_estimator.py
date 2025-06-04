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

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'depth_anything_v2_{self.encoder}.pth'
        )

        model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        return model.to(self.device).eval()

    def infer_depth(self, rgb_image: np.ndarray) -> torch.Tensor:
        """
        Estimates depth from an RGB input image.

        Args:
            rgb_image (np.ndarray): RGB image (shape: H x W x 3).

        Returns:
            torch.Tensor: Estimated depth map as tensor.
        """
        return self.model.infer_image(rgb_image)
