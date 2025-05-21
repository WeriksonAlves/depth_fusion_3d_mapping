"""
Module for estimating depth maps from RGB images using the DepthAnythingV2
model. Includes a class that loads the model, performs inference, and manages
configuration.
"""

import os
from typing import Optional

import numpy as np
import torch

from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2


class DepthEstimator:
    """
    A wrapper class for the DepthAnythingV2 model that performs depth
    estimation from RGB images.
    """

    def __init__(self,
                 encoder: str = 'vits',
                 checkpoint_dir: str = 'checkpoints',
                 device: Optional[str] = None) -> None:
        """
        Initializes the DepthEstimator.

        :param: encoder (str): Model encoder to use ('vits', 'vitb', 'vitl',
            'vitg').
        :param: checkpoint_dir (str): Path to the directory containing the
            model checkpoint.
        :param: device (str, optional): Computation device ('cuda', 'cpu').
            Defaults to best available.
        """
        self.encoder = encoder.lower()
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(checkpoint_dir)

    def _load_model(self, checkpoint_dir: str) -> DepthAnythingV2:
        """
        Loads the DepthAnythingV2 model from the specified checkpoint.

        :param: checkpoint_dir (str): Path to the checkpoint directory.

        :return: DepthAnythingV2: The configured model ready for inference.

        :raise: ValueError: If an unsupported encoder is provided.
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

        if self.encoder not in model_configs:
            raise ValueError(f"Unsupported encoder: {self.encoder}")

        model = DepthAnythingV2(**model_configs[self.encoder])
        checkpoint_path = os.path.join(
            checkpoint_dir, f'depth_anything_v2_{self.encoder}.pth')
        model.load_state_dict(torch.load(
            checkpoint_path, map_location=self.device))
        return model.to(self.device).eval()

    def infer_depth(self, rgb_image: np.ndarray) -> torch.Tensor:
        """
        Performs depth inference from an input RGB image.

        A:param: rgb_image (np.ndarray): RGB image array (HxWx3).

        :return: orch.Tensor: Estimated depth map.
        """
        return self.model.infer_image(rgb_image)
