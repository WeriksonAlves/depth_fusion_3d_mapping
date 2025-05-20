"""
    Script to estimate depth maps from RGB images using the DepthAnythingV2
    model. This module provides a class to load the model, perform depth
    inference, and handle the model's configuration.
"""
import os
# import sys
import torch
import numpy as np

from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2


class DepthEstimator:
    """
    A class to estimate depth maps from RGB images using the DepthAnythingV2
    model. This class provides methods to load the model, perform depth
    inference, and handle the model's configuration.
    """

    def __init__(self, encoder='vits',
                 checkpoint_dir='checkpoints',
                 device=None) -> None:
        """
        Initializes the DepthEstimator with the specified encoder, checkpoint
        directory, and device.

        Args:
            encoder (str): The encoder type to use for the model. Options are
                'vits', 'vitb', 'vitl', 'vitg'.
            checkpoint_dir (str): Directory path where the model checkpoint is
                stored.
            device (str, optional): The device to run the model on. If None,
                defaults to 'cuda' if available, otherwise 'cpu'.
        Raises:
            ValueError: If the specified encoder is not supported.
        """
        self.encoder = encoder
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(checkpoint_dir)

    def _load_model(self, checkpoint_dir: str) -> DepthAnythingV2:
        """
        Loads the DepthAnythingV2 model with the specified encoder and
        checkpoint directory.

        Args:
            checkpoint_dir (str): Directory path where the model checkpoint is
                stored.
        Returns:
            DepthAnythingV2: The loaded model.
        Raises:
            ValueError: If the specified encoder is not supported.
        """
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64,
                     'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128,
                     'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256,
                     'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384,
                     'out_channels': [1536, 1536, 1536, 1536]}
        }
        model = DepthAnythingV2(**model_configs[self.encoder])
        ckpt_path = os.path.join(
            checkpoint_dir, f'depth_anything_v2_{self.encoder}.pth')
        model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        return model.to(self.device).eval()

    def infer_depth(self, rgb_img: np.ndarray) -> torch.Tensor:
        """
        Infers the depth map from the input RGB image.

        Args:
            rgb_img: The input RGB image as a PyTorch tensor.
        Returns:
            torch.Tensor: The inferred depth map as a PyTorch tensor.
        """
        return self.model.infer_image(rgb_img)
