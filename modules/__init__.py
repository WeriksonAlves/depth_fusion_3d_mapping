"""
Top-level namespace for the SIBGRAPI2025_SLAM project modules.

This package exposes the core submodules:
- inference: monocular depth estimation
- reconstruction: 3D mapping and registration
- evaluation: comparison and alignment tools
- utils: ROS 2 and Open3D utilities
"""

from modules import inference
from modules import reconstruction
from modules import utils

__all__ = [
    "inference",
    "reconstruction",
    "evaluation",
    "utils"
]
