"""
Public API for the 'inference' submodule.

Provides access to monocular depth estimation using the DepthAnythingV2 model.
Includes batch processing tools and ROS 2 integration.
"""

from modules.inference.depth_estimator import DepthAnythingV2Estimator
from modules.inference.depth_batch_inferencer import DepthBatchInferencer

__all__ = [
    "DepthAnythingV2Estimator",
    "DepthBatchInferencer",
]
