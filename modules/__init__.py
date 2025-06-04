"""
Public API for the 'modules' package.

Exposes the main classes used for depth estimation, point cloud processing,
and live visualization from RGB-D data.
"""

from modules.depth_estimator import DepthAnythingV2Estimator
from modules.point_cloud_processor import PointCloudProcessor
from modules.live_point_cloud_visualizer import LivePointCloudVisualizer

__all__ = [
    "DepthAnythingV2Estimator",
    "PointCloudProcessor",
    "LivePointCloudVisualizer"
]
