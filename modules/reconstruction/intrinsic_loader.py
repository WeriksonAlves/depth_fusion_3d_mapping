# modules/reconstruction/intrinsic_loader.py

"""
Loads camera intrinsics from a JSON file in Open3D format.
This module provides a utility to read camera intrinsic parameters
from a JSON file and convert them into an Open3D camera intrinsic object.
"""

import json
import open3d as o3d
from pathlib import Path


class IntrinsicLoader:
    """
    Loads camera intrinsics from a JSON file formatted for Open3D.
    """

    @staticmethod
    def load_from_json(json_path: Path) -> o3d.camera.PinholeCameraIntrinsic:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fx, fy = data["K"][0], data["K"][4]
        cx, cy = data["K"][2], data["K"][5]
        width, height = data["width"], data["height"]
        return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
