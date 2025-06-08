"""
Loads camera intrinsics from a JSON file in Open3D format.
This module provides a utility to read camera intrinsic parameters
from a JSON file and convert them into an Open3D camera intrinsic object.
"""

import json
from pathlib import Path
import open3d as o3d


class IntrinsicLoader:
    """
    Utility to load camera intrinsic parameters from a JSON file
    formatted for Open3D.
    """

    @staticmethod
    def load_from_json(
        json_path: Path
    ) -> o3d.camera.PinholeCameraIntrinsic:
        """
        Loads Open3D camera intrinsics from a JSON file.

        Args:
            json_path (Path): Path to the JSON file.

        Returns:
            o3d.camera.PinholeCameraIntrinsic: Intrinsic camera parameters.
        """
        data = IntrinsicLoader._read_json(json_path)
        fx, fy, cx, cy = IntrinsicLoader._extract_parameters(data)
        width, height = data["width"], data["height"]

        return o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )

    @staticmethod
    def _read_json(json_path: Path) -> dict:
        """Reads and parses the intrinsic JSON file."""
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _extract_parameters(data: dict) -> tuple:
        """
        Extracts fx, fy, cx, cy from the 'K' matrix.

        Args:
            data (dict): Parsed JSON data.

        Returns:
            tuple: fx, fy, cx, cy values.
        """
        fx = data["K"][0]
        fy = data["K"][4]
        cx = data["K"][2]
        cy = data["K"][5]
        return fx, fy, cx, cy
