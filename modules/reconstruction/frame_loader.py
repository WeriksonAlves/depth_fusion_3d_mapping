"""
Module to load and preprocess RGB + depth image pairs as point clouds.

Supports both real RGB-D data and monocular depth maps (e.g., from inference).
Applies downsampling and normal estimation for reconstruction tasks.
"""

from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d


class FrameLoader:
    """
    Loads RGB and depth image pairs and converts them to Open3D point clouds.
    """

    def __init__(
        self,
        rgb_dir: Path,
        depth_dir: Path,
        intrinsics: o3d.camera.PinholeCameraIntrinsic,
        mode: str = "real",
        depth_scale: float = 5000.0,
        depth_trunc: float = 4.0,
        voxel_size: float = 0.02
    ) -> None:
        """
        Initializes frame loader configuration.

        Args:
            rgb_dir (Path): Path to RGB images.
            depth_dir (Path): Path to depth maps (.png or .npy).
            intrinsics (o3d.camera.PinholeCameraIntrinsic): Camera intrinsics.
            mode (str): 'real' for .png depths or 'mono' for .npy (in meters).
            depth_scale (float): Scaling factor to convert depth to mm.
            depth_trunc (float): Max depth value to accept in meters.
            voxel_size (float): Downsampling voxel size.
        """
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.intrinsics = intrinsics
        self.mode = mode
        self.depth_scale = depth_scale
        self.depth_trunc = depth_trunc
        self.voxel_size = voxel_size

    def load_point_clouds(self) -> List[o3d.geometry.PointCloud]:
        """
        Loads point clouds generated from RGB + depth image pairs.

        Returns:
            List[o3d.geometry.PointCloud]: Preprocessed point clouds.

        Raises:
            AssertionError: If the number of RGB and depth files differ.
        """
        ext = ".png" if self.mode == "real" else ".npy"
        rgb_paths = sorted(self.rgb_dir.glob("*.png"))
        depth_paths = sorted(self.depth_dir.glob(f"*{ext}"))

        assert len(rgb_paths) == len(depth_paths), (
            "[ERROR] Number of RGB and depth files must match."
        )

        return [
            self._load_single_pair(rgb_path, depth_path)
            for rgb_path, depth_path in zip(rgb_paths, depth_paths)
        ]

    def _load_single_pair(
        self,
        rgb_path: Path,
        depth_path: Path
    ) -> o3d.geometry.PointCloud:
        """
        Loads and processes a single RGB + depth pair into a point cloud.

        Args:
            rgb_path (Path): Path to RGB image (.png).
            depth_path (Path): Path to depth image (.png or .npy).

        Returns:
            o3d.geometry.PointCloud: Downsampled and normalized point cloud.
        """
        rgb = o3d.io.read_image(str(rgb_path))

        if self.mode == "real":
            depth = o3d.io.read_image(str(depth_path))
        else:
            raw_depth = np.load(depth_path).astype(np.float32)
            depth = o3d.geometry.Image(
                (raw_depth * self.depth_scale).astype(np.uint16)
            )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=self.depth_scale,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False
        )

        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, self.intrinsics
        )

        point_cloud = point_cloud.voxel_down_sample(self.voxel_size)
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
        )

        return point_cloud
