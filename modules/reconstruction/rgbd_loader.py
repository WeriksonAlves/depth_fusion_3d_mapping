"""
Loads RGB and Depth data, converts them into point clouds using Open3D.
This module provides functionality to read RGB images and depth maps,
and convert them into point clouds using the Open3D library.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List


class RGBDLoader:
    """
    Loads RGB and depth pairs (depth as .npy) and converts them into
    Open3D point clouds with optional voxel downsampling and normal estimation.
    """

    @staticmethod
    def load_point_clouds(
        rgb_dir: Path,
        depth_dir: Path,
        intrinsic: o3d.camera.PinholeCameraIntrinsic,
        voxel_size: float,
        scale_correction: float = 1.0
    ) -> List[o3d.geometry.PointCloud]:
        """
        Loads and processes RGB and depth pairs into downsampled point clouds.

        Args:
            rgb_dir (Path): Directory containing RGB images (.png).
            depth_dir (Path): Directory containing depth maps (.npy).
            intrinsic (o3d.camera.PinholeCameraIntrinsic): Camera intrinsics.
            voxel_size (float): Size of voxel for downsampling.
            scale_correction (float): Scale factor for depth correction.

        Returns:
            List[o3d.geometry.PointCloud]: List of processed point clouds.
        """
        rgb_paths = sorted(rgb_dir.glob("*.png"))
        depth_paths = sorted(depth_dir.glob("*.npy"))

        point_clouds = []
        for rgb_path, depth_path in zip(rgb_paths, depth_paths):
            color_image = RGBDLoader._load_color_image(rgb_path)
            depth_image = RGBDLoader._load_depth_image(
                depth_path, scale_correction
            )
            rgbd_image = RGBDLoader._create_rgbd_image(
                color_image, depth_image
            )
            pcd = RGBDLoader._create_point_cloud_from_rgbd(
                rgbd_image, intrinsic
            )  # Read .ply
            pcd_down = RGBDLoader._downsample_and_estimate_normals(
                pcd, voxel_size
            )
            point_clouds.append(pcd_down)

        return point_clouds

    @staticmethod
    def _load_color_image(rgb_path: Path) -> o3d.geometry.Image:
        """Loads an RGB image from disk."""
        return o3d.io.read_image(str(rgb_path))

    @staticmethod
    def _load_depth_image(
        depth_path: Path,
        scale_correction: float
    ) -> o3d.geometry.Image:
        """Loads a depth .npy file and converts it to Open3D Image in mm."""
        depth_np = np.load(depth_path) * scale_correction
        depth_mm = (depth_np * 1000.0).astype(np.uint16)
        return o3d.geometry.Image(depth_mm)

    @staticmethod
    def _create_rgbd_image(
        color: o3d.geometry.Image,
        depth: o3d.geometry.Image
    ) -> o3d.geometry.RGBDImage:
        """Creates an Open3D RGBD image from color and depth."""
        return o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1000.0,
            convert_rgb_to_intensity=False
        )

    @staticmethod
    def _create_point_cloud_from_rgbd(
        rgbd: o3d.geometry.RGBDImage,
        intrinsic: o3d.camera.PinholeCameraIntrinsic
    ) -> o3d.geometry.PointCloud:
        """Generates a point cloud from RGBD and intrinsics."""
        return o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsic
        )

    @staticmethod
    def _downsample_and_estimate_normals(
        pcd: o3d.geometry.PointCloud,
        voxel_size: float
    ) -> o3d.geometry.PointCloud:
        """
        Applies voxel downsampling and normal estimation.

        Args:
            pcd (o3d.geometry.PointCloud): Original point cloud.
            voxel_size (float): Size of voxel grid for downsampling.

        Returns:
            o3d.geometry.PointCloud: Processed point cloud.
        """
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2.0,
                max_nn=30
            )
        )
        return pcd_down
