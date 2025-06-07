# modules/reconstruction/rgbd_loader.py

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
    Converts RGB + depth pairs (depth in .npy) into Open3D point clouds.
    """

    @staticmethod
    def load_point_clouds(
        rgb_dir: Path,
        depth_dir: Path,
        intrinsic: o3d.camera.PinholeCameraIntrinsic,
        voxel_size: float,
        scale_correction: float = 1.0
    ) -> List[o3d.geometry.PointCloud]:
        rgb_paths = sorted(rgb_dir.glob("*.png"))
        depth_paths = sorted(depth_dir.glob("*.npy"))

        point_clouds = []
        for rgb_path, depth_path in zip(rgb_paths, depth_paths):
            color = o3d.io.read_image(str(rgb_path))
            depth_np = np.load(depth_path) * scale_correction
            depth_mm = (depth_np * 1000.0).astype(np.uint16)
            depth = o3d.geometry.Image(depth_mm)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0,
                convert_rgb_to_intensity=False
            )

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, intrinsic
            )
            pcd_down = pcd.voxel_down_sample(voxel_size)
            pcd_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 2.0, max_nn=30
                )
            )
            point_clouds.append(pcd_down)

        return point_clouds
