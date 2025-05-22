"""
Module for processing 3D point clouds from RGB and depth images
using Open3D. It provides utilities for creating and filtering
point clouds using voxel downsampling and statistical filtering.
"""


import numpy as np
import open3d as o3d


class PointCloudProcessor:
    """
    Handles creation and filtering of point clouds from RGB and
    depth image inputs.
    """

    def __init__(self,
                 fx: float = 525.0,
                 fy: float = 525.0,
                 cx: float = 319.5,
                 cy: float = 239.5,
                 width: int = 640,
                 height: int = 480) -> None:
        """
        Initializes the processor with pinhole camera intrinsics.

        :param: fx (float): Focal length along the X-axis.
        :param: fy (float): Focal length along the Y-axis.
        :param: cx (float): Principal point X-coordinate.
        :param: cy (float): Principal point Y-coordinate.
        :param: width (int): Image width.
        :param: height (int): Image height.
        """
        self._intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )

    def create_point_cloud(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        depth_scale: float = 1000.0,
        depth_trunc: float = 4.0
    ) -> o3d.geometry.PointCloud:
        """
        Creates a point cloud from RGB and depth input images.

        :param: rgb_image (np.ndarray): RGB image (H x W x 3).
        :param: depth_map (np.ndarray): Depth map (H x W) in meters.
        :param: depth_scale (float): Factor to scale depth values.
        :param: depth_trunc (float): Maximum depth range to keep.

        :return: o3d.geometry.PointCloud: Output point cloud.
        """
        rgb_o3d = o3d.geometry.Image(rgb_image)
        depth_scaled = (depth_map * depth_scale).astype(np.uint16)
        depth_o3d = o3d.geometry.Image(depth_scaled)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )

        return o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, self._intrinsics
        )

    def filter_point_cloud(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float = 0.01,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0
    ) -> o3d.geometry.PointCloud:
        """
        Applies voxel downsampling and statistical outlier removal.

        :param: pcd (o3d.geometry.PointCloud): Input point cloud.
        :param: voxel_size (float): Voxel size for downsampling.
        :param: nb_neighbors (int): Number of neighbors for filtering.
        :param: std_ratio (float): Threshold for statistical outliers.
        :return: o3d.geometry.PointCloud: Filtered point cloud.
        """
        downsampled = pcd.voxel_down_sample(voxel_size)
        filtered, _ = downsampled.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        return filtered

    def save_point_cloud(self,
                         pcd: o3d.geometry.PointCloud,
                         filename: str = "output_map.pcd") -> None:
        """
        Saves the point cloud to a file.

        :param: pcd (o3d.geometry.PointCloud): Input point cloud.
        :param: filename (str): Filename to save the point cloud.
        """
        o3d.io.write_point_cloud(filename, pcd)
