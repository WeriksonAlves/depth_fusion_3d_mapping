"""
    Script to process point clouds from RGB and depth images.
    This module provides a class to create point clouds from RGB and depth
    images, filter point clouds using voxel downsampling and statistical
    outlier removal, and visualize point clouds in a live window.
    The PointCloudProcessor class handles the creation and filtering of
    point clouds, while the LivePointCloudVisualizer class manages the
    visualization of point clouds in a live window.
"""

import numpy as np
import open3d as o3d


class PointCloudProcessor:
    """
    A class to process point clouds from RGB and depth images.
    This class provides methods to create point clouds from RGB and depth
    images, filter point clouds using voxel downsampling and statistical
    outlier removal, and visualize point clouds in a live window.
    """
    def __init__(self, fx=525.0, fy=525.0, cx=319.5, cy=239.5, width=640,
                 height=480) -> None:
        """
        Initializes the PointCloudProcessor with camera intrinsics.

        Args:
            fx (float): Focal length in x direction.
            fy (float): Focal length in y direction.
            cx (float): Optical center in x direction.
            cy (float): Optical center in y direction.
            width (int): Width of the image.
            height (int): Height of the image.
        """
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx,
                                                            fy, cx, cy)

    def create_point_cloud(self, rgb_img: np.ndarray, depth_map: np.ndarray,
                           depth_scale=1000.0, depth_trunc=4.0):
        """
        Creates a point cloud from RGB and depth images.

        Args:
            rgb_img (np.ndarray): The input RGB image as a NumPy array.
            depth_map (np.ndarray): The input depth map as a NumPy array.
            depth_scale (float): Scale factor for depth values.
            depth_trunc (float): Truncation distance for depth values.

        Returns:
            o3d.geometry.PointCloud: The generated point cloud.
        """
        rgb_o3d = o3d.geometry.Image(rgb_img)
        depth_o3d = o3d.geometry.Image(
            (depth_map * depth_scale).astype(np.uint16))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, depth_scale, depth_trunc,
            convert_rgb_to_intensity=False)
        return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,
                                                              self.intrinsics)

    def filter_point_cloud(self, pcd: o3d.geometry.PointCloud,
                           voxel_size=0.01) -> o3d.geometry.PointCloud:
        """
        Filters the point cloud using voxel downsampling and statistical
        outlier removal.

        Args:
            pcd (o3d.geometry.PointCloud): The input point cloud.
            voxel_size (float): Size of the voxel for downsampling.

        Returns:
            o3d.geometry.PointCloud: The filtered point cloud.
        """
        pcd = pcd.voxel_down_sample(voxel_size)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        return pcd


class LivePointCloudVisualizer:
    """
    A class to visualize point clouds in a live window using Open3D.
    This class provides methods to update the point cloud in the visualizer
    and close the visualizer window.
    """

    def __init__(self) -> None:
        """
        Initializes the visualizer with a window and a point cloud object.
        """
        # Create a visualizer window
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Live Point Cloud", 960, 540)
        self.pcd = o3d.geometry.PointCloud()
        self.initialized = False

    def update(self, new_pcd: o3d.geometry.PointCloud) -> None:
        """
        Updates the point cloud in the visualizer.

        Args:
            new_pcd: The new point cloud to display.
        """
        self.pcd.points = new_pcd.points
        self.pcd.colors = new_pcd.colors
        if not self.initialized:
            self.vis.add_geometry(self.pcd)
            self.initialized = True
        else:
            self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self) -> None:
        """
        Closes the visualizer window.
        """
        self.vis.destroy_window()
