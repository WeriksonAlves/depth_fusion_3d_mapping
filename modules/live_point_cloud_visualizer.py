"""
    Script to visualize point clouds in a live window using Open3D.

    This module provides a class to create a visualizer window, update the
    point cloud in the visualizer, and close the visualizer window.
    The visualizer is designed to work with the PointCloudProcessor class
    to display the processed point clouds in real-time.
    The visualizer window is created with a specified width and height,
    and the point cloud is updated with new data as it becomes available.
    The visualizer uses Open3D's visualization capabilities to render the
    point cloud, allowing for real-time updates and interaction.
    The visualizer can be closed gracefully when no longer needed.
"""

import open3d as o3d


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
