"""
Module for live point cloud visualization using Open3D.

This class creates an Open3D visualizer window, allowing continuous
updates of a 3D point cloud in real time. It is designed to be used in
conjunction with a point cloud generation system.
"""

import open3d as o3d


class LivePointCloudVisualizer:
    """
    A class to manage a real-time point cloud visualizer window using Open3D.
    """

    def __init__(self) -> None:
        """
        Initializes the visualizer window and an empty point cloud object.
        """
        self._visualizer = o3d.visualization.Visualizer()
        self._visualizer.create_window(
            window_name="Live Point Cloud",
            width=960,
            height=540
        )
        self._point_cloud = o3d.geometry.PointCloud()
        self._initialized = False

    def update(self, new_pcd: o3d.geometry.PointCloud) -> None:
        """
        Updates the displayed point cloud with new data.

        :param: new_pcd (o3d.geometry.PointCloud): The new point cloud data to
            display.
        """
        self._point_cloud.points = new_pcd.points
        self._point_cloud.colors = new_pcd.colors

        if not self._initialized:
            self._visualizer.add_geometry(self._point_cloud)
            self._initialized = True
        else:
            self._visualizer.update_geometry(self._point_cloud)

        self._visualizer.poll_events()
        self._visualizer.update_renderer()

    def close(self) -> None:
        """
        Gracefully closes the visualizer window.
        """
        self._visualizer.destroy_window()
