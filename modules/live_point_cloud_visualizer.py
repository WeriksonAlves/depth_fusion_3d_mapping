"""
Live visualization module for Open3D point clouds.

This class manages a real-time visualizer window using Open3D and updates
a 3D point cloud instance dynamically. It is intended to support any
point cloud generation pipeline.
"""

import open3d as o3d


class LivePointCloudVisualizer:
    """
    Handles real-time Open3D point cloud visualization.
    """

    def __init__(self) -> None:
        """
        Initializes the visualization window and internal state.
        """
        self._visualizer = o3d.visualization.Visualizer()
        self._visualizer.create_window(
            window_name="Live Point Cloud",
            width=960,
            height=540
        )
        self._point_cloud = o3d.geometry.PointCloud()
        self._initialized = False

    def update(self, point_cloud: o3d.geometry.PointCloud) -> None:
        """
        Updates the visualized geometry with the latest point cloud.

        Args:
            point_cloud (o3d.geometry.PointCloud): The new point cloud to
                render.
        """
        self._point_cloud.points = point_cloud.points
        self._point_cloud.colors = point_cloud.colors

        if not self._initialized:
            self._visualizer.add_geometry(self._point_cloud)
            self._initialized = True
        else:
            self._visualizer.update_geometry(self._point_cloud)

        self._visualizer.poll_events()
        self._visualizer.update_renderer()

    def close(self) -> None:
        """
        Closes the Open3D window and cleans up the visualizer.
        """
        self._visualizer.destroy_window()
