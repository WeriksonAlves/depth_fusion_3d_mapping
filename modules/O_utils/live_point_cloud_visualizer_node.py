"""
ROS 2 node for live point cloud visualization using Open3D.

Subscribes to a PointCloud2 topic and dynamically renders the 3D geometry.
Designed for SLAM, depth reconstruction, and RGB-D pipelines.
"""

from typing import Optional

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


class LivePointCloudVisualizer:
    """
    Manages Open3D real-time point cloud rendering.
    """

    def __init__(self) -> None:
        """
        Initializes the Open3D window and internal state.
        """
        self._visualizer = o3d.visualization.Visualizer()
        self._visualizer.create_window(
            window_name="Live Point Cloud Viewer",
            width=960,
            height=540
        )
        self._point_cloud = o3d.geometry.PointCloud()
        self._initialized = False

    def update(self, cloud: o3d.geometry.PointCloud) -> None:
        """
        Updates visualizer with a new point cloud.

        Args:
            cloud (o3d.geometry.PointCloud): Input cloud to display.
        """
        self._point_cloud.points = cloud.points
        self._point_cloud.colors = cloud.colors

        if not self._initialized:
            self._visualizer.add_geometry(self._point_cloud)
            self._initialized = True
        else:
            self._visualizer.update_geometry(self._point_cloud)

        self._visualizer.poll_events()
        self._visualizer.update_renderer()

    def close(self) -> None:
        """
        Destroys the Open3D window.
        """
        self._visualizer.destroy_window()


class VisualizePointCloudNode(Node):
    """
    ROS 2 node to subscribe and visualize PointCloud2 data using Open3D.
    """

    def __init__(self) -> None:
        super().__init__('visualize_pointcloud_node')

        topic = self.declare_parameter(
            'pointcloud_topic', '/o3d_points'
        ).get_parameter_value().string_value

        self.get_logger().info(f"Subscribing to: {topic}")

        self.subscription = self.create_subscription(
            PointCloud2,
            topic,
            self._callback,
            10
        )

        self.visualizer = LivePointCloudVisualizer()

    def _callback(self, msg: PointCloud2) -> None:
        """
        Callback to receive PointCloud2 and update the Open3D window.

        Args:
            msg (PointCloud2): Incoming point cloud message.
        """
        cloud = self._ros_to_o3d(msg)
        if cloud:
            self.visualizer.update(cloud)

    def _ros_to_o3d(
        self,
        msg: PointCloud2
    ) -> Optional[o3d.geometry.PointCloud]:
        """
        Converts ROS PointCloud2 message to Open3D point cloud.

        Args:
            msg (PointCloud2): Input ROS point cloud.

        Returns:
            o3d.geometry.PointCloud: Converted Open3D geometry.
        """
        dtype = np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('rgb', np.uint32)
        ])
        data = np.frombuffer(msg.data, dtype=dtype)

        if data.size == 0:
            return None

        points = np.vstack((data['x'], data['y'], data['z'])).T
        colors = np.zeros((len(data), 3), dtype=np.float32)

        for i, rgb in enumerate(data['rgb']):
            r = (rgb >> 16) & 255
            g = (rgb >> 8) & 255
            b = rgb & 255
            colors[i] = [r, g, b]
        colors /= 255.0

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)

        return cloud


def main() -> None:
    """
    Entry point to run the visualizer node.
    """
    rclpy.init()
    try:
        node = VisualizePointCloudNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("[Shutdown] Interrupted by user.")
    finally:
        node.visualizer.close()
        node.destroy_node()
        rclpy.shutdown()
