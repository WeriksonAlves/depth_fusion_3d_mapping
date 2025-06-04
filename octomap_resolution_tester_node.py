"""
ROS 2 node for testing different OctoMap resolutions by publishing
a point cloud generated from monocular depth and exporting the map.

This node uses DepthAnythingV2 for depth inference and Open3D to
generate a point cloud. It publishes this point cloud to /o3d_points
at different octomap resolutions and saves the result to disk.
"""

import time
import subprocess
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node

from modules.inference import DepthAnythingV2Estimator
from modules.utils import PointCloudProcessor


class OctomapResolutionTesterNode(Node):
    """
    ROS 2 node to perform a series of OctoMap resolution tests
    using monocular RGB input and DepthAnythingV2 inference.
    """

    def __init__(
        self,
        rgb_path: str = 'datasets/lab_scene_kinect_xyz/rgb/1305031102.175304.png',
        checkpoint_dir: str = 'checkpoints',
        output_dir: str = 'results/octomap',
        topic: str = '/o3d_points'
    ) -> None:
        super().__init__('octomap_resolution_tester_node')

        self.rgb_path = Path(rgb_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.estimator = DepthAnythingV2Estimator(
            checkpoint_dir=checkpoint_dir
        )
        self.processor = PointCloudProcessor(ros_node=self, topic=topic)

        self.publisher = self.create_publisher(
            msg_type=self.processor._publisher.msg_type,
            topic=self.processor._publisher.topic_name,
            qos_profile=10
        )

        self.get_logger().info("Octomap resolution tester initialized.")

    def set_octomap_resolution(self, resolution: float) -> None:
        """
        Updates the resolution of the OctoMap server via ROS 2 param.
        """
        self.get_logger().info(f"Setting OctoMap resolution to: {resolution}")
        subprocess.run([
            'ros2', 'param', 'set',
            '/octomap_server',
            'resolution',
            str(resolution)
        ])
        time.sleep(2.0)

    def publish_point_cloud(self, msg) -> None:
        """
        Publishes the point cloud multiple times for OctoMap to receive.
        """
        for _ in range(10):
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(msg)
            time.sleep(0.4)

        self.get_logger().info("Point cloud published to /o3d_points.")

    def export_octomap(self, filename: str, fmt: str = 'bt') -> None:
        """
        Exports the current OctoMap to disk using ros2 CLI.
        """
        self.get_logger().info(f"Saving OctoMap to: {filename}")
        subprocess.run([
            'ros2', 'service', 'call',
            '/octomap_server/save_map',
            'octomap_msgs/srv/SaveMap',
            f'{{filename: "{filename}", file_format: "{fmt}"}}'
        ])

    def run(self) -> None:
        """
        Executes tests at multiple OctoMap resolutions.
        """
        rgb = cv2.imread(str(self.rgb_path))
        if rgb is None:
            self.get_logger().error(f"Failed to load image: {self.rgb_path}")
            return

        self.get_logger().info("Inferring depth from RGB image...")
        depth = self.estimator.infer_depth(rgb)

        pcd = self.processor.create_point_cloud(rgb, depth)
        pcd_filtered = self.processor.filter_point_cloud(pcd)
        msg = self.processor._convert_to_ros_msg(pcd_filtered)

        for resolution in [0.05, 0.1, 0.2, 0.3]:
            self.set_octomap_resolution(resolution)
            self.publish_point_cloud(msg)

            filename = self.output_dir / f"octomap_res_{str(resolution).replace('.', '_')}.bt"
            self.export_octomap(str(filename))

            input("[NEXT] Press Enter to continue to next resolution...")


def main() -> None:
    """
    Entry point for running the OctoMap resolution test node.
    """
    rclpy.init()
    node = OctomapResolutionTesterNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
