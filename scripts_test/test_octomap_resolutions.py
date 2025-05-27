import os
import time
import subprocess

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from modules.depth_estimator import DepthAnythingV2Estimator
from modules.point_cloud_processor import PointCloudProcessor


class OctomapResolutionTester(Node):
    def __init__(self):
        super().__init__('octomap_resolution_tester')

        self.estimator = DepthAnythingV2Estimator(checkpoint_dir='checkpoints')
        self.processor = PointCloudProcessor(ros_node=self)

        self.publisher = self.create_publisher(
            msg_type=self.processor._publisher.msg_type,
            topic=self.processor._publisher.topic_name,
            qos_profile=10
        )

    def set_octomap_resolution(self, resolution: float) -> None:
        """
        Updates the resolution of the octomap_server via parameter set.
        """
        subprocess.run([
            'ros2', 'param', 'set',
            '/octomap_server',
            'resolution',
            str(resolution)
        ])
        self.get_logger().info(f"Set resolution: {resolution}")
        time.sleep(2.0)

    def publish_pointcloud(self, msg) -> None:
        """
        Publishes the given PointCloud2 multiple times to ensure reception.
        """
        for _ in range(10):
            msg.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(msg)
            time.sleep(0.4)
        self.get_logger().info("Published point cloud to /o3d_points.")

    def export_octomap(self, filename: str, file_format: str = 'bt') -> None:
        """
        Calls ros2 CLI to save the current octomap to disk.
        """
        print(f"Exporting octomap to {filename}...")
        subprocess.run([
            'ros2', 'service', 'call',
            '/octomap_server/save_map',
            'octomap_msgs/srv/SaveMap',
            f'{{filename: "{filename}", file_format: "{file_format}"}}'
        ])

    def run_tests(self):
        """
        Executes a series of resolution tests with map export.
        """
        img_path = 'datasets/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png'
        rgb = cv2.imread(img_path)
        if rgb is None:
            self.get_logger().error("Failed to load test image.")
            return

        depth = self.estimator.infer_depth(rgb)
        pcd = self.processor.create_point_cloud(rgb, depth)
        pcd_filtered = self.processor.filter_point_cloud(pcd)
        msg = self.processor._convert_to_ros_msg(pcd_filtered)

        resolutions = [0.05, 0.1, 0.2, 0.3]
        for res in resolutions:
            self.set_octomap_resolution(res)
            self.publish_pointcloud(msg)

            filename = os.path.abspath(
                f"results/octomap/octomap_res_{str(res).replace('.', '_')}.bt"
            )
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            self.export_octomap(filename)

            input(f"[NEXT] Press Enter to continue to next resolution...")


def main():
    rclpy.init()
    node = OctomapResolutionTester()
    try:
        node.run_tests()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
