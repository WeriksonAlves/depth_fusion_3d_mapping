import time
import subprocess

import rclpy
from modules.depth_estimator import DepthAnythingV2Estimator
from modules.point_cloud_processor import PointCloudProcessor

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import cv2
import numpy as np


def set_octomap_resolution(resolution: float) -> None:
    """
    Sets the resolution of the running octomap_server node.
    """
    subprocess.run([
        'ros2', 'param', 'set',
        '/octomap_server',
        'resolution',
        str(resolution)
    ])
    print(f"[INFO] Set resolution to {resolution} m")
    time.sleep(2)


def publish_static_pointcloud(pcd: PointCloud2, node, publisher) -> None:
    """
    Publishes the same point cloud multiple times to ensure reception.
    """
    for _ in range(10):
        pcd.header.stamp = node.get_clock().now().to_msg()
        publisher.publish(pcd)
        time.sleep(0.5)
    print("[INFO] Point cloud published.")


def main():
    rclpy.init()
    node = rclpy.create_node("octomap_resolution_tester")

    # Set up estimator and processor
    estimator = DepthAnythingV2Estimator(checkpoint_dir="checkpoints")
    processor = PointCloudProcessor(ros_node=node)

    publisher = node.create_publisher(PointCloud2, "/o3d_points", 10)

    # Load static image for test
    img_path = 'datasets/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png'
    rgb = cv2.imread(img_path)
    if rgb is None:
        print("[ERROR] Failed to load test image.")
        return

    # Generate depth and point cloud
    depth = estimator.infer_depth(rgb)
    pcd = processor.create_point_cloud(rgb, depth)
    pcd_filtered = processor.filter_point_cloud(pcd)
    ros_msg = processor._convert_to_ros_msg(pcd_filtered)


    # Test resolutions
    resolutions = [0.05, 0.1, 0.2, 0.3]
    for res in resolutions:
        set_octomap_resolution(res)
        publish_static_pointcloud(ros_msg, node, publisher)
        input(f"[NEXT] Press ENTER to continue to next resolution...")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
