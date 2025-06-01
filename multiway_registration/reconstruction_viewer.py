import argparse
import json
import os
import open3d as o3d
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import struct


def load_point_cloud(pcd_path: str) -> o3d.geometry.PointCloud:
    print(f"[INFO] Loading point cloud from: {pcd_path}")
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"[OK] Loaded {len(pcd.points)} points.")
    return pcd


def visualize_open3d(pcd: o3d.geometry.PointCloud):
    print("[INFO] Launching Open3D visualizer...")
    o3d.visualization.draw_geometries([pcd])


class RVizPublisher(Node):
    def __init__(self, frame_id: str = "map"):
        super().__init__('rviz_publisher')
        self.publisher = self.create_publisher(PointCloud2, '/reconstruction_points', 10)
        self.frame_id = frame_id

    def publish_pointcloud(self, cloud: o3d.geometry.PointCloud):
        points = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors)

        if len(colors) != len(points):
            colors = np.zeros_like(points)

        data = []
        for i in range(points.shape[0]):
            x, y, z = points[i]
            r, g, b = (colors[i] * 255).astype(np.uint8)
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0]
            data.append([x, y, z, rgb])

        data_array = np.array(data, dtype=np.float32)

        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = 1
        msg.width = data_array.shape[0]
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = data_array.tobytes()

        self.publisher.publish(msg)
        self.get_logger().info(f"Published {msg.width} points to RViz.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd', type=str, default='datasets/lab_scene_kinect_xyz/reconstruction_d435.ply',
                        help="Path to the point cloud file.")
    parser.add_argument('--mode', type=str, choices=['open3d', 'rviz'], default='open3d',
                        help="Choose visualization mode.")
    args = parser.parse_args()

    pcd = load_point_cloud(args.pcd)

    if args.mode == 'open3d':
        visualize_open3d(pcd)

    elif args.mode == 'rviz':
        rclpy.init()
        node = RVizPublisher()
        node.publish_pointcloud(pcd)
        rclpy.spin_once(node, timeout_sec=2.0)  # Publica e encerra
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
