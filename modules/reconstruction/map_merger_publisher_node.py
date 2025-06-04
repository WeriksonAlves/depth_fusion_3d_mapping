"""
ROS 2 node for merging a list of point clouds and publishing the final map.

This node loads all PCD files from a specified directory, merges them into
a single point cloud, saves the result to disk, and publishes it as a ROS 2
PointCloud2 message.
"""

from pathlib import Path
from typing import List

import open3d as o3d
import rclpy
from rclpy.node import Node

from modules.utils.point_cloud_processor import PointCloudProcessor
from modules.reconstruction.map_merger_publisher import MapMergerPublisher


class MapMergerPublisherNode(Node):
    """
    ROS 2 node to merge PCD files and publish the resulting map.
    """

    def __init__(
        self,
        pcd_dir: str,
        output_path: str = "point_clouds/merged_map.ply",
        voxel_size: float = 0.02
    ) -> None:
        super().__init__('map_merger_publisher_node')

        self.pcd_dir = Path(pcd_dir)
        self.output_path = Path(output_path)
        self.voxel_size = voxel_size

        self.processor = PointCloudProcessor(ros_node=self)
        self.merger = MapMergerPublisher(
            output_path=self.output_path,
            processor=self.processor,
            voxel_size=self.voxel_size
        )

        self.get_logger().info("Merging PCDs and publishing merged map...")
        point_clouds = self._load_point_clouds()
        self.merger.merge_and_publish(point_clouds)
        self.get_logger().info("Merged map published successfully.")

    def _load_point_clouds(self) -> List[o3d.geometry.PointCloud]:
        """
        Loads all .ply or .pcd files from the given directory.

        Returns:
            List[o3d.geometry.PointCloud]: Loaded point clouds.
        """
        files = sorted([
            f for f in self.pcd_dir.glob("*.ply")
            if f.is_file()
        ])

        if not files:
            self.get_logger().error(
                f"No .ply files found in directory: {self.pcd_dir}"
            )
            rclpy.shutdown()
            return []

        clouds = []
        for file in files:
            cloud = o3d.io.read_point_cloud(str(file))
            if len(cloud.points) == 0:
                self.get_logger().warn(f"Empty point cloud: {file}")
                continue
            clouds.append(cloud)

        self.get_logger().info(
            f"Loaded {len(clouds)} point clouds from: {self.pcd_dir}"
        )
        return clouds


def main() -> None:
    """
    Entry point for the map merger ROS 2 node.
    """
    rclpy.init()
    try:
        node = MapMergerPublisherNode(
            pcd_dir="point_clouds",
            output_path="point_clouds/merged_map.ply",
            voxel_size=0.02
        )
        rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("[Shutdown] Interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
