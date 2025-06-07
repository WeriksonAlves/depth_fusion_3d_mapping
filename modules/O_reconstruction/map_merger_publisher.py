"""
Map merger and ROS 2 publisher for Open3D point clouds.

Merges a list of point clouds, saves the final map to disk, and optionally
publishes it via a PointCloudProcessor to a ROS 2 topic.
"""

from pathlib import Path
from typing import List, Optional

import open3d as o3d

from modules.utils.point_cloud_processor import PointCloudProcessor


class MapMergerPublisher:
    """
    Merges point clouds and optionally publishes to ROS 2.
    """

    def __init__(
        self,
        output_path: Path,
        processor: Optional[PointCloudProcessor] = None,
        voxel_size: float = 0.02
    ) -> None:
        """
        Initializes the merger.

        Args:
            output_path (Path): Path to save the merged map (e.g., .ply file).
            processor (Optional[PointCloudProcessor]): ROS 2 point cloud
                publisher.
            voxel_size (float): Voxel size for final map downsampling.
        """
        self.output_path = output_path
        self.processor = processor
        self.voxel_size = voxel_size

    def merge_and_publish(
        self,
        point_clouds: List[o3d.geometry.PointCloud]
    ) -> None:
        """
        Merges all point clouds, saves to disk, and publishes if processor set.

        Args:
            point_clouds (List[o3d.geometry.PointCloud]): List of PCDs.
        """
        if not point_clouds:
            print("[Warning] No point clouds provided.")
            return

        merged = point_clouds[0]
        for cloud in point_clouds[1:]:
            merged += cloud

        merged = merged.voxel_down_sample(self.voxel_size / 2.0)
        o3d.io.write_point_cloud(str(self.output_path), merged)

        print(f"[✓] Merged map saved to: {self.output_path}")
        print(f"[INFO] Total points: {len(merged.points)}")

        bbox = merged.get_axis_aligned_bounding_box()
        print(f"[INFO] Bounding box volume: {bbox.volume():.4f} m³")

        if self.processor:
            print("[INFO] Publishing merged map to ROS 2...")
            self.processor.publish_point_cloud(merged)
