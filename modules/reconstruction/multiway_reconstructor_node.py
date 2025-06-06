"""
ROS 2 service node for multiway 3D reconstruction from RGB-D images.

Combines depth estimation (real or monocular), pose graph registration,
global optimization, and point cloud fusion with optional ROS 2 publishing.
"""

import json
from pathlib import Path
from typing import Optional

import open3d as o3d
from rclpy.node import Node

from modules.utils.point_cloud_processor import PointCloudProcessor
from modules.reconstruction.frame_loader import FrameLoader
from modules.reconstruction.pose_graph_builder import PoseGraphBuilder
from modules.reconstruction.graph_optimizer import GraphOptimizer
from modules.reconstruction.map_merger_publisher import MapMergerPublisher


class MultiwayReconstructor:
    """
    Orchestrates the multiway 3D reconstruction pipeline.
    """

    def __init__(
        self,
        dataset_dir: Path,
        output_dir: Path,
        mode: str = "real",
        ros_node: Optional[Node] = None,
        voxel_size: float = 0.02,
        depth_scale: float = 5000.0,
        depth_trunc: float = 4.0,
        frame_id: str = "map",
        topic: str = "/o3d_points"
    ) -> None:
        """
        Initializes the reconstruction pipeline.

        Args:
            dataset_dir (Path): Path to dataset root directory.
            mode (str): 'real' for RGB-D (.png), 'mono' for monocular depth
                (.npy).
            ros_node (Node, optional): ROS node for publishing output.
            voxel_size (float): Downsampling voxel size.
            depth_scale (float): Scaling factor for depth map.
            depth_trunc (float): Maximum depth value (meters).
            frame_id (str): ROS frame ID.
            topic (str): ROS topic to publish the final map.
        """
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.rgb_dir = dataset_dir / "rgb"
        if mode == "real":
            self.depth_dir = dataset_dir / "depth_npy"
        else:
            self.depth_dir = output_dir / "depth_mono"

        suffix = "sensor" if mode == "real" else "depthanything"
        self.output_path = output_dir / f"reconstruction_{suffix}.ply"
        self.intrinsics_path = dataset_dir / "intrinsics.json"

        self.voxel_size = voxel_size
        self.depth_scale = depth_scale
        self.depth_trunc = depth_trunc

        self.intrinsics = self._load_intrinsics()
        self.processor = None

        if ros_node:
            self.processor = PointCloudProcessor(
                fx=self.intrinsics.get_focal_length()[0],
                fy=self.intrinsics.get_focal_length()[1],
                cx=self.intrinsics.get_principal_point()[0],
                cy=self.intrinsics.get_principal_point()[1],
                width=self.intrinsics.width,
                height=self.intrinsics.height,
                ros_node=ros_node,
                frame_id=frame_id,
                topic=topic
            )

    def _load_intrinsics(self) -> o3d.camera.PinholeCameraIntrinsic:
        """
        Loads camera intrinsics from JSON.

        Returns:
            o3d.camera.PinholeCameraIntrinsic: Open3D intrinsics object.
        """
        with open(self.intrinsics_path, "r") as f:
            data = json.load(f)

        intr = o3d.camera.PinholeCameraIntrinsic()
        intr.set_intrinsics(
            width=data["width"],
            height=data["height"],
            fx=data["K"][0],
            fy=data["K"][4],
            cx=data["K"][2],
            cy=data["K"][5]
        )
        return intr

    def run(self) -> None:
        """
        Executes the full multiway reconstruction pipeline.
        """
        loader = FrameLoader(
            rgb_dir=self.rgb_dir,
            depth_dir=self.depth_dir,
            intrinsics=self.intrinsics,
            mode=self.mode,
            depth_scale=self.depth_scale,
            depth_trunc=self.depth_trunc,
            voxel_size=self.voxel_size
        )
        point_clouds = loader.load_point_clouds()

        builder = PoseGraphBuilder(voxel_size=self.voxel_size)
        pose_graph = builder.build(point_clouds)

        print("Optimizing PoseGraph ...")
        optimizer = GraphOptimizer(voxel_size=self.voxel_size)
        optimizer.optimize(pose_graph, point_clouds)

        merger = MapMergerPublisher(
            output_path=self.output_path,
            processor=self.processor,
            voxel_size=self.voxel_size
        )
        merger.merge_and_publish(point_clouds)
