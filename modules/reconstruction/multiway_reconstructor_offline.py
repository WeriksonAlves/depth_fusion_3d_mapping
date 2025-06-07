# modules/reconstruction/multiway_reconstructor_offline.py

"""
Offline multiway registration pipeline using Open3D.

Loads RGB and depth data, reconstructs 3D scenes, aligns frames using pairwise
registration, builds a pose graph, optimizes it, and merges all point clouds.
"""

import json
import numpy as np
import open3d as o3d
from pathlib import Path
from modules.reconstruction.intrinsic_loader import IntrinsicLoader
from modules.reconstruction.rgbd_loader import RGBDLoader


class MultiwayReconstructorOffline:
    """
    Runs full multiway reconstruction using RGB + depth frames (Numpy format),
    without relying on ROS 2. Outputs a merged point cloud.
    """

    def __init__(
        self,
        dataset_path: Path,
        output_path: Path,
        voxel_size: float = 0.02
    ) -> None:
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.voxel_size = voxel_size

        self.rgb_dir = dataset_path / "rgb"
        self.depth_dir = dataset_path / "depth_npy"
        self.intrinsics_path = dataset_path / "intrinsics.json"
        self.output_pcd = output_path / "reconstruction_sensor.ply"

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"[ERROR] Dataset not found: {self.dataset_path}"
            )
        self.output_path.mkdir(parents=True, exist_ok=True)

    def _pairwise_registration(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        max_dist_coarse: float,
        max_dist_fine: float
    ) -> tuple[np.ndarray, np.ndarray]:
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, max_dist_coarse, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        icp_fine = o3d.pipelines.registration.registration_icp(
            source, target, max_dist_fine, icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, max_dist_fine, icp_fine.transformation
        )
        return icp_fine.transformation, info

    def _build_pose_graph(
        self,
        point_clouds: list[o3d.geometry.PointCloud],
        max_dist_coarse: float,
        max_dist_fine: float
    ) -> o3d.pipelines.registration.PoseGraph:
        pose_graph = o3d.pipelines.registration.PoseGraph()
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.identity(4))
        )
        odometry = np.identity(4)

        for src_id in range(len(point_clouds)):
            for tgt_id in range(src_id + 1, len(point_clouds)):
                transform, info = self._pairwise_registration(
                    point_clouds[src_id], point_clouds[tgt_id],
                    max_dist_coarse, max_dist_fine
                )
                if tgt_id == src_id + 1:
                    odometry = transform @ odometry
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(
                            np.linalg.inv(odometry)
                        )
                    )
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            src_id, tgt_id, transform, info, uncertain=False
                        )
                    )
                else:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            src_id, tgt_id, transform, info, uncertain=True
                        )
                    )

        return pose_graph

    def run(self) -> None:
        print("[INFO] Loading intrinsics and images...")
        intrinsic = IntrinsicLoader.load_from_json(self.intrinsics_path)
        point_clouds = RGBDLoader.load_point_clouds(
            self.rgb_dir, self.depth_dir, intrinsic, self.voxel_size
        )

        print(f"[✓] Saving original reconstruction to: {self.output_path}")
        o3d.visualization.draw(point_clouds)

        print("[INFO] Building pose graph...")
        max_coarse = self.voxel_size * 15
        max_fine = self.voxel_size * 1.5
        pose_graph = self._build_pose_graph(point_clouds, max_coarse, max_fine)

        print("[INFO] Optimizing pose graph...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=max_fine,
            edge_prune_threshold=0.25,
            reference_node=0
        )
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option
        )

        print("[INFO] Merging aligned point clouds...")
        for i, pcd in enumerate(point_clouds):
            pcd.transform(pose_graph.nodes[i].pose)

        merged = o3d.geometry.PointCloud()
        for pcd in point_clouds:
            merged += pcd

        print(f"[✓] Saving final reconstruction to: {self.output_pcd}")
        o3d.io.write_point_cloud(self.output_pcd, merged)

        # Compute and save basic metrics
        aabb = merged.get_axis_aligned_bounding_box()
        metrics = {
            "num_points": len(merged.points),
            "volume_aabb": aabb.volume(),
            "extent_aabb": aabb.get_extent().tolist(),  # [x, y, z] extent
            "voxel_size": self.voxel_size
        }

        metrics_path = self.output_path / "reconstruction_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"[✓] Metrics saved to: {metrics_path}")

        o3d.visualization.draw([merged])
