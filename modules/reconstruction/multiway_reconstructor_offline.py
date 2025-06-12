"""
Offline multiway registration pipeline using Open3D.

Loads RGB and depth data, reconstructs 3D scenes, aligns frames using pairwise
registration, builds a pose graph, optimizes it, and merges all point clouds.
"""

from pathlib import Path
from typing import List, Tuple
import json
import numpy as np
import open3d as o3d

from modules.reconstruction.intrinsic_loader import IntrinsicLoader
from modules.reconstruction.rgbd_loader import RGBDLoader


class MultiwayReconstructorOffline:
    """
    Executes offline multiway registration pipeline using RGB and depth maps.

    Steps:
    - Load RGB-D data
    - Convert to point clouds
    - Perform pairwise ICP registration
    - Build and optimize pose graph
    - Merge all clouds and export output
    """

    def __init__(
        self,
        rgb_dir: Path,
        depth_dir: Path,
        intrinsics_path: Path,
        output_dir: Path,
        output_pcd_path: Path,
        voxel_size: float = 0.02,
        scale_correction: float = 1.0
    ) -> None:
        self._rgb_dir = rgb_dir
        self._depth_dir = depth_dir
        self._intrinsics_path = intrinsics_path
        self._output_dir = output_dir
        self._output_pcd_path = output_pcd_path
        self._voxel_size = voxel_size
        self._scale_correction = scale_correction

        self._validate_paths()

    def _validate_paths(self) -> None:
        """Checks for required directories and files."""
        for path in [
            self._rgb_dir,
            self._depth_dir,
            self._intrinsics_path
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Missing path: {path}")
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _pairwise_icp(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        dist_coarse: float,
        dist_fine: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies ICP (coarse-to-fine) and computes information matrix.

        Args:
            source (o3d.geometry.PointCloud): Source point cloud.
            target (o3d.geometry.PointCloud): Target point cloud.
            dist_coarse (float): Coarse ICP distance threshold.
            dist_fine (float): Fine ICP distance threshold.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformation matrix and
                information matrix.
        """
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source,
            target,
            dist_coarse,
            np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        icp_fine = o3d.pipelines.registration.registration_icp(
            source,
            target,
            dist_fine,
            icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        info = o3d.pipelines.registration.\
            get_information_matrix_from_point_clouds(
                source,
                target,
                dist_fine,
                icp_fine.transformation
            )
        return icp_fine.transformation, info

    def _build_pose_graph(
        self,
        clouds: List[o3d.geometry.PointCloud],
        dist_coarse: float,
        dist_fine: float
    ) -> o3d.pipelines.registration.PoseGraph:
        """Builds pose graph using pairwise transformations."""
        pose_graph = o3d.pipelines.registration.PoseGraph()
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.identity(4))
        )
        odometry = np.identity(4)

        for src_id in range(len(clouds)):
            for tgt_id in range(src_id + 1, len(clouds)):
                trans, info = self._pairwise_icp(
                    clouds[src_id],
                    clouds[tgt_id],
                    dist_coarse,
                    dist_fine
                )
                if tgt_id == src_id + 1:
                    odometry = np.dot(trans, odometry)
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(
                            np.linalg.inv(odometry)
                        )
                    )
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            src_id, tgt_id, trans, info, uncertain=False
                        )
                    )
                else:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            src_id, tgt_id, trans, info, uncertain=True
                        )
                    )
        return pose_graph

    def _merge_and_save(
        self,
        clouds: List[o3d.geometry.PointCloud],
        pose_graph: o3d.pipelines.registration.PoseGraph
    ) -> o3d.geometry.PointCloud:
        """Applies global poses and saves merged point cloud."""
        for i, cloud in enumerate(clouds):
            cloud.transform(pose_graph.nodes[i].pose)

        merged = o3d.geometry.PointCloud()
        for cloud in clouds:
            merged += cloud

        o3d.io.write_point_cloud(self._output_pcd_path, merged)
        print(f"[✓] Final point cloud saved to: {self._output_pcd_path}")

        return merged

    def _save_metrics(self, cloud: o3d.geometry.PointCloud) -> None:
        """Saves reconstruction statistics to disk."""
        aabb = cloud.get_axis_aligned_bounding_box()
        volume = aabb.volume()
        num_points = len(cloud.points)

        metrics = {
            "num_points": num_points,
            "volume_aabb": volume,
            "extent_aabb": aabb.get_extent().tolist(),  # [x, y, z] extent
            "voxel_size": self._voxel_size,
            "avg_density": num_points / volume if volume > 0 else 0.0
        }

        metrics_path = self._output_dir / "reconstruction_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"[✓] Metrics saved to: {metrics_path}")

    def _save_snapshot(self, cloud: o3d.geometry.PointCloud) -> None:
        """Saves a rendered snapshot image of the final cloud."""
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(cloud)
        vis.poll_events()
        vis.update_renderer()

        snapshot_path = self._output_dir / "reconstruction_snapshot.png"
        vis.capture_screen_image(str(snapshot_path))
        vis.destroy_window()
        print(f"[✓] Snapshot saved to: {snapshot_path}")

    def run(self) -> None:
        """Runs the full registration pipeline."""
        print("[INFO] Loading intrinsics and RGB-D data...")
        intrinsic = IntrinsicLoader.load_from_json(self._intrinsics_path)
        clouds = RGBDLoader.load_point_clouds(
            self._rgb_dir,
            self._depth_dir,
            intrinsic,
            self._voxel_size,
            scale_correction=self._scale_correction
        )

        print("[INFO] Building pose graph...")
        dist_coarse = self._voxel_size * 15.0
        dist_fine = self._voxel_size * 1.5
        for pcd in clouds:
            pcd.rotate(pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0)))
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            pose_graph = self._build_pose_graph(clouds, dist_coarse, dist_fine)

        print("[INFO] Optimizing pose graph globally...")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=dist_fine,
            edge_prune_threshold=0.25,
            reference_node=0
        )
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.
                GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.
                GlobalOptimizationConvergenceCriteria(),
                option
            )

        print("[INFO] Merging and saving result...")
        merged = self._merge_and_save(clouds, pose_graph)
        self._save_metrics(merged)
        self._save_snapshot(merged)

        o3d.visualization.draw([merged])
