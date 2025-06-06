# modules/reconstruction/multiway_registration_offline.py

"""
Offline multiway registration pipeline using Open3D.

This script loads RGB and Depth data from disk (depth as .npy),
builds a pose graph via pairwise registration, performs global optimization,
and saves the merged point cloud and metrics to disk.
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open3d as o3d


class IntrinsicLoader:
    """
    Loads camera intrinsics from a JSON file formatted for Open3D.
    """

    @staticmethod
    def load_from_json(json_path: Path) -> o3d.camera.PinholeCameraIntrinsic:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fx, fy = data["K"][0], data["K"][4]
        cx, cy = data["K"][2], data["K"][5]
        width, height = data["width"], data["height"]
        return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


class RGBDLoader:
    """
    Converts RGB + depth pairs (depth in .npy) into Open3D point clouds.
    """

    @staticmethod
    def load_point_clouds(
        rgb_dir: Path,
        depth_dir: Path,
        intrinsic: o3d.camera.PinholeCameraIntrinsic,
        voxel_size: float,
        scale_correction: float = 1.0
    ) -> List[o3d.geometry.PointCloud]:
        rgb_paths = sorted(rgb_dir.glob("*.png"))
        depth_paths = sorted(depth_dir.glob("*.npy"))

        point_clouds = []
        for rgb_path, depth_path in zip(rgb_paths, depth_paths):
            color = o3d.io.read_image(str(rgb_path))
            depth_np = np.load(depth_path) * scale_correction
            depth_mm = (depth_np * 1000.0).astype(np.uint16)
            depth = o3d.geometry.Image(depth_mm)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0,
                convert_rgb_to_intensity=False
            )

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, intrinsic
            )
            pcd_down = pcd.voxel_down_sample(voxel_size)
            pcd_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 2.0, max_nn=30
                )
            )
            point_clouds.append(pcd_down)

        return point_clouds


class MultiwayReconstructorOffline:
    """
    Orchestrates full offline multiway reconstruction from RGB-D input folders.
    """

    def __init__(
        self,
        rgb_dir: Path,
        depth_dir: Path,
        intrinsics_json: Path,
        output_dir: Path,
        output_pcd: Path,
        voxel_size: float = 0.02,
        scale_correction: float = 1.0
    ) -> None:
        self.rgb_dir = rgb_dir
        self.depth_dir = depth_dir
        self.intrinsics_path = intrinsics_json
        self.output_dir = output_dir
        self.output_pcd = output_pcd
        self.voxel_size = voxel_size
        self.scale_correction = scale_correction

        self._validate_paths()

    def _validate_paths(self) -> None:
        if not self.rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {self.rgb_dir}")
        if not self.depth_dir.exists():
            raise FileNotFoundError(
                f"Depth directory not found: {self.depth_dir}"
            )
        if not self.intrinsics_path.exists():
            raise FileNotFoundError(
                f"Intrinsics file not found: {self.intrinsics_path}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _pairwise_registration(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        dist_coarse: float,
        dist_fine: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, dist_coarse, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        icp_fine = o3d.pipelines.registration.registration_icp(
            source, target, dist_fine, icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source, target, dist_fine, icp_fine.transformation
        )
        return icp_fine.transformation, info

    def _build_pose_graph(
        self,
        clouds: List[o3d.geometry.PointCloud],
        dist_coarse: float,
        dist_fine: float
    ) -> o3d.pipelines.registration.PoseGraph:
        pose_graph = o3d.pipelines.registration.PoseGraph()
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(np.identity(4))
        )
        odometry = np.identity(4)

        for src_id in range(len(clouds)):
            for tgt_id in range(src_id + 1, len(clouds)):
                transform, info = self._pairwise_registration(
                    clouds[src_id], clouds[tgt_id], dist_coarse, dist_fine
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

    def _save_metrics(self, cloud: o3d.geometry.PointCloud) -> None:
        aabb = cloud.get_axis_aligned_bounding_box()
        volume = aabb.volume()
        num_points = len(cloud.points)

        metrics = {
            "num_points": num_points,
            "volume_aabb": volume,
            "extent_aabb": aabb.get_extent().tolist(),
            "voxel_size": self.voxel_size,
            "avg_density": num_points / volume if volume > 0 else 0.0
        }

        metrics_path = self.output_dir / "reconstruction_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"[✓] Metrics saved to: {metrics_path}")

    def run(self) -> None:
        print("[INFO] Loading intrinsics and RGB-D data...")
        intrinsic = IntrinsicLoader.load_from_json(self.intrinsics_path)
        clouds = RGBDLoader.load_point_clouds(
            self.rgb_dir,
            self.depth_dir,
            intrinsic,
            self.voxel_size,
            scale_correction=self.scale_correction
        )

        print("[INFO] Building pose graph...")
        dist_coarse = self.voxel_size * 15.0
        dist_fine = self.voxel_size * 1.5
        pose_graph = self._build_pose_graph(clouds, dist_coarse, dist_fine)

        print("[INFO] Optimizing pose graph globally...")
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=dist_fine,
                edge_prune_threshold=0.25,
                reference_node=0
            )
        )

        print("[INFO] Transforming and merging point clouds...")
        for i, cloud in enumerate(clouds):
            cloud.transform(pose_graph.nodes[i].pose)

        merged = o3d.geometry.PointCloud()
        for cloud in clouds:
            merged += cloud

        print(f"[✓] Saving final point cloud to: {self.output_pcd}")
        o3d.io.write_point_cloud(str(self.output_pcd), merged)
        self._save_metrics(merged)

        o3d.visualization.draw([merged])


# %%
# scripts_test/run_multiway_registration_offline.py

# from pathlib import Path
# from modules.reconstruction.multiway_registration_offline import (
#     MultiwayReconstructorOffline
# )

def main() -> None:
    scene = "lab_scene_f"

    reconstructor = MultiwayReconstructorOffline(
        rgb_dir=Path(f"datasets/{scene}/rgb"),
        depth_npy_dir=Path(f"results/{scene}/d4/depth_npy"),
        intrinsics_json=Path(f"datasets/{scene}/intrinsics.json"),
        output_path=Path(f"results/{scene}/d5"),
        output_pcd=Path(f"results/{scene}/d5/final_reconstruction.ply"),
        voxel_size=0.02
    )
    reconstructor.run()


if __name__ == "__main__":
    main()
