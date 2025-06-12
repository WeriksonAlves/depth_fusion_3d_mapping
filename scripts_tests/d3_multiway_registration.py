# modules/reconstruction/multiway_registration_offline.py

"""
Offline multiway registration pipeline using Open3D.

Loads RGB and depth data, reconstructs 3D scenes, aligns frames using pairwise
registration, builds a pose graph, optimizes it, and merges all point clouds.
"""

import json
import numpy as np
import open3d as o3d
from pathlib import Path


class IntrinsicLoader:
    """Loads camera intrinsics from JSON file in Open3D format."""

    @staticmethod
    def load_from_json(json_path: Path) -> o3d.camera.PinholeCameraIntrinsic:
        with open(json_path, "r") as f:
            data = json.load(f)
        fx, fy = data["K"][0], data["K"][4]
        cx, cy = data["K"][2], data["K"][5]
        width, height = data["width"], data["height"]
        return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


class RGBDLoader:
    """Converts RGB + Depth (.npy) images into point clouds using Open3D."""

    @staticmethod
    def load_point_clouds(
        rgb_dir: Path,
        depth_dir: Path,
        intrinsic: o3d.camera.PinholeCameraIntrinsic,
        voxel_size: float = 0.02
    ) -> list[o3d.geometry.PointCloud]:
        rgb_paths = sorted(rgb_dir.glob("*.png"))
        depth_paths = sorted(depth_dir.glob("*.npy"))
        point_clouds = []

        for rgb_path, depth_path in zip(rgb_paths, depth_paths):
            color = o3d.io.read_image(str(rgb_path))
            depth_np = np.load(depth_path)
            depth_img = o3d.geometry.Image((depth_np * 1000).astype(np.uint16))

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth_img, depth_scale=1000.0,
                convert_rgb_to_intensity=False
            )

            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, intrinsic
            )
            pcd = pcd.voxel_down_sample(voxel_size)
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * 2.0, max_nn=30
                )
            )
            point_clouds.append(pcd)

        return point_clouds


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

        for pcd in point_clouds:
            pcd.rotate(pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0)))
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


# %%
# scripts_test/run_multiway_registration_offline.py

# from pathlib import Path
# from modules.reconstruction.multiway_registration_offline import (
#     MultiwayReconstructorOffline
# )

def main() -> None:
    scene = "lab_scene_d"
    dataset = Path(f"datasets/{scene}")
    output = Path(f"comparation/results_test/{scene}/d3")

    reconstructor = MultiwayReconstructorOffline(
        dataset_path=dataset,
        output_path=output,
        voxel_size=0.02
    )
    reconstructor.run()


if __name__ == "__main__":
    main()
