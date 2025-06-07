import json
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d

from modules.reconstruction.intrinsic_loader import IntrinsicLoader


class FrameICPAligner:
    """
    Aligns a pair of point clouds (monocular vs real depth) for a single frame
    using ICP. Saves the transformation matrix and registration metrics.
    """

    def __init__(
        self,
        dataset_dir: Path,
        results_dir: Path,
        frame_index: int,
        voxel_size: float = 0.02,
        depth_scale: float = 1000.0
    ) -> None:
        self._dataset_dir = dataset_dir
        self._results_dir = results_dir
        self._frame_index = frame_index
        self._voxel_size = voxel_size
        self._depth_scale = depth_scale

        self._frame_name = f"frame_{frame_index:04d}"
        self._rgb_path = dataset_dir / "rgb" / f"{self._frame_name}.png"
        self._real_depth_path = dataset_dir / "depth_npy" / f"{self._frame_name}.npy"
        self._mono_depth_path = results_dir / "d4" / "depth_npy" / f"{self._frame_name}.npy"
        self._intr_path = dataset_dir / "intrinsics.json"

        self._validate_paths()
        self._intrinsics = IntrinsicLoader.load_from_json(self._intr_path)

    def _validate_paths(self) -> None:
        """Checks that all required input files exist."""
        for path in [
            self._rgb_path,
            self._real_depth_path,
            self._mono_depth_path,
            self._intr_path
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")

    def _create_point_cloud(self, depth_path: Path) -> o3d.geometry.PointCloud:
        """Creates a point cloud from RGB + depth."""
        rgb = o3d.io.read_image(str(self._rgb_path))
        depth_np = np.load(depth_path).astype(np.float32)
        depth_mm = (depth_np * self._depth_scale).astype(np.uint16)
        depth_img = o3d.geometry.Image(depth_mm)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth_img,
            depth_scale=self._depth_scale,
            convert_rgb_to_intensity=False
        )

        return o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, self._intrinsics
        )

    def _compute_icp(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud
    ) -> Tuple[np.ndarray, o3d.pipelines.registration.RegistrationResult]:
        """Applies point-to-plane ICP between downsampled clouds."""
        src_down = source.voxel_down_sample(self._voxel_size)
        tgt_down = target.voxel_down_sample(self._voxel_size)

        for pcd in [src_down, tgt_down]:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self._voxel_size * 2.0,
                    max_nn=30
                )
            )

        result = o3d.pipelines.registration.registration_icp(
            source=src_down,
            target=tgt_down,
            max_correspondence_distance=self._voxel_size * 2.5,
            init=np.eye(4),
            estimation_method=(
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
        )

        print(f"[✓] ICP Fitness: {result.fitness:.4f}")
        print(f"[✓] ICP RMSE: {result.inlier_rmse:.4f}")

        return result.transformation, result

    def _visualize_alignment(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        transform: np.ndarray
    ) -> None:
        """Visualizes the alignment result between two point clouds."""
        source_copy = source.transform(transform.copy())
        source_color = [0, 0.651, 0.929]  # Blue
        target_color = [1, 0.706, 0.0]    # Yellow

        o3d.visualization.draw_geometries(
            [
                target.paint_uniform_color(target_color),
                source_copy.paint_uniform_color(source_color)
            ],
            window_name="ICP Alignment: Mono (blue) → Real (yellow)"
        )

    def _save_results(
        self,
        transform: np.ndarray,
        result: o3d.pipelines.registration.RegistrationResult,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud
    ) -> None:
        """Saves the transformation matrix and registration metrics."""
        out_dir = self._results_dir / "d6"
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(out_dir / "T_d_to_m_frame0000.npy", transform)

        metrics = {
            "fitness": float(result.fitness),
            "inlier_rmse": float(result.inlier_rmse),
            "voxel_size": self._voxel_size,
            "num_points_source": len(source.points),
            "num_points_target": len(target.points)
        }

        with open(out_dir / "icp_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        print(
            f"[✓] Transformation saved to: {out_dir/'T_d_to_m_frame0000.npy'}"
        )
        print(f"[✓] ICP metrics saved to: {out_dir/'icp_metrics.json'}")

    def run(self) -> None:
        """Runs the full ICP alignment pipeline for a single frame."""
        print("[INFO] Creating point clouds from RGB + depth...")
        pcd_real = self._create_point_cloud(self._real_depth_path)
        pcd_mono = self._create_point_cloud(self._mono_depth_path)

        print("[INFO] Running ICP alignment...")
        transform, result = self._compute_icp(pcd_mono, pcd_real)

        print("[INFO] Saving results...")
        self._save_results(transform, result, pcd_mono, pcd_real)

        print("[INFO] Visualizing alignment...")
        self._visualize_alignment(pcd_mono, pcd_real, transform)
