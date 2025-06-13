import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open3d as o3d

from modules.reconstruction.intrinsic_loader import IntrinsicLoader
from modules.reconstruction.rgbd_loader import RGBDLoader


class FrameICPAlignerBatch:
    """
    Batch processor for ICP alignment between sensor and estimated
    depth maps. Performs pairwise alignment, saves transformations
    and computes scale estimation.
    """

    def __init__(
        self,
        rgb_dir: Path,
        depth_sensor_dir: Path,
        depth_estimation_dir: Path,
        intrinsics_path: Path,
        output_dir: Path,
        voxel_size: float = 0.02,
        depth_scale: float = 1000.0
    ) -> None:
        """
        Initializes the batch ICP aligner with paths and parameters.

        Args:
            rgb_dir (Path): Directory containing RGB images.
            depth_sensor_dir (Path): Directory with sensor-based depth maps.
            depth_estimation_dir (Path): Directory with estimated depth maps.
            intrinsics_path (Path): Path to the camera intrinsics JSON file.
            output_dir (Path): Directory to save results.
            voxel_size (float): Voxel size for downsampling point clouds.
            depth_scale (float): Scale factor for depth maps.
        """
        self._rgb_dir = rgb_dir
        self._depth_sensor_dir = depth_sensor_dir
        self._depth_estimation_dir = depth_estimation_dir
        self._intrinsics_path = intrinsics_path
        self._output_dir = output_dir
        self._voxel_size = voxel_size
        self._depth_scale = depth_scale

        self._validate_paths()

    def _validate_paths(self) -> None:
        """Validates input paths and ensures output directory exists."""
        for path in [
            self._rgb_dir,
            self._depth_sensor_dir,
            self._depth_estimation_dir,
            self._intrinsics_path
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Missing path: {path}")
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _rotate_point_clouds(
        self,
        clouds: List[o3d.geometry.PointCloud],
        rotation: Tuple[float, float, float] = (np.pi, 0, 0)
    ) -> None:
        """
        Applies rotation to align point clouds with camera frame.

        Args:
            clouds (List[o3d.geometry.PointCloud]): List of point clouds to
                rotate.
            rotation (Tuple[float, float, float]): Rotation angles in radians
                around x, y, z axes.
        """
        R = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
        for pcd in clouds:
            pcd.rotate(R, center=(0, 0, 0))

    def _pairwise_icp(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        dist_coarse: float,
        dist_fine: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs two-stage ICP and returns transformation + info matrix.

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
                source, target, dist_fine, icp_fine.transformation
            )
        return icp_fine.transformation, info

    def _visualize_alignment(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        transform: np.ndarray
    ) -> None:
        """
        Displays aligned point clouds in Open3D viewer.

        Args:
            source (o3d.geometry.PointCloud): Source point cloud.
            target (o3d.geometry.PointCloud): Target point cloud.
            transform (np.ndarray): Transformation matrix to apply to source.
        """
        src_trans = source.transform(transform.copy())
        src_color = [0, 0.651, 0.929]
        tgt_color = [1, 0.706, 0.0]

        o3d.visualization.draw_geometries(
            [
                target.paint_uniform_color(tgt_color),
                src_trans.paint_uniform_color(src_color)
            ],
            window_name="ICP Alignment: Estimated → Sensor"
        )

    def _compute_icp(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        dist_coarse: float,
        dist_fine: float,
        visualization: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper for ICP computation and visualization.

        Args:
            source (o3d.geometry.PointCloud): Source point cloud.
            target (o3d.geometry.PointCloud): Target point cloud.
            dist_coarse (float): Coarse ICP distance threshold.
            dist_fine (float): Fine ICP distance threshold.
            visualization (bool): Whether to visualize the alignment.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformation matrix and
                information matrix.
        """
        trans, info = self._pairwise_icp(
            source, target, dist_coarse, dist_fine
        )
        if visualization:
            self._visualize_alignment(source, target, trans)
        return trans, info

    def _save_results(
        self,
        transforms: List[np.ndarray],
        infos: List[np.ndarray]
    ) -> None:
        """
        Saves transformations and registration info to disk, both in batch and
        per-frame format.

        Args:
            transforms (List[np.ndarray]): Transformation matrices.
            infos (List[np.ndarray]): ICP information matrices.
        """
        trans_path = self._output_dir / "full_transformations.npy"
        np.save(trans_path, transforms)

        json_path = self._output_dir / "icp_metrics.json"
        metrics = {
            "voxel_size": self._voxel_size,
            "depth_scale": self._depth_scale,
            "transformations": [t.tolist() for t in transforms],
            "infos": [i.tolist() for i in infos]
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        print(f"[✓] Transformations saved to: {trans_path}")
        print(f"[✓] ICP metrics saved to: {json_path}")

        # Save per-frame transformation matrices
        frame_dir = self._output_dir / "transforms"
        frame_dir.mkdir(parents=True, exist_ok=True)

        for i, T in enumerate(transforms):
            frame_path = frame_dir / f"transforms_frame{i:04d}.npy"
            np.save(frame_path, T)

        print(f"[✓] Per-frame transformations saved to: {frame_dir}")

    def _estimate_scale(self, transforms: np.ndarray) -> float:
        """
        Estimates average scale factor from transformation matrices.

        Args:
            transforms (np.ndarray): Array of shape (N, 4, 4) with
                homogeneous transformation matrices.
        Returns:
            float: Estimated average scale factor.
        """
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            raise ValueError("Expected array of shape (N, 4, 4)")

        scales = []
        for i, T in enumerate(transforms):
            sx = np.linalg.norm(T[:3, 0])
            sy = np.linalg.norm(T[:3, 1])
            sz = np.linalg.norm(T[:3, 2])
            avg = (sx + sy + sz) / 3.0
            scales.append(avg)
            print(
                f"[Frame {i:03d}] Scale x={sx:.4f}, "
                f"y={sy:.4f}, z={sz:.4f} → avg={avg:.4f}")

        overall = np.mean(scales)
        print(f"\n[✓] Estimated average scale factor: {overall:.6f}")
        return overall

    def run(self) -> None:
        """Runs ICP batch alignment for all frame pairs."""
        print("[INFO] Loading data and intrinsics...")
        intrinsics = IntrinsicLoader.load_from_json(self._intrinsics_path)

        clouds_sensor = RGBDLoader.load_point_clouds(
            self._rgb_dir,
            self._depth_sensor_dir,
            intrinsics,
            self._voxel_size,
            scale_correction=1.0
        )
        clouds_estimated = RGBDLoader.load_point_clouds(
            self._rgb_dir,
            self._depth_estimation_dir,
            intrinsics,
            self._voxel_size,
            scale_correction=1.0
        )

        self._rotate_point_clouds(clouds_sensor)
        self._rotate_point_clouds(clouds_estimated)

        print("[INFO] Running ICP alignments...")
        transforms, infos = [], []
        visualization = True  # Set to False to skip visualizations

        for i, (src, tgt) in enumerate(zip(clouds_estimated, clouds_sensor)):
            print(f"[Frame {i:04d}]")
            trans, info = self._compute_icp(
                src, tgt,
                dist_coarse=self._voxel_size * 15.0,
                dist_fine=self._voxel_size * 1.5,
                visualization=visualization
            )
            transforms.append(trans)
            infos.append(info)

            if visualization:
                user_input = input("Press Enter to continue or 'q' to quit: ")
                if user_input.strip().lower() == "q":
                    visualization = False

        print("[INFO] Saving results...")
        self._save_results(transforms, infos)

        overall_scale = self._estimate_scale(np.array(transforms))
        print(f"[✓] Overall scale factor: {overall_scale:.6f}")


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
            estimation_method=o3d.pipelines.registration.
            TransformationEstimationPointToPlane()
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

        np.save(out_dir / "T_d_to_m_frame0000.txt", transform)

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
            f"[✓] Transformation saved to: "
            f"{out_dir/'T_d_to_m_frame{self._frame_index:04d}.npy'}"
        )
        print(f"[✓] ICP metrics saved to: {out_dir/'icp_metrics.json'}")

    def run(self, inv_transform: bool = False) -> None:
        """Runs the full ICP alignment pipeline for a single frame."""
        print("[INFO] Creating point clouds from RGB + depth...")
        pcd_real = self._create_point_cloud(self._real_depth_path)
        pcd_mono = self._create_point_cloud(self._mono_depth_path)

        print("[INFO] Running ICP alignment...")
        transform, result = self._compute_icp(pcd_mono, pcd_real)

        if inv_transform:  # If you want to save the inverse transformation
            transform = np.linalg.inv(transform)

        print("[INFO] Saving results...")
        self._save_results(transform, result, pcd_mono, pcd_real)

        print("[INFO] Visualizing alignment...")
        self._visualize_alignment(pcd_mono, pcd_real, transform)
