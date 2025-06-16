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
        """
        self._rgb_dir = rgb_dir
        self._depth_sensor_dir = depth_sensor_dir
        self._depth_estimation_dir = depth_estimation_dir
        self._intrinsics_path = intrinsics_path
        self._output_dir = output_dir
        self._voxel_size = voxel_size
        self._visualization = True  # Set to False to skip visualizations

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
        self._ply_path = self._output_dir / "depth_ply"
        self._ply_path.mkdir(parents=True, exist_ok=True)

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

    def normalize_png_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Normalizes the depth map to a range of 0-255.

        Args:
            depth (np.ndarray): Depth map to normalize.

        Returns:
            np.ndarray: Normalized depth map.
        """
        aux_1 = depth - depth.min()
        aux_2 = depth.max() - depth.min()
        normalized = (aux_1) / (aux_2) * 255.0
        return normalized.astype(np.uint8)

    def _visualize_alignment(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        transform: np.ndarray,
    ) -> None:
        """
        Displays aligned point clouds and saves the transformed cloud and
        depth.

        Args:
            source (o3d.geometry.PointCloud): Source point cloud.
            target (o3d.geometry.PointCloud): Target point cloud.
            transform (np.ndarray): Transformation matrix to apply to source.
        """
        # Apply transformation to source
        src_trans = source.transform(transform.copy())
        src_color = [0, 0.651, 0.929]
        tgt_color = [1, 0.706, 0.0]

        # Optional: visualize alignment
        if self._visualization:
            o3d.visualization.draw_geometries(
                [
                    target.paint_uniform_color(tgt_color),
                    src_trans.paint_uniform_color(src_color)
                ],
                window_name="ICP Alignment: Estimated → Sensor"
            )

        # Save transformed point cloud as .ply
        ply_path = self._ply_path / f"{self.name}_aligned.ply"
        o3d.io.write_point_cloud(str(ply_path), src_trans)
        print(f"[✓] Transformed point cloud saved to: {ply_path}")

    def _compute_icp(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
        dist_coarse: float,
        dist_fine: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Wrapper for ICP computation and visualization.

        Args:
            source (o3d.geometry.PointCloud): Source point cloud.
            target (o3d.geometry.PointCloud): Target point cloud.
            dist_coarse (float): Coarse ICP distance threshold.
            dist_fine (float): Fine ICP distance threshold.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Transformation matrix and
                information matrix.
        """
        trans, info = self._pairwise_icp(
            source, target, dist_coarse, dist_fine
        )
        # trans = np.linalg.inv(trans)
        self._visualize_alignment(source, target, trans)
        return trans, info

    def _save_results(
        self,
        transforms: List[np.ndarray],
        infos: List[np.ndarray],
        estimated_scale_factor: float
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
            "estimated_scale_factor": float(estimated_scale_factor),
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

    def run(self, estimated_scale_factor) -> None:
        """Runs ICP batch alignment for all frame pairs."""
        print("[INFO] Loading data and intrinsics...")
        intrinsics = IntrinsicLoader.load_from_json(self._intrinsics_path)

        clouds_sensor = RGBDLoader.load_point_clouds(
            self._rgb_dir,
            self._depth_sensor_dir,
            intrinsics,
            self._voxel_size
        )
        clouds_estimated = RGBDLoader.load_point_clouds(
            self._rgb_dir,
            self._depth_estimation_dir,
            intrinsics,
            self._voxel_size,
        )

        if estimated_scale_factor != 1.0:
            print(
                f"[INFO] Applying scale: {estimated_scale_factor:.4f}")
            for cloud in clouds_estimated:
                cloud.points = o3d.utility.Vector3dVector(
                    np.asarray(cloud.points) * estimated_scale_factor
                )

        self._rotate_point_clouds(clouds_sensor)
        self._rotate_point_clouds(clouds_estimated)

        print("[INFO] Running ICP alignments...")
        transforms, infos = [], []

        for i, (src, tgt) in enumerate(zip(clouds_estimated, clouds_sensor)):
            print(f"[Frame {i:04d}]")
            self.name = f"frame_{i:04d}"
            trans, info = self._compute_icp(
                src, tgt,
                dist_coarse=self._voxel_size * 2.5,
                dist_fine=self._voxel_size * 1.5,
            )
            transforms.append(trans)
            infos.append(info)

            if self._visualization:
                user_input = input("Press Enter to continue or 'q' to quit: ")
                if user_input.strip().lower() == "q":
                    self._visualization = False

        print("[INFO] Saving results...")
        self._save_results(transforms, infos, estimated_scale_factor)

    def estimate_scale_from_depth_maps(
        self,
        max_samples: int = 50000,
        min_depth: float = 0.2,
        max_depth: float = 4.0
    ) -> Tuple[float, float]:
        """
        Estimates scale and shift between estimated and real depths via
        regression.

        Args:
            max_samples (int): Max number of pixels sampled across all frames.
            min_depth (float): Minimum valid depth (in meters).
            max_depth (float): Maximum valid depth (in meters).

        Returns:
            Tuple[float, float]: scale (s) and shift (t) such that:
                1 / d_real ≈ s * d_mono + t
        """
        print("[INFO] Estimating scale using depth map regression...")

        rgb_paths = sorted(self._rgb_dir.glob("*.png"))
        real_paths = sorted(self._depth_sensor_dir.glob("*.npy"))
        mono_paths = sorted(self._depth_estimation_dir.glob("*.npy"))

        if not (len(rgb_paths) == len(real_paths) == len(mono_paths)):
            raise ValueError(
                "Frame count mismatch between RGB and depth maps.")

        all_real = []
        all_mono = []

        for rgb, real, mono in zip(rgb_paths, real_paths, mono_paths):
            d_real = np.load(real).astype(np.float32)
            d_mono = np.load(mono).astype(np.float32)

            valid_mask = (d_real > min_depth) & (d_real < max_depth
                                                 ) & (d_mono > 0.01)

            d_real_valid = d_real[valid_mask]
            d_mono_valid = d_mono[valid_mask]

            if d_real_valid.size == 0:
                continue

            all_real.append(d_real_valid)
            all_mono.append(d_mono_valid)

        # Concatenate all samples
        d_real_all = np.concatenate(all_real)
        d_mono_all = np.concatenate(all_mono)

        if d_real_all.size > max_samples:
            idx = np.random.choice(d_real_all.size, max_samples, replace=False)
            d_real_all = d_real_all[idx]
            d_mono_all = d_mono_all[idx]

        # Regression: 1/d_real = s * d_mono + t
        A = np.vstack([d_mono_all, np.ones_like(d_mono_all)]).T
        y = 1.0 / d_real_all

        scale, shift = np.linalg.lstsq(A, y, rcond=None)[0]

        print(f"[✓] Estimated scale: {scale:.6f}, shift: {shift:.6f}")
        return scale, shift
