"""
Depth map fusion utility: combines real and monocular depth using projection.

This module projects the monocular depth map into the real camera space,
fuses it with the real depth (e.g., from D435), and saves fused outputs.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import cv2

from modules.reconstruction.intrinsic_loader import IntrinsicLoader
from modules.reconstruction.rgbd_loader import RGBDLoader


class DepthFusionProcessor:
    """
    Performs fusion between real and monocular depth maps using inverse-depth
    linear regression to compute scale and shift.
    """

    def __init__(
        self,
        rgb_dir: Path,
        depth_real_dir: Path,
        depth_mono_dir: Path,
        transform_path: Path,
        intrinsics_path: Path,
        output_dir: Path,
        depth_scale: float = 1000.0,
        depth_trunc: float = 5.0,
        visualize: bool = True,
        voxel_size: float = 0.02
    ) -> None:
        self._rgb_dir = rgb_dir
        self._depth_real_dir = depth_real_dir
        self._depth_mono_dir = depth_mono_dir
        self._transform = np.load(transform_path)
        self._intrinsics = IntrinsicLoader.load_from_json(intrinsics_path)
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._depth_scale = depth_scale
        self._depth_trunc = depth_trunc
        self._visualize = visualize
        self._voxel_size = voxel_size

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

    def _project_point_cloud_to_depth(
        self,
        pcd: o3d.geometry.PointCloud
    ) -> np.ndarray:
        """
        Projects a point cloud into a 2D depth image using camera intrinsics.
        """
        width, height = self._intrinsics.width, self._intrinsics.height
        fx, fy = self._intrinsics.get_focal_length()
        cx, cy = self._intrinsics.get_principal_point()

        depth_img = np.zeros((height, width), dtype=np.float32)
        z_buffer = np.full((height, width), np.inf)

        for x, y, z in np.asarray(pcd.points):
            if z <= 0 or z > self._depth_trunc:
                continue
            u = int(round((x * fx) / z + cx))
            v = int(round((y * fy) / z + cy))
            if 0 <= u < width and 0 <= v < height and z < z_buffer[v, u]:
                z_buffer[v, u] = z
                depth_img[v, u] = z

        return (depth_img * self._depth_scale).astype(np.uint16)

    def _fuse_maps_least_squares(
        self, real: np.ndarray, mono: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Fuses two depth maps using inverse-depth linear regression.
        """
        mask = (real > 0) & (mono > 0)
        if not np.any(mask):
            raise ValueError("No valid pixels for fusion.")

        x = mono[mask].astype(np.float32)
        y = 1.0 / real[mask].astype(np.float32)

        A = np.vstack([x, np.ones_like(x)]).T
        scale, shift = np.linalg.lstsq(A, y, rcond=None)[0]

        corrected = np.zeros_like(mono, dtype=np.float32)
        valid = mono > 0
        corrected[valid] = 1.0 / (scale * mono[valid] + shift)

        fused = np.where(real > 0, real, corrected)

        stats = {
            "scale": float(scale),
            "shift": float(shift),
            "num_valid_pairs": int(np.count_nonzero(mask))
        }
        return fused, stats

    def _visualize_comparison(
        self,
        name: str,
        depth_real: np.ndarray,
        projected: np.ndarray,
        fused: np.ndarray
    ) -> None:
        """
        Saves a side-by-side comparison of real, projected and fused depth
        maps.
        """
        def normalize_depth(depth):
            d_norm = cv2.normalize(
                depth.astype(np.float32),
                None, 0, 255,
                cv2.NORM_MINMAX
            )
            return d_norm.astype(np.uint8)

        real_vis = normalize_depth(depth_real)
        proj_vis = normalize_depth(projected)
        fused_vis = normalize_depth(fused)
        diff_vis = normalize_depth(np.abs(depth_real - fused))

        real_col = cv2.applyColorMap(real_vis, cv2.COLORMAP_JET)
        proj_col = cv2.applyColorMap(proj_vis, cv2.COLORMAP_JET)
        fused_col = cv2.applyColorMap(fused_vis, cv2.COLORMAP_JET)
        diff_col = cv2.applyColorMap(diff_vis, cv2.COLORMAP_HOT)

        top = np.hstack([real_col, proj_col])
        bottom = np.hstack([fused_col, diff_col])
        grid = np.vstack([top, bottom])

        save_path = self._output_dir / f"{name}_comparison.png"
        cv2.imwrite(str(save_path), grid)
        print(f"[✓] Comparison image saved to: {save_path}")

    def run(self) -> None:
        """
        Alternative run method that processes frames in a different way.
        This is a placeholder for future modifications or different processing.
        """
        clouds_sensor = RGBDLoader.load_point_clouds(
            self._rgb_dir,
            self._depth_real_dir,
            self._intrinsics,
            self._voxel_size
        )
        clouds_estimated = RGBDLoader.load_point_clouds(
            self._rgb_dir,
            self._depth_mono_dir,
            self._intrinsics,
            self._voxel_size
        )

        assert len(clouds_sensor) == len(clouds_estimated), (
            "[ERROR] Frame count mismatch between sensor and estimated clouds."
        )

        self._rotate_point_clouds(clouds_sensor)
        self._rotate_point_clouds(clouds_estimated)

        stats_all = []

        for i, (pcd_real, pcd_mono) in enumerate(zip(clouds_sensor, clouds_estimated)):
            print(f"[INFO] Processing frame {i + 1}/{len(clouds_sensor)}")

            name = f"frame_{i:04d}"
            real_path = self._depth_real_dir / f"{name}.npy"
            mono_path = self._depth_mono_dir / f"{name}.npy"

            depth_real = np.load(real_path).astype(np.float32)
            depth_mono = np.load(mono_path).astype(np.float32)

            # Apply transformation
            pcd_mono.transform(self._transform[i])
            projected = self._project_point_cloud_to_depth(
                pcd_mono).astype(
                    np.float32
                ) / self._depth_scale

            fused, fusion_stats = self._fuse_maps_least_squares(depth_real,
                                                                depth_mono)

            if self._visualize:
                self._visualize_comparison(
                    name=name,
                    depth_real=depth_real,
                    projected=projected,
                    fused=fused
                )

            # Save outputs
            np.save(self._output_dir / f"{name}.npy", fused.astype(np.float32))
            cv2.imwrite(str(self._output_dir / f"{name}.png"),
                        (fused * self._depth_scale).astype(np.uint16))

            # Compute error
            valid = (depth_real > 0) & (fused > 0)
            error_abs = np.abs((depth_real - fused)[valid]) * 1000.0

            stats = {
                "frame": name,
                "valid_pixels": int(valid.sum()),
                "depth_min_m": float(fused[valid].min()) if valid.any() else None,
                "depth_max_m": float(fused[valid].max()) if valid.any() else None,
                "depth_mean_m": float(fused[valid].mean()) if valid.any() else None,
                "error_abs_mean_mm": float(error_abs.mean()) if valid.any() else None,
                "error_abs_max_mm": float(error_abs.max()) if valid.any() else None,
                "scale": fusion_stats["scale"],
                "shift": fusion_stats["shift"],
                "num_valid_pairs": fusion_stats["num_valid_pairs"]
            }
            stats_all.append(stats)

        with open(self._output_dir / "fusion_statistics.json", "w", encoding="utf-8") as f:
            json.dump(stats_all, f, indent=4)

        print(f"[✓] Fusion statistics saved to: {self._output_dir / 'fusion_statistics.json'}")
