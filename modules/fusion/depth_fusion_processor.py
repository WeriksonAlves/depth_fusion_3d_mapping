# modules/ ...

"""
Depth map fusion utility: combines real and monocular depth using projection.

This module projects the monocular depth map into the real camera space,
fuses it with the real depth (e.g., from D435), and saves fused outputs.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import open3d as o3d
import cv2

from modules.reconstruction.intrinsic_loader import IntrinsicLoader
from modules.reconstruction.rgbd_loader import RGBDLoader


class DepthFusionProcessor:
    """
    Performs fusion between real and monocular depth maps by projecting
    monocular data and applying conditional merging rules.
    """

    def __init__(
        self,
        rgb_dir: Path,
        depth_real_dir: Path,
        depth_mono_dir: Path,
        transform_path: Path,
        intrinsics_path: Path,
        output_dir: Path,
        depth_scale: float = 5000.0,
        depth_trunc: float = 4.0,
        mode: str = "min",
        visualize: bool = True
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
        self._mode = mode
        self._visualize = visualize

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

    def _fuse_maps(
        self,
        real: np.ndarray,
        projected: np.ndarray
    ) -> np.ndarray:
        """Fuses two depth maps based on selected mode."""
        if self._mode == "min":
            return np.where(
                real == 0,
                projected,
                np.where((real > 0) & (projected > 0),
                         np.minimum(real, projected),
                         real)
            )
        if self._mode == "mean":
            return np.where(
                real == 0,
                projected,
                np.where((real > 0) & (projected > 0),
                         (real + projected) / 2.0,
                         real)
            )
        if self._mode == "real-priority":
            return np.where(real > 0, real, projected)
        if self._mode == "mono-priority":
            return np.where(projected > 0, projected, real)

        raise ValueError(f"Unsupported fusion mode: {self._mode}")

    def _process_single_frame(
        self,
        rgb_path: Path,
        depth_real_path: Path,
        depth_mono_path: Path
    ) -> Dict:
        """Processes a single RGB frame and fuses its depth maps."""
        rgb = o3d.io.read_image(str(rgb_path))
        real = np.load(depth_real_path).astype(np.float32)
        mono = np.load(depth_mono_path).astype(np.float32)

        depth_img = o3d.geometry.Image(
            (mono * self._depth_scale).astype(np.uint16)
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth_img,
            depth_scale=self._depth_scale,
            convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, self._intrinsics
        )
        pcd.transform(self._transform)

        projected = self._project_point_cloud_to_depth(pcd)
        fused = self._fuse_maps(real * self._depth_scale, projected)

        if self._visualize:
            self._visualize_comparison(
                name=rgb_path.stem,
                depth_real=real * self._depth_scale,
                projected=projected,
                fused=fused
            )

        # Save depth outputs
        name = rgb_path.stem
        png_path = self._output_dir / f"{name}.png"
        npy_path = self._output_dir / f"{name}.npy"
        cv2.imwrite(str(png_path), fused)
        np.save(npy_path, fused.astype(np.float32) / self._depth_scale)

        # Compute stats
        valid_mask = fused > 0
        depth_valid = fused[valid_mask].astype(np.float32) / self._depth_scale

        # Save error map
        error_abs = np.abs((real * self._depth_scale) - fused)
        error_map_path = self._output_dir / f"{name}_error_abs.npy"
        np.save(error_map_path, error_abs)

        # Return statistics
        return {
            "frame": name,
            "valid_pixels": int(valid_mask.sum()),
            "depth_min_m": float(depth_valid.min()
                                 ) if depth_valid.size else None,
            "depth_max_m": float(depth_valid.max()
                                 ) if depth_valid.size else None,
            "depth_mean_m": float(depth_valid.mean()
                                  ) if depth_valid.size else None,
            "error_abs_mean_mm": float(error_abs[valid_mask].mean()
                                       ) if valid_mask.any() else None,
            "error_abs_max_mm": float(error_abs[valid_mask].max()
                                      ) if valid_mask.any() else None
        }

    def run(self) -> None:
        """Executes fusion for all frames in the dataset."""
        rgb_files = sorted(self._rgb_dir.glob("*.png"))
        real_files = sorted(self._depth_real_dir.glob("*.npy"))
        mono_files = sorted(self._depth_mono_dir.glob("*.npy"))

        assert len(rgb_files) == len(real_files) == len(mono_files), (
            "[ERROR] Frame count mismatch between RGB and depth files."
        )

        stats_all = []
        for rgb_path, real_path, mono_path in zip(rgb_files,
                                                  real_files,
                                                  mono_files):
            print(f"[INFO] Processing frame: {rgb_path.name}")
            stats = self._process_single_frame(rgb_path, real_path, mono_path)
            stats_all.append(stats)
            print(f"[✓] Frame processed: {rgb_path.stem}")

        stats_path = self._output_dir / "fusion_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_all, f, indent=4)

        print(f"[✓] Fusion statistics saved to: {stats_path}")

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

        # Resize all depth maps to match RGB image shape
        target_shape = (depth_real.shape[1], depth_real.shape[0])
        real_col = cv2.resize(real_col, target_shape)
        proj_col = cv2.resize(proj_col, target_shape)
        fused_col = cv2.resize(fused_col, target_shape)

        top = np.hstack([real_col, proj_col])
        bottom = np.hstack([fused_col, diff_col])
        grid = np.vstack([top, bottom])

        save_path = self._output_dir / f"{name}_comparison.png"
        cv2.imwrite(str(save_path), grid)

        print(f"[✓] Comparison image saved to: {save_path}")
