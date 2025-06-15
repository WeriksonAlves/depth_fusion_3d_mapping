"""
Utility for side-by-side comparison of multiple point clouds.

Loads .ply files, aligns them spatially along the X-axis,
assigns distinct colors, and renders them using Open3D.
"""

from pathlib import Path
from typing import List, Optional
import open3d as o3d
import numpy as np


class PointCloudComparer:
    """
    Utility class to load and compare multiple point clouds side by side.
    Each cloud is translated along the X-axis and painted a unique color.
    """

    def __init__(self, offset_apply: Optional[bool] = False) -> None:
        """
        Initializes the comparer with an empty list of geometries and
        predefined colors for visualization.

        Args:
            offset (bool, optional): If True, applies an offset to the
                point clouds for side-by-side comparison. Defaults to False.
        """
        self._offset_apply = offset_apply
        self._geometries: List[o3d.geometry.PointCloud] = []
        self._offset = 0.0
        self._colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
        ]

    def _load_and_transform(self, path: Path, index: int) -> None:
        """
        Loads, centers, translates, and colors a point cloud.

        Args:
            path (Path): Path to the .ply file.
            index (int): Index used to assign a predefined color.
        """
        if not path.exists():
            print(f"[WARNING] File not found: {path}")
            return

        cloud = o3d.io.read_point_cloud(str(path))
        bbox = cloud.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        if self._offset_apply:
            extent_x = bbox.get_extent()[0]

        # Translate to origin and offset for side-by-side comparison
        cloud.translate([-center[0] + self._offset, -center[1], -center[2]])

        color = self._colors[index % len(self._colors)]
        # cloud.paint_uniform_color(color)
        self._geometries.append(cloud)

        if self._offset_apply:
            self._offset += extent_x * 1.2  # Add margin between clouds

        color_str = (
            f"RGB({int(color[0]*255)}, {int(color[1]*255)}, "
            f"{int(color[2]*255)})"
        )
        print(f"  - [{color_str}] {path.name}")

    def visualize(self, paths: List[Path]) -> None:
        """
        Loads and renders all valid point clouds for comparison.

        Args:
            paths (List[Path]): List of .ply file paths.
        """
        print("\n[Legend] Point Cloud Colors:")
        for idx, path in enumerate(paths):
            self._load_and_transform(path, idx)

        if not self._geometries:
            print("[ERROR] No valid point clouds to visualize.")
            return

        print("[INFO] Launching Open3D viewer...")
        o3d.visualization.draw_geometries(self._geometries)

    def _load_point_clouds(self,
                          estimated_path: Path,
                          real_path: Path) -> None:
        """
        Loads point clouds from specified paths.

        Args:
            estimated_path (Path): Path to the estimated point cloud (.ply).
            real_path (Path): Path to the real point cloud (.ply).
        """
        estimated = o3d.io.read_point_cloud(str(estimated_path))
        real = o3d.io.read_point_cloud(str(real_path))
        return estimated, real

    def _estimate_scale_ratio(self, pcd_estimated: o3d.geometry.PointCloud,
                             pcd_real: o3d.geometry.PointCloud) -> float:
        """
        Estimates a scale ratio between two point clouds based on mean distance
        to centroid (as a simple heuristic).
        """
        center_est = pcd_estimated.get_center()
        center_real = pcd_real.get_center()

        dist_est = np.mean(np.linalg.norm(
            np.asarray(pcd_estimated.points) - center_est, axis=1
        ))
        dist_real = np.mean(np.linalg.norm(
            np.asarray(pcd_real.points) - center_real, axis=1
        ))

        scale = dist_real / dist_est
        print(f"[✓] Estimated scale correction: {scale:.4f}")
        return scale

    def _estimate_scale_from_aabb(self, pcd_estimated, pcd_real) -> float:
        aabb_est = pcd_estimated.get_axis_aligned_bounding_box()
        aabb_real = pcd_real.get_axis_aligned_bounding_box()
        vol_est = aabb_est.volume()
        vol_real = aabb_real.volume()
        scale = (vol_real / vol_est) ** (1.0 / 3.0)
        print(f"[✓] Estimated scale from AABB volume: {scale:.4f}")
        return scale

    def run(self, paths: List[Path], mode=0) -> None:
        """
        Main method to load, align, and visualize point clouds.

        Args:
            paths (List[Path]): List of .ply file paths to compare.
        """
        estimated_pcd, real_pcd = self._load_point_clouds(
            estimated_path=paths[0],
            real_path=paths[1]
        )

        if mode == 0:
            print("[INFO] Using scale ratio based on mean distance to centroid.")
            scale_ratio = self._estimate_scale_ratio(
                pcd_estimated=estimated_pcd,
                pcd_real=real_pcd
            )
            estimated_pcd.scale(scale_ratio, estimated_pcd.get_center())
        else:
            print("[INFO] Using scale based on AABB volume.")
            aabb_scale = self._estimate_scale_from_aabb(
                pcd_estimated=estimated_pcd,
                pcd_real=real_pcd
            )
            estimated_pcd.scale(aabb_scale, estimated_pcd.get_center())

        self._process_single_cloud(estimated_pcd, 0)
        self._process_single_cloud(real_pcd, 1)
        
        if not self._geometries:
            print("[ERROR] No valid point clouds to visualize.")
            return
        print("[INFO] Launching Open3D viewer...")
        o3d.visualization.draw_geometries(self._geometries)

    def _process_single_cloud(
        self,
        cloud: o3d.geometry.PointCloud,
        index: int
    ) -> None:
        
        bbox = cloud.get_axis_aligned_bounding_box()
        center = bbox.get_center()
        if self._offset_apply:
            extent_x = bbox.get_extent()[0]

        # Translate to origin and offset for side-by-side comparison
        cloud.translate([-center[0] + self._offset, -center[1], -center[2]])

        color = self._colors[index % len(self._colors)]
        cloud.paint_uniform_color(color)
        self._geometries.append(cloud)

        if self._offset_apply:
            self._offset += extent_x * 1.2  # Add margin between clouds

        color_str = (
            f"RGB({int(color[0]*255)}, {int(color[1]*255)}, "
            f"{int(color[2]*255)})"
        )
        print(f"  - [{color_str}] {index}")