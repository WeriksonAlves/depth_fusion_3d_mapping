# modules/evaluation/visualizer.py

"""
Utility for side-by-side comparison of multiple point clouds.

Loads .ply files, aligns them spatially along the X-axis,
assigns distinct colors, and renders them using Open3D.
"""

from pathlib import Path
from typing import List
import open3d as o3d


class PointCloudComparer:
    """
    Handles visualization and alignment of multiple point clouds for
    comparison.
    """

    def __init__(self) -> None:
        self.geometries = []
        self.offset = 0.0
        self.colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
        ]

    def load_and_transform(self, path: Path, index: int) -> None:
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

        cloud.translate([-center[0] + self.offset, -center[1], -center[2]])
        color = self.colors[index % len(self.colors)]
        cloud.paint_uniform_color(color)
        self.geometries.append(cloud)

        extent = bbox.get_extent()[0]
        # self.offset += extent * 1.2  # Add margin for next cloud

        color_str = f"RGB({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"
        print(f"  - [{color_str}] {path.name}")

    def visualize(self, paths: List[Path]) -> None:
        """
        Loads and renders all valid point clouds for comparison.

        Args:
            paths (List[Path]): List of .ply file paths.
        """
        print("\n[Legend] Point Cloud Colors:")
        for idx, path in enumerate(paths):
            self.load_and_transform(path, idx)

        if not self.geometries:
            print("[ERROR] No valid point clouds to visualize.")
            return

        print("\n[INFO] Launching Open3D viewer...")
        o3d.visualization.draw_geometries(self.geometries)


def main() -> None:
    """
    Example usage: Compare point clouds from D435 vs DepthAnything.
    """
    scene = "lab_scene_f"
    path_d435 = Path(f"results/{scene}/d3/reconstruction_sensor.ply")
    path_mono = Path(f"results/{scene}/d5/reconstruction_estimated.ply")

    comparer = PointCloudComparer()
    comparer.visualize([path_d435, path_mono])


if __name__ == "__main__":
    main()
