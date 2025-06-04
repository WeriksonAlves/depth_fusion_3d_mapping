"""
ROS 2 node to compare two reconstructed point clouds.

Overlays a real depth-based reconstruction and a monocular reconstruction
for visual comparison using Open3D.
"""

from pathlib import Path

import open3d as o3d
import rclpy
from rclpy.node import Node


class ReconstructionComparator:
    """
    Handles comparison and visualization of two point cloud reconstructions.
    """

    def __init__(
        self,
        real_pcd_path: Path,
        mono_pcd_path: Path,
        scale_mono: float = 1.0
    ) -> None:
        """
        Initializes the comparator.

        Args:
            real_pcd_path (Path): Path to real depth-based PCD (.ply).
            mono_pcd_path (Path): Path to monocular PCD (.ply).
            scale_mono (float): Optional scale adjustment for mono PCD.
        """
        self.real_pcd_path = real_pcd_path
        self.mono_pcd_path = mono_pcd_path
        self.scale_mono = scale_mono

    def _load_and_colorize(
        self,
        path: Path,
        color: list[float]
    ) -> o3d.geometry.PointCloud:
        """
        Loads a point cloud and paints it with a uniform color.

        Args:
            path (Path): Path to the .ply file.
            color (list[float]): RGB values in [0, 1].

        Returns:
            o3d.geometry.PointCloud: Colored point cloud.
        """
        pcd = o3d.io.read_point_cloud(str(path))
        pcd.paint_uniform_color(color)
        return pcd

    def run(self) -> None:
        """
        Executes the comparison and launches the visualizer.
        """
        print("[INFO] Loading point clouds...")
        real_pcd = self._load_and_colorize(
            self.real_pcd_path, [0.0, 0.6, 1.0]
        )
        mono_pcd = self._load_and_colorize(
            self.mono_pcd_path, [1.0, 0.3, 0.3]
        )

        if self.scale_mono != 1.0:
            print(f"[INFO] Scaling mono PCD by factor {self.scale_mono}")
            mono_pcd.scale(
                self.scale_mono, center=mono_pcd.get_center()
            )

        print("[INFO] Launching Open3D visualizer...")
        o3d.visualization.draw_geometries(
            [real_pcd, mono_pcd],
            window_name='Comparison: Real (blue) vs Mono (red)',
            width=960,
            height=540,
            point_show_normal=False
        )


class ReconstructionComparatorNode(Node):
    """
    ROS 2 node to run PCD comparison and visualization.
    """

    def __init__(self) -> None:
        super().__init__('reconstruction_comparator_node')

        real_pcd_path = Path(self.declare_parameter(
            'real_pcd_path',
            'datasets/lab_scene_kinect_xyz/reconstruction_d435.ply'
        ).get_parameter_value().string_value)

        mono_pcd_path = Path(self.declare_parameter(
            'mono_pcd_path',
            'datasets/lab_scene_kinect_xyz/reconstruction_depthanything.ply'
        ).get_parameter_value().string_value)

        scale_factor = self.declare_parameter(
            'scale_mono',
            1.0
        ).get_parameter_value().double_value

        self.get_logger().info("Starting reconstruction comparison...")

        comparator = ReconstructionComparator(
            real_pcd_path=real_pcd_path,
            mono_pcd_path=mono_pcd_path,
            scale_mono=scale_factor
        )
        comparator.run()

        self.get_logger().info("Comparison visualization finished.")


def main() -> None:
    """
    Entry point for ROS 2 node execution.
    """
    rclpy.init()
    try:
        node = ReconstructionComparatorNode()
        rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("[Shutdown] Interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
