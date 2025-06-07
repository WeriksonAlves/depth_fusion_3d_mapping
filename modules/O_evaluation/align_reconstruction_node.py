"""
ROS 2 node to align monocular depth-based reconstruction to a real sensor map.

Uses ICP with scaling and saves the aligned point cloud and transformation
matrix.
"""

from pathlib import Path

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node


class ReconstructionAligner:
    """
    Aligns monocular and real sensor-based point clouds using ICP.
    """

    def __init__(
        self,
        real_path: Path,
        mono_path: Path,
        output_aligned_path: Path,
        output_matrix_path: Path,
        voxel_size: float = 0.02
    ) -> None:
        self.real_path = real_path
        self.mono_path = mono_path
        self.output_aligned_path = output_aligned_path
        self.output_matrix_path = output_matrix_path
        self.voxel_size = voxel_size

    def _run_icp(self, source: o3d.geometry.PointCloud,
                 target: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Performs ICP alignment between two point clouds.

        Args:
            source (o3d.geometry.PointCloud): The source (mono) point cloud.
            target (o3d.geometry.PointCloud): The target (real) point cloud.

        Returns:
            np.ndarray: Transformation matrix (4x4).
        """
        print("[INFO] Running ICP with scaling...")
        threshold = self.voxel_size * 1.5
        source.estimate_normals()
        target.estimate_normals()

        result = o3d.pipelines.registration.registration_icp(
            source, target, threshold,
            np.eye(4),
            o3d.pipelines.registration.
            TransformationEstimationForGeneralizedICP()
        )
        print(f"[OK] Fitness: {result.fitness:.4f}")
        print(f"[OK] RMSE: {result.inlier_rmse:.4f}")
        return result.transformation

    def _apply_transformation(
        self,
        pcd: o3d.geometry.PointCloud,
        T: np.ndarray
    ) -> o3d.geometry.PointCloud:
        """
        Applies transformation matrix to a point cloud.

        Args:
            pcd (o3d.geometry.PointCloud): Point cloud to align.
            T (np.ndarray): 4x4 transformation matrix.

        Returns:
            o3d.geometry.PointCloud: Transformed point cloud.
        """
        return pcd.transform(T)

    def run(self) -> None:
        """
        Executes alignment pipeline: load, register, save, visualize.
        """
        print("[INFO] Loading point clouds...")
        pcd_real = o3d.io.read_point_cloud(str(self.real_path))
        pcd_mono = o3d.io.read_point_cloud(str(self.mono_path))

        pcd_real_d = pcd_real.voxel_down_sample(self.voxel_size)
        pcd_mono_d = pcd_mono.voxel_down_sample(self.voxel_size)

        T = self._run_icp(pcd_mono_d, pcd_real_d)

        np.save(str(self.output_matrix_path), T)
        print(f"[✓] Transformation matrix saved to: {self.output_matrix_path}")

        aligned = self._apply_transformation(pcd_mono, T)
        o3d.io.write_point_cloud(str(self.output_aligned_path), aligned)
        print(f"[✓] Aligned point cloud saved to: {self.output_aligned_path}")

        # Visualization
        pcd_real.paint_uniform_color([0.0, 0.6, 1.0])
        aligned.paint_uniform_color([1.0, 0.3, 0.3])
        o3d.visualization.draw_geometries(
            [pcd_real, aligned],
            window_name='Aligned: Real (blue) vs Mono (red)',
            width=960,
            height=540
        )

        # Save visualization of real point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_real)
        vis.run()
        vis.capture_screen_image(str(self.real_path.with_suffix('.png')))
        print(f"[✓] Visualization saved to: {self.real_path.with_suffix('.png')}")

        # Save visualization of aligned point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(aligned)
        vis.run()
        vis.capture_screen_image(str(self.mono_path.with_suffix('.png')))
        print(f"[✓] Visualization saved to: {self.mono_path.with_suffix('.png')}")

        # Save visualization both together
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_real)
        vis.add_geometry(aligned)
        vis.run()
        vis.capture_screen_image(str(self.output_aligned_path.with_suffix('.png')))
        print(f"[✓] Visualization saved to: {self.output_aligned_path.with_suffix('.png')}")


class ReconstructionAlignerNode(Node):
    """
    ROS 2 node to align mono depth to real sensor map.
    """

    def __init__(self) -> None:
        super().__init__('reconstruction_aligner_node')

        dataset_path = 'datasets/lab_scene_kinect_xyz'
        output_path = 'results/lab_scene_kinect_xyz'
        real_path = Path(
            self.declare_parameter(
                'real_pcd_path',
                f'{dataset_path}/reconstruction_d435.ply'
            ).get_parameter_value().string_value
        )
        mono_path = Path(
            self.declare_parameter(
                'mono_pcd_path',
                f'{dataset_path}/reconstruction_depthanything.ply'
            ).get_parameter_value().string_value
        )
        out_pcd_path = Path(
            self.declare_parameter(
                'output_aligned_pcd',
                f'{output_path}/reconstruction_depthanything_aligned.ply'
            ).get_parameter_value().string_value
        )
        out_matrix_path = Path(
            self.declare_parameter(
                'output_transform_matrix',
                f'{output_path}//T_d_to_m.npy'
            ).get_parameter_value().string_value
        )

        self.get_logger().info("Running alignment pipeline...")

        aligner = ReconstructionAligner(
            real_path=real_path,
            mono_path=mono_path,
            output_aligned_path=out_pcd_path,
            output_matrix_path=out_matrix_path
        )
        aligner.run()

        self.get_logger().info("Alignment complete.")


def main() -> None:
    """
    Launches the ROS 2 node.
    """
    rclpy.init()
    try:
        node = ReconstructionAlignerNode()
        rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("[Shutdown] Interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
