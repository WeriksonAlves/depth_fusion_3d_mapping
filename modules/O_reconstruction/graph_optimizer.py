"""
Module for global pose graph optimization using Open3D.

Refines the poses of RGB-D point clouds using pairwise constraints
and global optimization (Levenberg-Marquardt).
"""

from typing import List

import open3d as o3d


class GraphOptimizer:
    """
    Refines pose graph transformations using global optimization.
    """

    def __init__(self, voxel_size: float = 0.02) -> None:
        """
        Initializes the optimizer with default parameters.

        Args:
            voxel_size (float): Voxel size used to define correspondence
                distance.
        """
        self.voxel_size = voxel_size

    def optimize(
        self,
        pose_graph: o3d.pipelines.registration.PoseGraph,
        point_clouds: List[o3d.geometry.PointCloud]
    ) -> None:
        """
        Applies global optimization and updates each point cloud pose.

        Args:
            pose_graph (o3d.pipelines.registration.PoseGraph): Pose graph
                with odometry and loop closures.
            point_clouds (List[o3d.geometry.PointCloud]): Point clouds to
                be transformed using the optimized poses.
        """
        optimization_method = (
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
        )
        convergence_criteria = (
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
        )
        optimization_option = (
            o3d.pipelines.registration.GlobalOptimizationOption(
                max_correspondence_distance=self.voxel_size * 1.5,
                edge_prune_threshold=0.25,
                reference_node=0
            )
        )

        o3d.pipelines.registration.global_optimization(
            pose_graph,
            optimization_method,
            convergence_criteria,
            optimization_option
        )

        for pcd, node in zip(point_clouds, pose_graph.nodes):
            pcd.transform(node.pose)
