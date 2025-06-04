"""
Module for building pose graphs using pairwise ICP registration.

Creates a sequential pose graph with odometry edges between consecutive
point clouds using Open3D's ICP registration.
"""

from typing import List

import numpy as np
import open3d as o3d


class PoseGraphBuilder:
    """
    Constructs a pose graph from a list of point clouds.
    """

    def __init__(self, voxel_size: float = 0.02) -> None:
        """
        Initializes the builder with voxel size for ICP registration.

        Args:
            voxel_size (float): Voxel size to define correspondence distance.
        """
        self.voxel_size = voxel_size

    def build(
        self,
        point_clouds: List[o3d.geometry.PointCloud]
    ) -> o3d.pipelines.registration.PoseGraph:
        """
        Builds a pose graph using pairwise ICP registration.

        Args:
            point_clouds (List[o3d.geometry.PointCloud]): List of input point
                clouds to register.

        Returns:
            o3d.pipelines.registration.PoseGraph: Constructed pose graph.
        """
        pose_graph = o3d.pipelines.registration.PoseGraph()
        current_odometry = np.eye(4)
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(current_odometry)
        )

        for idx in range(1, len(point_clouds)):
            registration_result = self._register_pairwise(
                source=point_clouds[idx - 1],
                target=point_clouds[idx]
            )
            current_odometry = (
                current_odometry @ registration_result.transformation
            )

            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(
                    np.linalg.inv(current_odometry)
                )
            )
            pose_graph.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(
                    source_node_id=idx - 1,
                    target_node_id=idx,
                    transformation=registration_result.transformation,
                    uncertain=False
                )
            )

        return pose_graph

    def _register_pairwise(
        self,
        source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud
    ) -> o3d.pipelines.registration.RegistrationResult:
        """
        Registers two point clouds using point-to-plane ICP.

        Args:
            source (o3d.geometry.PointCloud): Source point cloud.
            target (o3d.geometry.PointCloud): Target point cloud.

        Returns:
            o3d.pipelines.registration.RegistrationResult: ICP result.
        """
        max_dist = self.voxel_size * 1.5
        estimation = (
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        return o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance=max_dist,
            estimation_method=estimation
        )
