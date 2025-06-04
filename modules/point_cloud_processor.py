"""
Point cloud processing module for RGB-D inputs using Open3D and ROS 2.

Provides methods to generate, filter, save, and publish point clouds
derived from RGB and depth images.
"""

import struct
from typing import Optional

import numpy as np
import open3d as o3d
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


class PointCloudProcessor:
    """
    Handles 3D point cloud creation, filtering, and publishing to ROS 2.
    """

    def __init__(
        self,
        fx: float = 525.0,
        fy: float = 525.0,
        cx: float = 319.5,
        cy: float = 239.5,
        width: int = 640,
        height: int = 480,
        ros_node: Optional[Node] = None,
        frame_id: str = "map",
        topic: str = "/o3d_points"
    ) -> None:
        """
        Initializes camera intrinsics and ROS publisher if a node is provided.

        Args:
            fx (float): Focal length along the x-axis.
            fy (float): Focal length along the y-axis.
            cx (float): Principal point x-coordinate.
            cy (float): Principal point y-coordinate.
            width (int): Image width in pixels.
            height (int): Image height in pixels.
            ros_node (Node, optional): ROS 2 node used to create the publisher.
            frame_id (str): Frame of reference for the published point cloud.
            topic (str): ROS 2 topic to publish point clouds on.
        """
        self._intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )
        self._ros_node = ros_node
        self._frame_id = frame_id
        self._publisher = None

        if ros_node is not None:
            self._publisher = ros_node.create_publisher(
                PointCloud2, topic, 10
            )

    def create_point_cloud(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        depth_scale: float = 1000.0,
        depth_trunc: float = 4.0
    ) -> o3d.geometry.PointCloud:
        """
        Creates a 3D point cloud from aligned RGB and depth images.

        Args:
            rgb_image (np.ndarray): Input RGB image of shape (H, W, 3).
            depth_map (np.ndarray): Depth image in meters of shape (H, W).
            depth_scale (float): Depth scale to convert meters to millimeters.
            depth_trunc (float): Maximum depth threshold.

        Returns:
            o3d.geometry.PointCloud: Generated point cloud.
        """
        rgb = o3d.geometry.Image(rgb_image)
        depth_scaled = (depth_map * depth_scale).astype(np.uint16)
        depth = o3d.geometry.Image(depth_scaled)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )

        return o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, self._intrinsics
        )

    def filter_point_cloud(
        self,
        cloud: o3d.geometry.PointCloud,
        voxel_size: float = 0.01,
        nb_neighbors: int = 20,
        std_ratio: float = 2.0
    ) -> o3d.geometry.PointCloud:
        """
        Applies voxel downsampling and statistical outlier removal.

        Args:
            cloud (o3d.geometry.PointCloud): Input point cloud.
            voxel_size (float): Size of the voxel grid.
            nb_neighbors (int): Number of neighbors to analyze per point.
            std_ratio (float): Threshold for outlier removal.

        Returns:
            o3d.geometry.PointCloud: Filtered point cloud.
        """
        downsampled = cloud.voxel_down_sample(voxel_size)
        filtered, _ = downsampled.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        return filtered

    def save_point_cloud(
        self,
        cloud: o3d.geometry.PointCloud,
        filename: str = "output_map.pcd"
    ) -> None:
        """
        Saves a point cloud to a file in PCD format.

        Args:
            cloud (o3d.geometry.PointCloud): Point cloud to save.
            filename (str): Destination filename.
        """
        o3d.io.write_point_cloud(filename, cloud)

    def publish_point_cloud(
        self,
        cloud: o3d.geometry.PointCloud
    ) -> None:
        """
        Publishes a point cloud to the configured ROS 2 topic.

        Args:
            cloud (o3d.geometry.PointCloud): Point cloud to publish.

        Raises:
            RuntimeError: If publisher is not initialized.
        """
        if self._publisher is None:
            raise RuntimeError("ROS 2 publisher not initialized.")

        msg = self._convert_to_ros_msg(cloud)
        self._publisher.publish(msg)
        self._ros_node.get_logger().info("Point cloud published.")

    def _convert_to_ros_msg(
        self,
        cloud: o3d.geometry.PointCloud
    ) -> PointCloud2:
        """
        Converts an Open3D point cloud to a ROS 2 PointCloud2 message.

        Args:
            cloud (o3d.geometry.PointCloud): Input Open3D point cloud.

        Returns:
            PointCloud2: ROS 2 formatted point cloud message.
        """
        points = np.asarray(cloud.points)
        colors = np.asarray(cloud.colors)

        if colors.shape[0] != points.shape[0]:
            colors = np.zeros_like(points)

        data = []
        for i in range(points.shape[0]):
            x, y, z = points[i]
            r, g, b = (colors[i] * 255).astype(np.uint8)
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0]
            data.append([x, y, z, rgb])
        data_array = np.array(data, dtype=np.float32)

        msg = PointCloud2()
        msg.header = Header()
        msg.header.stamp = self._ros_node.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.height = 1
        msg.width = data_array.shape[0]
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32,
                       count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32,
                       count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32,
                       count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32,
                       count=1)
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        msg.is_dense = True
        msg.data = data_array.tobytes()

        return msg
