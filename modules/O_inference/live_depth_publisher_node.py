"""
ROS 2 node for real-time monocular depth inference and point cloud publishing.

Supports both webcam input and batch image directory processing. The generated
point cloud is visualized with Open3D and published to ROS 2 as PointCloud2.
"""

import os
from typing import Optional, List

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from modules.inference import DepthAnythingV2Estimator
from modules.utils import PointCloudProcessor, LivePointCloudVisualizer


class LiveDepthPublisherNode(Node):
    """
    ROS 2 node to perform real-time depth estimation and point cloud
    publishing.
    """

    def __init__(
        self,
        image_dir: Optional[str] = None,
        use_webcam: bool = False,
        cam_index: int = 0,
        frame_rate: float = 1.0,
        topic: str = "/o3d_points"
    ) -> None:
        super().__init__('live_depth_publisher_node')

        self.use_webcam = use_webcam
        self.cam_index = cam_index
        self.image_dir = image_dir
        self.frame_index = 0
        self.image_list: List[str] = []

        self.estimator = DepthAnythingV2Estimator(checkpoint_dir='checkpoints')
        self.processor = PointCloudProcessor(ros_node=self, topic=topic)
        self.visualizer = LivePointCloudVisualizer()

        self._initialize_input_source()
        self.timer = self.create_timer(frame_rate, self._process_next_frame)

        self.get_logger().info("Live depth publisher node started.")

    def _initialize_input_source(self) -> None:
        """
        Initializes input source: either webcam or image directory.
        """
        if self.use_webcam:
            self.cap = cv2.VideoCapture(self.cam_index)
            if not self.cap.isOpened():
                msg = f"Webcam index {self.cam_index} could not be opened."
                self.get_logger().error(msg)
                raise RuntimeError(msg)
        else:
            if not self.image_dir or not os.path.isdir(self.image_dir):
                raise ValueError("Valid image_dir must be provided.")
            self.image_list = sorted([
                os.path.join(self.image_dir, f)
                for f in os.listdir(self.image_dir)
                if f.endswith('.png')
            ])
            if not self.image_list:
                raise FileNotFoundError("No PNG images found in directory.")

    def _process_next_frame(self) -> None:
        """
        Captures the next frame, estimates depth, and publishes point cloud.
        """
        if self.use_webcam:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warning("Webcam frame not available.")
                return
        else:
            if self.frame_index >= len(self.image_list):
                self.get_logger().info("All images processed. Shutting down.")
                self._shutdown()
                return
            frame = cv2.imread(self.image_list[self.frame_index])
            self.frame_index += 1

        if frame is None:
            self.get_logger().warning("Invalid image frame encountered.")
            return

        self._process_and_publish(frame)

    def _process_and_publish(self, frame: np.ndarray) -> None:
        """
        Processes the frame to estimate depth and publish a point cloud.
        """
        depth = self.estimator.infer_depth(frame)
        pcd = self.processor.create_point_cloud(frame, depth)
        pcd_filtered = self.processor.filter_point_cloud(pcd)

        self.visualizer.update(pcd_filtered)
        self.processor.publish_point_cloud(pcd_filtered)

    def _shutdown(self) -> None:
        """
        Gracefully shuts down the node and visualizer.
        """
        self.visualizer.close()
        self.destroy_node()
        rclpy.shutdown()


def main() -> None:
    """
    Entry point to run the depth estimation node.
    """
    rclpy.init()
    try:
        node = LiveDepthPublisherNode(
            use_webcam=True,
            image_dir='datasets/rgbd_dataset_freiburg1_xyz/rgb',
            cam_index=0,
            frame_rate=1.0
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("[Shutdown] Interrupted by user.")
    finally:
        if 'node' in locals():
            node._shutdown()


if __name__ == '__main__':
    main()
