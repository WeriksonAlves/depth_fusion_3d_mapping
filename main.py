"""
Monocular depth estimation and point cloud processing application.

This script supports real-time point cloud generation using webcam or batch
processing from image directories. It includes both local visualization and
ROS 2 publication to topics.
"""

import os
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node

from modules.depth_estimator import DepthAnythingV2Estimator
from modules.point_cloud_processor import PointCloudProcessor
from modules.live_point_cloud_visualizer import LivePointCloudVisualizer


class DepthPointCloudApp:
    """
    Application for generating and visualizing point clouds from RGB images.
    """

    def __init__(
        self,
        current_dir: str,
        mode: str = 'images',
        image_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        save_output: bool = False,
        camera_idx: int = 0
    ) -> None:
        """
        Initializes the app with depth estimator, processor, and visualizer.

        Args:
            current_dir (str): Base directory path.
            mode (str): Mode of operation ('images' or 'webcam').
            image_dir (Optional[str]): Input directory for images.
            output_dir (Optional[str]): Output directory for results.
            save_output (bool): Whether to save outputs.
            camera_idx (int): Webcam index.
        """
        self.mode = mode
        self.image_dir = os.path.join(current_dir, image_dir or "")
        self.output_dir = os.path.join(current_dir, output_dir or "")
        self.save_output = save_output
        self.camera_idx = camera_idx

        if mode not in ['images', 'webcam']:
            raise ValueError("Mode must be 'images' or 'webcam'.")
        if mode == 'images' and not image_dir:
            raise ValueError("image_dir must be provided in 'images' mode.")
        if save_output and not output_dir:
            raise ValueError("output_dir required if save_output is True.")

        self.estimator = DepthAnythingV2Estimator(
            checkpoint_dir=os.path.join(current_dir, 'checkpoints')
        )
        self.processor = PointCloudProcessor()
        self.visualizer = LivePointCloudVisualizer()

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def process_frame(
        self,
        frame: np.ndarray,
        name: Optional[str] = None
    ) -> None:
        """
        Processes a frame and visualizes it.

        Args:
            frame (np.ndarray): RGB image frame.
            name (Optional[str]): Base name for saved outputs.
        """
        depth = self.estimator.infer_depth(frame)
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_color = cv2.applyColorMap(
            depth_norm.astype(np.uint8), cv2.COLORMAP_INFERNO
        )

        view = cv2.hconcat([cv2.flip(frame, 0), cv2.flip(depth_color, 0)])
        cv2.imshow('RGB + Depth Map', view)

        pcd = self.processor.create_point_cloud(frame, depth)
        pcd_filtered = self.processor.filter_point_cloud(pcd)
        self.visualizer.update(pcd_filtered)

        if self.save_output and name:
            cv2.imwrite(os.path.join(self.output_dir, f'rgb{name}'), frame)
            cv2.imwrite(
                os.path.join(self.output_dir, f'depth_{name}'), depth_color
            )
            self.processor.save_point_cloud(
                pcd, os.path.join(self.output_dir, f'pcd_{name}.ply')
            )

    def _run_from_directory(self) -> None:
        files = sorted(
            f for f in os.listdir(self.image_dir) if f.endswith('.png')
        )

        for filename in files:
            path = os.path.join(self.image_dir, filename)
            frame = cv2.imread(path)
            if frame is None:
                print(f"[Warning] Could not load image: {path}")
                continue

            self.process_frame(frame, name=filename)
            if cv2.waitKey(1) == 27:
                break

    def _run_from_webcam(self) -> None:
        cap = cv2.VideoCapture(self.camera_idx)
        if not cap.isOpened():
            print("[Error] Unable to access webcam.")
            return

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Warning] Failed to capture frame.")
                break

            filename = f'frame_{frame_count:04d}.png'
            self.process_frame(frame, name=filename)
            frame_count += 1

            if cv2.waitKey(1) == 27:
                break

        cap.release()

    def run(self) -> None:
        """
        Runs the application in the selected mode.
        """
        try:
            if self.mode == 'images':
                self._run_from_directory()
            else:
                self._run_from_webcam()
        finally:
            cv2.destroyAllWindows()
            self.visualizer.close()


class LivePublisherNode(Node):
    """
    ROS 2 node for real-time depth estimation and point cloud publishing.
    """

    def __init__(
        self,
        image_dir: Optional[str] = None,
        use_webcam: bool = False,
        cam_index: int = 0,
        frame_rate: float = 1.0
    ) -> None:
        super().__init__('live_pointcloud_publisher')

        self.use_webcam = use_webcam
        self.image_dir = image_dir
        self.cam_index = cam_index
        self.image_list = []
        self.frame_idx = 0

        self.estimator = DepthAnythingV2Estimator(checkpoint_dir='checkpoints')
        self.processor = PointCloudProcessor(ros_node=self)
        self.visualizer = LivePointCloudVisualizer()

        if self.use_webcam:
            self.cap = cv2.VideoCapture(self.cam_index)
            if not self.cap.isOpened():
                self.get_logger().error("Webcam not accessible.")
                raise RuntimeError("Webcam unavailable.")
        else:
            self.image_list = sorted([
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.endswith('.png')
            ])
            if not self.image_list:
                raise FileNotFoundError("No input images found.")

        self.timer = self.create_timer(frame_rate, self.timer_callback)
        self.get_logger().info("Point cloud publisher initialized.")

    def timer_callback(self) -> None:
        if self.use_webcam:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("Webcam frame not captured.")
                return
        else:
            if self.frame_idx >= len(self.image_list):
                self.get_logger().info("All images processed.")
                self.visualizer.close()
                rclpy.shutdown()
                return
            frame = cv2.imread(self.image_list[self.frame_idx])
            self.frame_idx += 1

        if frame is None:
            self.get_logger().warn("Invalid image frame.")
            return

        depth = self.estimator.infer_depth(frame)
        pcd = self.processor.create_point_cloud(frame, depth)
        pcd_filtered = self.processor.filter_point_cloud(pcd)

        self.visualizer.update(pcd_filtered)
        self.processor.publish_point_cloud(pcd_filtered)

        if cv2.waitKey(1) == 27:
            self.visualizer.close()
            rclpy.shutdown()


def main() -> None:
    """
    Main function to launch either the ROS node or local visualization app.
    """
    ros_mode = True
    use_webcam = True
    image_dir = 'datasets/rgbd_dataset_freiburg1_xyz/rgb'

    if ros_mode:
        rclpy.init()
        try:
            node = LivePublisherNode(
                use_webcam=use_webcam,
                image_dir=image_dir,
                cam_index=0,
                frame_rate=1.0
            )
            rclpy.spin(node)
        except KeyboardInterrupt:
            print("[Shutdown] Interrupted by user.")
        finally:
            node.visualizer.close()
            node.destroy_node()
            rclpy.shutdown()
    else:
        app = DepthPointCloudApp(
            mode='webcam',
            current_dir=os.path.dirname(os.path.abspath(__file__)),
            image_dir=image_dir,
            output_dir='results/test/',
            save_output=False,
            camera_idx=0
        )
        app.run()


if __name__ == '__main__':
    main()
