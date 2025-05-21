"""
Application to perform monocular depth estimation and generate 3D point clouds.

It supports real-time visualization from webcam input or batch processing from
an image directory, using the DepthAnythingV2 model and Open3D for rendering.
"""

import os
from typing import Optional

import cv2
import numpy as np

from modules.depth_estimator import DepthEstimator
from modules.point_cloud_processor import PointCloudProcessor
from modules.live_point_cloud_visualizer import LivePointCloudVisualizer


class DepthPointCloudApp:
    """
    Main application for estimating depth maps and visualizing point clouds
    from RGB images in real-time or batch mode.
    """

    def __init__(self,
                 mode: str = 'images',
                 image_dir: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 save_output: bool = False,
                 camera_idx: int = 0) -> None:
        """
        Initializes the application.

        :param: mode (str): Operation mode: 'images' or 'webcam'.
        :param: image_dir (str, optional): Directory containing images for
            batch processing. Required if mode is 'images'.
        :param: output_dir (str, optional): Directory to save output images.
            Required if save_output is True.
        :param: save_output (bool): Whether to save images and results.
        :param: camera_idx (int): Index of the webcam for live mode.

        :raises: ValueError: If mode is not 'images' or 'webcam', or if
            image_dir is not specified in 'images' mode, or if output_dir is
            not specified when save_output is True.
        """
        self.mode = mode
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.save_output = save_output
        self.camera_idx = camera_idx

        if self.mode not in ['images', 'webcam']:
            raise ValueError("Invalid mode. Choose 'images' or 'webcam'.")
        if self.mode == 'images' and not self.image_dir:
            raise ValueError("image_dir must be specified in 'images' mode.")
        if self.save_output and not self.output_dir:
            raise ValueError(
                "output_dir must be specified if save_output is True.")

        self.estimator = DepthEstimator()
        self.processor = PointCloudProcessor()
        self.visualizer = LivePointCloudVisualizer()

        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def process_frame(self,
                      frame: np.ndarray,
                      name: Optional[str] = None) -> None:
        """
        Processes a single RGB frame to compute depth and display
        visualizations.

        :param: frame (np.ndarray): The RGB image frame to process.
        :param: name (str, optional): The name of the image file for saving
            output. If None, the frame will not be saved.
        """
        depth = self.estimator.infer_depth(frame)
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(
            depth_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO
        )

        display = cv2.hconcat([cv2.flip(frame, 0), cv2.flip(depth_colored, 0)])

        if self.save_output and name:
            out_path = os.path.join(self.output_dir, f'depth_{name}')
            cv2.imwrite(out_path, display)

        cv2.imshow('RGB + Depth Map', display)

        pcd = self.processor.create_point_cloud(frame, depth)
        pcd_filtered = self.processor.filter_point_cloud(pcd)
        self.visualizer.update(pcd_filtered)

    def _run_image_directory_mode(self) -> None:
        """
        Runs the app in image directory mode.
        """
        if not self.image_dir:
            raise ValueError("Image directory is not specified.")

        image_files = sorted(
            f for f in os.listdir(self.image_dir) if f.endswith('.png')
        )

        for filename in image_files:
            path = os.path.join(self.image_dir, filename)
            frame = cv2.imread(path)
            if frame is None:
                print(f"[Warning] Could not load image: {path}")
                continue
            self.process_frame(frame, name=filename)

            # Stop if 'Esc' key is pressed
            if cv2.waitKey(1) == 27:
                break

    def _run_webcam_mode(self) -> None:
        """
        Runs the app using webcam input.
        """
        cap = cv2.VideoCapture(self.camera_idx)
        if not cap.isOpened():
            print("[Error] Failed to open webcam.")
            return

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Warning] Failed to capture frame.")
                break

            filename = f'camera_frame_{frame_count:04d}.png'
            self.process_frame(frame, name=filename)
            frame_count += 1

            # Stop if 'Esc' key is pressed
            if cv2.waitKey(1) == 27:
                break

        cap.release()

    def run(self) -> None:
        """
        Starts the application in the selected mode.
        """
        try:
            if self.mode == 'images':
                self._run_image_directory_mode()
            elif self.mode == 'webcam':
                self._run_webcam_mode()
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        finally:
            cv2.destroyAllWindows()
            self.visualizer.close()


if __name__ == '__main__':
    app = DepthPointCloudApp(
        mode='webcam',
        image_dir='datasets/rgbd_dataset_freiburg1_xyz/rgb',
        output_dir='results/depth_maps/test_run',
        save_output=False,
        camera_idx=0
    )
    app.run()
