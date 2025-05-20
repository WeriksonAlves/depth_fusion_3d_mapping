"""

"""

import os
import cv2
import numpy as np

from modules.depth_estimator import DepthEstimator
from modules.point_cloud_processor import PointCloudProcessor
from modules.live_point_cloud_visualizer import LivePointCloudVisualizer


class DepthPointCloudApp:
    """
    A class to process images or webcam frames to infer depth maps and
    visualize point clouds in real-time. This class provides methods to
    process a single frame, run the application in different modes (images or
    webcam), and manage the output directory for saving results. The
    application uses a depth estimator to infer depth maps from RGB images and
    a point cloud processor to create and filter point clouds. The visualizer
    displays the point clouds in a live window using Open3D.
    """

    def __init__(self, mode='images', image_dir: str = None,
                 output_dir: str = None, save_output=False,
                 camera_idx=0) -> None:
        """
        Initializes the DepthPointCloudApp with the specified mode, image
        directory, output directory, save output flag, and camera index.

        Args:
            mode: The mode of operation ('images' or 'webcam').
            image_dir: Directory containing input images (if mode is 'images').
            output_dir: Directory to save output images and point clouds.
            save_output: Flag to save output images.
            camera_idx: Index of the webcam to use (if mode is 'webcam').
        """
        self.mode = mode
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.save_output = save_output
        self.camera_idx = camera_idx

        self.estimator = DepthEstimator()
        self.processor = PointCloudProcessor()
        self.visualizer = LivePointCloudVisualizer()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def process_frame(self, frame: np.ndarray, name: str = None) -> None:
        """
        Processes a single frame to infer its depth map, displays the
        concatenated RGB and depth images, and optionally saves the result to
        disk.

        Args:
            frame (np.ndarray): The input RGB image as a NumPy array.
            name (str, optional): The name used for saving the output image. If
                None, the image is not saved.
        Returns:
            None
        Side Effects:
            - Displays a window showing the concatenated RGB and depth images.
            - Saves the concatenated image to disk if save_output is True and a
                name is provided.
        Notes:
            - The depth map is inferred using the 'estimator' object's
                'infer_depth' method.
            - The depth map is normalized, color-mapped, and concatenated with
                the original image for visualization.
        """
        depth = self.estimator.infer_depth(frame)
        depth_norm = cv2.normalize(
            depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
        vis_image = cv2.hconcat(
            [cv2.flip(frame, 0), cv2.flip(depth_colored, 0)])

        if self.save_output and name:
            out_path = os.path.join(self.output_dir, f'depth_{name}')
            cv2.imwrite(out_path, vis_image)

        cv2.imshow('RGB + Depth Map', vis_image)

        pcd = self.processor.create_point_cloud(frame, depth)
        pcd_filtered = self.processor.filter_point_cloud(pcd)
        self.visualizer.update(pcd_filtered)

    def run(self) -> None:
        """
        Runs the application in the specified mode (images or webcam). In
        'images' mode, it processes all images in the specified directory. In
        'webcam' mode, it captures frames from the webcam and processes them
        in real-time. The application continues running until the user
        presses the 'Esc' key.
        """
        if self.mode == 'images':
            files = sorted(
                f for f in os.listdir(self.image_dir) if f.endswith('.png'))
            for filename in files:
                img_path = os.path.join(self.image_dir, filename)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"[Erro] Imagem corrompida: {img_path}")
                    continue
                self.process_frame(image, filename)
                if cv2.waitKey(1) == 27:
                    break

        elif self.mode == 'webcam':
            cap = cv2.VideoCapture(self.camera_idx)
            if not cap.isOpened():
                print("[Erro] Câmera não encontrada.")
                return
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[Aviso] Frame não capturado.")
                    break
                self.process_frame(
                    frame, f'camera_frame_{frame_count:04d}.png')
                frame_count += 1
                if cv2.waitKey(1) == 27:
                    break
            cap.release()

        cv2.destroyAllWindows()
        self.visualizer.close()


if __name__ == '__main__':
    app = DepthPointCloudApp(
        mode='webcam',  # images or webcam
        image_dir='SIBGRAPI2025/datasets/rgbd_dataset_freiburg1_xyz/rgb',
        output_dir='SIBGRAPI2025/results/depth_maps_2/test_1',
        save_output=False,
        camera_idx=0
    )
    app.run()
