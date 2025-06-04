"""
Intrinsic parameter evaluator for monocular depth estimation and point clouds.

Tests multiple sets of pinhole camera intrinsics using a fixed RGB-D image
pipeline. Results are visualized and exported to a CSV file.
"""

import os
import cv2
import numpy as np
import pandas as pd

from modules.depth_estimator import DepthAnythingV2Estimator
from modules.point_cloud_processor import PointCloudProcessor
from modules.live_point_cloud_visualizer import LivePointCloudVisualizer


class IntrinsicsTester:
    """
    Evaluates multiple intrinsic configurations for 3D point cloud generation.
    """

    def __init__(
        self,
        image_dir: str,
        output_csv: str,
        intrinsics_list: list[dict]
    ) -> None:
        """
        Initializes the test environment.

        Args:
            image_dir (str): Path to a directory containing RGB images.
            output_csv (str): Output CSV file to store test results.
            intrinsics_list (list): List of intrinsic parameter dictionaries.
        """
        self.image_dir = image_dir
        self.output_csv = output_csv
        self.intrinsics_list = intrinsics_list
        self.results = []

        self.rgb_image = self._load_sample_image()
        self.depth_map = self._estimate_depth(self.rgb_image)
        self.visualizer = LivePointCloudVisualizer()

    def _load_sample_image(self) -> np.ndarray:
        """
        Loads the first available RGB image from the directory.

        Returns:
            np.ndarray: Loaded RGB image.

        Raises:
            FileNotFoundError: If no valid image is found.
        """
        files = sorted(
            f for f in os.listdir(self.image_dir) if f.endswith('.png')
        )
        if not files:
            raise FileNotFoundError("No .png images found in the directory.")

        path = os.path.join(self.image_dir, files[0])
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")

        return image

    def _estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Performs monocular depth estimation on the input image.

        Args:
            image (np.ndarray): Input RGB image.

        Returns:
            np.ndarray: Predicted depth map.
        """
        estimator = DepthAnythingV2Estimator()
        return estimator.infer_depth(image)

    def _evaluate_intrinsics_set(
        self,
        idx: int,
        params: dict
    ) -> None:
        """
        Evaluates a single intrinsic parameter configuration.

        Args:
            idx (int): Index of the test configuration.
            params (dict): Dictionary with fx, fy, cx, cy values.
        """
        print(f"\n[INFO] Testing intrinsics set {idx}: {params}")

        processor = PointCloudProcessor(
            fx=params["fx"],
            fy=params["fy"],
            cx=params["cx"],
            cy=params["cy"],
            width=self.rgb_image.shape[1],
            height=self.rgb_image.shape[0]
        )

        point_cloud = processor.create_point_cloud(
            self.rgb_image, self.depth_map
        )
        filtered_cloud = processor.filter_point_cloud(point_cloud)

        print(f"[INFO] Visualizing point cloud for test {idx}")
        self.visualizer.update(filtered_cloud)

        points = np.asarray(filtered_cloud.points)
        if points.size == 0:
            print("[Warning] No valid points generated.")
            return

        bbox = filtered_cloud.get_axis_aligned_bounding_box()
        center = bbox.get_center()

        self.results.append({
            "Test ID": idx,
            "fx": params["fx"],
            "fy": params["fy"],
            "cx": params["cx"],
            "cy": params["cy"],
            "Num Points": len(points),
            "BBox Volume": bbox.volume(),
            "BBox Center X": center[0],
            "BBox Center Y": center[1],
            "BBox Center Z": center[2],
        })

    def run(self) -> None:
        """
        Executes the test sequence for all defined intrinsic sets.
        """
        for idx, intrinsics in enumerate(self.intrinsics_list):
            self._evaluate_intrinsics_set(idx, intrinsics)

            key = input(
                "[INFO] Press Enter to continue or 'q' to quit: "
            )
            if key.lower() == 'q':
                break

        self.visualizer.close()
        self._export_results()

    def _export_results(self) -> None:
        """
        Saves all test statistics to a CSV file.
        """
        if not self.results:
            print("[Warning] No data to save.")
            return

        df = pd.DataFrame(self.results)
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        df.to_csv(self.output_csv, index=False)
        print(f"[OK] Results saved to: {self.output_csv}")


def main() -> None:
    """
    Entrypoint for running intrinsic calibration tests.
    """
    image_dir = 'datasets/rgbd_dataset_freiburg1_xyz/rgb'
    output_csv = 'results/intrinsics_test_results.csv'

    intrinsics_list = [
        {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5},
        {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0},
        {"fx": 450.0, "fy": 450.0, "cx": 310.0, "cy": 230.0},
        {"fx": 500.0, "fy": 480.0, "cx": 300.0, "cy": 250.0},
    ]

    tester = IntrinsicsTester(image_dir, output_csv, intrinsics_list)
    tester.run()


if __name__ == '__main__':
    main()
