"""
Module for recording RGB-D frames and intrinsics from a RealSense sensor.

Captures synchronized color and depth images and saves them along with
camera intrinsics in Open3D-compatible format.
"""

import time
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

import logging
logging.basicConfig(level=logging.INFO)


class RealSenseRecorder:
    """
    Captures RGB and depth frames using Intel RealSense and exports:
    - RGB as .png
    - Depth as .png (in millimeters) and .npy (in meters)
    - Intrinsics as intrinsics.json
    """

    def __init__(
        self,
        output_path: str = "datasets/lab_scene",
        width: int = 640,
        height: int = 480,
        max_frames: int = 50,
        fps: int = 30,
        warmup_sec: int = 5
    ) -> None:
        """
        Initializes RealSense configuration and output directories.

        Args:
            output_dir (str): Path to save RGB, depth and intrinsics.
            width (int): Frame width.
            height (int): Frame height.
            max_frames (int): Number of frames to capture.
            fps (int): Target frames per second.
            warmup_sec (int): Warm-up time before starting capture.
        """
        self.output_path = Path(output_path)
        self.rgb_dir = self.output_path / "rgb"
        self.depth_png_dir = self.output_path / "depth_png"
        self.depth_npy_dir = self.output_path / "depth_npy"
        self.intrinsics_file = self.output_path / "intrinsics.json"

        self.width = width
        self.height = height
        self.max_frames = max_frames
        self.fps = fps
        self.warmup_sec = warmup_sec

        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.depth_png_dir.mkdir(parents=True, exist_ok=True)
        self.depth_npy_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.color, width, height, rs.format.bgr8, fps
        )
        self.config.enable_stream(
            rs.stream.depth, width, height, rs.format.z16, fps
        )

    def start(self) -> None:
        """
        Starts the RealSense pipeline and records intrinsics.
        """
        logging.info("[INFO] Starting RealSense pipeline...")
        self.pipeline.start(self.config)
        time.sleep(self.warmup_sec)
        self._save_intrinsics()

    def _save_intrinsics(self) -> None:
        """
        Extracts intrinsics from color stream and saves to JSON.
        """
        profile = self.pipeline.get_active_profile()
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )

        data = {
            "width": intr.width,
            "height": intr.height,
            "K": [intr.fx, 0, intr.ppx, 0, intr.fy, intr.ppy, 0, 0, 1],
            "D": [0, 0, 0, 0, 0],
            "distortion_model": intr.model.name
        }

        with open(self.intrinsics_file, "w") as f:
            json.dump(data, f, indent=4)

        logging.info(f"[✓] Intrinsics saved to: {self.intrinsics_file}")

    def normalize_png_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Normalizes the depth map to a range of 0-255.

        Args:
            depth (np.ndarray): Depth map to normalize.

        Returns:
            np.ndarray: Normalized depth map.
        """
        aux_1 = depth - depth.min()
        aux_2 = depth.max() - depth.min()
        normalized = (aux_1) / (aux_2) * 255.0
        return normalized.astype(np.uint8)

    def _save_frame(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        index: int
    ) -> None:
        """
        Saves RGB and depth images.

        Args:
            color (np.ndarray): BGR image (uint8).
            depth (np.ndarray): Depth map in meters (float32).
            index (int): Frame index.
        """
        name = f"frame_{index:04d}"
        cv2.imwrite(str(self.rgb_dir / f"{name}.png"), color)
        cv2.imwrite(
            str(self.depth_png_dir / f"{name}.png"),
            (self.normalize_png_depth(depth)).astype(np.uint16)
        )
        np.save(self.depth_npy_dir / f"{name}.npy", depth)

    def capture(self) -> None:
        """
        Captures RGB-D frames and saves them to disk.
        """
        logging.info("[INFO] Starting frame capture...")
        frame_count = 0
        last_time = time.time()

        try:
            while frame_count < self.max_frames:
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                now = time.time()

                if not depth_frame or not color_frame:
                    continue

                if now - last_time < 1.0 / self.fps:
                    continue
                last_time = now

                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data()) / 1000.0

                self._save_frame(color, depth, frame_count)

                vis = color.copy()
                cv2.putText(
                    vis,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )
                cv2.imshow("RealSense RGB", vis)

                if cv2.waitKey(1) == 27:
                    break

                frame_count += 1

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            logging.info(f"[✓] Capture finished: {frame_count} frames saved.")
