"""
Depth map fusion utility: combines real and monocular depth using projection.

This module projects the monocular depth map into the real camera space,
fuses it with the real depth (e.g., from D435), and saves fused outputs.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2


class DepthFusionProcessor:
    """
    Batch processor for depth fusion, handling multiple scenes and
    configurations.
    """

    def __init__(
        self,
        depth_real_dir: Path,
        depth_estimated_dir: Path,
        output_dir: Path,
    ) -> None:
        """
        Initializes the batch processor with directories and transformation
        path.

        Args:
            depth_real_dir (Path): Directory containing real depth maps.
            depth_estimated_dir (Path): Directory containing estimated depth
                maps.
            output_dir (Path): Directory to save the fused depth maps and
                statistics.
        """
        self._depth_real_dir = depth_real_dir
        self._depth_estimated_dir = depth_estimated_dir
        self._output_dir = output_dir

        self._validate_directories()

    def _validate_directories(self) -> None:
        """
        Validates the existence of the required directories and files.
        Raises:
            FileNotFoundError: If any required directory or file is missing.
        """
        for directory in [self._depth_real_dir, self._depth_estimated_dir]:
            if not directory.exists():
                raise FileNotFoundError(f"Directory not found: {directory}")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._npy_path = self._output_dir / "npy"
        self._npy_path.mkdir(parents=True, exist_ok=True)
        self._png_dir = self._output_dir / "png"
        self._png_dir.mkdir(parents=True, exist_ok=True)

    def _load_depth_maps(
        self,
        directory: Path,
        extension: str
    ) -> List[np.ndarray]:
        """
        Loads depth maps from the specified directory.

        Args:
            directory (Path): Directory containing depth maps.
            extension (str): File extension to filter depth maps.

        Returns:
            List[np.ndarray]: List of loaded depth maps nx(height, width).
        """
        depth_maps = []
        sorted_directory = sorted(directory.glob(f"*{extension}"))
        for file in sorted_directory:
            depth_map = np.load(file)
            if depth_map.ndim == 2:
                depth_maps.append(depth_map)
            else:
                raise ValueError(
                    f"Invalid depth map shape in {file}: {depth_map.shape}"
                )
        return depth_maps

    def _fuse_maps_mean_std(
        self,
        depth_real: np.ndarray,
        depth_estimated: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Fuses two depth maps using mean and standard deviation.

        Args:
            depth_real (np.ndarray): Real depth map.
            depth_estimated (np.ndarray): Estimated depth map.
        Returns:
            Tuple[np.ndarray, Dict]: Fused depth map and statistics.
        """
        mask_real = depth_real > 0
        mask_estimated = depth_estimated > 0
        if not np.any(mask_real) or not np.any(mask_estimated):
            raise ValueError("No valid pixels for fusion.")

        estimated_mean = depth_estimated.mean()
        estimated_std = depth_estimated.std()

        real_mean = depth_real.mean()
        real_std = depth_real.std()

        # mask = mask_real & mask_estimated
        mask = mask_real

        fused = np.zeros_like(depth_real, dtype=np.float32)
        fused[mask] = (
            (depth_real[mask] - real_mean) / real_std +
            (depth_estimated[mask] - estimated_mean) / estimated_std
        ) * estimated_std + estimated_mean

        stats = {
            "real_mean": float(real_mean),
            "real_std": float(real_std),
            "estimated_mean": float(estimated_mean),
            "estimated_std": float(estimated_std),
            "num_valid_pairs": int(np.count_nonzero(mask))
        }
        return fused, stats

    def _fuse_maps_least_squares(
        self, real: np.ndarray, mono: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Fuses two depth maps using inverse-depth linear regression.
        """
        mask = (real > 0) & (mono > 0)
        if not np.any(mask):
            raise ValueError("No valid pixels for fusion.")

        x = mono[mask].astype(np.float32)
        y = 1.0 / real[mask].astype(np.float32)

        A = np.vstack([x, np.ones_like(x)]).T
        scale, shift = np.linalg.lstsq(A, y, rcond=None)[0]

        corrected = np.zeros_like(mono, dtype=np.float32)
        valid = mono > 0
        corrected[valid] = 1.0 / (scale * mono[valid] + shift)

        fused = np.where(real > 0, real, corrected)

        stats = {
            "scale": float(scale),
            "shift": float(shift),
            "num_valid_pairs": int(np.count_nonzero(mask))
        }
        return fused, stats

    def _normalize_png_depth(self, depth: np.ndarray) -> np.ndarray:
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

    def _combine_depth_maps(
        self,
        depth_real: np.ndarray,
        depth_estimated: np.ndarray,
        fused: np.ndarray
    ) -> np.ndarray:
        """
        Combines real, estimated, and fused depth maps into a single image.

        Args:
            depth_real (np.ndarray): Real depth map.
            depth_estimated (np.ndarray): Estimated depth map.
            fused (np.ndarray): Fused depth map.

        Returns:
            np.ndarray: Combined image of the three depth maps.
        """
        real_normalized = self._normalize_png_depth(depth_real)
        estimated_normalized = self._normalize_png_depth(depth_estimated)
        fused_normalized = self._normalize_png_depth(fused)

        combined = np.hstack(
            (real_normalized[:, :, np.newaxis],
             estimated_normalized[:, :, np.newaxis],
             fused_normalized[:, :, np.newaxis])
        )
        return combined

    def _save_combined_image(
        self,
        combined_img: np.ndarray,
        index: int
    ) -> None:
        """
        Saves the combined image of real, estimated, and fused depth maps.

        Args:
            combined_img (np.ndarray): Combined image to save.
            index (int): Index for naming the output file.
        """
        output_path = self._png_dir / f"combined_depth_{index:04d}.png"
        cv2.imwrite(str(output_path), combined_img)
        print(f"[✓] Combined depth map saved to: {output_path}")

    def _save_fused_depth_map(
        self,
        fused: np.ndarray,
        index: int
    ) -> None:
        """
        Saves the fused depth map as a .npy file.

        Args:
            fused (np.ndarray): Fused depth map to save.
            index (int): Index for naming the output file.
        """
        fused_path = self._npy_path / f"fused_depth_{index:04d}.npy"
        np.save(fused_path, fused.astype(np.float32))
        print(f"[✓] Fused depth map saved to: {fused_path}")

    def run(self, mode: int) -> None:
        """
        Loads depth maps from both real and estimated directories, applies
        transformations, and fuses them using least squares method.
        Saves the fused depth maps and statistics to the output directory.
        """
        # Load depth maps from both directories in .npy format (height, width)
        print("[INFO] Loading depth maps...")
        depth_real_maps = self._load_depth_maps(self._depth_real_dir, '.npy')
        depth_estimated_maps = self._load_depth_maps(self._depth_estimated_dir,
                                                     '.npy')

        # Check if the number of depth maps matches
        if len(depth_real_maps) != len(depth_estimated_maps):
            raise ValueError(
                "Mismatch in number of depth maps "
                "between real and est directories."
            )

        # Fuse depth maps using mean and std
        stats_all = []
        for i, (depth_real, depth_estimated) in enumerate(
            zip(depth_real_maps, depth_estimated_maps)
        ):
            print(
                f"[INFO] Processing depth map {i + 1}/{len(depth_real_maps)}")
            if mode == 0:
                # Fuse maps using least squares
                fused, fusion_stats = self._fuse_maps_mean_std(
                    depth_real,
                    depth_estimated
                )
            elif mode == 1:
                # Fuse maps using least squares
                fused, fusion_stats = self._fuse_maps_least_squares(
                    depth_real,
                    depth_estimated
                )

            # Concatenate depth maps for visualization
            concateneted_img = self._combine_depth_maps(
                depth_real, depth_estimated, fused
            )

            # Save concatenated image as png
            self._save_combined_image(concateneted_img, i)

            # Save fused depth map as npy
            self._save_fused_depth_map(fused, i)

            # Save statistics
            stats = {
                "frame": f"fused_depth_{i:04d}",
                "depth_min_m": float(fused.min()),
                "depth_max_m": float(fused.max()),
                "depth_mean_m": float(fused.mean()),
                "depth_std_m": float(fused.std()),
                "stats_method": fusion_stats
            }
            stats_all.append(stats)

        # Save all statistics to a JSON file
        stats_path = self._output_dir / "fusion_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_all, f, indent=4)
        print(f"[✓] Fusion statistics saved to: {stats_path}")
