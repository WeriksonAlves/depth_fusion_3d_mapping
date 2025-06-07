# modules/ ...

"""
Depth map fusion utility: combines real and monocular depth using projection.

This module projects the monocular depth map into the real camera space,
fuses it with the real depth (e.g., from D435), and saves fused outputs.
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import cv2


def load_intrinsics(json_path: Path) -> o3d.camera.PinholeCameraIntrinsic:
    """
    Loads Open3D intrinsics from a JSON file in OpenCV-compatible format.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        width=data["width"],
        height=data["height"],
        fx=data["K"][0],
        fy=data["K"][4],
        cx=data["K"][2],
        cy=data["K"][5]
    )
    return intrinsic


def project_point_cloud_to_depth(
    pcd: o3d.geometry.PointCloud,
    intr: o3d.camera.PinholeCameraIntrinsic,
    depth_scale: float = 5000.0,
    depth_trunc: float = 4.0
) -> np.ndarray:
    """
    Projects a 3D point cloud to a depth image using z-buffering.

    Args:
        pcd: Input 3D point cloud (transformed).
        intr: Camera intrinsics.
        depth_scale: Depth scaling factor to convert to uint16.
        depth_trunc: Max depth in meters to keep in projection.

    Returns:
        2D numpy array with depth values in uint16 format.
    """
    width, height = intr.width, intr.height
    fx, fy = intr.get_focal_length()
    cx, cy = intr.get_principal_point()

    depth_image = np.zeros((height, width), dtype=np.float32)
    z_buffer = np.full((height, width), np.inf)

    for x, y, z in np.asarray(pcd.points):
        if z <= 0 or z > depth_trunc:
            continue
        u = int(round((x * fx) / z + cx))
        v = int(round((y * fy) / z + cy))
        if 0 <= u < width and 0 <= v < height and z < z_buffer[v, u]:
            z_buffer[v, u] = z
            depth_image[v, u] = z

    return (depth_image * depth_scale).astype(np.uint16)


def fuse_depth_maps(real: np.ndarray,
                    projected_mono: np.ndarray) -> np.ndarray:
    """
    Performs conditional fusion:
    - If real == 0 → use mono
    - If mono == 0 → use real
    - Else         → use min(real, mono)
    """
    fused = np.where(
        real == 0,
        projected_mono,
        np.where((real > 0) & (projected_mono > 0),
                 np.minimum(real, projected_mono),
                 real)
    )
    return fused


def fuse_dataset_depth_maps(
    rgb_dir: Path,
    depth_real_dir: Path,
    depth_mono_dir: Path,
    transform_path: Path,
    output_dir: Path,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    depth_scale: float = 5000.0
) -> None:
    """
    Fuses real and monocular depth maps for a dataset using a known
    transformation.

    Args:
        rgb_dir: Directory with RGB images (for geometry consistency).
        depth_real_dir: Directory with .npy real depth maps.
        depth_mono_dir: Directory with .npy monocular depth maps.
        transform_path: Path to .npy file with 4x4 matrix.
        output_dir: Output directory for fused PNG + NPY files.
        intrinsics: Open3D intrinsics.
        depth_scale: Scale factor (e.g., 5000 for meters to uint16).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = np.load(transform_path)

    rgb_files = sorted(rgb_dir.glob("*.png"))
    real_files = sorted(depth_real_dir.glob("*.npy"))
    mono_files = sorted(depth_mono_dir.glob("*.npy"))

    assert len(rgb_files) == len(real_files) == len(mono_files), (
        "[ERROR] Frame count mismatch between RGB and depth sources."
    )

    for rgb_path, real_path, mono_path in zip(rgb_files,
                                              real_files,
                                              mono_files):
        print(f"[INFO] Processing: {rgb_path.name}")

        rgb_image = o3d.io.read_image(str(rgb_path))
        depth_real = np.load(real_path).astype(np.float32)
        depth_mono = np.load(mono_path).astype(np.float32)

        # Create point cloud from monocular depth
        depth_img = o3d.geometry.Image(
            (depth_mono * depth_scale).astype(np.uint16)
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_image, depth_img, depth_scale=depth_scale,
            convert_rgb_to_intensity=False
        )
        pcd_mono = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,
                                                                  intrinsics)
        pcd_mono.transform(transform)

        # Project to real camera space
        depth_proj = project_point_cloud_to_depth(pcd_mono,
                                                  intrinsics,
                                                  depth_scale)

        # Fuse depth maps
        fused = fuse_depth_maps(depth_real * depth_scale, depth_proj)

        # Save as PNG
        png_path = output_dir / f"{rgb_path.stem}.png"
        cv2.imwrite(str(png_path), fused)

        # Save as NPY (in meters)
        npy_path = output_dir / f"{rgb_path.stem}.npy"
        np.save(npy_path, fused.astype(np.float32) / depth_scale)

        print(f"[✓] Saved fused maps: {png_path.name} | valid pixels: {np.count_nonzero(fused)}")


def main() -> None:
    """
    Runs fusion for a specific dataset scene.
    """
    scene = "lab_scene_f"
    scale = 1000

    fuse_dataset_depth_maps(
        rgb_dir=Path(f"datasets/{scene}/rgb"),
        depth_real_dir=Path(f"datasets/{scene}/depth_npy"),
        depth_mono_dir=Path(f"results/{scene}/d4/depth_npy"),
        transform_path=Path(f"results/{scene}/d6/T_d_to_m_frame0000.npy"),
        output_dir=Path(f"results/{scene}/d8/T_d_to_m__depth_fused_{scale}"),
        intrinsics=load_intrinsics(Path(f"datasets/{scene}/intrinsics.json")),
        depth_scale=scale  # Scale for meters to uint16
    )


if __name__ == "__main__":
    main()
