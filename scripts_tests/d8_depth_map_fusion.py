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
                    projected_mono: np.ndarray,
                    mode: str = "min") -> np.ndarray:
    """
    Fuses real and projected monocular depth maps.

    Args:
        real: Real depth map (e.g., from D435).
        projected_mono: Projected monocular depth map.
        mode: Fusion mode, 'min', 'mean', 'real-priority' and mono-priority.

    Returns:
        Fused depth map as a numpy array.
    """

    # Performs conditional fusion:
    # - If real == 0 → use mono
    # - If mono == 0 → use real
    # - Else         → use min(real, mono)

    if mode == "min":
        fused = np.where(
            real == 0,
            projected_mono,
            np.where((real > 0) & (projected_mono > 0),
                     np.minimum(real, projected_mono),
                     real)
        )
    elif mode == "mean":
        fused = np.where(
            real == 0,
            projected_mono,
            np.where((real > 0) & (projected_mono > 0),
                     (real + projected_mono) / 2.0,
                     real)
        )
    elif mode == "real-priority":
        fused = np.where(
            real > 0,
            real,
            projected_mono
        )
    elif mode == "mono-priority":
        fused = np.where(
            projected_mono > 0,
            projected_mono,
            real
        )
    else:
        raise ValueError(f"Unsupported fusion mode: {mode}")
    return fused


def fuse_dataset_depth_maps(
    rgb_dir: Path,
    depth_real_dir: Path,
    depth_mono_dir: Path,
    transform_path: Path,
    output_dir: Path,
    intrinsics: o3d.camera.PinholeCameraIntrinsic,
    depth_scale: float = 5000.0,
    depth_trunc: float = 4.0,
    mode: str = "min"
) -> None:
    """
    Fuses real and monocular depth maps using a known transformation.

    Args:
        rgb_dir: Directory with RGB images.
        depth_real_dir: Directory with .npy real depth maps.
        depth_mono_dir: Directory with .npy monocular depth maps.
        transform_path: Path to 4x4 transformation matrix.
        output_dir: Output directory for fused PNG + NPY files.
        intrinsics: Open3D intrinsics.
        depth_scale: Scale factor for uint16 conversion.
        depth_trunc: Maximum valid depth (in meters).
        mode: Fusion mode ('min', 'mean', 'real-priority', 'mono-priority').
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = np.load(transform_path)

    rgb_files = sorted(rgb_dir.glob("*.png"))
    real_files = sorted(depth_real_dir.glob("*.npy"))
    mono_files = sorted(depth_mono_dir.glob("*.npy"))

    assert len(rgb_files) == len(real_files) == len(mono_files), (
        "[ERROR] Frame count mismatch between RGB and depth sources."
    )

    stats_per_frame = []
    for rgb_path, real_path, mono_path in zip(rgb_files,
                                              real_files,
                                              mono_files):
        print(f"[INFO] Processing: {rgb_path.name}")

        rgb_image = o3d.io.read_image(str(rgb_path))
        depth_real = np.load(real_path).astype(np.float32)
        depth_mono = np.load(mono_path).astype(np.float32)

        # Build point cloud from monocular depth
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

        # Project to image plane
        depth_proj = project_point_cloud_to_depth(
            pcd=pcd_mono,
            intr=intrinsics,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc
        )

        # Fuse
        fused = fuse_depth_maps(depth_real * depth_scale, depth_proj, mode)

        # Save outputs
        png_path = output_dir / f"{rgb_path.stem}.png"
        npy_path = output_dir / f"{rgb_path.stem}.npy"

        cv2.imwrite(str(png_path), fused)
        np.save(npy_path, fused.astype(np.float32) / depth_scale)

        # Coletar estatísticas do frame atual
        valid_mask = fused > 0
        depth_valid = fused[valid_mask].astype(np.float32) / depth_scale

        frame_stats = {
            "frame": rgb_path.stem,
            "valid_pixels": int(np.count_nonzero(valid_mask)),
            "depth_min_m": float(depth_valid.min()
                                 ) if depth_valid.size else None,
            "depth_max_m": float(depth_valid.max()
                                 ) if depth_valid.size else None,
            "depth_mean_m": float(depth_valid.mean()
                                  ) if depth_valid.size else None
        }
        stats_per_frame.append(frame_stats)

        print(f"[✓] Saved fused maps: {png_path.name} | \
              valid pixels: {np.count_nonzero(fused)}")

    stats_path = output_dir / "fusion_statistics.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_per_frame, f, indent=4)
    print(f"[✓] Saved fusion statistics: {stats_path}")


def visualize_depth_maps(
    rgb_path: Path,
    depth_real_path: Path,
    depth_mono_path: Path,
    depth_fused_path: Path,
    depth_scale: float = 10000.0,
    max_display_depth: float = 3.0
) -> None:
    """
    Visualiza mapa de profundidade real, monocular e fundido lado a lado.

    Args:
        rgb_path (Path): Caminho para imagem RGB.
        depth_real_path (Path): .npy com profundidade real (em metros).
        depth_mono_path (Path): .npy com profundidade monocular (em metros).
        depth_fused_path (Path): .npy com profundidade fundida (em metros).
        depth_scale (float): Fator de conversão metros → milímetros.
        max_display_depth (float): Profundidade máxima para visualização.
    """
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2))  # opcional

    def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
        depth_vis = np.clip(depth, 0.0, max_display_depth)
        norm = (depth_vis * 255 / max_display_depth).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    depth_real = np.load(depth_real_path)
    depth_mono = np.load(depth_mono_path)
    depth_fused = np.load(depth_fused_path)

    vis_real = depth_to_colormap(depth_real)
    vis_mono = depth_to_colormap(depth_mono)
    vis_fused = depth_to_colormap(depth_fused)

    # Resize all depth maps to match RGB image shape
    target_shape = (rgb.shape[1], rgb.shape[0])  # width x height
    vis_real = cv2.resize(vis_real, target_shape)
    vis_mono = cv2.resize(vis_mono, target_shape)
    vis_fused = cv2.resize(vis_fused, target_shape)

    top = np.hstack([rgb, vis_real])
    bottom = np.hstack([vis_mono, vis_fused])
    collage = np.vstack([top, bottom])

    cv2.imshow("RGB | Real Depth | Mono Depth | Fused Depth", collage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    """
    Runs fusion for a specific dataset scene.
    """
    scene = "lab_scene_d"
    scale = 100
    trunc = 3.0  # Adjust as needed
    mode = "min"  # Fusion mode: 'min', 'mean', 'real-priority', 'mono-priority'

    rgb_dir = Path(f"datasets/{scene}/rgb")
    depth_real_dir = Path(f"datasets/{scene}/depth_npy")
    depth_mono_dir = Path(f"results/{scene}/d4/depth_npy")
    transform_path = Path(f"results/{scene}/d6/T_d_to_m_frame0000.npy")
    output_dir = Path(
        f"results/{scene}/d8/fused_depth_Tdm_{mode}_{scale}_{trunc:.1f}"
    )
    intrinsics = load_intrinsics(Path(f"datasets/{scene}/intrinsics.json"))

    fuse_dataset_depth_maps(
        rgb_dir=rgb_dir,
        depth_real_dir=depth_real_dir,
        depth_mono_dir=depth_mono_dir,
        transform_path=transform_path,
        output_dir=output_dir,
        intrinsics=intrinsics,
        depth_scale=scale,
        depth_trunc=trunc
    )

    visualize_depth_maps(
        rgb_path=rgb_dir / "frame_0000.png",
        depth_real_path=depth_real_dir / "frame_0000.npy",
        depth_mono_path=depth_mono_dir / "frame_0000.npy",
        depth_fused_path=output_dir / "frame_0000.npy"
    )


if __name__ == "__main__":
    main()
