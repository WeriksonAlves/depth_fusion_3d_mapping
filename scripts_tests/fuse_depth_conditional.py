from pathlib import Path
import numpy as np
import cv2
import open3d as o3d
import json


def load_intrinsics(json_path: Path) -> o3d.camera.PinholeCameraIntrinsic:
    with open(json_path, "r") as f:
        data = json.load(f)
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(
        width=data["width"],
        height=data["height"],
        fx=data["K"][0],
        fy=data["K"][4],
        cx=data["K"][2],
        cy=data["K"][5]
    )
    return intr


def project_point_cloud_to_depth(
    pcd: o3d.geometry.PointCloud,
    intr: o3d.camera.PinholeCameraIntrinsic,
    depth_scale: float = 5000.0,
    depth_trunc: float = 4.0
) -> np.ndarray:
    """
    Projects a 3D point cloud to a depth image using z-buffering.

    Args:
        pcd (o3d.geometry.PointCloud): Transformed point cloud.
        intr (o3d.camera.PinholeCameraIntrinsic): Camera intrinsics.
        depth_scale (float): Scale for converting depth to uint16.
        depth_trunc (float): Max depth to keep (in meters).

    Returns:
        np.ndarray: 2D depth image in uint16 format.
    """
    width, height = intr.width, intr.height
    fx, fy = intr.get_focal_length()
    cx, cy = intr.get_principal_point()

    depth_image = np.zeros((height, width), dtype=np.float32)
    z_buffer = np.full((height, width), np.inf)

    points = np.asarray(pcd.points)
    for x, y, z in points:
        if z <= 0 or z > depth_trunc:
            continue
        u = int(round((x * fx) / z + cx))
        v = int(round((y * fy) / z + cy))
        if 0 <= u < width and 0 <= v < height:
            if z < z_buffer[v, u]:
                z_buffer[v, u] = z
                depth_image[v, u] = z

    return (depth_image * depth_scale).astype(np.uint16)


def fuse_pixelwise(real: np.ndarray, mono: np.ndarray) -> np.ndarray:
    """
    Applies conditional fusion:
    - If real == 0 → use mono
    - If mono == 0 → use real
    - Else         → use min(real, mono)
    """
    fused = np.where(real == 0, mono, real)
    fused = np.where((real > 0) & (mono > 0), np.minimum(real, mono), fused)
    return fused


def fuse_maps_conditional(
    rgb_dir, depth_real_dir, depth_mono_dir, T_path, output_dir,
    intrinsics, depth_scale=5000.0
):
    output_dir.mkdir(exist_ok=True, parents=True)
    T = np.load(T_path)

    rgb_files = sorted(rgb_dir.glob("*.png"))
    real_files = sorted(depth_real_dir.glob("*.png"))
    mono_files = sorted(depth_mono_dir.glob("*.npy"))

    assert len(rgb_files) == len(real_files) == len(mono_files), "[ERROR] Frame count mismatch."

    for i, (rgb_path, real_path, mono_path) in enumerate(zip(rgb_files, real_files, mono_files)):
        print(f"[INFO] Processing frame: {rgb_path.name}")

        rgb = o3d.io.read_image(str(rgb_path))
        depth_real = o3d.io.read_image(str(real_path))
        depth_mono = np.load(mono_path).astype(np.float32)

        # Point cloud from mono, transformed
        depth_mono_img = o3d.geometry.Image((depth_mono * depth_scale).astype(np.uint16))
        rgbd_mono = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_mono_img, depth_scale=depth_scale, convert_rgb_to_intensity=False
        )
        pcd_mono = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_mono, intrinsics)
        pcd_mono.transform(T)

        # Project mono cloud back to depth map
        depth_mono_proj = project_point_cloud_to_depth(pcd_mono, intrinsics, depth_scale)

        real_np = np.asarray(depth_real).astype(np.uint16)
        fused = fuse_pixelwise(real_np, depth_mono_proj)

        out_path = output_dir / f"{rgb_path.stem}.png"
        cv2.imwrite(str(out_path), fused)
        print(f"[✓] Fused depth saved: {out_path}")


def main():
    base = Path("datasets/lab_scene_kinect_xyz")
    fuse_maps_conditional(
        rgb_dir=base / "rgb",
        depth_real_dir=base / "depth",
        depth_mono_dir=base / "depth_mono",
        T_path=Path("results/lab_scene_kinect_xyz/T_d_to_m.npy"),
        output_dir=base / "depth_fused_refined",
        intrinsics=load_intrinsics(base / "intrinsics.json")
    )


if __name__ == "__main__":
    main()
