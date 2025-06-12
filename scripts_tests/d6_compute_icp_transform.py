# scripts_test/align_icp_frame_pair.py

"""
ICP-based alignment between monocular and real depth point clouds.

Loads RGB image and two depth maps (.npy), computes corresponding point clouds,
and performs ICP to estimate transformation between them.
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d


def load_intrinsics(json_path: Path) -> o3d.camera.PinholeCameraIntrinsic:
    """
    Loads camera intrinsics from JSON into Open3D PinholeCameraIntrinsic.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fx, fy = data["K"][0], data["K"][4]
    cx, cy = data["K"][2], data["K"][5]
    width, height = data["width"], data["height"]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def create_point_cloud(
    rgb_path: Path,
    depth_path: Path,
    intr: o3d.camera.PinholeCameraIntrinsic,
    depth_scale: float = 1000.0
) -> o3d.geometry.PointCloud:
    """
    Generates an Open3D point cloud from RGB image and depth map in .npy
    format.
    """
    rgb = o3d.io.read_image(str(rgb_path))
    depth_np = np.load(depth_path).astype(np.float32)
    depth_mm = (depth_np * depth_scale).astype(np.uint16)
    depth = o3d.geometry.Image(depth_mm)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, depth_scale=depth_scale, convert_rgb_to_intensity=False
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)


def compute_icp_alignment(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float = 0.02
) -> tuple[np.ndarray, o3d.pipelines.registration.RegistrationResult]:
    """
    Aligns source to target using ICP (point-to-plane) and returns
    transformation matrix.
    """
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)

    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30)
    )
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30)
    )

    icp_result = o3d.pipelines.registration.registration_icp(
        source=source_down,
        target=target_down,
        max_correspondence_distance=voxel_size * 2.5,
        init=np.eye(4),
        estimation_method=(
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
    )

    print(f"[✓] ICP Fitness: {icp_result.fitness:.4f}")
    print(f"[✓] ICP RMSE: {icp_result.inlier_rmse:.4f}")
    return icp_result.transformation, icp_result


def visualize_alignment(source, target, transform):
    source_temp = source.transform(transform.copy())
    o3d.visualization.draw_geometries(
        [target.paint_uniform_color([1, 0.706, 0]),
         source_temp.paint_uniform_color([0, 0.651, 0.929])],
        window_name="ICP Alignment: Mono (blue) → Real (yellow)"
    )


def run_icp_alignment_for_frame(
    scene: str,
    frame_index: int,
    voxel_size: float = 0.02,
    depth_scale: float = 1000.0
) -> None:
    """
    Loads a single RGB frame and two depth maps (mono vs real), runs ICP,
    and saves the transformation matrix to disk.
    """
    dataset_dir = Path(f"datasets/{scene}")
    results_dir = Path(f"comparation/results_test/{scene}")
    frame_name = f"frame_{frame_index:04d}"

    rgb_path = dataset_dir / "rgb" / f"{frame_name}.png"
    real_depth_path = dataset_dir / "depth_npy" / f"{frame_name}.npy"
    mono_depth_path = results_dir / "d4" / "depth_npy" / f"{frame_name}.npy"
    intr_path = dataset_dir / "intrinsics.json"

    for path in [rgb_path, real_depth_path, mono_depth_path, intr_path]:
        assert path.exists(), f"Missing file: {path}"

    intrinsics = load_intrinsics(intr_path)

    print("[INFO] Creating point clouds from depth maps...")
    pcd_real = create_point_cloud(rgb_path, real_depth_path,
                                  intrinsics, depth_scale)
    pcd_mono = create_point_cloud(rgb_path, mono_depth_path,
                                  intrinsics, depth_scale)

    print("[INFO] Running ICP to align monocular to real depth...")
    transformation, icp_result = compute_icp_alignment(
        pcd_mono, pcd_real, voxel_size
    )

    output_dir = results_dir / "d6"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "T_d_to_m_frame0000.npy"
    # transformation = np.linalg.inv(transformation)  # T_m_from_d
    np.save(output_file, transformation)
    print(f"[✓] Transformation saved to: {output_file}")

    # Visualização
    visualize_alignment(pcd_mono, pcd_real, transformation)

    # Salva métricas
    icp_metrics = {
        "fitness": float(icp_result.fitness),
        "inlier_rmse": float(icp_result.inlier_rmse),
        "voxel_size": voxel_size,
        "num_points_source": len(pcd_mono.points),
        "num_points_target": len(pcd_real.points)
    }
    with open(output_dir / "icp_metrics.json", "w", encoding="utf-8") as f:
        json.dump(icp_metrics, f, indent=4)
    print(f"[✓] ICP metrics saved to: {output_dir/'icp_metrics.json'}")


def main() -> None:
    """
    Example: aligns frame_0000 from scene 'lab_scene_f'.
    """
    run_icp_alignment_for_frame(scene="lab_scene_d", frame_index=0)


if __name__ == "__main__":
    main()
