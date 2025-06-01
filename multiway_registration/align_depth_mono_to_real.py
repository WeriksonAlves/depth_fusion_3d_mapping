import open3d as o3d
import numpy as np
import argparse
import os


def run_icp_with_scaling(source: o3d.geometry.PointCloud,
                         target: o3d.geometry.PointCloud,
                         voxel_size: float = 0.02) -> np.ndarray:
    print("[INFO] Running ICP with scaling...")
    threshold = voxel_size * 1.5

    # Normalize center and orientation
    source.estimate_normals()
    target.estimate_normals()

    result = o3d.pipelines.registration.registration_icp(
        source, target, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()
    )

    print("[OK] Fitness:", result.fitness)
    print("[OK] RMSE:", result.inlier_rmse)
    return result.transformation


def save_transformation_matrix(T: np.ndarray, output_path: str):
    np.save(output_path, T)
    print(f"[✓] Saved transformation matrix to {output_path}")


def apply_transformation(pcd: o3d.geometry.PointCloud, T: np.ndarray) -> o3d.geometry.PointCloud:
    aligned = pcd.transform(T)
    return aligned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', type=str, default='datasets/lab_scene_kinect_xyz/reconstruction_d435.ply')
    parser.add_argument('--mono', type=str, default='datasets/lab_scene_kinect_xyz/reconstruction_depthanything.ply')
    parser.add_argument('--out_cloud', type=str, default='datasets/lab_scene_kinect_xyz/reconstruction_depthanything_aligned.ply')
    parser.add_argument('--out_transform', type=str, default='datasets/lab_scene_kinect_xyz/T_d_to_m.npy')
    args = parser.parse_args()

    print("[INFO] Loading point clouds...")
    pcd_real = o3d.io.read_point_cloud(args.real)
    pcd_mono = o3d.io.read_point_cloud(args.mono)

    # Downsample for registration
    voxel_size = 0.02
    pcd_real_d = pcd_real.voxel_down_sample(voxel_size)
    pcd_mono_d = pcd_mono.voxel_down_sample(voxel_size)

    T = run_icp_with_scaling(pcd_mono_d, pcd_real_d)
    save_transformation_matrix(T, args.out_transform)

    aligned_pcd = apply_transformation(pcd_mono, T)
    o3d.io.write_point_cloud(args.out_cloud, aligned_pcd)
    print(f"[✓] Saved aligned point cloud to {args.out_cloud}")

    # Visual check
    pcd_real.paint_uniform_color([0.0, 0.6, 1.0])
    aligned_pcd.paint_uniform_color([1.0, 0.3, 0.3])
    o3d.visualization.draw_geometries(
        [pcd_real, aligned_pcd],
        window_name='Aligned: Real (blue) vs Mono Aligned (red)',
        width=960,
        height=540
    )


if __name__ == '__main__':
    main()
