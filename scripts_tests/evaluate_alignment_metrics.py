from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def compute_rmse(source: o3d.geometry.PointCloud,
                 target: o3d.geometry.PointCloud,
                 max_points: int = 100_000) -> float:
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)

    # Subsampling
    if source_points.shape[0] > max_points:
        idx = np.random.choice(source_points.shape[0], size=max_points, replace=False)
        source_points = source_points[idx]
    if target_points.shape[0] > max_points:
        idx = np.random.choice(target_points.shape[0], size=max_points, replace=False)
        target_points = target_points[idx]

    print(f"[INFO] Subsampled: {len(source_points)} source, {len(target_points)} target points")

    tree = cKDTree(target_points)
    distances, _ = tree.query(source_points)
    return np.sqrt(np.mean(distances ** 2))


def compute_hausdorff(source: o3d.geometry.PointCloud,
                      target: o3d.geometry.PointCloud,
                      max_points: int = 100_000) -> float:
    src_pts = np.asarray(source.points)
    tgt_pts = np.asarray(target.points)

    if src_pts.shape[0] > max_points:
        idx = np.random.choice(src_pts.shape[0], size=max_points, replace=False)
        src_pts = src_pts[idx]
    if tgt_pts.shape[0] > max_points:
        idx = np.random.choice(tgt_pts.shape[0], size=max_points, replace=False)
        tgt_pts = tgt_pts[idx]

    dist_src_to_tgt = cKDTree(tgt_pts).query(src_pts)[0]
    dist_tgt_to_src = cKDTree(src_pts).query(tgt_pts)[0]
    return max(np.max(dist_src_to_tgt), np.max(dist_tgt_to_src))


def print_and_export_metrics(real_pcd, aligned_pcd, output_path):
    print("[INFO] Computing RMSE...")
    rmse = compute_rmse(aligned_pcd, real_pcd)

    print("[INFO] Computing Hausdorff distance...")
    hausdorff = compute_hausdorff(aligned_pcd, real_pcd)

    num_points_mono = np.asarray(aligned_pcd.points).shape[0]
    num_points_real = np.asarray(real_pcd.points).shape[0]

    volume_mono = aligned_pcd.get_axis_aligned_bounding_box().volume()
    volume_real = real_pcd.get_axis_aligned_bounding_box().volume()

    print("\n=== Evaluation Metrics ===")
    print(f"RMSE:             {rmse:.4f}")
    print(f"Hausdorff dist.:  {hausdorff:.4f}")
    print(f"# Points (Real):  {num_points_real}")
    print(f"# Points (Mono):  {num_points_mono}")
    print(f"Volume (Real):    {volume_real:.4f} m³")
    print(f"Volume (Mono):    {volume_mono:.4f} m³")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("# Evaluation Report\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"Hausdorff Distance: {hausdorff:.4f}\n")
        f.write(f"Points (Real): {num_points_real}\n")
        f.write(f"Points (Mono): {num_points_mono}\n")
        f.write(f"Volume (Real): {volume_real:.4f} m³\n")
        f.write(f"Volume (Mono): {volume_mono:.4f} m³\n")

    print(f"[✓] Report saved to {output_path}")


def main():
    real_path = Path("datasets/lab_scene_kinect_xyz/reconstruction_d435.ply")
    aligned_path = Path("datasets/lab_scene_kinect_xyz/reconstruction_depthanything_aligned.ply")
    report_path = Path("results/lab_scene_kinect_xyz/evaluation_report.txt")

    print("[INFO] Loading point clouds...")
    real_pcd = o3d.io.read_point_cloud(str(real_path))
    aligned_pcd = o3d.io.read_point_cloud(str(aligned_path))

    print("[INFO] Applying voxel downsampling...")
    real_pcd = real_pcd.voxel_down_sample(0.01)
    aligned_pcd = aligned_pcd.voxel_down_sample(0.01)

    print_and_export_metrics(real_pcd, aligned_pcd, report_path)


if __name__ == "__main__":
    main()
