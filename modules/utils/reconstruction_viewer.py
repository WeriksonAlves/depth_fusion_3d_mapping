import open3d as o3d
from pathlib import Path


def visualize_open3d(pcd_path: Path | str) -> None:
    """
    Loads and visualizes a point cloud using Open3D.

    Args:
        pcd_path (str or Path): Path to the .ply file.
    """
    print(f"[INFO] Loading: {pcd_path}")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    print(f"[INFO] Total points: {len(pcd.points)}")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name='Open3D Viewer',
        width=960,
        height=540
    )
