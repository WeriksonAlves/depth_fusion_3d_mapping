import open3d as o3d
import argparse
import numpy as np
import os


def load_and_colorize(path: str, color: list) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    pcd.paint_uniform_color(color)
    return pcd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real', type=str, default='datasets/lab_scene_kinect_xyz/reconstruction_d435.ply',
                        help='Path to point cloud from real depth sensor.')
    parser.add_argument('--mono', type=str, default='datasets/lab_scene_kinect_xyz/reconstruction_depthanything.ply',
                        help='Path to point cloud from DepthAnythingV2.')
    parser.add_argument('--scale_mono', type=float, default=1.0,
                        help='Optional manual scaling factor for mono reconstruction.')
    args = parser.parse_args()

    print("[INFO] Loading point clouds...")
    pcd_real = load_and_colorize(args.real, [0.0, 0.6, 1.0])    # azul
    pcd_mono = load_and_colorize(args.mono, [1.0, 0.3, 0.3])    # vermelho

    if args.scale_mono != 1.0:
        print(f"[INFO] Scaling mono depth by factor {args.scale_mono}")
        pcd_mono.scale(args.scale_mono, center=pcd_mono.get_center())

    print("[INFO] Visualizing...")
    o3d.visualization.draw_geometries([pcd_real, pcd_mono],
        window_name='Depth Comparison — Real (blue) vs Monocular (red)',
        width=960, height=540,
        point_show_normal=False)

if __name__ == '__main__':
    main()
