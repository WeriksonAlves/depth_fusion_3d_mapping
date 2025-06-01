import os
import json
import open3d as o3d
import numpy as np
from glob import glob
from tqdm import tqdm

DATASET_DIR = 'datasets/lab_scene_kinect_xyz'
OUTPUT_PATH = os.path.join(DATASET_DIR, 'reconstruction_d435.ply')

RGB_DIR = os.path.join(DATASET_DIR, 'rgb')
DEPTH_DIR = os.path.join(DATASET_DIR, 'depth')
INTRINSICS_FILE = os.path.join(DATASET_DIR, 'intrinsics.json')

VOXEL_SIZE = 0.02  # in meters


def load_camera_intrinsics(path):
    with open(path) as f:
        data = json.load(f)
    K = data["K"]
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(data["width"], data["height"], K[0], K[4], K[2], K[5])
    return intrinsic


def preprocess_frame(rgb_path, depth_path, intrinsic):
    rgb = o3d.io.read_image(rgb_path)
    depth = o3d.io.read_image(depth_path)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, depth_scale=5000.0, depth_trunc=4.0, convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    return pcd


def register_pairwise(source, target):
    return o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance=VOXEL_SIZE * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())


def build_pose_graph(pcds):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odom = np.eye(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odom))

    for i in range(1, len(pcds)):
        icp_result = register_pairwise(pcds[i - 1], pcds[i])
        odom = np.dot(odom, icp_result.transformation)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odom)))
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i - 1, i, icp_result.transformation, 
            uncertain=False))

    return pose_graph


def apply_global_optimization(pose_graph, pcds):
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=VOXEL_SIZE * 1.5,
        edge_prune_threshold=0.25,
        reference_node=0)
    
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)

    for pcd, node in zip(pcds, pose_graph.nodes):
        pcd.transform(node.pose)

    return pcds


def main():
    intrinsic = load_camera_intrinsics(INTRINSICS_FILE)
    rgb_files = sorted(glob(os.path.join(RGB_DIR, '*.png')))
    depth_files = sorted(glob(os.path.join(DEPTH_DIR, '*.png')))

    assert len(rgb_files) == len(depth_files), "Mismatch in RGB and depth count."

    print(f"[INFO] Loading {len(rgb_files)} frames...")

    pcds = []
    for rgb_path, depth_path in tqdm(zip(rgb_files, depth_files), total=len(rgb_files)):
        pcd = preprocess_frame(rgb_path, depth_path, intrinsic)
        pcds.append(pcd)

    print("[INFO] Registering all pairs...")
    pose_graph = build_pose_graph(pcds)

    print("[INFO] Applying global optimization...")
    pcds = apply_global_optimization(pose_graph, pcds)

    print(f"[INFO] Merging and saving as {OUTPUT_PATH}")
    full_map = pcds[0]
    for pcd in pcds[1:]:
        full_map += pcd
    full_map = full_map.voxel_down_sample(voxel_size=VOXEL_SIZE / 2)

    o3d.io.write_point_cloud(OUTPUT_PATH, full_map)

    # Métricas básicas
    print(f"[OK] Total points: {np.asarray(full_map.points).shape[0]}")
    bbox = full_map.get_axis_aligned_bounding_box()
    print(f"Bounding Box Volume: {bbox.volume():.4f} m³")


if __name__ == "__main__":
    main()
