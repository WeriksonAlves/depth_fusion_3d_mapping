import json
import numpy as np
import open3d as o3d
from pathlib import Path


def load_intrinsics(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    fx, fy = data["K"][0], data["K"][4]
    cx, cy = data["K"][2], data["K"][5]
    width, height = data["width"], data["height"]
    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def load_rgbd_images(
    rgb_dir, depth_dir, intrinsic, depth_scale=1.0, voxel_size=0.02
):
    rgb_paths = sorted(Path(rgb_dir).glob("*.png"))
    depth_paths = sorted(Path(depth_dir).glob("*.npy"))

    pcds = []
    for rgb_path, depth_path in zip(rgb_paths, depth_paths):
        color = o3d.io.read_image(str(rgb_path))
        depth_np = np.load(depth_path)
        depth_o3d = o3d.geometry.Image((depth_np * 1).astype(np.uint16))  # em mm
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth_o3d, depth_scale=1000.0, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2.0, max_nn=30
            )
        )
        pcds.append(pcd_down)

    return pcds


def pairwise_registration(source, target, max_dist_coarse, max_dist_fine):
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_dist_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_dist_fine, icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    transformation = icp_fine.transformation
    info = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_dist_fine, icp_fine.transformation
    )
    return transformation, info


def build_pose_graph(pcds, max_dist_coarse, max_dist_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    for src_id in range(len(pcds)):
        for tgt_id in range(src_id + 1, len(pcds)):
            transformation, info = pairwise_registration(
                pcds[src_id], pcds[tgt_id], max_dist_coarse, max_dist_fine
            )
            if tgt_id == src_id + 1:
                odometry = transformation @ odometry
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry))
                )
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        src_id, tgt_id, transformation, info, uncertain=False
                    )
                )
            else:
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        src_id, tgt_id, transformation, info, uncertain=True
                    )
                )
    return pose_graph


def run_multiway_registration(
    dataset_path=Path("datasets/lab_scene"),
    output_path=Path("results/lab_scene"),
    voxel_size=0.02
) -> None:
    rgb_dir = dataset_path / "rgb"
    depth_dir = dataset_path / "depth_npy"
    intr_dir = dataset_path / "intrinsics.json"
    output_dir = output_path / "reconstruction_sensor.ply"

    intrinsic = load_intrinsics(intr_dir)
    pcds = load_rgbd_images(
        rgb_dir, depth_dir, intrinsic, voxel_size=voxel_size
    )
    o3d.visualization.draw(pcds)

    print("[INFO] Building pose graph...")
    max_dist_coarse = voxel_size * 15
    max_dist_fine = voxel_size * 1.5
    pose_graph = build_pose_graph(pcds, max_dist_coarse, max_dist_fine)

    print("[INFO] Optimizing pose graph...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_dist_fine,
        edge_prune_threshold=0.25,
        reference_node=0
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option
    )

    print("[INFO] Applying transformations and combining point clouds...")
    for i in range(len(pcds)):
        pcds[i].transform(pose_graph.nodes[i].pose)

    pcd_combined = o3d.geometry.PointCloud()
    for pcd in pcds:
        pcd_combined += pcd

    print(f"[✓] Saving final point cloud to: {output_dir}")
    o3d.io.write_point_cloud(str(output_dir), pcd_combined)
    o3d.visualization.draw([pcd_combined])


if __name__ == "__main__":
    run_multiway_registration(
        dataset_path=Path("datasets/lab_scene_f"),
        output_path=Path("results/lab_scene_f"),
        voxel_size=0.02
    )
