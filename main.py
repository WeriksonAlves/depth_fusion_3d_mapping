from pathlib import Path
import numpy as np
import open3d as o3d

from modules.utils.realsense_recorder import RealSenseRecorder
from modules.reconstruction.multiway_reconstructor_offline import (
    MultiwayReconstructorOffline
)
from modules.inference.depth_batch_inferencer import DepthBatchInferencer
from modules.utils.point_cloud_comparer import PointCloudComparer
from modules.utils.frame_icp_aligner import FrameICPAlignerBatch
from modules.utils.depth_fusion_processor import DepthFusionProcessor


def visualize_camera_trajectory(
    trajectory_path: Path,
    reconstruction_path: Path = None,
    axis_size: float = 0.1,
    axis_every: int = 1  # mostra eixo em todo ponto (ou a cada N)
) -> None:
    """
    Visualizes the camera trajectory with coordinate axes and optional scene.

    Args:
        trajectory_path (Path): Path to .npy file with camera centers (N, 3).
        reconstruction_path (Path, optional): Path to .ply file with the
            reconstructed scene (optional).
    """
    # Load trajectory points
    trajectory = np.load(trajectory_path)

    # Point cloud of camera centers (red dots)
    traj_pcd = o3d.geometry.PointCloud()
    traj_pcd.points = o3d.utility.Vector3dVector(trajectory)
    traj_pcd.paint_uniform_color([1, 0, 0])  # red

    # Line path (green)
    lines = [[i, i + 1] for i in range(len(trajectory) - 1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(trajectory),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(
        [[0, 1, 0] for _ in lines]  # green
    )

    # Coordinate frames along trajectory
    axes = []
    for i, pos in enumerate(trajectory):
        if i % axis_every == 0:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=axis_size if i != 0 else 1,
                origin=pos.tolist()
            )
            axes.append(axis)

    geometries = [traj_pcd, line_set] + axes

    # Optional scene overlay
    if reconstruction_path and reconstruction_path.exists():
        scene = o3d.io.read_point_cloud(str(reconstruction_path))
        geometries.insert(0, scene)
        scene.rotate(o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0)))

    print("[INFO] Visualizing trajectory with axes...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Camera Trajectory with Axes"
    )


def run_stage_1_capture_and_reconstruct(
    scene: str,
    max_frames: int = 60,
    fps: int = 15,
    voxel_size: float = 0.02,
    recoder_bool: bool = False
) -> None:
    dataset = Path(f"datasets/{scene}")
    rgb_dir = dataset / "rgb"
    depth_dir = dataset / "depth_npy"
    intrinsics = dataset / "intrinsics.json"

    output_dir = Path(f"results/{scene}/step_1")
    output_pcd = output_dir / "reconstruction_sensor.ply"

    print("[INFO] Starting RealSense capture...")
    recorder = RealSenseRecorder(
        output_path=dataset,
        max_frames=max_frames,
        fps=fps
    )
    if recoder_bool:
        recorder.start()
        recorder.capture()

    print("[INFO] Running reconstruction using sensor depth...")
    reconstructor = MultiwayReconstructorOffline(
        rgb_dir, depth_dir, intrinsics, output_dir, output_pcd, voxel_size
    )
    reconstructor.run()


def run_stage_2_monocular_inference_and_reconstruction(
    scene: str,
    voxel_size: float = 0.02
) -> None:
    dataset = Path(f"datasets/{scene}")
    rgb_dir = dataset / "rgb"
    intrinsics = dataset / "intrinsics.json"
    output_dir = Path(f"results/{scene}/step_2")
    checkpoint_dir = Path("checkpoints")
    depth_output = output_dir / "depth_npy"
    output_pcd = output_dir / "reconstruction_est.ply"

    print("[INFO] Running monocular depth inference...")
    inferencer = DepthBatchInferencer(
        rgb_dir=rgb_dir,
        output_path=output_dir,
        encoder="vits",
        checkpoint_dir=checkpoint_dir,
    )
    inferencer.run()

    print("[INFO] Running reconstruction using estimated depth...")
    reconstructor = MultiwayReconstructorOffline(
        rgb_dir=rgb_dir,
        depth_dir=depth_output,
        intrinsics_path=intrinsics,
        output_dir=output_dir,
        output_pcd_path=output_pcd,
        voxel_size=voxel_size
    )
    reconstructor.run()


def run_stage_3_alignment_and_fusion(
    scene: str,
    voxel_size: float = 0.02
) -> None:
    dataset = Path(f"datasets/{scene}")
    results = Path(f"results/{scene}")
    rgb_dir = dataset / "rgb"
    depth_real = dataset / "depth_npy"
    depth_est = results / "step_2/depth_npy"
    intrinsics = dataset / "intrinsics.json"
    output_dir = results / "step_3"
    fused_dir = output_dir / "mean_std_fused"

    print("[INFO] Running ICP alignment and scale estimation...")
    batch = FrameICPAlignerBatch(
        rgb_dir=rgb_dir,
        depth_sensor_dir=depth_real,
        depth_estimation_dir=depth_est,
        intrinsics_path=intrinsics,
        output_dir=output_dir,
        voxel_size=voxel_size,
    )
    scale, _ = batch.estimate_scale_from_depth_maps(
        max_samples=50000,
        min_depth=0.01,
        max_depth=5.0
    )
    # batch.run(1/scale)

    print("[INFO] Fusing depth maps (real and estimated)...")
    fusion = DepthFusionProcessor(
        depth_real_dir=depth_real,
        depth_estimated_dir=depth_est,
        output_dir=fused_dir
    )
    fusion.run(mode=4)

    print("[INFO] Reconstructing from fused depth...")
    reconstructor = MultiwayReconstructorOffline(
        rgb_dir=rgb_dir,
        depth_dir=fused_dir / "npy",
        intrinsics_path=intrinsics,
        output_dir=output_dir / "reconstruction",
        output_pcd_path=output_dir / "reconstruction/reconstruction.ply",
        voxel_size=voxel_size
    )
    reconstructor.run()


def run_compare_sensor_vs_estimated(
    scene: str,
    offset: bool = False
) -> None:
    """
    Visual comparison of point clouds: sensor vs monocular model.
    """
    results = Path(f"results/{scene}")
    path_sensor = results / "step_1/reconstruction_sensor.ply"
    path_est = results / "step_2/reconstruction_est.ply"

    comparer = PointCloudComparer(offset_apply=offset)
    comparer.run([path_est, path_sensor], mode=0)


def visualize_trajectory_and_reconstruction(
    scene: str,
    recosnstruction: bool = True
) -> None:
    """
    Visualizes the camera trajectory and reconstruction point cloud.
    """
    print("[INFO] Visualizing camera trajectory...")
    visualize_camera_trajectory(
        trajectory_path=Path(f"results/{scene}/step_1/camera_trajectory.npy"),
        reconstruction_path=Path(
            f"results/{scene}/step_1/reconstruction_sensor.ply"
            ) if recosnstruction else None
    )

    print("[INFO] Visualizing estimated trajectory...")
    visualize_camera_trajectory(
        trajectory_path=Path(
            f"results/{scene}/step_3/reconstruction/camera_trajectory.npy"),
        reconstruction_path=Path(
            f"results/{scene}/step_3/reconstruction/reconstruction.ply"
            ) if recosnstruction else None
    )


def main() -> None:
    scene = "validate_rs_Is_low"
    recoder_bool = False  # Set to True to record new data
    voxel_size = 0.09  # Adjust voxel size as needed
    stage = 3
    print(f"[✓] Running pipeline for scene: {scene}")
    print(f"[✓] Executing stage: {stage}")

    if stage == 1:
        run_stage_1_capture_and_reconstruct(scene, 100, 4, voxel_size,
                                            recoder_bool)
    elif stage == 2:
        run_stage_2_monocular_inference_and_reconstruction(scene, voxel_size)
    elif stage == 3:
        run_stage_3_alignment_and_fusion(scene, voxel_size)
    elif stage == 4:
        visualize_trajectory_and_reconstruction(scene, True)
    else:
        run_compare_sensor_vs_estimated(scene, offset=False)


if __name__ == "__main__":
    main()
