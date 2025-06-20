"""
Main execution script for the 3D reconstruction pipeline.

Supports the following stages:
1. RGB-D data capture and reconstruction from RealSense depth
2. Depth estimation from monocular images and reconstruction
3. Frame-wise ICP alignment, depth fusion, and final reconstruction
4. Visualization of camera trajectories and comparisons
"""

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
    axis_every: int = 1
) -> None:
    """
    Visualizes camera poses and optional 3D reconstruction.

    Args:
        trajectory_path (Path): Path to .npy file with camera centers (N, 3).
        reconstruction_path (Path, optional): Path to .ply file with the
            reconstructed scene (optional).
    """
    trajectory = np.load(trajectory_path)
    traj_pcd = o3d.geometry.PointCloud()
    traj_pcd.points = o3d.utility.Vector3dVector(trajectory)
    traj_pcd.paint_uniform_color([1, 0, 0])

    lines = [[i, i + 1] for i in range(len(trajectory) - 1)]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(trajectory),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(lines))

    axes = []
    for i, pos in enumerate(trajectory):
        if i % axis_every == 0:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=axis_size if i != 0 else 1.0,
                origin=pos.tolist()
            )
            axes.append(frame)

    geometries = [traj_pcd, line_set] + axes
    if reconstruction_path and reconstruction_path.exists():
        scene = o3d.io.read_point_cloud(str(reconstruction_path))
        geometries.insert(0, scene)

    print("[INFO] Displaying trajectory...")
    o3d.visualization.draw_geometries(geometries)


def run_stage_1_capture_and_reconstruct(scene: str, max_frames: int = 60,
                                        fps: int = 15, record: bool = False,
                                        voxel_size: float = 0.02) -> None:
    """Captures RGB-D data and reconstructs using RealSense depth."""
    base = Path(f"datasets/{scene}")
    rgb_dir, depth_dir = base / "rgb", base / "depth_npy"
    intrinsics = base / "intrinsics.json"
    output = Path(f"results/{scene}/step_1")
    output_pcd = output / "reconstruction_sensor.ply"

    if record:
        recorder = RealSenseRecorder(output_path=base,
                                     max_frames=max_frames,
                                     fps=fps)
        recorder.start()
        recorder.capture()

    reconstructor = MultiwayReconstructorOffline(rgb_dir, depth_dir,
                                                 intrinsics, output,
                                                 output_pcd, voxel_size)
    reconstructor.run()


def run_stage_2_monocular_inference_and_reconstruction(
    scene: str, voxel_size: float = 0.02
) -> None:
    """Estimates monocular depth and reconstructs 3D scene."""
    base = Path(f"datasets/{scene}")
    rgb_dir, intrinsics = base / "rgb", base / "intrinsics.json"
    output = Path(f"results/{scene}/step_2")
    depth_output = output / "depth_npy"
    checkpoint = Path("checkpoints")
    output_pcd = output / "reconstruction_est.ply"

    inferencer = DepthBatchInferencer(rgb_dir, output, "vits", checkpoint)
    inferencer.run()

    reconstructor = MultiwayReconstructorOffline(rgb_dir, depth_output,
                                                 intrinsics, output,
                                                 output_pcd, voxel_size)
    reconstructor.run()


def run_stage_3_alignment_and_fusion(scene: str,
                                     voxel_size: float = 0.02) -> None:
    """Aligns monocular and sensor depth, fuses maps, and reconstructs."""
    base = Path(f"datasets/{scene}")
    results = Path(f"results/{scene}")
    rgb, d_real = base / "rgb", base / "depth_npy"
    d_est = results / "step_2/depth_npy"
    intrinsics = base / "intrinsics.json"
    step3_dir = results / "step_3"
    fused_dir = step3_dir / "mean_std_fused"

    batch = FrameICPAlignerBatch(rgb, d_real, d_est, intrinsics,
                                 step3_dir, voxel_size)
    scale, _ = batch.estimate_scale_from_depth_maps(50000, 0.01, 5.0)
    # batch.run(scale)

    fusion = DepthFusionProcessor(d_real, d_est, fused_dir)
    fusion.run(mode=4)

    reconstructor = MultiwayReconstructorOffline(
        rgb, fused_dir / "npy", intrinsics,
        step3_dir / "reconstruction",
        step3_dir / "reconstruction/reconstruction.ply",
        voxel_size
    )
    reconstructor.run()


def run_compare_sensor_vs_estimated(scene: str, offset: bool = False) -> None:
    """Compares point clouds from sensor and estimated depth."""
    base = Path(f"results/{scene}")
    pc1 = base / "step_1/reconstruction_sensor.ply"
    pc2 = base / "step_2/reconstruction_est.ply"

    comparer = PointCloudComparer(offset_apply=offset)
    comparer.run([pc2, pc1], mode=0)


def visualize_trajectory_and_reconstruction(scene: str,
                                            show_reconstruction: bool = True
                                            ) -> None:
    """Visualizes both original and estimated camera trajectories."""
    print("[INFO] Visualizing sensor trajectory...")
    visualize_camera_trajectory(
        trajectory_path=Path(f"results/{scene}/step_1/camera_trajectory.npy"),
        reconstruction_path=Path(
            f"results/{scene}/step_1/reconstruction_sensor.ply"
        ) if show_reconstruction else None
    )

    print("[INFO] Visualizing estimated trajectory...")
    visualize_camera_trajectory(
        trajectory_path=Path(
            f"results/{scene}/step_3/reconstruction/camera_trajectory.npy"
        ),
        reconstruction_path=Path(
            f"results/{scene}/step_3/reconstruction/reconstruction.ply"
        ) if show_reconstruction else None
    )


def main() -> None:
    scene = "validate_rs_Il_low_0.09"
    record = False
    voxel_size = 0.09
    stage = 4

    print(f"[✓] Running pipeline for: {scene} | Stage: {stage}")

    if stage == 1:
        run_stage_1_capture_and_reconstruct(scene, 100, 4, voxel_size, record)
    elif stage == 2:
        run_stage_2_monocular_inference_and_reconstruction(scene, voxel_size)
    elif stage == 3:
        run_stage_3_alignment_and_fusion(scene, voxel_size)
    elif stage == 4:
        visualize_trajectory_and_reconstruction(scene)
    else:
        run_compare_sensor_vs_estimated(scene, offset=False)


if __name__ == "__main__":
    main()
