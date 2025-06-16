from pathlib import Path

from modules.utils.realsense_recorder import RealSenseRecorder
from modules.reconstruction.multiway_reconstructor_offline import (
    MultiwayReconstructorOffline
)
from modules.inference.depth_batch_inferencer import DepthBatchInferencer
from modules.utils.point_cloud_comparer import PointCloudComparer
from modules.utils.frame_icp_aligner import FrameICPAlignerBatch
from modules.utils.depth_fusion_processor import DepthFusionProcessor


def run_stage_1_capture_and_reconstruct(
    scene: str,
    max_frames: int = 60,
    fps: int = 15,
    voxel_size: float = 0.02
) -> None:
    dataset = Path(f"datasets/{scene}")
    rgb_dir = dataset / "rgb"
    depth_dir = dataset / "depth_npy"
    intrinsics = dataset / "intrinsics.json"

    output_dir = Path(f"results_new/{scene}/step_1")
    output_pcd = output_dir / "reconstruction_sensor.ply"

    print("[INFO] Starting RealSense capture...")
    recorder = RealSenseRecorder(
        output_path=dataset,
        max_frames=max_frames,
        fps=fps
    )
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
    output_dir = Path(f"results_new/{scene}/step_2")
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
    results = Path(f"results_new/{scene}")
    rgb_dir = dataset / "rgb"
    depth_real = dataset / "depth_npy"
    depth_est = results / "step_2/depth_npy"
    intrinsics = dataset / "intrinsics.json"
    output_dir = results / "step_3"
    fused_dir = output_dir / "both"

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
    batch.run(scale)

    print("[INFO] Fusing depth maps (real and estimated)...")
    fusion = DepthFusionProcessor(
        depth_real_dir=depth_real,
        depth_estimated_dir=depth_est,
        output_dir=fused_dir
    )
    fusion.run(0)

    print("[INFO] Reconstructing from fused depth...")
    reconstructor = MultiwayReconstructorOffline(
        rgb_dir=rgb_dir,
        depth_dir=fused_dir / "npy",
        intrinsics_path=intrinsics,
        output_dir=output_dir / "reconstruction",
        output_pcd_path=output_dir / "reconstruction.ply",
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
    results = Path(f"results_new/{scene}")
    path_sensor = results / "step_1/reconstruction_sensor.ply"
    path_est = results / "step_2/reconstruction_est.ply"

    comparer = PointCloudComparer(offset_apply=offset)
    comparer.run([path_est, path_sensor], mode=0)


def main() -> None:
    scene = "lab_scene_kinect_xyz"
    print(f"[✓] Running pipeline for scene: {scene}")

    # run_stage_1_capture_and_reconstruct(scene, max_frames=60, fps=15,
    #                                     voxel_size=0.02)
    run_stage_2_monocular_inference_and_reconstruction(scene, voxel_size=0.05)
    # run_stage_3_alignment_and_fusion(scene, voxel_size=0.01)
    # run_compare_sensor_vs_estimated(scene, offset=False)


if __name__ == "__main__":
    main()

    # modules/reconstruction/rgbd_loader.py # Read .ply
