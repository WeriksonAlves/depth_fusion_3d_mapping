from pathlib import Path

from modules.utils.realsense_recorder import RealSenseRecorder
from modules.reconstruction.multiway_reconstructor_offline import (
    MultiwayReconstructorOffline
)
from modules.inference.depth_batch_inferencer import DepthBatchInferencer
from modules.utils.point_cloud_comparer import PointCloudComparer
from modules.align.frame_icp_aligner import FrameICPAlignerBatch
from modules.fusion.depth_fusion_processor import DepthFusionProcessor


def stage1(scene: str, max_frames=60, fps=15, voxel_size=0.02) -> None:
    dataset_base = Path(f"datasets/{scene}")
    rgb_dir = dataset_base / "rgb"
    depth_dir = dataset_base / "depth_npy"
    intrinsics_path = dataset_base / "intrinsics.json"

    output_dir = Path(f"results_new/{scene}/step_1")
    output_pcd_path = output_dir / "reconstruction_sensor.ply"

    # recorder = RealSenseRecorder(
    #     output_path=dataset_base,
    #     max_frames=max_frames,
    #     fps=fps
    # )
    # recorder.start()
    # recorder.capture()

    reconstructor = MultiwayReconstructorOffline(
        rgb_dir, depth_dir, intrinsics_path, output_dir,
        output_pcd_path, voxel_size)
    reconstructor.run()


def stage2(scene: str, voxel_size=0.02) -> None:
    dataset_base = Path(f"datasets/{scene}")
    rgb_dir = dataset_base / "rgb"
    output_dir = Path(f"results_new/{scene}/step_2")
    checkpoint_dir = Path("checkpoints")

    inferencer = DepthBatchInferencer(
        rgb_dir=rgb_dir,
        output_path=output_dir,
        encoder="vits",
        checkpoint_dir=checkpoint_dir,
    )
    inferencer.run()

    depth_dir = output_dir / "depth_npy"
    intrinsics_path = dataset_base / "intrinsics.json"
    output_pcd_path = output_dir / "reconstruction_est.ply"

    reconstructor = MultiwayReconstructorOffline(
        rgb_dir=rgb_dir,
        depth_dir=depth_dir,
        intrinsics_path=intrinsics_path,
        output_dir=output_dir,
        output_pcd_path=output_pcd_path,
        voxel_size=voxel_size
    )
    reconstructor.run()


def stage3(scene: str, voxel_size=0.02) -> None:
    dataset_base = Path(f"datasets/{scene}")
    results_base = Path(f"results_new/{scene}")

    rgb_dir = dataset_base / "rgb"
    depth_sensor_dir = dataset_base / "depth_npy"
    depth_estimation_dir = results_base / "step_2/depth_npy"
    intrinsics_path = dataset_base / "intrinsics.json"
    output_dir = results_base / "step_3"

    batch = FrameICPAlignerBatch(
        rgb_dir=rgb_dir,
        depth_sensor_dir=depth_sensor_dir,
        depth_estimation_dir=depth_estimation_dir,
        intrinsics_path=intrinsics_path,
        output_dir=output_dir,
        voxel_size=voxel_size,
    )
    scale, _ = batch.estimate_scale_from_depth_maps(
        max_samples=50000,
        min_depth=0.01,
        max_depth=5.0
    )
    batch.run(scale+0.1)

    processor = DepthFusionProcessor(
        depth_real_dir=depth_sensor_dir,
        depth_estimated_dir=depth_estimation_dir,
        output_dir=output_dir / "both",
    )
    processor.run(0)


def run_compare_d5(
    scene: str,
    offset_apply: bool = False
) -> None:
    """
    Compares point clouds from sensor vs monocular estimation.
    """
    path_d435 = Path(f"results_new/{scene}/d3/reconstruction_sensor.ply")
    path_mono = Path(f"results_new/{scene}/d5/reconstruction_est.ply")

    comparer = PointCloudComparer(offset_apply=offset_apply)
    comparer.run([path_mono, path_d435], mode=0)


def run_fused_reconstruction_d9(
    scene: str,
    scale: int = 100,
    trunc: float = 3.0,
    mode: str = "min",
    voxel_size: float = 0.02
) -> None:
    """
    Reconstructs from fused depth maps.
    """
    results = Path(f"results_new/{scene}")
    output = results / "d8/fused_depth"

    reconstructor = MultiwayReconstructorOffline(
        rgb_dir=Path(f"datasets/{scene}/rgb"),
        depth_dir=output,
        intrinsics_path=Path(f"datasets/{scene}/intrinsics.json"),
        output_dir=Path(f"results_new/{scene}/d9_2"),
        output_pcd_path=Path(f"results_new/{scene}/d9/reconstruction_fused.ply"),
        voxel_size=voxel_size
    )
    reconstructor.run()


def run_pipeline_sequence() -> None:
    """
    Executes a full pipeline step-by-step.
    """
    scene = "lab_scene_r"
    voxel_size = 0.02

    print(f"Running pipeline for scene: {scene}")

    # Stage 1: Capture and reconstruct using RealSense depth
    # stage1(scene, max_frames=60, fps=15, voxel_size=voxel_size)

    # # Stage 2: Infer depth using monocular model and reconstruct
    # stage2(scene, voxel_size=voxel_size)

    # # Stage 3: Align frames using ICP and fuse depth maps
    stage3(scene, voxel_size=voxel_size)

    # modules/reconstruction/rgbd_loader.py # Read .ply
    # run_fused_reconstruction_d9(
    #     scene,
    #     voxel_size
    # )
    # run_compare_d5(
    #     scene,
    #     offset_apply=False
    # )


if __name__ == "__main__":
    run_pipeline_sequence()
