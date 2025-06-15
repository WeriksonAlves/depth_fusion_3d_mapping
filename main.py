from pathlib import Path

from modules.utils.realsense_recorder import RealSenseRecorder
from modules.reconstruction.multiway_reconstructor_offline import (
    MultiwayReconstructorOffline
)
from modules.inference.depth_batch_inferencer import DepthBatchInferencer
from modules.utils.point_cloud_comparer import PointCloudComparer
from modules.align.frame_icp_aligner import FrameICPAlignerBatch
from modules.fusion.depth_fusion_processor import DepthFusionProcessor


def run_capture_d12(
    output_path,
    max_frames=60,
    fps=15
) -> None:
    """
    Captures frames using RealSense and saves RGB, depth, and intrinsics.
    """
    recorder = RealSenseRecorder(
        output_path=output_path,
        max_frames=max_frames,
        fps=fps
    )
    recorder.start()
    recorder.capture()


def run_reconstruction_d3(
    scene: str,
    voxel_size=0.02
) -> None:
    """
    Runs reconstruction using RealSense depth.
    """
    reconstructor = MultiwayReconstructorOffline(
        rgb_dir=Path(f"datasets/{scene}/rgb"),
        depth_dir=Path(f"datasets/{scene}/depth_npy"),
        intrinsics_path=Path(f"datasets/{scene}/intrinsics.json"),
        output_dir=Path(f"results_new/{scene}/d3"),
        output_pcd_path=Path(f"results_new/{scene}/d3/reconstruction_sensor.ply"),
        voxel_size=voxel_size
    )
    reconstructor.run()


def run_monodepth_d4(scene: str) -> None:
    """
    Performs monocular depth inference using DepthAnythingV2.
    """

    input_dir = Path(f"datasets/{scene}/rgb")
    output_dir = Path(f"results_new/{scene}/d4")
    checkpoint_dir = Path("checkpoints")

    inferencer = DepthBatchInferencer(
        rgb_dir=input_dir,
        output_path=output_dir,
        encoder="vits",
        checkpoint_dir=checkpoint_dir,
    )
    inferencer.run()


def run_reconstruction_d5(
    scene: str,
    voxel_size=0.02,
    scale_correction=1.0
) -> None:
    """
    Runs reconstruction using monocular depth estimates.
    """

    reconstructor = MultiwayReconstructorOffline(
        rgb_dir=Path(f"datasets/{scene}/rgb"),
        depth_dir=Path(f"results_new/{scene}/d4/depth_npy"),
        intrinsics_path=Path(f"datasets/{scene}/intrinsics.json"),
        output_dir=Path(f"results_new/{scene}/d5"),
        output_pcd_path=Path(f"results_new/{scene}/d5/reconstruction_est.ply"),
        voxel_size=voxel_size,
        scale_correction=scale_correction
    )
    reconstructor.run()


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


def run_batch_alignment_d6(scene: str,
                           voxel_size: float = 0.02,
                           depth_scale=1000.0) -> None:
    """
    Runs ICP alignment in batch for multiple frames.
    """
    batch = FrameICPAlignerBatch(
        rgb_dir=Path(f"datasets/{scene}/rgb"),
        depth_sensor_dir=Path(f"datasets/{scene}/depth_npy"),
        depth_estimation_dir=Path(f"results_new/{scene}/d4/depth_npy"),
        intrinsics_path=Path(f"datasets/{scene}/intrinsics.json"),
        output_dir=Path(f"results_new/{scene}/d6"),
        voxel_size=voxel_size,
    )
    scale, _ = batch.estimate_scale_from_depth_maps(
        max_samples=50000,
        min_depth=0.01,
        max_depth=5.0
    )
    batch.run(scale)


def run_fusion_d8(
    scene: str,
    scale: int = 1000,
    trunc: float = 3.0,
    mode: str = "min",
    visualize: bool = True
) -> None:
    """
    Performs depth map fusion between real and monocular.
    """

    results = Path(f"results_new/{scene}")
    processor = DepthFusionProcessor(
        depth_real_dir=Path(f"datasets/{scene}/depth_npy"),
        depth_estimated_dir=results / "d4/depth_npy",
        output_dir=results / "d8/test_square",
    )
    processor.run(1)


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
    scene = "lab_scene_d"
    voxel_size = 0.02

    print(f"Running pipeline for scene: {scene}")

    # modules/reconstruction/rgbd_loader.py # Read .ply

    # run_capture_d12(
    #     output_path=Path(f"datasets/{scene}"),
    #     max_frames=60,
    #     fps=15
    # )
    # run_reconstruction_d3(scene, voxel_size)

    run_monodepth_d4(scene)
    # run_reconstruction_d5(
    #     scene,
    #     voxel_size,
    #     scale_correction=1  # Adjust as needed
    # )

    # run_batch_alignment_d6(
    #     scene,
    #     voxel_size=voxel_size,
    #     depth_scale=1000.0
    # )

    # run_fusion_d8(
    #     scene,
    #     scale=1000,
    #     trunc=4.0,
    #     mode='min',
    #     visualize=True
    # )

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
