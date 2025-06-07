from pathlib import Path

from modules.utils.realsense_recorder import RealSenseRecorder
from modules.reconstruction.multiway_reconstructor_offline import (
    MultiwayReconstructorOffline
)
from modules.inference.depth_batch_inferencer import DepthBatchInferencer
from modules.utils.point_cloud_comparer import PointCloudComparer
from modules.align.frame_icp_aligner import FrameICPAligner
from modules.align.frame_icp_batch_aligner import FrameICPAlignerBatch
from modules.fusion.depth_fusion_processor import DepthFusionProcessor


def run_capture_d12(output_path,
                    max_frames=60,
                    fps=15) -> None:
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


def run_reconstruction_d3(scene: str, voxel_size=0.02) -> None:
    """
    Runs reconstruction using RealSense depth.
    """
    reconstructor = MultiwayReconstructorOffline(
        rgb_dir=Path(f"datasets/{scene}/rgb"),
        depth_dir=Path(f"datasets/{scene}/depth_npy"),
        intrinsics_path=Path(f"datasets/{scene}/intrinsics.json"),
        output_dir=Path(f"results/{scene}/d3"),
        output_pcd_path=Path(f"results/{scene}/d3/reconstruction_sensor.ply"),
        voxel_size=voxel_size
    )
    reconstructor.run()


def run_monodepth_d4(scene: str, scaling_factor=1.0) -> None:
    """
    Performs monocular depth inference using DepthAnythingV2.
    """

    input_dir = Path(f"datasets/{scene}/rgb")
    output_dir = Path(f"results/{scene}/d4")
    checkpoint_dir = Path("checkpoints")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input not found: {input_dir}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    inferencer = DepthBatchInferencer(
        input_dir=input_dir,
        output_path=output_dir,
        encoder="vits",
        checkpoint_dir=checkpoint_dir,
        scaling_factor=scaling_factor
    )
    inferencer.run()


def run_reconstruction_d5(scene: str, voxel_size=0.02) -> None:
    """
    Runs reconstruction using monocular depth estimates.
    """

    reconstructor = MultiwayReconstructorOffline(
        rgb_dir=Path(f"datasets/{scene}/rgb"),
        depth_dir=Path(f"results/{scene}/d4/depth_npy"),
        intrinsics_path=Path(f"datasets/{scene}/intrinsics.json"),
        output_dir=Path(f"results/{scene}/d5"),
        output_pcd_path=Path(f"results/{scene}/d5/reconstruction_est.ply"),
        voxel_size=voxel_size
    )
    reconstructor.run()


def run_compare_d5(scene: str, offset_apply: bool = False) -> None:
    """
    Compares point clouds from sensor vs monocular estimation.
    """
    path_d435 = Path(f"results/{scene}/d3/reconstruction_sensor.ply")
    path_mono = Path(f"results/{scene}/d5/reconstruction_est.ply")

    comparer = PointCloudComparer(offset_apply=offset_apply)
    comparer.visualize([path_d435, path_mono])


def run_alignment_d6(scene: str, frame_index: int = 0) -> None:
    """
    Runs ICP alignment for a single frame.
    """
    aligner = FrameICPAligner(
        dataset_dir=Path(f"datasets/{scene}"),
        results_dir=Path(f"results/{scene}"),
        frame_index=frame_index
    )
    aligner.run()


def run_batch_alignment_d6(scene: str,
                           len_range: int = 10,
                           voxel_size: float = 0.02,
                           depth_scale=1000.0) -> None:
    """
    Runs ICP alignment in batch for multiple frames.
    """
    batch = FrameICPAlignerBatch(
        scene_name=scene,
        frame_indices=list(range(len_range)),
        voxel_size=voxel_size,
        depth_scale=depth_scale
    )
    batch.run()


def run_fusion_d8(
    scene: str,
    scale: int = 100,
    trunc: float = 3.0,
    mode: str = "min",
    visualize=True) -> None:
    """
    Performs depth map fusion between real and monocular.
    
    Args:
        scene (str): Name of the scene to process.
        scale (int): Scale factor for depth maps.
        trunc (float): Truncation value for depth maps.
        mode (str): Fusion mode, e.g., "min", "mean", "real-priority" and
            "mono-priority".
    """
    base = Path(f"datasets/{scene}")
    results = Path(f"results/{scene}")
    output = results / f"d8/fused_depth_Tdm_{mode}_{scale}_{trunc:.1f}"

    processor = DepthFusionProcessor(
        rgb_dir=base / "rgb",
        depth_real_dir=base / "depth_npy",
        depth_mono_dir=results / "d4/depth_npy",
        transform_path=results / "d6/T_d_to_m_frame0000.npy",
        intrinsics_path=base / "intrinsics.json",
        output_dir=output,
        depth_scale=scale,
        depth_trunc=trunc,
        mode=mode,
        visualize=visualize
    )
    processor.run()


def run_fused_reconstruction_d9(
    scene: str,
    scale: int = 100,
    trunc: float = 3.0,
    mode: str = "min",
    voxel_size=0.02) -> None:
    """
    Reconstructs from fused depth maps.
    """
    results = Path(f"results/{scene}")
    output = results / f"d8/fused_depth_Tdm_{mode}_{scale}_{trunc:.1f}"

    reconstructor = MultiwayReconstructorOffline(
        rgb_dir=Path(f"datasets/{scene}/rgb"),
        depth_dir=output,
        intrinsics_path=Path(f"datasets/{scene}/intrinsics.json"),
        output_dir=Path(f"results/{scene}/d9"),
        output_pcd_path=Path(f"results/{scene}/d9/reconstruction_fused.ply"),
        voxel_size=voxel_size
    )
    reconstructor.run()


def run_pipeline_sequence(scene: str) -> None:
    """
    Executes a full pipeline step-by-step.
    """
    scene = "lab_scene_r"
    voxel_size = 0.02
    scaling_factor = 1.0
    frame_index = 0
    len_range = 10
    scale = 100
    trunc = 3.0
    mode = "min"
    visualize = True

    print(f"Running pipeline for scene: {scene}")

    # run_capture_d12(
    #     output_path=Path(f"datasets/{scene}"),
    #     max_frames=60,
    #     fps=15
    # )

    run_reconstruction_d3(scene, voxel_size)
    run_monodepth_d4(scene, scaling_factor)
    run_reconstruction_d5(scene, voxel_size)
    run_alignment_d6(scene, frame_index)
    run_batch_alignment_d6(scene, len_range, voxel_size)
    run_fusion_d8(scene, scale, trunc, mode, visualize)
    run_fused_reconstruction_d9(scene, scale, trunc, mode, voxel_size)
    run_compare_d5(scene, offset_apply=True)


if __name__ == "__main__":
    run_pipeline_sequence()
