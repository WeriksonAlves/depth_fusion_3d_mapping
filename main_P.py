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


def run_capture_d12() -> None:
    """
    Captures frames using RealSense and saves RGB, depth, and intrinsics.
    """
    recorder = RealSenseRecorder(
        output_path="datasets/lab_scene_d",
        max_frames=60,
        fps=15
    )
    recorder.start()
    recorder.capture()


def run_reconstruction_d3() -> None:
    """
    Runs reconstruction using RealSense depth.
    """
    scene = "lab_scene_d"
    reconstructor = MultiwayReconstructorOffline(
        rgb_dir=Path(f"datasets/{scene}/rgb"),
        depth_dir=Path(f"datasets/{scene}/depth_npy"),
        intrinsics_path=Path(f"datasets/{scene}/intrinsics.json"),
        output_dir=Path(f"results/{scene}/d3"),
        output_pcd_path=Path(f"results/{scene}/d3/reconstruction_sensor.ply"),
        voxel_size=0.02
    )
    reconstructor.run()


def run_monodepth_d4() -> None:
    """
    Performs monocular depth inference using DepthAnythingV2.
    """
    scene = "lab_scene_d"
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
        scaling_factor=1.0
    )
    inferencer.run()


def run_reconstruction_d5() -> None:
    """
    Runs reconstruction using monocular depth estimates.
    """
    scene = "lab_scene_f"
    reconstructor = MultiwayReconstructorOffline(
        rgb_dir=Path(f"datasets/{scene}/rgb"),
        depth_dir=Path(f"results/{scene}/d4/depth_npy"),
        intrinsics_path=Path(f"datasets/{scene}/intrinsics.json"),
        output_dir=Path(f"results/{scene}/d5"),
        output_pcd_path=Path(f"results/{scene}/d5/reconstruction_est.ply"),
        voxel_size=0.02
    )
    reconstructor.run()


def run_compare_d5() -> None:
    """
    Compares point clouds from sensor vs monocular estimation.
    """
    scene = "lab_scene_f"
    path_d435 = Path(f"results/{scene}/d3/reconstruction_sensor.ply")
    path_mono = Path(f"results/{scene}/d5/reconstruction_estimated.ply")

    comparer = PointCloudComparer(offset_apply=True)
    comparer.visualize([path_d435, path_mono])


def run_alignment_d6() -> None:
    """
    Runs ICP alignment for a single frame.
    """
    aligner = FrameICPAligner(
        dataset_dir=Path("datasets/lab_scene_d"),
        results_dir=Path("results/lab_scene_d"),
        frame_index=0
    )
    aligner.run()


def run_batch_alignment_d6() -> None:
    """
    Runs ICP alignment in batch for multiple frames.
    """
    batch = FrameICPAlignerBatch(
        scene_name="lab_scene_d",
        frame_indices=list(range(10)),  # frames 0000–0009
        voxel_size=0.02,
        depth_scale=1000.0
    )
    batch.run()


def run_fusion_d8() -> None:
    """
    Performs depth map fusion between real and monocular.
    """
    scene = "lab_scene_d"
    scale = 100
    trunc = 3.0
    mode = "min"

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
        visualize=True
    )
    processor.run()


def run_fused_reconstruction_d9() -> None:
    """
    Reconstructs from fused depth maps.
    """
    scene = "lab_scene_f"
    reconstructor = MultiwayReconstructorOffline(
        rgb_dir=Path(f"datasets/{scene}/rgb"),
        depth_dir=Path(f"results/{scene}/d8/fused_depth_Tdm_min_100_3.0"),
        intrinsics_path=Path(f"datasets/{scene}/intrinsics.json"),
        output_dir=Path(f"results/{scene}/d9"),
        output_pcd_path=Path(f"results/{scene}/d9/reconstruction_fused.ply"),
        voxel_size=0.02
    )
    reconstructor.run()


def run_pipeline_sequence() -> None:
    """
    Executes a full pipeline step-by-step.
    """
    run_capture_d12()
    run_reconstruction_d3()
    run_monodepth_d4()
    run_reconstruction_d5()
    run_alignment_d6()
    run_batch_alignment_d6()
    run_fusion_d8()
    run_fused_reconstruction_d9()
    run_compare_d5()


if __name__ == "__main__":
    run_pipeline_sequence()
