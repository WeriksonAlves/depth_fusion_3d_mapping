from pathlib import Path
from modules.reconstruction.multiway_reconstructor_offline import (
    MultiwayReconstructorOffline
)
from modules.inference.depth_batch_inferencer import DepthBatchInferencer


def main_d3() -> None:
    scene = "lab_scene_d"

    reconstructor = MultiwayReconstructorOffline(
        rgb_dir=Path(f"datasets/{scene}/rgb"),
        depth_dir=Path(f"datasets/{scene}/depth_npy"),
        intrinsics_json=Path(f"datasets/{scene}/intrinsics.json"),
        output_dir=Path(f"results/{scene}/d3"),
        output_pcd=Path(f"results/{scene}/d3/reconstruction_sensor.ply"),
        voxel_size=0.02,
        scale_correction=1.0
    )
    reconstructor.run()


def main_d4() -> None:
    scene = "lab_scene_f"
    input_dir = Path(f"datasets/{scene}/rgb")
    output_dir = Path(f"results/{scene}/estimated")
    checkpoint_dir = Path("checkpoints")
    encoder = "vits"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input not found: {input_dir}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    inferencer = DepthBatchInferencer(
        input_dir=input_dir,
        output_path=output_dir,
        encoder=encoder,
        checkpoint_dir=checkpoint_dir,
        scaling_factor=1
    )

    print("[INFO] Running depth inference...")
    inferencer.run()
    print("[✓] Inference complete.")


def main_d5() -> None:
    scene = "lab_scene_f"

    reconstructor = MultiwayReconstructorOffline(
        rgb_dir=Path(f"datasets/{scene}/rgb"),
        depth_dir=Path(f"results/{scene}/d4/depth_npy"),
        intrinsics_json=Path(f"datasets/{scene}/intrinsics.json"),
        output_dir=Path(f"results/{scene}/d5"),
        output_pcd=Path(f"results/{scene}/d5/final_reconstruction.ply"),
        voxel_size=0.02,
        scale_correction=1.0
    )
    reconstructor.run()


if __name__ == "__main__":
    main_d3()
