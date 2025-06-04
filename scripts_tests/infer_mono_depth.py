import os
from pathlib import Path

# Adiciona raiz do projeto ao sys.path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.inference.depth_batch_inferencer import DepthBatchInferencer


def main():
    # Dataset base path
    dataset_path = Path("datasets/lab_scene_kinect_xyz")

    input_dir = dataset_path / "rgb"
    output_dir = dataset_path / "depth_mono"
    checkpoint_dir = Path("checkpoints")
    encoder = "vits"

    # Verifica estrutura de diretórios
    if not input_dir.exists():
        raise FileNotFoundError(f"[ERROR] RGB directory not found: {input_dir}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"[ERROR] Checkpoint directory not found: {checkpoint_dir}")

    print("[INFO] Running monocular depth inference...")
    inferencer = DepthBatchInferencer(
        input_dir=input_dir,
        output_dir=output_dir,
        encoder=encoder,
        checkpoint_dir=checkpoint_dir
    )
    inferencer.run()
    print("[✓] Depth inference complete.")


if __name__ == "__main__":
    main()
