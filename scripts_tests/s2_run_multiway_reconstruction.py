from pathlib import Path

# Adiciona raiz do projeto ao sys.path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.reconstruction.multiway_reconstructor_node import MultiwayReconstructor


def main():
    # Dataset base path
    scene_dir = Path("lab_scene_01")

    dataset_path = Path(f"datasets/{scene_dir}")
    output_path = Path(f"results/{scene_dir}")

    # Verifica estrutura de diretórios
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Dataset directory not found: {dataset_path}")
    if not output_path.exists():
        raise FileNotFoundError(
            f"[ERROR] Output directory with depth mono not found: {output_path}")

    print("[INFO] Running multiway reconstruction...")
    reconstructor = MultiwayReconstructor(
        dataset_dir=dataset_path,
        output_dir=output_path,
        mode="real",  # real or mono"
        ros_node=None,
        voxel_size=0.02,
        depth_scale=500.0,
        depth_trunc=4.0,
        frame_id="map",
        topic="/o3d_points"
    )
    print(f"[INFO] Starting reconstruction in '{reconstructor.mode}' mode...")
    reconstructor.run()
    print("[✓] Reconstruction complete.")


if __name__ == "__main__":
    main()
