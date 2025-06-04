import argparse
from pathlib import Path

# Adiciona raiz do projeto ao sys.path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.reconstruction.multiway_reconstructor_node import MultiwayReconstructor


def main():
    parser = argparse.ArgumentParser(
        description="Run multiway 3D reconstruction using real or mono depth."
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to dataset directory (must contain rgb/, depth/, depth_mono/)."
    )
    parser.add_argument(
        "--mode", type=str, choices=["real", "mono"], default="real",
        help="Depth mode: 'real' (from sensor) or 'mono' (from DepthAnything)."
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.02,
        help="Voxel size for downsampling and registration."
    )
    parser.add_argument(
        "--depth_scale", type=float, default=5000.0,
        help="Depth scale used to convert meters to mm (real) or apply gain (mono)."
    )
    parser.add_argument(
        "--depth_trunc", type=float, default=4.0,
        help="Maximum depth in meters to consider."
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"[ERROR] Dataset path not found: {dataset_path}")

    print(f"[INFO] Starting multiway reconstruction ({args.mode})...")
    reconstructor = MultiwayReconstructor(
        dataset_dir=dataset_path,
        mode=args.mode,
        ros_node=None,
        voxel_size=args.voxel_size,
        depth_scale=args.depth_scale,
        depth_trunc=args.depth_trunc
    )
    reconstructor.run()
    print("[✓] Reconstruction complete.")


if __name__ == "__main__":
    main()
