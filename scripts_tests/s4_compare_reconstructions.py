from pathlib import Path
# Adiciona raiz do projeto ao sys.path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.evaluation.compare_reconstructions_node import ReconstructionComparator


def main():
    dataset_dir = Path("datasets/lab_scene_kinect_xyz")

    real_pcd_path = dataset_dir / "reconstruction_d435.ply"
    mono_pcd_path = dataset_dir / "reconstruction_depthanything.ply"

    comparator = ReconstructionComparator(
        real_pcd_path=real_pcd_path,
        mono_pcd_path=mono_pcd_path,
        scale_mono=1.0  # Ajuste se necessário
    )
    comparator.run()


if __name__ == "__main__":
    main()
