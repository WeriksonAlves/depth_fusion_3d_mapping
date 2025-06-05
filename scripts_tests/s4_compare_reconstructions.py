from pathlib import Path
# Adiciona raiz do projeto ao sys.path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.evaluation.compare_reconstructions_node import ReconstructionComparator


def main():
    dataset_dir = Path("results/lab_scene_03/1000")

    real_pcd_path = dataset_dir / "reconstruction_sensor.ply"
    mono_pcd_path = dataset_dir / "reconstruction_depthanything.ply"
    aligned_mono_pcd = dataset_dir / "reconstruction_depthanything_aligned.ply"

    comparator = ReconstructionComparator(
        real_pcd_path=real_pcd_path,
        mono_pcd_path=aligned_mono_pcd,
        scale_mono=1  # Ajuste se necessário
    )
    comparator.run()


if __name__ == "__main__":
    main()
