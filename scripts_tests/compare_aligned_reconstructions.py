from pathlib import Path
# Adiciona raiz do projeto ao sys.path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.evaluation.compare_reconstructions_node import ReconstructionComparator


def main():
    real_pcd = Path("datasets/lab_scene_kinect_xyz/reconstruction_d435.ply")
    aligned_mono_pcd = Path("results/lab_scene_kinect_xyz/reconstruction_depthanything_aligned.ply")

    comparator = ReconstructionComparator(
        real_pcd_path=real_pcd,
        mono_pcd_path=aligned_mono_pcd,
        scale_mono=1.0  # Já está alinhado, não escalar novamente
    )
    comparator.run()


if __name__ == "__main__":
    main()
