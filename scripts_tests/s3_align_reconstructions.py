from pathlib import Path
import numpy as np
import open3d as o3d

# Adiciona raiz do projeto ao sys.path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.evaluation.align_reconstruction_node import ReconstructionAligner


def main():
    output_dir = Path("results/lab_scene_03")

    real_pcd_path = output_dir / "reconstruction_sensor.ply"
    mono_pcd_path = output_dir / "reconstruction_depthanything.ply"
    output_pcd = output_dir / "reconstruction_depthanything_aligned.ply"
    output_matrix = output_dir / "T_d_to_m.npy"

    aligner = ReconstructionAligner(
        real_path=real_pcd_path,
        mono_path=mono_pcd_path,
        output_aligned_path=output_pcd,
        output_matrix_path=output_matrix,
        voxel_size=0.02
    )
    aligner.run()


if __name__ == "__main__":
    main()
