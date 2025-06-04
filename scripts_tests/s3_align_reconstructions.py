from pathlib import Path
import numpy as np
import open3d as o3d

# Adiciona raiz do projeto ao sys.path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from modules.evaluation.align_reconstruction_node import ReconstructionAligner


def main():
    base_dir = Path("datasets/lab_scene_kinect_xyz")
    output_dir = Path("results/lab_scene_kinect_xyz")
    output_dir.mkdir(parents=True, exist_ok=True)

    real_pcd_path = base_dir / "reconstruction_d435.ply"
    mono_pcd_path = base_dir / "reconstruction_depthanything.ply"
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
