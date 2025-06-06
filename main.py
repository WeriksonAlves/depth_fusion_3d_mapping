import argparse
from pathlib import Path

import rclpy

from modules.inference import DepthBatchInferencer
from modules.reconstruction import MultiwayReconstructor
from modules.evaluation import ReconstructionAligner
from modules.evaluation import ReconstructionComparator


def run_monocular_inference(dataset_dir: Path) -> None:
    input_dir = dataset_dir / "rgb"
    output_dir = dataset_dir / "depth_mono"
    inferencer = DepthBatchInferencer(
        input_dir=input_dir,
        output_dir=output_dir,
        encoder="vits",
        checkpoint_dir=Path("checkpoints")
    )
    inferencer.run()


def run_reconstruction(dataset_dir: Path, mode: str, use_ros: bool) -> None:
    node = None
    if use_ros:
        rclpy.init()
        node = rclpy.create_node(f"reconstructor_{mode}_node")

    reconstructor = MultiwayReconstructor(
        dataset_dir=dataset_dir,
        mode=mode,
        ros_node=node if use_ros else None
    )
    reconstructor.run()

    if use_ros:
        node.destroy_node()
        rclpy.shutdown()


def run_alignment(dataset_dir: Path, output_dir: Path) -> None:
    aligner = ReconstructionAligner(
        real_path=dataset_dir / "reconstruction_d435.ply",
        mono_path=dataset_dir / "reconstruction_depthanything.ply",
        output_aligned_path=output_dir / "reconstruction_depthanything_aligned.ply",
        output_matrix_path=output_dir / "T_d_to_m.npy"
    )
    aligner.run()


def run_comparison(dataset_dir: Path, scale_mono: float = 1.0) -> None:
    comparator = ReconstructionComparator(
        real_pcd_path=dataset_dir / "reconstruction_d435.ply",
        mono_pcd_path=dataset_dir / "reconstruction_depthanything.ply",
        scale_mono=scale_mono
    )
    comparator.run()


def main():
    parser = argparse.ArgumentParser(description="3D Reconstruction Pipeline")
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to dataset directory.")
    parser.add_argument(
        "--mode", type=str, choices=["real", "mono"], default="real",
        help="Depth mode: 'real' or 'mono'")
    parser.add_argument(
        "--infer", action="store_true",
        help="Run monocular depth inference (required for mono)")
    parser.add_argument(
        "--align", action="store_true",
        help="Align monocular reconstruction to real.")
    parser.add_argument(
        "--compare", action="store_true",
        help="Visual comparison between real and mono reconstruction.")
    parser.add_argument(
        "--ros", action="store_true",
        help="Enable ROS 2 point cloud publishing.")
    parser.add_argument(
        "--visualizer", type=str, choices=["open3d", "rviz"],
        help="Optional point cloud visualizer.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path("results") / dataset_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "mono" and args.infer:
        run_monocular_inference(dataset_dir)

    run_reconstruction(dataset_dir, args.mode, use_ros=args.ros)

    if args.align:
        run_alignment(dataset_dir, output_dir)

    if args.compare:
        run_comparison(dataset_dir)

    if args.visualizer == "rviz":
        print(
            "[INFO] Please launch RViz manually and subscribe to /o3d_points.")

    elif args.visualizer == "open3d":
        from modules.utils.reconstruction_viewer import visualize_open3d
        ply_file = (
            dataset_dir / "reconstruction_d435.ply"
            if args.mode == "real"
            else dataset_dir / "reconstruction_depthanything_aligned.ply"
        )
        visualize_open3d(ply_file)


if __name__ == "__main__":

    main()
