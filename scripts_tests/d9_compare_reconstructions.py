import json
from pathlib import Path

import open3d as o3d


def load_point_clouds(pcd_paths: dict) -> dict:
    """
    Load point clouds from given file paths.

    Args:
        pcd_paths (dict): Dictionary {name: Path} to each .ply file.

    Returns:
        dict: Dictionary {name: PointCloud}
    """
    clouds = {}
    for name, path in pcd_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"[ERROR] Missing file: {path}")
        clouds[name] = o3d.io.read_point_cloud(str(path))
        print(f"[✓] Loaded: {name} | {len(clouds[name].points):,} points")
    return clouds


def load_metrics(metrics_paths: dict) -> dict:
    """
    Load reconstruction metrics from JSON files.

    Args:
        metrics_paths (dict): Dictionary {name: Path} to each .json file.

    Returns:
        dict: Dictionary {name: metrics_dict}
    """
    metrics = {}
    for name, path in metrics_paths.items():
        if not path.exists():
            print(f"[!] Metrics not found for {name}: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            metrics[name] = json.load(f)
    return metrics


def print_comparison(metrics: dict) -> None:
    """
    Print side-by-side metrics comparison.

    Args:
        metrics (dict): {name: metrics_dict}
    """
    print("\n=== Reconstruction Metrics Comparison ===\n")
    headers = ["Reconstruction", "Points", "Volume (m³)", "Avg. Density"]
    row_format = "{:<15} | {:>10} | {:>13.4f} | {:>13.4f}"

    print("{:<15} | {:>10} | {:>13} | {:>13}".format(*headers))
    print("-" * 60)

    for name, m in metrics.items():
        print(
            row_format.format(
                name,
                f"{m['num_points']:,}",
                m["volume_aabb"],
                m["avg_density"]
            )
        )
    print()


def visualize_side_by_side(clouds: dict) -> None:
    """
    Displays all point clouds in Open3D with colors.

    Args:
        clouds (dict): {name: PointCloud}
    """
    colors = {
        "D435": [1.0, 0.706, 0.0],       # yellow
        "Mono": [0.0, 0.651, 0.929],     # blue
        "Fused": [0.0, 0.8, 0.0],        # green
    }

    geometries = []
    for name, pcd in clouds.items():
        pcd_copy = o3d.geometry.PointCloud(pcd)  # cria uma cópia manual
        pcd_copy.paint_uniform_color(colors.get(name, [0.7, 0.7, 0.7]))
        geometries.append(pcd_copy)

    o3d.visualization.draw_geometries(
        geometries,
        window_name="Reconstruction Comparison (D435 / Mono / Fused)",
        width=1600,
        height=900,
    )


def main() -> None:
    scene = "lab_scene_f"
    base = Path(f"results/{scene}")

    pcd_paths = {
        "D435": base / "d3/reconstruction_sensor.ply",
        "Mono": base / "d5/reconstruction_estimated.ply",
        "Fused": base / "d9/reconstruction_fused.ply"
    }

    metrics_paths = {
        "D435": base / "d3/reconstruction_metrics.json",
        "Mono": base / "d5/reconstruction_metrics.json",
        "Fused": base / "d9/reconstruction_metrics.json"
    }

    clouds = load_point_clouds(pcd_paths)
    metrics = load_metrics(metrics_paths)

    print_comparison(metrics)
    visualize_side_by_side(clouds)


if __name__ == "__main__":
    main()
