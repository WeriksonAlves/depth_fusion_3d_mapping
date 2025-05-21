import os
import cv2
import numpy as np
import pandas as pd
import open3d as o3d

from modules.depth_estimator import DepthEstimator
from modules.point_cloud_processor import PointCloudProcessor
from modules.live_point_cloud_visualizer import LivePointCloudVisualizer



def main():
    image_dir = 'datasets/rgbd_dataset_freiburg1_xyz/rgb'
    output_csv = 'results/intrinsics_test_results.csv'

    # Lista de intrínsecos para testar
    intrinsics_list = [
        {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5},
        {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0},
        {"fx": 450.0, "fy": 450.0, "cx": 310.0, "cy": 230.0},
        {"fx": 500.0, "fy": 480.0, "cx": 300.0, "cy": 250.0},
    ]

    # Carrega primeira imagem RGB
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    if not image_files:
        raise FileNotFoundError("Nenhuma imagem PNG encontrada no diretório.")
    test_image_path = os.path.join(image_dir, image_files[0])
    rgb_image = cv2.imread(test_image_path)

    # Estimar profundidade
    estimator = DepthEstimator()
    depth_map = estimator.infer_depth(rgb_image)

    # Visualizador 3D
    visualizer = LivePointCloudVisualizer()

    # Lista de resultados para exportação
    results = []

    for idx, intr in enumerate(intrinsics_list):
        print(f"\n[INFO] Testando conjunto {idx}: {intr}")
        processor = PointCloudProcessor(
            fx=intr["fx"], fy=intr["fy"],
            cx=intr["cx"], cy=intr["cy"],
            width=rgb_image.shape[1],
            height=rgb_image.shape[0]
        )

        # Criar e filtrar a nuvem de pontos
        pcd = processor.create_point_cloud(rgb_image, depth_map)
        pcd_filtered = processor.filter_point_cloud(pcd)

        # Visualizar
        print(f"[INFO] Visualizando nuvem de pontos para Test {idx}")
        visualizer.update(pcd_filtered)

        # Extrair estatísticas
        points = np.asarray(pcd_filtered.points)
        if points.size == 0:
            print("[Aviso] Nenhum ponto gerado.")
            continue

        bbox = pcd_filtered.get_axis_aligned_bounding_box()
        center = bbox.get_center()

        results.append({
            "Test ID": idx,
            "fx": intr["fx"], "fy": intr["fy"],
            "cx": intr["cx"], "cy": intr["cy"],
            "Num Points": len(points),
            "BBox Volume": bbox.volume(),
            "BBox Center X": center[0],
            "BBox Center Y": center[1],
            "BBox Center Z": center[2],
        })

        key = input("[INFO] Pressione Enter para continuar ou 'q' para sair: ")
        if key.lower() == 'q':
            break

    visualizer.close()

    # Exportar resultados
    if results:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[OK] Resultados salvos em: {output_csv}")
    else:
        print("[Aviso] Nenhum dado para salvar.")


if __name__ == '__main__':
    main()
