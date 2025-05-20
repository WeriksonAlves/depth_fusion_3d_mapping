'''

'''
import os
import sys
import cv2
import torch
import numpy as np
import open3d as o3d

# Adiciona o path do DepthAnythingV2
sys.path.append(os.path.join(os.getcwd(), "Depth_Anything_V2"))
from depth_anything_v2.dpt import DepthAnythingV2

# ==== CONFIGURAÇÕES ====
MODE = 'images'  # 'webcam' ou 'images'
IMAGE_DIR = 'SIBGRAPI2025/datasets/rgbd_dataset_freiburg1_xyz/rgb'
OUTPUT_DIR = 'SIBGRAPI2025/results/depth_maps'
SAVE_DEPTH = False
ENCODER = 'vits'  # 'vits', 'vitb', 'vitl', 'vitg'
IDX_CAM = 1  # ID da câmera (0 para webcam padrão, mude se necessário)

# Exemplo de matriz intrínseca (ajuste conforme sua câmera)
fx, fy = 525.0, 525.0
cx, cy = 319.5, 239.5
width, height = 640, 480




# %% depth inference in real time

# ==== CONFIGURAÇÃO DO MODELO ====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = f'SIBGRAPI2025/checkpoints/depth_anything_v2_{ENCODER}.pth'
print(f"Using device: {DEVICE} and encoder: {ENCODER}")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192,
                                                                 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384,
                                                                  768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512,
                                                                  1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536,
                                                                  1536, 1536]}
}

model = DepthAnythingV2(**model_configs[ENCODER])
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==== FUNÇÃO DE PROCESSAMENTO ====
def process_and_display(img_input: np.ndarray, name=None) -> None:
    """
    Processes an input RGB image to infer its depth map, displays the
    concatenated RGB and depth images, and optionally saves the result to disk.
    Args:
        img_input (np.ndarray): The input RGB image as a NumPy array.
        name (str, optional): The name used for saving the output image. If
                                None, the image is not saved.
    Returns:
        None
    Side Effects:
        - Displays a window showing the concatenated RGB and depth images.
        - Saves the concatenated image to disk if SAVE_DEPTH is True and a
            name is provided.
    Notes:
        - Requires global variables: model, SAVE_DEPTH, OUTPUT_DIR, cv2, np,
            os.
        - The depth map is inferred using the 'model' object's 'infer_image'
            method.
        - The depth map is normalized, color-mapped, and concatenated with the
            original image for visualization.
    """
    depth = model.infer_image(img_input)
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    concat_img = cv2.hconcat([cv2.flip(img_input,0), cv2.flip(depth_color,0)])

    if SAVE_DEPTH and name:
        output_path = os.path.join(OUTPUT_DIR, f'depth_{name}')
        cv2.imwrite(output_path, concat_img)

    cv2.imshow('RGB + Depth Map', concat_img)

    # depth = depth map (HxW float32), rgb = original RGB frame
    pcd = create_point_cloud(img_input, depth)
    pcd_filtered = filter_point_cloud(pcd)
    live_vis.update(pcd_filtered)




# %% point cloud generation

intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def create_point_cloud(rgb_img, depth_map, depth_scale=1000.0, depth_trunc=4.0):
    # Converte imagens para formato Open3D
    rgb_o3d = o3d.geometry.Image(rgb_img)
    depth_o3d = o3d.geometry.Image((depth_map * depth_scale).astype(np.uint16))  # Open3D espera uint16

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=rgb_o3d,
        depth=depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    return pcd


def filter_point_cloud(pcd, voxel_size=0.01):
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd

class LivePointCloudVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Live Point Cloud", width=960, height=540)
        self.pcd = o3d.geometry.PointCloud()
        self.is_initialized = False

    def update(self, new_pcd):
        if not self.is_initialized:
            self.pcd.points = new_pcd.points
            self.pcd.colors = new_pcd.colors
            self.vis.add_geometry(self.pcd)
            self.is_initialized = True
        else:
            self.pcd.points = new_pcd.points
            self.pcd.colors = new_pcd.colors
            self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()

live_vis = LivePointCloudVisualizer()

# %% main loop

def main():
    # ==== MODO 1: PROCESSAR PASTA DE IMAGENS ====
    if MODE == 'images':
        image_files = sorted([
            f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')])
        for filename in image_files:
            img_path = os.path.join(IMAGE_DIR, filename)
            raw_img = cv2.imread(img_path)
            if raw_img is None:
                print(f"[Aviso] Imagem não encontrada ou corrompida: {img_path}")
                continue
            process_and_display(raw_img, name=filename)
            if cv2.waitKey(1) == 27:
                break  # ESC

    # ==== MODO 2: WEBCAM (Intel RealSense ou qualquer câmera RGB) ====
    elif MODE == 'webcam':
        cap = cv2.VideoCapture(IDX_CAM)
        if not cap.isOpened():
            print("[Erro] Não foi possível abrir a câmera.")
            exit()

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Aviso] Frame não capturado.")
                break
            process_and_display(frame, name=f'camera_frame_{frame_count:04d}.png')
            frame_count += 1
            if cv2.waitKey(1) == 27:
                break  # ESC

        cap.release()

    cv2.destroyAllWindows()
    live_vis.close()


main()
