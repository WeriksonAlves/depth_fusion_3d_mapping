"""
    Script para inferência de profundidade em tempo real usando o modelo
    DepthAnythingV2. O script pode processar imagens de uma pasta ou de uma
    câmera (webcam). O resultado é exibido em uma janela e pode ser salvo em
    disco. O modelo é carregado a partir de um checkpoint e utiliza o encoder
    especificado. O script é configurável para diferentes modos de operação e
    parâmetros do modelo.
"""
import os
import sys
import cv2
import torch
import numpy as np

# Adiciona o path do DepthAnythingV2
sys.path.append(os.path.join(os.getcwd(), "Depth_Anything_V2"))
from depth_anything_v2.dpt import DepthAnythingV2

# ==== CONFIGURAÇÕES ====
MODE = 'images'  # 'webcam' ou 'images'
IMAGE_DIR = 'SIBGRAPI2025/datasets/rgbd_dataset_freiburg1_xyz/rgb'
OUTPUT_DIR = 'SIBGRAPI2025/results/depth_maps'
SAVE_DEPTH = True
ENCODER = 'vits'  # 'vits', 'vitb', 'vitl', 'vitg'
IDX_CAM = 1  # ID da câmera (0 para webcam padrão, mude se necessário)

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
    concat_img = cv2.hconcat([img_input, depth_color])

    if SAVE_DEPTH and name:
        output_path = os.path.join(OUTPUT_DIR, f'depth_{name}')
        cv2.imwrite(output_path, concat_img)

    cv2.imshow('RGB + Depth Map', concat_img)


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
