import os
import sys

import cv2
import torch
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "Depth_Anything_V2"))
from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits' # or 'vitl', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'SIBGRAPI2025\checkpoints\depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('SIBGRAPI2025\datasets\\rgbd_dataset_freiburg1_xyz\\rgb\\1305031102.175304.png')
depth = model.infer_image(raw_img)  # HxW raw depth map in numpy

# Normalização e coloração
depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_uint8 = depth_normalized.astype(np.uint8)
depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

# Visualização
concatenate_img = cv2.hconcat([raw_img, depth_colored])
cv2.imshow('RGB + Depth Map', concatenate_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
