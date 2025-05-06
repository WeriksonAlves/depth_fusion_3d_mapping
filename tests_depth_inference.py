import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "Depth_Anything_V2"))

from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from depth_anything_v2.dpt import DepthAnything

# Caminho para a imagem de entrada
img_path = "SIBGRAPI2025\datasets\\rgbd_dataset_freiburg1_xyz\\rgb\\1305031102.243211.png"

# Carregamento e pré-processamento
transform = T.Compose([
    Resize(384, 512),
    NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    PrepareForNet()
])

# Carregar imagem
image = Image.open(img_path).convert('RGB')
image_np = np.array(image)  # <- CONVERSÃO CRÍTICA

# Aplicar transformações
sample = transform({'image': image_np})
image_tensor = torch.from_numpy(sample['image']).unsqueeze(0).float()
img_input = image_tensor

# Carregar modelo
model = DepthAnything.from_pretrained(
    "SIBGRAPI2025/checkpoints/depth_anything_vits.pth"
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()

with torch.no_grad():
    depth = model(img_input.to(model.device))

depth_map = depth.squeeze().cpu().numpy()
np.save("SIBGRAPI2025/depth_output.npy", depth_map)

# Visualizar profundidade
plt.imshow(depth.squeeze().cpu(), cmap='inferno')
plt.title("Depth Map")
plt.axis("off")
plt.show()
