# 🛰️ SIBGRAPI2025_SLAM

Projeto para reconstrução 3D com base em imagens monoculares utilizando estimativa de profundidade e geração de nuvens de pontos integradas ao OctoMap via ROS 2 Humble (Ubuntu 22.04).
Este projeto tem como objetivo realizar **reconstrução 3D a partir de imagens monoculares** utilizando:

* **Inferência de profundidade com DepthAnythingV2**
* **Back-projection para gerar nuvens de pontos**
* **Visualização com Open3D**
* **Publicação via ROS 2 para o `octomap_server`**
* **Visualização final no RViz2**

O sistema já cobre as etapas de aquisição de imagens, estimação de profundidade, geração e visualização de nuvem de pontos. Está preparado para integração com o ROS 2 para mapeamento 3D em tempo real.

---

## 🗂️ **Estrutura do Projeto (diretórios e scripts principais)**

```bash
SIBGRAPI2025_slam/
│
├── main.py                         # Script principal: executa a geração e visualização de nuvem em tempo real
├── test_depth_inference.py        # Testes com DepthAnythingV2 (webcam ou imagens)
├── intrinsics_tester.py           # Avaliação de diferentes parâmetros intrínsecos da câmera
│
├── modules/
│   ├── depth_estimator.py             # Módulo de inferência de profundidade com DepthAnythingV2
│   ├── point_cloud_processor.py       # Geração, filtragem e salvamento da nuvem de pontos com Open3D
│   └── live_point_cloud_visualizer.py # Visualização ao vivo da nuvem com Open3D
│
├── Depth_Anything_V2/             # Repositório clonado do modelo de profundidade (renomeado)
├── checkpoints
│
├── point_clouds/                  # Diretório esperado para salvar nuvens em `.ply` para o ROS2
│   └── pcd.ply                    # Nuvem a ser publicada no ROS
│
├── results/                       # Saídas de testes, imagens e CSVs de resultados
│   └── test/                      # Imagens e nuvens salvas durante testes com `main.py`
│
├── datasets/                      # Dataset de imagens RGB utilizadas (ex: TUM)
│   └── rgbd_dataset_freiburg1_xyz/
│       └── rgb/
│
├── README.md                      # Instruções completas de instalação, execução e publicação ROS
└── requirements.txt               # (Dentro de Depth_Anything_V2) dependências do modelo
```

## Execução

### 1. Clonar os repositórios

```bash
cd SIBGRAPI2025_slam
git clone https://github.com/DepthAnything/Depth-Anything-V2
```

**Atenção:** Renomear a pasta `Depth-Anything-V2` para `Depth_Anything_V2`.

### 2. Criar ambiente virtual Python

```bash
cd ~/octomap_ws/
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependências Python

```bash
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128  # ou cu121/cu128 conforme sua GPU
cd SIBGRAPI2025_slam/Depth_Anything_V2
pip3 install -r requirements.txt
pip3 install open3d opencv-python numpy "numpy<2" pandas