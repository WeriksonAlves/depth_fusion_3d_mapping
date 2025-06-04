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
├── checkpoints/
│   └── depth_anything_v2_vits.pth
├── datasets/
│   └── lab_scene_kinect_xyz/
│       └── ...
├── installs/
│   ├── requirements.txt
│   └── README_OCTOMAP.md
├── launch/
│   ├── align_reconstruction.launch.py
│   ├── compare_reconstructions.launch.py
│   └── check_realsense_topics.launch.py
├── modules/
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── align_reconstruction_node.py
│   │   └── compare_reconstructions_node.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── depth_estimator.py
│   │   └── depth_batch_inferencer.py
│   ├── reconstruction/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── frame_loader.py
│   │   ├── graph_optimizer.py
│   │   ├── map_merger_publisher_node.py
│   │   ├── map_merger_publisher.py
│   │   ├── pose_graph_builder.py
│   │   └── multiway_reconstructor_node.py
│   └── utils/
│       ├── __init__.py
│       ├── README.md
│       ├── point_cloud_processor.py
│       ├── live_point_cloud_visualizer.py
│       ├── reconstruction_viewer.py
│       └── realsense_topic_checker_node.py
├── results/
│   └──  ...
├── scripts_test/
│   └──  ...
├── main.py
└── README.md  
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
```

### Gravar um .bag do sensor de entrada (Intel Realsense)


### Extrair os dados .bag gravados

### Infererir a nuvem de pontos com os dados do sensor

### Substituir o D do sensor pelo D no estimafor e inferir novemente

### Comparar ambas as nuvens de pontos armazenadas