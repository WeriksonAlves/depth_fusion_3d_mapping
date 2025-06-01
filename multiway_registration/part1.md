
## ✅ **Etapa 1 — Aquisição de Dados com ROS 2 e Intel RealSense D435**

### 🎯 Objetivo:

Capturar um conjunto de **\~100 frames RGB-D sincronizados** usando a câmera Intel RealSense D435 no ambiente do laboratório, com **10 FPS e resolução 640×480**, salvando tudo em um único arquivo `.bag`.

---

## 🛠️ **1. Preparação do Ambiente**

### a) Instale os pacotes do RealSense ROS 2

Caso ainda não tenha:

```bash
sudo apt update
sudo apt install ros-humble-realsense2-camera
```

> 🔁 Se quiser compilar do código-fonte: [Instruções oficiais](https://github.com/IntelRealSense/realsense-ros/tree/ros2-development)

---

## 📦 **2. Lançamento do driver da câmera**

Abra um terminal:

```bash
ros2 launch realsense2_camera rs_camera.launch.py \
    enable_depth:=true \
    enable_color:=true \
    align_depth.enable:=true \
    depth_module.profile:=640x480x10 \
    color_module.profile:=640x480x10 \
    unite_imu_method:=none
```

> Isso ativa os streams de profundidade e cor alinhados com resolução e taxa de quadros desejadas.

---

## 🎥 **3. Gravação do .bag com ros2 bag**

Abra outro terminal e execute:

```bash
ros2 bag record /camera/color/image_raw /camera/depth/image_rect_raw /camera/aligned_depth_to_color/image_raw /camera/color/camera_info -o lab_scene_01
```

> 📁 Isso criará uma pasta chamada `rgbd_lab_capture` com os dados gravados.

Você pode interromper com `Ctrl+C` após os \~10 segundos.

---


## 🗂️ **4. Estrutura de Pastas Recomendada**

```bash
datasets/
└── lab_scene_01/
    ├── raw.bag                      # Dados brutos
    ├── color/                       # (Será extraído depois)
    ├── depth/                       # (Será extraído depois)
    └── intrinsics.json             # Parâmetros da câmera
```

---

## ⏭️ **Próximas Etapas**

Na próxima etapa, vamos:

* Extrair as **imagens RGB e mapas de profundidade** do `.bag` para `.png` ou `.npy`.
* Salvar os intrínsecos reais da D435 para uso no Open3D.
* Organizar os dados para uso no pipeline de reconstrução.

Deseja que eu gere agora o script para essa **extração do .bag para imagens RGB e profundidade**?
