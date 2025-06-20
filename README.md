# 🛰️ SIBGRAPI2025\_SLAM

**3D Reconstruction from Monocular Images using Depth Estimation and Point Cloud Fusion**
Modular pipeline for scene reconstruction from RGB images, leveraging monocular depth inference, point cloud generation, fusion, and visualization using Open3D — all without ROS2.

---

## 🎯 Objective

This project aims to build a robust and modular pipeline for **3D reconstruction from monocular RGB images**, including the following components:

* 📷 RGB + Depth capture with RealSense (offline)
* 🔍 Monocular depth estimation with **DepthAnythingV2**
* 🔁 Point cloud reconstruction using **Open3D**
* 🔄 Frame alignment via **ICP**
* 🧩 Depth map fusion (real + estimated)
* 🗺️ Final visualization and export of the fused 3D map

---

## 📁 Project Structure

```bash
SIBGRAPI2025_slam/
│
├── checkpoints/                  # Pre-trained weights for DepthAnything
├── datasets/                     # Input images, depth maps, and intrinsics
│   └── <scene>/
│       ├── rgb/
│       ├── depth_npy/
│       └── intrinsics.json
│
├── results/                      # Output organized by step
│   └── <scene>/
│       └── step_1/step_2/...step_3/
│
├── modules/                      # Modular implementation
│   ├── inference/              # Monocular depth inference
│   ├── reconstruction/        # Point cloud generation and registration
│   └── utils/                 # ICP, fusion, visualization, RealSense tools
│
├── installs/                     # Setup instructions and dependencies
├── theoric/                      # Supplementary theoretical material
├── main.py                       # Pipeline entry point
└── README.md
```

---

## ⚙️ Installation

### 1. Clone this repository

```bash
git clone https://github.com/your-user/SIBGRAPI2025_slam.git
cd SIBGRAPI2025_slam
```

### 2. Clone DepthAnythingV2

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
mv Depth-Anything-V2 Depth_Anything_V2
```

### 3. Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # cu121 cu128
pip install -r Depth_Anything_V2/requirements.txt
pip install open3d opencv-python numpy "numpy<2" pandas
```

---

## 🚀 Execution

Each stage of the pipeline is callable via functions defined in `main.py`.

### Available Stages

| Stage | Description                                                |
| ----- | ---------------------------------------------------------- |
| 1     | RealSense data capture + reconstruction using real depth   |
| 2     | Monocular depth inference + reconstruction                 |
| 3     | ICP alignment + depth map fusion + final 3D reconstruction |
| 4     | Trajectory visualization                                   |
| 5     | Side-by-side point cloud comparison                        |

### Example:

```python
scene = "lab_scene_kinect_xyz"
stage = 1  # or 2, 3, 4, 5
main()
```

---

## ✅ Expected Results

* Accurate 3D point cloud reconstructions
* Fused depth maps for completeness
* Trajectory visualization of estimated camera motion
* Visual and quantitative comparison between real and monocular depth

---

## 📌 Notes

* No ROS2 is required; the system is ready for extension to OctoMap or RViz if needed.
* Depth inference runs offline on `.png` images.
* CUDA-compatible GPU is recommended for efficient processing.

---

## 👨‍🔬 Author

Developed as part of a research project for **SIBGRAPI 2025**.

> Academic work under the graduate course **INF791 — Computer Vision for 3D Data**
