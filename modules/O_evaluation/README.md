## 📁 `modules/evaluation/README.md`

### 🎯 Overview

The `evaluation` submodule provides tools for **comparing** and **aligning** 3D reconstructions obtained from:

* Real sensor data (e.g. Intel RealSense D435)
* Monocular depth estimation (e.g. DepthAnythingV2)

---

## 🔍 Components

| Tool / Node                    | Description                                                       |
| ------------------------------ | ----------------------------------------------------------------- |
| `compare_reconstructions_node` | Visualizes and compares two 3D reconstructions                    |
| `align_reconstruction_node`    | Aligns monocular reconstruction to real depth using ICP + scaling |

---

### 🧩 Compare Reconstructions: RealSense vs DepthAnythingV2

This tool visualizes both reconstructions simultaneously to assess overlap, scale, and alignment.

#### ✅ Direct Execution

```bash
ros2 run sibgrapi2025_slam compare_reconstructions_node
```

#### 🚀 With Launch File

```bash
ros2 launch sibgrapi2025_slam compare_reconstructions.launch.py
```

#### ⚙️ Configurable Parameters

```bash
ros2 launch sibgrapi2025_slam compare_reconstructions.launch.py \
  real_pcd_path:=datasets/lab_scene_kinect_xyz/reconstruction_d435.ply \
  mono_pcd_path:=datasets/lab_scene_kinect_xyz/reconstruction_depthanything.ply \
  scale_mono:=0.95
```

#### 🖍️ Color Convention

* **Blue** → RealSense-based reconstruction
* **Red** → Monocular-based reconstruction

---

### 🔄 Align Reconstructions (Monocular ➜ Real)

Aligns the monocular-based 3D point cloud to the RealSense reference using **Generalized ICP + scaling**.

#### ✅ Execution with Launch File

```bash
ros2 launch sibgrapi2025_slam align_reconstruction.launch.py
```

#### ⚙️ Parameters

```bash
ros2 launch sibgrapi2025_slam align_reconstruction.launch.py \
  real_pcd_path:=datasets/lab_scene_kinect_xyz/reconstruction_d435.ply \
  mono_pcd_path:=datasets/lab_scene_kinect_xyz/reconstruction_depthanything.ply \
  output_aligned_pcd:=datasets/lab_scene_kinect_xyz/aligned_mono.ply \
  output_transform_matrix:=datasets/lab_scene_kinect_xyz/T_mono_to_real.npy
```

#### 💾 Outputs

* `T_mono_to_real.npy`: 4×4 transformation matrix
* `aligned_mono.ply`: Aligned version of the monocular point cloud

#### 🖍️ Color Convention

* **Blue** → RealSense-based cloud
* **Red** → Aligned monocular cloud

---

### ♻️ Reusing the Transformation Matrix

The transformation matrix can be applied to any future monocular reconstruction captured in the same environment.

#### 🧪 Example in Python

```python
import numpy as np
import open3d as o3d

# Load transformation
T = np.load('datasets/lab_scene_kinect_xyz/T_mono_to_real.npy')

# Load new monocular cloud
pcd = o3d.io.read_point_cloud('datasets/lab_scene_kinect_xyz/depthanything_raw.ply')

# Apply transformation
pcd.transform(T)

# Save or visualize
o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud('datasets/lab_scene_kinect_xyz/depthanything_aligned.ply', pcd)
```

---

### 🔗 Integration

These tools integrate seamlessly with:

* `MultiwayReconstructor` in `modules.reconstruction`
* `DepthBatchInferencer` in `modules.inference`
