## 📁 `modules/inference/README.md`

modules/inference/
├── depth_estimator.py            # DepthAnythingV2Estimator (modelo)
├── depth_batch_inferencer.py     # DepthBatchInferencer (inference em lote)
├── depth_offline_runner.py       # DepthOfflineInference (wrapper CLI)
└── __init__.py                   # API pública

### 🎯 Overview

The `modules.inference` submodule provides tools for monocular depth estimation using the **DepthAnythingV2** model. It supports both:

* Standalone usage for batch processing of RGB images;
* ROS 2 service integration for depth map generation.

---

### 📦 File Structure

| File                        | Description                                   |
| --------------------------- | --------------------------------------------- |
| `depth_estimator.py`        | Wrapper around the DepthAnythingV2 model      |
| `depth_batch_inferencer.py` | Batch inference utility and ROS 2 integration |
| `__init__.py`               | Public API exposure                           |

---

## 🔍 Core Components

### ✅ `DepthAnythingV2Estimator`

A reusable, high-level wrapper around the DepthAnythingV2 model.

#### 🔧 Usage:

```python
from modules.inference import DepthAnythingV2Estimator

estimator = DepthAnythingV2Estimator(
    encoder='vits',
    checkpoint_dir='checkpoints'
)
depth_tensor = estimator.infer_depth(rgb_image)
```

* **Input:** RGB image (`np.ndarray`)
* **Output:** Depth tensor (`torch.Tensor`) in meters
* **Supported encoders:** `vits`, `vitb`, `vitl`, `vitg`

---

### ✅ `DepthBatchInferencer`

Processes a full directory of `.png` images to generate `.npy` depth maps.

#### 🔧 Usage:

```python
from pathlib import Path
from modules.inference import DepthBatchInferencer

inferencer = DepthBatchInferencer(
    input_dir=Path("datasets/lab_scene_kinect_xyz/rgb"),
    output_dir=Path("datasets/lab_scene_kinect_xyz/depth_mono")
)
inferencer.run()
```

* **Output:** Depth maps saved in `.npy` format (meters)
* **Inference device:** GPU if available, otherwise CPU

---

### ✅ `DepthBatchInferencerNode` (ROS 2)

ROS 2 node version of the batch inferencer for integration in SLAM pipelines.

#### 🚀 Launch via terminal:

```bash
ros2 run sibgrapi2025_slam depth_batch_inferencer_node
```

Or use a launch file that includes:

```python
Node(
    package='sibgrapi2025_slam',
    executable='depth_batch_inferencer_node',
    name='depth_batch_inferencer_node',
    parameters=[{
        "input_dir": "datasets/scene/rgb",
        "output_dir": "datasets/scene/depth_mono"
    }]
)
```

---

## 📚 Related

You can use the generated `.npy` depth maps with the `MultiwayReconstructor` class in:

```text
modules.reconstruction
```

---

## 📝 Notes

* Output depth values are in **meters** (`float32`)
* RGB images must be `.png` format
* ROS 2 node terminates after processing all frames (non-continuous)
