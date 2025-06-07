## 📁 `modules/reconstruction/README.md`

### 🔧 Overview

This submodule provides the core components for 3D reconstruction pipelines based on RGB-D or monocular inputs. It supports:

* Frame loading from RGB and depth images
* Multiway pose graph construction and optimization
* Point cloud merging and publishing via ROS 2
* Execution in both offline and ROS-integrated modes

---

### 📦 Available Classes (Public API)

| Class / Node                | Description                                           |
| --------------------------- | ----------------------------------------------------- |
| `FrameLoader`               | Loads and preprocesses RGB-D or monocular frame pairs |
| `PoseGraphBuilder`          | Builds a pose graph using pairwise ICP registration   |
| `GraphOptimizer`            | Performs global optimization on pose graphs           |
| `MapMergerPublisher`        | Merges and saves point clouds                         |
| `MapMergerPublisherNode`    | ROS 2 node that publishes the merged cloud            |
| `MultiwayReconstructor`     | Standalone reconstruction pipeline                    |
| `MultiwayReconstructorNode` | ROS 2-compatible reconstruction node                  |

---

## ✅ Example of Usage in `main.py`

```python
"""
Main script for triggering 3D reconstruction from RGB-D or monocular data.

Supports both ROS 2 node execution and offline reconstruction using the
public API defined in modules.reconstruction.__init__.py
"""

import rclpy
from rclpy.node import Node
from modules.reconstruction import (
    MultiwayReconstructor,
    MultiwayReconstructorNode
)


def run_standalone_pipeline() -> None:
    """
    Runs the reconstruction pipeline without ROS 2.
    """
    from pathlib import Path
    dataset_path = Path("datasets/lab_scene_kinect_xyz")
    reconstructor = MultiwayReconstructor(
        dataset_dir=dataset_path,
        mode="real",  # or "mono"
        ros_node=None,
        voxel_size=0.02,
        depth_scale=5000.0,
        depth_trunc=4.0
    )
    reconstructor.run()


def run_ros_pipeline() -> None:
    """
    Runs the reconstruction pipeline as a ROS 2 node.
    """
    rclpy.init()
    try:
        node = MultiwayReconstructorNode()
        rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("[Shutdown] Interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    ros_mode = True
    if ros_mode:
        run_ros_pipeline()
    else:
        run_standalone_pipeline()


if __name__ == '__main__':
    main()
```

---

## 🧠 What This Example Demonstrates

| Component                   | Explanation                                                                |
| --------------------------- | -------------------------------------------------------------------------- |
| `MultiwayReconstructor`     | Used for non-ROS standalone usage (batch execution)                        |
| `MultiwayReconstructorNode` | Enables integration with ROS 2 and publishing of `PointCloud2`             |
| `rclpy.spin_once()`         | Executes the pipeline and shuts down the node immediately after completion |
| `ros_mode` flag             | Switches between ROS and non-ROS modes                                     |

---

## 🛰️ When to Use ROS or Standalone Mode?

| Scenario                                 | Recommended Mode              |
| ---------------------------------------- | ----------------------------- |
| Use with OctoMap / RViz2                 | ROS (`ros_mode=True`)         |
| Offline `.ply` generation and testing    | Standalone (`ros_mode=False`) |
| Integration in embedded ROS environments | ROS                           |
| Development and debugging without ROS    | Standalone                    |
