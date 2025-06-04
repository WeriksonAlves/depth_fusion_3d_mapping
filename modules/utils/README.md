## 📁 `modules/utils/README.md`

### 🔧 Overview

This subpackage provides auxiliary utilities for:

* Checking sensor topics in ROS 2;
* Real-time point cloud visualization using Open3D;
* Publishing `PointCloud2` from RGB-D or monocular depth data.

---

### 🧪 RealSense Topic Verification

Before starting data acquisition with a RealSense camera, use the topic checker to ensure that all required topics are being published correctly.

#### 📦 Direct execution:

```bash
ros2 run sibgrapi2025_slam realsense_topic_checker_node
```

#### 🚀 Using a launch file:

```bash
ros2 launch sibgrapi2025_slam check_realsense_topics.launch.py
```

#### ✅ Expected topics:

* `/camera/color/image_raw` (RGB image)
* `/camera/aligned_depth_to_color/image_raw` (depth aligned to RGB)
* `/camera/color/camera_info` (camera intrinsic parameters)

---

### 👁️ Real-Time Visualization with Open3D

Visualize `PointCloud2` messages in real time, directly from ROS 2 topics using an interactive Open3D window.

#### 📦 Direct execution:

```bash
ros2 run sibgrapi2025_slam live_point_cloud_visualizer_node
```

#### 🚀 Using a launch file:

```bash
ros2 launch sibgrapi2025_slam visualize_pointcloud.launch.py
```

#### ⚙️ Optional parameter:

```bash
ros2 launch sibgrapi2025_slam visualize_pointcloud.launch.py \
  pointcloud_topic:=/my_custom_cloud
```

---

### 🛰️ Integrated Pipeline Example

```python
"""
Example: publish and visualize a reconstructed point cloud.
"""

import open3d as o3d
import rclpy
from rclpy.node import Node
from modules.utils import (
    PointCloudProcessor,
    LivePointCloudVisualizer
)


class TestPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__('test_publisher')
        self.processor = PointCloudProcessor(
            ros_node=self,
            frame_id='map',
            topic='/o3d_points'
        )
        self.visualizer = LivePointCloudVisualizer()

        cloud = o3d.io.read_point_cloud("point_clouds/merged_map.ply")
        self.processor.publish_point_cloud(cloud)
        self.visualizer.update(cloud)


def main():
    rclpy.init()
    try:
        node = TestPublisherNode()
        rclpy.spin_once(node, timeout_sec=2.0)
    finally:
        node.visualizer.close()
        node.destroy_node()
        rclpy.shutdown()
```

---

### 📦 `PointCloudProcessor`: Main Features

| Method                  | Description                                                             |
| ----------------------- | ----------------------------------------------------------------------- |
| `create_point_cloud()`  | Generates a point cloud from RGB and depth images using Open3D          |
| `filter_point_cloud()`  | Applies voxel grid downsampling and outlier removal                     |
| `save_point_cloud()`    | Saves the cloud as `.ply` or `.pcd`                                     |
| `publish_point_cloud()` | Publishes the cloud to ROS 2 as a `sensor_msgs/msg/PointCloud2` message |

---

### 📝 Notes

* RGB and depth images must be aligned and have the same resolution.
* Colors are extracted from the `rgb` field and visualized in Open3D.
* The node shuts down automatically when using `spin_once()`.
