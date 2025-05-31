import os
import json
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image, CameraInfo
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

import cv2
from cv_bridge import CvBridge


def extract_images_from_ros2_bag(bag_path, output_dir):
    os.makedirs(os.path.join(output_dir, 'color'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'depth'), exist_ok=True)

    rclpy.init()
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    bridge = CvBridge()
    frame_count = 0
    intrinsics_written = False

    type_map = {}

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in type_map:
            type_map[topic] = get_message(topic)

        msg_type = get_message(topic)
        msg = deserialize_message(data, msg_type)

        if topic.endswith("color/image_raw"):
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            fname = f'frame_{frame_count:04d}.png'
            cv2.imwrite(os.path.join(output_dir, 'color', fname), cv_image)
            print(f'[RGB] Saved {fname}')

        elif topic.endswith("aligned_depth_to_color/image_raw"):
            cv_depth = bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
            fname = f'frame_{frame_count:04d}.png'
            cv2.imwrite(os.path.join(output_dir, 'depth', fname), cv_depth)
            print(f'[DEPTH] Saved {fname}')
            frame_count += 1

        elif topic.endswith("camera_info") and not intrinsics_written:
            intrinsics = {
                'fx': msg.k[0],
                'fy': msg.k[4],
                'cx': msg.k[2],
                'cy': msg.k[5],
                'width': msg.width,
                'height': msg.height,
            }
            with open(os.path.join(output_dir, 'intrinsics.json'), 'w') as f:
                json.dump(intrinsics, f, indent=4)
            intrinsics_written = True
            print('[INFO] Camera intrinsics saved.')

    rclpy.shutdown()
    print(f"[DONE] Extracted {frame_count} frames.")


if __name__ == "__main__":
    bag_path = 'datasets/lab_scene_02'  # path to folder with .db3 and .yaml
    output_dir = 'datasets/lab_scene_02'
    extract_images_from_ros2_bag(bag_path, output_dir)
