import os
import cv2
import json
import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions

INPUT_BAG = 'datasets/lab_scene_01'
OUTPUT_RGB = os.path.join(INPUT_BAG, 'rgb')
OUTPUT_DEPTH = os.path.join(INPUT_BAG, 'depth_d435')
INTRINSICS_FILE = os.path.join(INPUT_BAG, 'intrinsics.json')

COLOR_TOPIC = '/camera/color/image_raw'
DEPTH_TOPIC = '/camera/aligned_depth_to_color/image_raw'
INFO_TOPIC = '/camera/color/camera_info'

os.makedirs(OUTPUT_RGB, exist_ok=True)
os.makedirs(OUTPUT_DEPTH, exist_ok=True)

bridge = CvBridge()
rclpy.init()

reader = SequentialReader()
storage_options = StorageOptions(uri=INPUT_BAG, storage_id='sqlite3')
converter_options = ConverterOptions(
    input_serialization_format='cdr', output_serialization_format='cdr'
)
reader.open(storage_options, converter_options)

camera_intrinsics = None
rgb_count = 0
depth_count = 0

while reader.has_next():
    topic, data, t = reader.read_next()
    if topic == COLOR_TOPIC:
        msg = deserialize_message(data, Image)
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        filename = os.path.join(OUTPUT_RGB, f'frame_{rgb_count:04d}.png')
        cv2.imwrite(filename, cv_img)
        rgb_count += 1

    elif topic == DEPTH_TOPIC:
        msg = deserialize_message(data, Image)
        cv_depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        filename = os.path.join(OUTPUT_DEPTH, f'depth_{depth_count:04d}.png')
        cv2.imwrite(filename, cv_depth)
        depth_count += 1

    elif topic == INFO_TOPIC and camera_intrinsics is None:
        msg = deserialize_message(data, CameraInfo)
        camera_intrinsics = {
            'width': msg.width,
            'height': msg.height,
            'K': list(msg.k),
            'D': list(msg.d),
            'distortion_model': msg.distortion_model
        }

if camera_intrinsics:
    with open(INTRINSICS_FILE, 'w') as f:
        json.dump(camera_intrinsics, f, indent=4)

print(f"[✓] RGB frames: {rgb_count} | Depth frames: {depth_count}")
print(f"[✓] Intrinsics saved to: {INTRINSICS_FILE}")
