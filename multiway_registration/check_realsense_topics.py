import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo

EXPECTED_TOPICS = {
    '/camera/color/image_raw': Image,
    '/camera/aligned_depth_to_color/image_raw': Image,
    '/camera/color/camera_info': CameraInfo,
}


class RealSenseTopicChecker(Node):
    def __init__(self):
        super().__init__('realsense_topic_checker')
        self.declared = set()

        self.get_logger().info("Checking RealSense topics...")

        for topic, msg_type in EXPECTED_TOPICS.items():
            self.create_subscription(
                msg_type,
                topic,
                self._make_callback(topic),
                qos_profile_sensor_data
            )

    def _make_callback(self, topic_name):
        def callback(msg):
            if topic_name not in self.declared:
                self.declared.add(topic_name)
                self.get_logger().info(
                    f"[✓] Receiving messages on: {topic_name}"
                )
            if len(self.declared) == len(EXPECTED_TOPICS):
                self.get_logger().info("All expected topics are active ✅")
                rclpy.shutdown()
        return callback


def main():
    rclpy.init()
    node = RealSenseTopicChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
