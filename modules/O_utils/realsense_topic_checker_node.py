"""
ROS 2 node to verify if required RealSense topics are active.

This utility node subscribes to expected RealSense topics and logs whether
each one is being published correctly. Shuts down automatically if all are
active.
"""

from typing import Callable

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo


class RealSenseTopicChecker(Node):
    """
    Subscribes to RealSense topics and checks if they are active.
    """

    EXPECTED_TOPICS = {
        '/camera/color/image_raw': Image,
        '/camera/aligned_depth_to_color/image_raw': Image,
        '/camera/color/camera_info': CameraInfo,
    }

    def __init__(self) -> None:
        super().__init__('realsense_topic_checker')

        self._received_topics = set()
        self._total_expected = len(self.EXPECTED_TOPICS)

        self.get_logger().info("Waiting for RealSense topics...")

        for topic_name, msg_type in self.EXPECTED_TOPICS.items():
            self.create_subscription(
                msg_type=msg_type,
                topic=topic_name,
                callback=self._make_callback(topic_name),
                qos_profile=qos_profile_sensor_data
            )

    def _make_callback(self, topic_name: str) -> Callable:
        """
        Creates a callback that logs the first received message per topic.

        Args:
            topic_name (str): The topic this callback is bound to.

        Returns:
            Callable: The callback function.
        """
        def callback(msg) -> None:
            if topic_name not in self._received_topics:
                self._received_topics.add(topic_name)
                self.get_logger().info(
                    f"[✓] Topic active: {topic_name}"
                )
            if len(self._received_topics) == self._total_expected:
                self.get_logger().info("✅ All expected topics are active.")
                rclpy.shutdown()
        return callback


def main() -> None:
    """
    ROS 2 node entry point.
    """
    rclpy.init()
    try:
        node = RealSenseTopicChecker()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("[Shutdown] Interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
