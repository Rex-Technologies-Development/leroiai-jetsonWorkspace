import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("enable_fake_detections", True)

        self.publish_rate_hz = (
            self.get_parameter("publish_rate_hz")
            .get_parameter_value()
            .double_value
        )

        self.enable_fake_detections = (
            self.get_parameter("enable_fake_detections")
            .get_parameter_value()
            .bool_value
        )

        self.status_pub = self.create_publisher(
            String,
            "/perception/status",
            10,
        )

        self.detection_pub = self.create_publisher(
            String,
            "/perception/detections",
            10,
        )

        timer_period = 1.0 / self.publish_rate_hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.frame_count = 0

        self.get_logger().info("Perception node started.")
        self.get_logger().info(f"Publish rate: {self.publish_rate_hz} Hz")
        self.get_logger().info(f"Fake detections enabled: {self.enable_fake_detections}")

    def timer_callback(self):
        self.frame_count += 1

        status_msg = String()
        status_msg.data = "perception_node alive"
        self.status_pub.publish(status_msg)

        if self.enable_fake_detections:
            detection_packet = {
                "frame_id": self.frame_count,
                "source": "fake_perception",
                "detections": [
                    {
                        "class_name": "block",
                        "class_id": 0,
                        "confidence": 0.92,
                        "x": 1.2,
                        "y": 0.15,
                        "z": 2.4,
                        "bbox_x": 320,
                        "bbox_y": 180,
                        "bbox_width": 80,
                        "bbox_height": 60,
                    }
                ],
            }

            detection_msg = String()
            detection_msg.data = json.dumps(detection_packet)
            self.detection_pub.publish(detection_msg)

        self.get_logger().info(
            f"Published perception update. frame_count={self.frame_count}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = PerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Perception node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()