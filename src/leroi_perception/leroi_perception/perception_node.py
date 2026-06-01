import json
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from leroi_perception.camera_sources import create_camera_source
from leroi_perception.yolo_detector import YoloDetector


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        # Camera parameters
        self.declare_parameter("camera_name", "oak_front")
        self.declare_parameter("camera_type", "oak")  # oak, usb, fake
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_fps", 30)
        self.declare_parameter("oak_mxid", "")

        # Runtime parameters
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("publish_every_frame", True)

        # YOLO parameters
        self.declare_parameter("enable_yolo", False)
        self.declare_parameter("yolo_model_path", "")
        self.declare_parameter("yolo_confidence", 0.35)
        self.declare_parameter("yolo_image_size", 640)
        self.declare_parameter("yolo_device", "0")
        self.declare_parameter("yolo_use_half", False)

        self.camera_name = str(self.get_parameter("camera_name").value)
        self.camera_type = str(self.get_parameter("camera_type").value)
        self.camera_index = int(self.get_parameter("camera_index").value)
        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.camera_fps = int(self.get_parameter("camera_fps").value)
        self.oak_mxid = str(self.get_parameter("oak_mxid").value)

        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.publish_every_frame = bool(self.get_parameter("publish_every_frame").value)

        self.enable_yolo = bool(self.get_parameter("enable_yolo").value)
        self.yolo_model_path = str(self.get_parameter("yolo_model_path").value)
        self.yolo_confidence = float(self.get_parameter("yolo_confidence").value)
        self.yolo_image_size = int(self.get_parameter("yolo_image_size").value)
        self.yolo_device = str(self.get_parameter("yolo_device").value)
        self.yolo_use_half = bool(self.get_parameter("yolo_use_half").value)

        self.status_topic = f"/perception/{self.camera_name}/status"
        self.detection_topic = f"/perception/{self.camera_name}/detections"

        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.detection_pub = self.create_publisher(String, self.detection_topic, 10)

        self.camera = create_camera_source(
            camera_type=self.camera_type,
            camera_index=self.camera_index,
            width=self.camera_width,
            height=self.camera_height,
            fps=self.camera_fps,
            oak_mxid=self.oak_mxid,
        )

        self.detector: Optional[YoloDetector] = None

        if self.enable_yolo:
            self.detector = YoloDetector(
                model_path=self.yolo_model_path,
                confidence_threshold=self.yolo_confidence,
                image_size=self.yolo_image_size,
                device=self.yolo_device,
                use_half=self.yolo_use_half,
            )

        self.frame_count = 0
        self.dropped_frame_count = 0
        self.last_log_time = time.time()
        self.last_status_time = time.time()

        timer_period = 1.0 / self.publish_rate_hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info("Perception node started.")
        self.get_logger().info(f"Camera name: {self.camera_name}")
        self.get_logger().info(f"Camera type: {self.camera_type}")
        self.get_logger().info(f"Camera index: {self.camera_index}")
        self.get_logger().info(f"Detection topic: {self.detection_topic}")
        self.get_logger().info(f"YOLO enabled: {self.enable_yolo}")
        self.get_logger().info(f"YOLO model path: {self.yolo_model_path}")

    def timer_callback(self):
        frame = self.camera.read()

        if frame is None:
            self.dropped_frame_count += 1
            self.publish_status("no_frame")
            return

        self.frame_count += 1

        detections = []

        if self.detector is not None:
            try:
                detections = self.detector.detect(frame)
            except Exception as exc:
                self.get_logger().error(f"YOLO inference failed: {exc}")
                self.publish_status("yolo_error")
                return

        if self.publish_every_frame or detections:
            packet = {
                "stamp_ns": self.get_clock().now().nanoseconds,
                "camera_name": self.camera_name,
                "camera_type": self.camera_type,
                "frame_id": self.frame_count,
                "image_width": int(frame.shape[1]),
                "image_height": int(frame.shape[0]),
                "yolo_enabled": self.enable_yolo,
                "detection_count": len(detections),
                "detections": detections,
            }

            detection_msg = String()
            detection_msg.data = json.dumps(packet)
            self.detection_pub.publish(detection_msg)

        self.publish_status("running")
        self.log_periodically(len(detections))

    def publish_status(self, status: str):
        now = time.time()

        # Do not spam status too quickly.
        if now - self.last_status_time < 0.5:
            return

        msg = String()
        msg.data = json.dumps(
            {
                "stamp_ns": self.get_clock().now().nanoseconds,
                "camera_name": self.camera_name,
                "camera_type": self.camera_type,
                "status": status,
                "frame_count": self.frame_count,
                "dropped_frame_count": self.dropped_frame_count,
            }
        )

        self.status_pub.publish(msg)
        self.last_status_time = now

    def log_periodically(self, detection_count: int):
        now = time.time()

        if now - self.last_log_time < 2.0:
            return

        self.get_logger().info(
            f"{self.camera_name}: frames={self.frame_count}, "
            f"dropped={self.dropped_frame_count}, "
            f"detections={detection_count}"
        )

        self.last_log_time = now

    def destroy_node(self):
        if hasattr(self, "camera") and self.camera is not None:
            self.camera.close()

        super().destroy_node()


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