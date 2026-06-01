import json
import time
from typing import Optional

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from leroi_perception.camera_sources import create_camera_source
from leroi_perception.preview_streamer import MjpegPreviewStreamer
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
        self.declare_parameter("yolo_device", "0")  # "0" for CUDA GPU, "cpu" for CPU
        self.declare_parameter("yolo_use_half", False)

        # Preview stream parameters
        self.declare_parameter("enable_preview_stream", True)
        self.declare_parameter("preview_port", 8080)
        self.declare_parameter("preview_jpeg_quality", 80)
        self.declare_parameter("draw_detections", True)

        # Read camera parameters
        self.camera_name = str(self.get_parameter("camera_name").value)
        self.camera_type = str(self.get_parameter("camera_type").value)
        self.camera_index = int(self.get_parameter("camera_index").value)
        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.camera_fps = int(self.get_parameter("camera_fps").value)
        self.oak_mxid = str(self.get_parameter("oak_mxid").value)

        # Read runtime parameters
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.publish_every_frame = bool(self.get_parameter("publish_every_frame").value)

        # Read YOLO parameters
        self.enable_yolo = bool(self.get_parameter("enable_yolo").value)
        self.yolo_model_path = str(self.get_parameter("yolo_model_path").value)
        self.yolo_confidence = float(self.get_parameter("yolo_confidence").value)
        self.yolo_image_size = int(self.get_parameter("yolo_image_size").value)
        self.yolo_device = str(self.get_parameter("yolo_device").value)
        self.yolo_use_half = bool(self.get_parameter("yolo_use_half").value)

        # Read preview parameters
        self.enable_preview_stream = bool(
            self.get_parameter("enable_preview_stream").value
        )
        self.preview_port = int(self.get_parameter("preview_port").value)
        self.preview_jpeg_quality = int(
            self.get_parameter("preview_jpeg_quality").value
        )
        self.draw_detections = bool(self.get_parameter("draw_detections").value)

        # ROS topics
        self.status_topic = f"/perception/{self.camera_name}/status"
        self.detection_topic = f"/perception/{self.camera_name}/detections"

        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.detection_pub = self.create_publisher(String, self.detection_topic, 10)

        # Camera source
        self.camera = create_camera_source(
            camera_type=self.camera_type,
            camera_index=self.camera_index,
            width=self.camera_width,
            height=self.camera_height,
            fps=self.camera_fps,
            oak_mxid=self.oak_mxid,
        )

        # YOLO detector
        self.detector: Optional[YoloDetector] = None

        if self.enable_yolo:
            self.detector = YoloDetector(
                model_path=self.yolo_model_path,
                confidence_threshold=self.yolo_confidence,
                image_size=self.yolo_image_size,
                device=self.yolo_device,
                use_half=self.yolo_use_half,
            )

        # Web preview streamer
        self.preview_streamer: Optional[MjpegPreviewStreamer] = None

        if self.enable_preview_stream:
            self.preview_streamer = MjpegPreviewStreamer(
                host="0.0.0.0",
                port=self.preview_port,
                jpeg_quality=self.preview_jpeg_quality,
            )
            self.preview_streamer.start()
            self.get_logger().info(
                f"Preview stream enabled at http://0.0.0.0:{self.preview_port}"
            )

        # Internal counters
        self.frame_count = 0
        self.dropped_frame_count = 0
        self.last_log_time = time.time()
        self.last_status_time = 0.0

        if self.publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be greater than 0")

        timer_period = 1.0 / self.publish_rate_hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info("Perception node started.")
        self.get_logger().info(f"Camera name: {self.camera_name}")
        self.get_logger().info(f"Camera type: {self.camera_type}")
        self.get_logger().info(f"Camera index: {self.camera_index}")
        self.get_logger().info(f"Camera resolution: {self.camera_width}x{self.camera_height}")
        self.get_logger().info(f"Camera FPS: {self.camera_fps}")
        self.get_logger().info(f"Status topic: {self.status_topic}")
        self.get_logger().info(f"Detection topic: {self.detection_topic}")
        self.get_logger().info(f"YOLO enabled: {self.enable_yolo}")
        self.get_logger().info(f"YOLO model path: {self.yolo_model_path}")
        self.get_logger().info(f"Preview enabled: {self.enable_preview_stream}")

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

        if self.preview_streamer is not None:
            preview_frame = frame.copy()

            if self.draw_detections:
                self.draw_detection_overlay(preview_frame, detections)

            self.preview_streamer.update_frame(preview_frame)

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

    def draw_detection_overlay(self, frame, detections):
        for det in detections:
            x = int(det.get("bbox_x", 0))
            y = int(det.get("bbox_y", 0))
            w = int(det.get("bbox_width", 0))
            h = int(det.get("bbox_height", 0))

            class_name = str(det.get("class_name", "object"))
            confidence = float(det.get("confidence", 0.0))

            x2 = x + w
            y2 = y + h

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)

            label = f"{class_name} {confidence:.2f}"
            label_y = max(20, y - 8)

            cv2.putText(
                frame,
                label,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"{self.camera_name} | frame {self.frame_count}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"detections: {len(detections)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

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
                "yolo_enabled": self.enable_yolo,
                "preview_enabled": self.enable_preview_stream,
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
        if hasattr(self, "preview_streamer") and self.preview_streamer is not None:
            self.preview_streamer.stop()
            self.preview_streamer = None

        if hasattr(self, "camera") and self.camera is not None:
            self.camera.close()
            self.camera = None

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