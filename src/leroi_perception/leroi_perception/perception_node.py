import json
import threading
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

        # Preview/status parameters
        self.declare_parameter("preview_rate_hz", 30.0)
        self.declare_parameter("preview_stream_fps", 15.0)
        self.declare_parameter("status_rate_hz", 2.0)

        # YOLO parameters
        self.declare_parameter("enable_yolo", False)
        self.declare_parameter("yolo_rate_hz", 2.0)
        self.declare_parameter("yolo_model_path", "")
        self.declare_parameter("yolo_confidence", 0.35)
        self.declare_parameter("yolo_image_size", 320)
        self.declare_parameter("yolo_device", "cpu")  # cpu for now; later "0" or TensorRT
        self.declare_parameter("yolo_use_half", False)

        # Web preview parameters
        self.declare_parameter("enable_preview_stream", True)
        self.declare_parameter("preview_port", 8080)
        self.declare_parameter("preview_jpeg_quality", 60)
        self.declare_parameter("draw_detections", True)

        # Read camera parameters
        self.camera_name = str(self.get_parameter("camera_name").value)
        self.camera_type = str(self.get_parameter("camera_type").value)
        self.camera_index = int(self.get_parameter("camera_index").value)
        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.camera_fps = int(self.get_parameter("camera_fps").value)
        self.oak_mxid = str(self.get_parameter("oak_mxid").value)

        # Read preview/status parameters
        self.preview_rate_hz = float(self.get_parameter("preview_rate_hz").value)
        self.preview_stream_fps = float(self.get_parameter("preview_stream_fps").value)
        self.status_rate_hz = float(self.get_parameter("status_rate_hz").value)

        # Read YOLO parameters
        self.enable_yolo = bool(self.get_parameter("enable_yolo").value)
        self.yolo_rate_hz = float(self.get_parameter("yolo_rate_hz").value)
        self.yolo_model_path = str(self.get_parameter("yolo_model_path").value)
        self.yolo_confidence = float(self.get_parameter("yolo_confidence").value)
        self.yolo_image_size = int(self.get_parameter("yolo_image_size").value)
        self.yolo_device = str(self.get_parameter("yolo_device").value)
        self.yolo_use_half = bool(self.get_parameter("yolo_use_half").value)

        # Read web preview parameters
        self.enable_preview_stream = bool(
            self.get_parameter("enable_preview_stream").value
        )
        self.preview_port = int(self.get_parameter("preview_port").value)
        self.preview_jpeg_quality = int(
            self.get_parameter("preview_jpeg_quality").value
        )
        self.draw_detections = bool(self.get_parameter("draw_detections").value)

        self._validate_parameters()

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
                stream_fps=self.preview_stream_fps,
            )
            self.preview_streamer.start()
            self.get_logger().info(
                f"Preview stream enabled at http://0.0.0.0:{self.preview_port}"
            )

        # Locks
        self.frame_lock = threading.Lock()
        self.detection_lock = threading.Lock()

        # Shared camera frame state
        self.latest_frame = None
        self.latest_frame_id = 0

        # Shared detection state
        self.latest_detections = []
        self.latest_yolo_frame_id = 0
        self.last_processed_yolo_frame_id = 0

        # Counters
        self.frame_count = 0
        self.dropped_frame_count = 0
        self.yolo_count = 0
        self.yolo_error_count = 0

        # FPS tracking
        self.fps_last_time = time.time()
        self.fps_last_frame_count = 0
        self.fps_last_yolo_count = 0

        self.camera_fps_actual = 0.0
        self.yolo_fps_actual = 0.0

        # Threads
        self.running = True

        self.camera_thread = threading.Thread(
            target=self.camera_loop,
            name=f"{self.camera_name}_camera_thread",
            daemon=True,
        )
        self.camera_thread.start()

        self.yolo_thread = None

        if self.enable_yolo:
            self.yolo_thread = threading.Thread(
                target=self.yolo_loop,
                name=f"{self.camera_name}_yolo_thread",
                daemon=True,
            )
            self.yolo_thread.start()

        # ROS timers
        self.preview_timer = self.create_timer(
            1.0 / self.preview_rate_hz,
            self.preview_timer_callback,
        )

        self.status_timer = self.create_timer(
            1.0 / self.status_rate_hz,
            self.status_timer_callback,
        )

        self.get_logger().info("Perception node started.")
        self.get_logger().info(f"Camera: {self.camera_name} ({self.camera_type})")
        self.get_logger().info(f"Camera target FPS: {self.camera_fps}")
        self.get_logger().info(f"Preview target FPS: {self.preview_rate_hz}")
        self.get_logger().info(f"Preview stream FPS: {self.preview_stream_fps}")
        self.get_logger().info(f"YOLO enabled: {self.enable_yolo}")
        self.get_logger().info(f"YOLO target FPS: {self.yolo_rate_hz}")
        self.get_logger().info(f"YOLO device: {self.yolo_device}")
        self.get_logger().info(f"YOLO image size: {self.yolo_image_size}")
        self.get_logger().info(f"YOLO model: {self.yolo_model_path}")
        self.get_logger().info(f"Status topic: {self.status_topic}")
        self.get_logger().info(f"Detection topic: {self.detection_topic}")

    def _validate_parameters(self):
        if self.camera_fps <= 0:
            raise ValueError("camera_fps must be greater than 0")

        if self.preview_rate_hz <= 0.0:
            raise ValueError("preview_rate_hz must be greater than 0")

        if self.preview_stream_fps <= 0.0:
            raise ValueError("preview_stream_fps must be greater than 0")

        if self.status_rate_hz <= 0.0:
            raise ValueError("status_rate_hz must be greater than 0")

        if self.yolo_rate_hz <= 0.0:
            raise ValueError("yolo_rate_hz must be greater than 0")

    def camera_loop(self):
        target_period = 1.0 / float(self.camera_fps)

        while self.running:
            start_time = time.time()

            frame = self.camera.read()

            if frame is None:
                with self.frame_lock:
                    self.dropped_frame_count += 1
            else:
                with self.frame_lock:
                    self.latest_frame = frame
                    self.frame_count += 1
                    self.latest_frame_id = self.frame_count

            elapsed = time.time() - start_time
            sleep_time = target_period - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    def yolo_loop(self):
        target_period = 1.0 / float(self.yolo_rate_hz)

        while self.running:
            start_time = time.time()

            if self.detector is None:
                time.sleep(target_period)
                continue

            with self.frame_lock:
                if self.latest_frame is None:
                    frame = None
                    frame_id = 0
                else:
                    frame = self.latest_frame.copy()
                    frame_id = self.latest_frame_id

            if frame is None:
                time.sleep(0.01)
                continue

            # Avoid processing the same camera frame multiple times.
            if frame_id == self.last_processed_yolo_frame_id:
                time.sleep(0.005)
                continue

            self.last_processed_yolo_frame_id = frame_id

            try:
                detections = self.detector.detect(frame)
            except Exception as exc:
                with self.detection_lock:
                    self.yolo_error_count += 1

                self.get_logger().error(f"YOLO inference failed: {exc}")
                time.sleep(0.05)
                continue

            with self.detection_lock:
                self.latest_detections = detections
                self.latest_yolo_frame_id = frame_id
                self.yolo_count += 1
                yolo_count = self.yolo_count

            packet = {
                "stamp_ns": self.get_clock().now().nanoseconds,
                "camera_name": self.camera_name,
                "camera_type": self.camera_type,
                "frame_id": frame_id,
                "image_width": int(frame.shape[1]),
                "image_height": int(frame.shape[0]),
                "yolo_enabled": self.enable_yolo,
                "yolo_device": self.yolo_device,
                "yolo_image_size": self.yolo_image_size,
                "yolo_count": yolo_count,
                "detection_count": len(detections),
                "detections": detections,
            }

            msg = String()
            msg.data = json.dumps(packet)
            self.detection_pub.publish(msg)

            elapsed = time.time() - start_time
            sleep_time = target_period - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    def preview_timer_callback(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return

            frame = self.latest_frame.copy()
            frame_id = self.latest_frame_id

        with self.detection_lock:
            detections = list(self.latest_detections)
            yolo_frame_id = self.latest_yolo_frame_id

        if self.draw_detections:
            self.draw_detection_overlay(
                frame=frame,
                detections=detections,
                frame_id=frame_id,
                yolo_frame_id=yolo_frame_id,
            )

        if self.preview_streamer is not None:
            self.preview_streamer.update_frame(frame, frame_id=frame_id)

    def status_timer_callback(self):
        self.update_fps_estimates()

        with self.frame_lock:
            frame_count = self.frame_count
            latest_frame_id = self.latest_frame_id
            dropped_frame_count = self.dropped_frame_count

        with self.detection_lock:
            detection_count = len(self.latest_detections)
            yolo_frame_id = self.latest_yolo_frame_id
            yolo_count = self.yolo_count
            yolo_error_count = self.yolo_error_count

        frame_lag = max(0, latest_frame_id - yolo_frame_id)

        msg = String()
        msg.data = json.dumps(
            {
                "stamp_ns": self.get_clock().now().nanoseconds,
                "camera_name": self.camera_name,
                "camera_type": self.camera_type,
                "status": "running",
                "camera_fps_actual": self.camera_fps_actual,
                "yolo_fps_actual": self.yolo_fps_actual,
                "frame_count": frame_count,
                "latest_frame_id": latest_frame_id,
                "dropped_frame_count": dropped_frame_count,
                "yolo_enabled": self.enable_yolo,
                "yolo_device": self.yolo_device,
                "yolo_count": yolo_count,
                "yolo_error_count": yolo_error_count,
                "latest_yolo_frame_id": yolo_frame_id,
                "yolo_frame_lag": frame_lag,
                "latest_detection_count": detection_count,
                "preview_enabled": self.enable_preview_stream,
                "preview_stream_fps": self.preview_stream_fps,
            }
        )
        self.status_pub.publish(msg)

        self.get_logger().info(
            f"{self.camera_name}: "
            f"cam_fps={self.camera_fps_actual:.1f}, "
            f"yolo_fps={self.yolo_fps_actual:.1f}, "
            f"frames={frame_count}, "
            f"dropped={dropped_frame_count}, "
            f"yolo_runs={yolo_count}, "
            f"yolo_errors={yolo_error_count}, "
            f"frame_lag={frame_lag}, "
            f"detections={detection_count}"
        )

    def update_fps_estimates(self):
        now = time.time()
        dt = now - self.fps_last_time

        if dt < 0.5:
            return

        with self.frame_lock:
            frame_count = self.frame_count

        with self.detection_lock:
            yolo_count = self.yolo_count

        frame_delta = frame_count - self.fps_last_frame_count
        yolo_delta = yolo_count - self.fps_last_yolo_count

        self.camera_fps_actual = frame_delta / dt
        self.yolo_fps_actual = yolo_delta / dt

        self.fps_last_time = now
        self.fps_last_frame_count = frame_count
        self.fps_last_yolo_count = yolo_count

    def draw_detection_overlay(self, frame, detections, frame_id, yolo_frame_id):
        for det in detections:
            x = int(det.get("bbox_x", 0))
            y = int(det.get("bbox_y", 0))
            w = int(det.get("bbox_width", 0))
            h = int(det.get("bbox_height", 0))

            class_name = str(det.get("class_name", "object"))
            confidence = float(det.get("confidence", 0.0))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            label = f"{class_name} {confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        frame_lag = max(0, frame_id - yolo_frame_id)

        cv2.putText(
            frame,
            f"{self.camera_name} | cam {self.camera_fps_actual:.1f} FPS | frame {frame_id}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"YOLO {self.yolo_fps_actual:.1f} FPS | frame {yolo_frame_id} | lag {frame_lag} | det {len(detections)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    def destroy_node(self):
        self.running = False

        if hasattr(self, "camera_thread") and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)

        if self.yolo_thread is not None and self.yolo_thread.is_alive():
            self.yolo_thread.join(timeout=1.0)

        if self.preview_streamer is not None:
            self.preview_streamer.stop()
            self.preview_streamer = None

        if self.camera is not None:
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