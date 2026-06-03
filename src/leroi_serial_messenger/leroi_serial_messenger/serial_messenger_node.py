import json
import math
import queue
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import serial
import serial.tools.list_ports


@dataclass
class TxPacket:
    payload_type: int
    payload_text: str
    source: str
    created_time: float


class SerialMessengerNode(Node):
    def __init__(self):
        super().__init__("serial_messenger_node")

        # Serial connection parameters
        self.declare_parameter("serial_port", "")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("read_timeout_sec", 0.05)
        self.declare_parameter("reconnect_period_sec", 2.0)
        self.declare_parameter(
            "auto_port_keywords",
            "VEX V5 User Port,VEX Robotics User Port,VEX Robotics",
        )

        # TX protocol parameters
        self.declare_parameter("tx_append_newline", True)
        self.declare_parameter("max_payload_bytes", 1024)
        self.declare_parameter("drop_tx_when_disconnected", True)

        # Heartbeat/debug parameters
        self.declare_parameter("send_heartbeat", True)
        self.declare_parameter("heartbeat_period_sec", 1.0)
        self.declare_parameter("status_period_sec", 1.0)
        self.declare_parameter("debug_log_rx", False)
        self.declare_parameter("debug_log_tx", True)
        self.declare_parameter("debug_log_scan", True)

        # Perception forwarding parameters
        self.declare_parameter("detection_topic", "/perception/oak_front/detections")
        self.declare_parameter("forward_detections_to_v5", True)
        self.declare_parameter("debug_log_detections", False)

        self.serial_port = str(self.get_parameter("serial_port").value)
        self.baud_rate = int(self.get_parameter("baud_rate").value)
        self.read_timeout_sec = float(self.get_parameter("read_timeout_sec").value)
        self.reconnect_period_sec = float(
            self.get_parameter("reconnect_period_sec").value
        )

        keywords_raw = str(self.get_parameter("auto_port_keywords").value)
        self.auto_port_keywords = [
            keyword.strip()
            for keyword in keywords_raw.split(",")
            if keyword.strip()
        ]

        self.tx_append_newline = bool(self.get_parameter("tx_append_newline").value)
        self.max_payload_bytes = int(self.get_parameter("max_payload_bytes").value)
        self.drop_tx_when_disconnected = bool(
            self.get_parameter("drop_tx_when_disconnected").value
        )

        self.send_heartbeat = bool(self.get_parameter("send_heartbeat").value)
        self.heartbeat_period_sec = float(
            self.get_parameter("heartbeat_period_sec").value
        )
        self.status_period_sec = float(self.get_parameter("status_period_sec").value)
        self.debug_log_rx = bool(self.get_parameter("debug_log_rx").value)
        self.debug_log_tx = bool(self.get_parameter("debug_log_tx").value)
        self.debug_log_scan = bool(self.get_parameter("debug_log_scan").value)

        self.detection_topic = str(self.get_parameter("detection_topic").value)
        self.forward_detections_to_v5 = bool(
            self.get_parameter("forward_detections_to_v5").value
        )
        self.debug_log_detections = bool(
            self.get_parameter("debug_log_detections").value
        )

        # This matches your V5-side serial framing test.
        self.sync_bytes = b"\xaa\x55\xcc\x33"

        # ROS publishers
        self.status_pub = self.create_publisher(
            String,
            "/serial_messenger/status",
            10,
        )

        self.rx_raw_pub = self.create_publisher(
            String,
            "/serial_messenger/rx_raw",
            10,
        )

        self.rx_parsed_pub = self.create_publisher(
            String,
            "/serial_messenger/rx_parsed",
            10,
        )

        self.odom_pub = self.create_publisher(
            String,
            "/v5/odom",
            10,
        )

        self.action_ack_pub = self.create_publisher(
            String,
            "/robot/action_ack",
            10,
        )

        # ROS subscribers
        self.tx_sub = self.create_subscription(
            String,
            "/serial_messenger/tx",
            self.tx_callback,
            10,
        )

        self.robot_command_sub = self.create_subscription(
            String,
            "/robot/command",
            self.robot_command_callback,
            10,
        )

        self.path_command_sub = self.create_subscription(
            String,
            "/robot/path_command",
            self.path_command_callback,
            10,
        )

        self.detection_sub = self.create_subscription(
            String,
            self.detection_topic,
            self.detection_callback,
            10,
        )

        # Runtime state
        self.serial_handle: Optional[serial.Serial] = None
        self.serial_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.tx_queue: queue.Queue[TxPacket] = queue.Queue(maxsize=100)

        self.connected = False
        self.current_port = ""
        self.last_connect_time = 0.0
        self.last_rx_time = 0.0
        self.last_tx_time = 0.0

        self.rx_line_count = 0
        self.rx_odom_count = 0
        self.rx_ack_count = 0
        self.rx_parse_error_count = 0
        self.tx_count = 0
        self.tx_error_count = 0
        self.tx_drop_disconnected_count = 0
        self.reconnect_count = 0

        self.latest_robot_pose_lock = threading.Lock()
        self.latest_robot_pose = {
            "valid": False,
            "x": 0.0,
            "y": 0.0,
            "heading": 0.0,
            "frame": 0,
            "stamp_ns": 0,
        }

        self.command_id_lock = threading.Lock()
        self.next_command_id = 1

        self.start_time = time.time()

        self.io_thread = threading.Thread(
            target=self.serial_io_loop,
            name="serial_messenger_io_thread",
            daemon=True,
        )
        self.io_thread.start()

        self.status_timer = self.create_timer(
            self.status_period_sec,
            self.publish_status,
        )

        if self.send_heartbeat:
            self.heartbeat_timer = self.create_timer(
                self.heartbeat_period_sec,
                self.heartbeat_callback,
            )
        else:
            self.heartbeat_timer = None

        self.get_logger().info("Serial messenger node started.")
        self.get_logger().info(f"serial_port: {self.serial_port or 'auto'}")
        self.get_logger().info(f"baud_rate: {self.baud_rate}")
        self.get_logger().info(f"auto_port_keywords: {self.auto_port_keywords}")
        self.get_logger().info(
            f"drop_tx_when_disconnected: {self.drop_tx_when_disconnected}"
        )
        self.get_logger().info(f"detection_topic: {self.detection_topic}")
        self.get_logger().info(
            f"forward_detections_to_v5: {self.forward_detections_to_v5}"
        )

    def allocate_command_id(self) -> int:
        with self.command_id_lock:
            command_id = self.next_command_id
            self.next_command_id += 1
            return command_id

    def tx_callback(self, msg: String):
        self.enqueue_payload(
            payload_text=msg.data,
            payload_type=1,
            source="/serial_messenger/tx",
        )

    def robot_command_callback(self, msg: String):
        """
        Accepts either a plain string command or a JSON command.

        Plain string:
          STOP

        JSON:
          {"type":"CMD","id":1,"command":"STOP"}
        """
        text = msg.data.strip()

        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                data = {
                    "type": "CMD",
                    "id": self.allocate_command_id(),
                    "command": text,
                }
        except json.JSONDecodeError:
            data = {
                "type": "CMD",
                "id": self.allocate_command_id(),
                "command": text,
            }

        if "id" not in data:
            data["id"] = self.allocate_command_id()

        if "type" not in data:
            data["type"] = "CMD"

        payload_text = json.dumps(data)

        self.enqueue_payload(
            payload_text=payload_text,
            payload_type=2,
            source="/robot/command",
        )

        self.get_logger().info(
            f"Queued robot command id={data.get('id')} type={data.get('type')}"
        )

    def path_command_callback(self, msg: String):
        """
        Expected JSON:

        {
          "type": "PATH",
          "id": 12,
          "mode": "tank",
          "points": [
            {"x": 24.0, "y": 12.0, "heading": 0.0},
            {"x": 48.0, "y": 24.0, "heading": 45.0}
          ]
        }
        """
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Invalid /robot/path_command JSON.")
            return

        if not isinstance(data, dict):
            self.get_logger().warning("/robot/path_command must be a JSON object.")
            return

        if "id" not in data:
            data["id"] = self.allocate_command_id()

        data["type"] = "PATH"

        points = data.get("points", [])
        if not isinstance(points, list) or len(points) == 0:
            self.get_logger().warning("PATH command has no target points.")
            return

        normalized_points = []

        for i, point in enumerate(points):
            try:
                if isinstance(point, dict):
                    x = float(point["x"])
                    y = float(point["y"])
                    heading = float(point.get("heading", 0.0))
                elif isinstance(point, list) or isinstance(point, tuple):
                    x = float(point[0])
                    y = float(point[1])
                    heading = float(point[2]) if len(point) >= 3 else 0.0
                else:
                    raise ValueError("point must be dict or list")

                normalized_points.append(
                    {
                        "x": x,
                        "y": y,
                        "heading": heading,
                    }
                )

            except (KeyError, ValueError, TypeError, IndexError) as exc:
                self.get_logger().warning(f"Invalid path point at index {i}: {exc}")
                return

        data["points"] = normalized_points

        payload_text = json.dumps(data)

        self.enqueue_payload(
            payload_text=payload_text,
            payload_type=4,
            source="/robot/path_command",
        )

        self.get_logger().info(
            f"Queued PATH command id={data.get('id')} points={len(normalized_points)}"
        )

    def detection_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if isinstance(data, list):
            detections = data
        elif isinstance(data, dict):
            detections = data.get("detections", [])
        else:
            return

        for det in detections:
            if not isinstance(det, dict):
                continue

            name = det.get("class_name", det.get("class", "?"))
            z_cm = det.get("z_cm")
            x_cm = det.get("x_cm")
            conf = float(det.get("confidence", det.get("conf", 0.0)))

            if z_cm is None:
                continue

            try:
                z_cm = float(z_cm)
                x_cm = float(x_cm) if x_cm is not None else None
            except (ValueError, TypeError):
                continue

            if self.debug_log_detections:
                self.get_logger().info(
                    f"[DETECTION] {name} dist={z_cm:.1f}cm "
                    f"x={x_cm if x_cm is not None else 0.0:.1f}cm conf={conf:.2f}"
                )

            if self.forward_detections_to_v5:
                payload = {
                    "type": "DETECTION",
                    "class": name,
                    "z_cm": round(z_cm, 1),
                    "x_cm": round(x_cm, 1) if x_cm is not None else None,
                    "conf": round(conf, 2),
                }

                self.enqueue_payload(
                    payload_text=json.dumps(payload),
                    payload_type=3,
                    source="detection",
                )

    def heartbeat_callback(self):
        # Do not fill the TX queue while disconnected.
        if not self.connected:
            return

        heartbeat = {
            "type": "HEARTBEAT",
            "stamp_ns": self.get_clock().now().nanoseconds,
            "uptime_sec": time.time() - self.start_time,
        }

        self.enqueue_payload(
            payload_text=json.dumps(heartbeat),
            payload_type=10,
            source="heartbeat",
        )

    def enqueue_payload(self, payload_text: str, payload_type: int, source: str):
        if self.drop_tx_when_disconnected and not self.connected:
            self.tx_drop_disconnected_count += 1

            if source != "heartbeat":
                self.get_logger().warning(
                    f"Not connected to V5. Dropping TX payload from {source}."
                )

            return

        packet = TxPacket(
            payload_type=payload_type,
            payload_text=payload_text,
            source=source,
            created_time=time.time(),
        )

        try:
            self.tx_queue.put_nowait(packet)
        except queue.Full:
            self.tx_error_count += 1

            if source != "heartbeat":
                self.get_logger().warning(
                    f"TX queue full. Dropping payload from {source}."
                )

    def serial_io_loop(self):
        rx_buffer = bytearray()

        while not self.stop_event.is_set():
            if self.serial_handle is None or not self.connected:
                self.connect_or_wait()
                continue

            try:
                self.flush_tx_queue()

                data = self.serial_handle.read(
                    max(1, self.serial_handle.in_waiting)
                )

                if data:
                    rx_buffer.extend(data)

                    while b"\n" in rx_buffer:
                        raw_line, _, remaining = rx_buffer.partition(b"\n")
                        rx_buffer = bytearray(remaining)

                        self.handle_rx_line(bytes(raw_line + b"\n"))

            except serial.SerialException as exc:
                self.get_logger().warning(f"Serial disconnected: {exc}")
                self.close_serial()
                time.sleep(self.reconnect_period_sec)

            except OSError as exc:
                self.get_logger().warning(f"Serial OS error: {exc}")
                self.close_serial()
                time.sleep(self.reconnect_period_sec)

            except Exception as exc:
                self.get_logger().error(f"Unexpected serial IO error: {exc}")
                self.close_serial()
                time.sleep(self.reconnect_period_sec)

    def connect_or_wait(self):
        port = self.find_v5_port()

        if not port:
            self.connected = False
            self.current_port = ""

            if self.debug_log_scan:
                self.get_logger().info(
                    f"Scanning for V5 brain... no matching port found. "
                    f"Retrying in {self.reconnect_period_sec:.1f}s"
                )

            time.sleep(self.reconnect_period_sec)
            return

        try:
            self.get_logger().info(f"Connecting to V5 on {port}...")

            ser = serial.Serial(
                port=port,
                baudrate=self.baud_rate,
                timeout=self.read_timeout_sec,
                write_timeout=0.25,
            )

            time.sleep(0.5)

            with self.serial_lock:
                self.serial_handle = ser
                self.connected = True
                self.current_port = port
                self.last_connect_time = time.time()
                self.reconnect_count += 1

            self.get_logger().info(f"Connected to V5 on {port}.")

        except serial.SerialException as exc:
            self.connected = False
            self.current_port = ""
            self.get_logger().warning(f"Failed to connect to {port}: {exc}")
            time.sleep(self.reconnect_period_sec)

    def find_v5_port(self) -> Optional[str]:
        if self.serial_port:
            return self.serial_port

        ports = list(serial.tools.list_ports.comports())

        if not ports:
            if self.debug_log_scan:
                self.get_logger().info("No serial ports found.")
            return None

        if self.debug_log_scan:
            self.get_logger().info("Available serial ports:")

        for port in ports:
            description = port.description or ""
            manufacturer = port.manufacturer or ""
            device = port.device or ""

            if self.debug_log_scan:
                self.get_logger().info(
                    f"  device={device}, "
                    f"description={description}, "
                    f"manufacturer={manufacturer}"
                )

            searchable = f"{description} {manufacturer} {device}"

            for keyword in self.auto_port_keywords:
                if keyword in searchable:
                    self.get_logger().info(
                        f"Found V5 candidate: device={device}, "
                        f"description={description}"
                    )
                    return device

        return None

    def flush_tx_queue(self):
        while not self.tx_queue.empty():
            try:
                packet = self.tx_queue.get_nowait()
            except queue.Empty:
                return

            self.write_packet(packet)

    def write_packet(self, packet: TxPacket):
        payload_bytes = packet.payload_text.encode("utf-8", errors="ignore")

        if self.tx_append_newline and not payload_bytes.endswith(b"\n"):
            payload_bytes += b"\n"

        if len(payload_bytes) > self.max_payload_bytes:
            self.tx_error_count += 1
            self.get_logger().warning(
                f"TX payload too large: {len(payload_bytes)} bytes. "
                f"Max is {self.max_payload_bytes}. Dropping."
            )
            return

        crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF

        with self.serial_lock:
            if self.serial_handle is None or not self.connected:
                self.tx_error_count += 1
                return

            self.serial_handle.write(self.sync_bytes)
            self.serial_handle.write(struct.pack("<h", len(payload_bytes)))
            self.serial_handle.write(struct.pack("<h", int(packet.payload_type)))
            self.serial_handle.write(struct.pack("<I", crc))
            self.serial_handle.write(payload_bytes)
            self.serial_handle.flush()

        self.tx_count += 1
        self.last_tx_time = time.time()

        if self.debug_log_tx:
            self.get_logger().info(
                f"TX type={packet.payload_type}, "
                f"bytes={len(payload_bytes)}, "
                f"crc={crc}, "
                f"source={packet.source}, "
                f"payload={packet.payload_text[:160]}"
            )

    def handle_rx_line(self, raw_line: bytes):
        self.rx_line_count += 1
        self.last_rx_time = time.time()

        text = raw_line.decode("utf-8", errors="ignore").strip()

        if self.debug_log_rx:
            self.get_logger().info(f"RX raw: {text}")

        raw_msg = String()
        raw_msg.data = json.dumps(
            {
                "stamp_ns": self.get_clock().now().nanoseconds,
                "raw": text,
            }
        )
        self.rx_raw_pub.publish(raw_msg)

        idx = text.find("soutV5,")
        if idx == -1:
            return

        cleaned = text[idx:].strip()
        parts = cleaned.split(",")

        if len(parts) < 2 or parts[0] != "soutV5":
            self.rx_parse_error_count += 1
            return

        message_type = parts[1]

        if message_type == "ODOM":
            self.parse_odom(parts, cleaned)
        elif message_type == "ACK":
            self.parse_ack(parts, cleaned)
        else:
            self.publish_parsed_message(
                {
                    "stamp_ns": self.get_clock().now().nanoseconds,
                    "source": "v5",
                    "message_type": message_type,
                    "parts": parts,
                    "raw": cleaned,
                }
            )

    def parse_odom(self, parts, raw_text: str):
        if len(parts) < 6:
            self.rx_parse_error_count += 1
            self.get_logger().warning(f"Invalid ODOM format: {raw_text}")
            return

        try:
            pos_x = float(parts[2])
            pos_y = float(parts[3])
            heading = float(parts[4])
            frame = int(parts[5])
        except ValueError:
            self.rx_parse_error_count += 1
            self.get_logger().warning(f"Could not parse ODOM values: {raw_text}")
            return

        valid = (
            math.isfinite(pos_x)
            and math.isfinite(pos_y)
            and math.isfinite(heading)
        )

        self.rx_odom_count += 1

        stamp_ns = self.get_clock().now().nanoseconds

        packet = {
            "stamp_ns": stamp_ns,
            "source": "v5",
            "message_type": "ODOM",
            "valid": valid,
            "x": pos_x,
            "y": pos_y,
            "heading": heading,
            "frame": frame,
            "raw": raw_text,
        }

        if valid:
            with self.latest_robot_pose_lock:
                self.latest_robot_pose = {
                    "valid": True,
                    "x": pos_x,
                    "y": pos_y,
                    "heading": heading,
                    "frame": frame,
                    "stamp_ns": stamp_ns,
                }

        msg = String()
        msg.data = json.dumps(packet)
        self.odom_pub.publish(msg)

        self.publish_parsed_message(packet)

    def parse_ack(self, parts, raw_text: str):
        """
        Expected V5 format:
          soutV5,ACK,command_id,status,detail

        Examples:
          soutV5,ACK,12,DONE,path complete
          soutV5,ACK,12,FAILED,blocked
        """
        if len(parts) < 4:
            self.rx_parse_error_count += 1
            self.get_logger().warning(f"Invalid ACK format: {raw_text}")
            return

        try:
            command_id = int(parts[2])
        except ValueError:
            command_id = -1

        status = parts[3]
        detail = ",".join(parts[4:]) if len(parts) > 4 else ""

        self.rx_ack_count += 1

        packet = {
            "stamp_ns": self.get_clock().now().nanoseconds,
            "source": "v5",
            "message_type": "ACK",
            "command_id": command_id,
            "status": status,
            "detail": detail,
            "raw": raw_text,
        }

        msg = String()
        msg.data = json.dumps(packet)
        self.action_ack_pub.publish(msg)

        self.publish_parsed_message(packet)

        self.get_logger().info(
            f"Received ACK command_id={command_id} status={status} detail={detail}"
        )

    def publish_parsed_message(self, packet: dict):
        msg = String()
        msg.data = json.dumps(packet)
        self.rx_parsed_pub.publish(msg)

    def publish_status(self):
        now = time.time()

        with self.latest_robot_pose_lock:
            latest_pose = dict(self.latest_robot_pose)

        packet = {
            "stamp_ns": self.get_clock().now().nanoseconds,
            "connected": self.connected,
            "current_port": self.current_port,
            "baud_rate": self.baud_rate,
            "rx_line_count": self.rx_line_count,
            "rx_odom_count": self.rx_odom_count,
            "rx_ack_count": self.rx_ack_count,
            "rx_parse_error_count": self.rx_parse_error_count,
            "tx_count": self.tx_count,
            "tx_error_count": self.tx_error_count,
            "tx_drop_disconnected_count": self.tx_drop_disconnected_count,
            "tx_queue_size": self.tx_queue.qsize(),
            "reconnect_count": self.reconnect_count,
            "latest_robot_pose": latest_pose,
            "last_rx_age_sec": None
            if self.last_rx_time == 0.0
            else now - self.last_rx_time,
            "last_tx_age_sec": None
            if self.last_tx_time == 0.0
            else now - self.last_tx_time,
        }

        msg = String()
        msg.data = json.dumps(packet)
        self.status_pub.publish(msg)

    def close_serial(self):
        with self.serial_lock:
            self.connected = False

            if self.serial_handle is not None:
                try:
                    self.serial_handle.close()
                except Exception:
                    pass

            self.serial_handle = None
            self.current_port = ""

    def destroy_node(self):
        self.stop_event.set()

        if self.io_thread.is_alive():
            self.io_thread.join(timeout=1.0)

        self.close_serial()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = SerialMessengerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Serial messenger stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()