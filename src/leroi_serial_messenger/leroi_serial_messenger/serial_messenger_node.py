import json
import math
import queue
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Dict, Optional

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

        self.declare_parameter("serial_port", "")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("read_timeout_sec", 0.05)
        self.declare_parameter("reconnect_period_sec", 2.0)
        self.declare_parameter(
            "auto_port_keywords",
            "VEX V5 User Port,VEX Robotics User Port,VEX Robotics",
        )

        self.declare_parameter("tx_append_newline", True)
        self.declare_parameter("max_payload_bytes", 1024)
        self.declare_parameter("drop_tx_when_disconnected", True)

        self.declare_parameter("send_heartbeat", True)
        self.declare_parameter("heartbeat_period_sec", 5.0)
        self.declare_parameter("status_period_sec", 1.0)
        self.declare_parameter("debug_log_rx", False)
        self.declare_parameter("debug_log_tx", True)
        self.declare_parameter("debug_log_scan", True)

        self.declare_parameter("detection_topic", "/perception/oak_front/detections")
        self.declare_parameter("forward_detections_to_v5", False)
        self.declare_parameter("debug_log_detections", False)

        # --- Gyro/odom-board reset coupling ---
        # When the V5 ACKs a RESET_GYRO action (it has stopped and is holding
        # still), write the reset character to the ESP32 odom board over its own
        # USB-C serial port. The board scans incoming bytes for the command:
        #   'R' (or 'r') -> full reset: recalibrate bias + zero heading (~1.2 s)
        #   'Z' (or 'z') -> zero heading only (instant)
        self.declare_parameter("gyro_reset_on_ack", True)
        self.declare_parameter("gyro_reset_action_id", 24)
        # Explicit board port; leave empty to auto-detect by keyword. ACM numbers
        # are not stable, so auto-detect (by the Espressif by-id string) is safer.
        self.declare_parameter("gyro_board_port", "")
        self.declare_parameter(
            "gyro_board_keywords",
            "USB JTAG/serial debug unit,Espressif",
        )
        self.declare_parameter("gyro_reset_char", "R")
        # Optional wait after the ACK before writing (e.g. let the robot settle).
        self.declare_parameter("gyro_reset_delay_sec", 0.0)
        # Fire one reset directly at node startup (no V5, no ACK). Use when you
        # have squared the robot up at heading 0 and just want to zero the board.
        self.declare_parameter("gyro_reset_on_startup", False)
        self.declare_parameter("gyro_startup_reset_delay_sec", 2.0)

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

        self.gyro_reset_on_ack = bool(self.get_parameter("gyro_reset_on_ack").value)
        self.gyro_reset_action_id = int(
            self.get_parameter("gyro_reset_action_id").value
        )
        self.gyro_board_port = str(self.get_parameter("gyro_board_port").value)
        gyro_keywords_raw = str(self.get_parameter("gyro_board_keywords").value)
        self.gyro_board_keywords = [
            keyword.strip()
            for keyword in gyro_keywords_raw.split(",")
            if keyword.strip()
        ]
        self.gyro_reset_char = str(self.get_parameter("gyro_reset_char").value) or "R"
        self.gyro_reset_delay_sec = float(
            self.get_parameter("gyro_reset_delay_sec").value
        )
        self.gyro_reset_on_startup = bool(
            self.get_parameter("gyro_reset_on_startup").value
        )
        self.gyro_startup_reset_delay_sec = float(
            self.get_parameter("gyro_startup_reset_delay_sec").value
        )
        self.gyro_reset_count = 0

        self.sync_bytes = b"\xaa\x55\xcc\x33"

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

        self.detection_sub = self.create_subscription(
            String,
            self.detection_topic,
            self.detection_callback,
            10,
        )

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

        # One-shot startup gyro reset (self-cancels on first fire).
        self._startup_gyro_timer = None
        if self.gyro_reset_on_startup:
            self._startup_gyro_timer = self.create_timer(
                self.gyro_startup_reset_delay_sec,
                self.startup_gyro_reset,
            )

        self.get_logger().info("Serial messenger node started.")
        self.get_logger().info(f"serial_port: {self.serial_port or 'auto'}")
        self.get_logger().info(f"baud_rate: {self.baud_rate}")
        self.get_logger().info(f"heartbeat_period_sec: {self.heartbeat_period_sec}")
        self.get_logger().info(f"auto_port_keywords: {self.auto_port_keywords}")
        self.get_logger().info(
            f"forward_detections_to_v5: {self.forward_detections_to_v5}"
        )
        self.get_logger().info(
            f"gyro_reset_on_ack: {self.gyro_reset_on_ack} "
            f"(action_id={self.gyro_reset_action_id}, char='{self.gyro_reset_char}', "
            f"board={self.gyro_board_port or 'auto'})"
        )

    def tx_callback(self, msg: String):
        self.enqueue_payload(
            payload_text=msg.data,
            payload_type=1,
            source="/serial_messenger/tx",
        )

    def robot_command_callback(self, msg: String):
        try:
            command = self.normalize_robot_command(msg.data)
        except ValueError as exc:
            self.get_logger().warning(f"Invalid /robot/command: {exc}")
            return

        payload_text = json.dumps(command)

        self.enqueue_payload(
            payload_text=payload_text,
            payload_type=5,
            source="/robot/command",
        )

        self.get_logger().info(
            f"Queued ACTION for V5 id={command.get('id')} "
            f"action_id={command.get('action_id')} "
            f"action={command.get('action_name')} "
            f"points={len(command.get('points', []))}"
        )

    def normalize_robot_command(self, raw_text: str) -> Dict:
        text = raw_text.strip()

        if not text:
            raise ValueError("empty command")

        try:
            command = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not JSON: {exc}") from exc

        if not isinstance(command, dict):
            raise ValueError("command must be a JSON object")

        command_type = str(command.get("type", "")).upper()
        if command_type != "ACTION":
            raise ValueError("command type must be ACTION")

        if "id" not in command:
            raise ValueError("missing id")

        if "action_id" not in command:
            raise ValueError("missing action_id")

        try:
            command["id"] = int(command["id"])
        except (ValueError, TypeError) as exc:
            raise ValueError("id must be integer") from exc

        try:
            command["action_id"] = int(command["action_id"])
        except (ValueError, TypeError) as exc:
            raise ValueError("action_id must be integer") from exc

        command["type"] = "ACTION"

        if "action_name" not in command:
            command["action_name"] = f"ACTION_{command['action_id']}"

        points = command.get("points", [])
        if points is None:
            points = []

        if not isinstance(points, list):
            raise ValueError("points must be a list")

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

                if not (
                    math.isfinite(x)
                    and math.isfinite(y)
                    and math.isfinite(heading)
                ):
                    raise ValueError("point values must be finite")

                normalized_points.append(
                    {
                        "x": x,
                        "y": y,
                        "heading": heading,
                    }
                )

            except (KeyError, ValueError, TypeError, IndexError) as exc:
                raise ValueError(f"invalid point {i}: {exc}") from exc

        command["points"] = normalized_points

        params = command.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params must be an object")

        command["params"] = params

        return command

    def detection_callback(self, msg: String):
        if not self.forward_detections_to_v5:
            return

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
                f"payload={packet.payload_text[:180]}"
            )

    def handle_rx_line(self, raw_line: bytes):
        self.rx_line_count += 1
        self.last_rx_time = time.time()

        text = raw_line.decode("utf-8", errors="ignore").strip()

        if self.debug_log_rx:
            safe_text = text.encode("unicode_escape", errors="ignore").decode("ascii")
            self.get_logger().info(f"RX raw escaped: {safe_text}")

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
        if len(parts) < 4:
            self.rx_parse_error_count += 1
            self.get_logger().warning(f"Invalid ACK format: {raw_text}")
            return

        command_id = self.safe_int(parts[2], default=-1)
        status = parts[3].upper().strip()

        action_id = None
        holding_blocks = None
        detail_parts = []

        remaining = parts[4:]

        if remaining:
            first = remaining[0].strip()

            try:
                action_id = int(first)
                remaining = remaining[1:]
            except ValueError:
                action_id = None

        key_values = {}

        for item in remaining:
            item = item.strip()

            if "=" in item:
                key, value = item.split("=", 1)
                key = key.strip()
                value = value.strip()
                key_values[key] = value
            else:
                detail_parts.append(item)

        if "holding_blocks" in key_values:
            holding_blocks = self.parse_bool(key_values["holding_blocks"])

        detail = ",".join(detail_parts)

        self.rx_ack_count += 1

        packet = {
            "stamp_ns": self.get_clock().now().nanoseconds,
            "source": "v5",
            "message_type": "ACK",
            "command_id": command_id,
            "status": status,
            "action_id": action_id,
            "holding_blocks": holding_blocks,
            "detail": detail,
            "fields": key_values,
            "raw": raw_text,
        }

        msg = String()
        msg.data = json.dumps(packet)
        self.action_ack_pub.publish(msg)

        self.publish_parsed_message(packet)

        self.get_logger().info(
            f"ACK command_id={command_id} status={status} "
            f"action_id={action_id} holding_blocks={holding_blocks} "
            f"detail={detail}"
        )

        if (
            self.gyro_reset_on_ack
            and status == "DONE"
            and action_id is not None
            and int(action_id) == self.gyro_reset_action_id
        ):
            self.trigger_gyro_reset(command_id, action_id)

    def trigger_gyro_reset(self, command_id, action_id):
        """Write the reset char to the ESP32 odom board on a RESET_GYRO ACK.

        The robot has stopped and is holding still (that's why it ACKed), so it
        is safe to recalibrate the gyro now.
        """
        if self.gyro_reset_delay_sec > 0:
            time.sleep(self.gyro_reset_delay_sec)

        self.write_gyro_reset(
            reason=f"ack command_id={command_id} action_id={action_id}"
        )

    def startup_gyro_reset(self):
        """One-shot reset at node startup (no V5, no ACK).

        Use when the robot has been squared up at heading 0 by hand and you just
        want the board zeroed/recalibrated before the run begins.
        """
        self.destroy_timer(self._startup_gyro_timer)
        self.write_gyro_reset(reason="startup")

    def write_gyro_reset(self, reason: str) -> bool:
        """Open the odom board's own USB-C port, write the reset char, close.

        We never read the board, so manual tools (``printf 'R' > /dev/ttyACMx``)
        keep working. Returns True if the byte was written.
        """
        port = self.gyro_board_port or self.find_gyro_board_port()

        if not port:
            self.get_logger().warning(
                f"Gyro reset ({reason}) requested but no odom board port was "
                f"found; skipping. Set gyro_board_port or check the board's USB-C."
            )
            return False

        char = self.gyro_reset_char[:1] if self.gyro_reset_char else "R"

        try:
            with serial.Serial(
                port=port,
                baudrate=self.baud_rate,  # ignored for native USB CDC
                write_timeout=1.0,
            ) as board:
                board.write(char.encode("ascii", errors="ignore"))
                board.flush()

            self.gyro_reset_count += 1
            self.get_logger().info(
                f"Gyro reset ({reason}): wrote '{char}' to odom board on {port}. "
                f"Keep the board still ~1.2 s for full recalibration."
            )
            return True
        except (serial.SerialException, OSError) as exc:
            self.get_logger().warning(
                f"Gyro reset ({reason}) write failed on {port}: {exc}"
            )
            return False

    def find_gyro_board_port(self) -> Optional[str]:
        if self.gyro_board_port:
            return self.gyro_board_port

        try:
            ports = list(serial.tools.list_ports.comports())
        except Exception as exc:  # noqa: BLE001 - listing must never crash IO loop
            self.get_logger().warning(f"Could not list serial ports: {exc}")
            return None

        for port in ports:
            device = port.device or ""
            # Never pick the V5 port we're already talking to.
            if device and device == self.current_port:
                continue

            searchable = f"{port.description or ''} {port.manufacturer or ''} {device}"
            for keyword in self.gyro_board_keywords:
                if keyword and keyword in searchable:
                    return device

        return None

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
            "gyro_reset_count": self.gyro_reset_count,
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

    @staticmethod
    def safe_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def parse_bool(value: str) -> Optional[bool]:
        text = str(value).strip().lower()

        if text in ("1", "true", "yes", "y", "on"):
            return True

        if text in ("0", "false", "no", "n", "off"):
            return False

        return None


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