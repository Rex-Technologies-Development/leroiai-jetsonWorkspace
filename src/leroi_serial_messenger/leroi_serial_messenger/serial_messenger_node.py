import json
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
        self.declare_parameter("max_payload_bytes", 512)
        self.declare_parameter("drop_tx_when_disconnected", True)

        # Heartbeat/debug parameters
        self.declare_parameter("send_heartbeat", True)
        self.declare_parameter("heartbeat_period_sec", 1.0)
        self.declare_parameter("status_period_sec", 1.0)
        self.declare_parameter("debug_log_rx", False)
        self.declare_parameter("debug_log_tx", True)
        self.declare_parameter("debug_log_scan", True)

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

        # This matches the sync bytes from your original test script.
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

        self.odom_pub = self.create_publisher(
            String,
            "/v5/odom",
            10,
        )

        self.rx_parsed_pub = self.create_publisher(
            String,
            "/serial_messenger/rx_parsed",
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
        self.rx_parse_error_count = 0
        self.tx_count = 0
        self.tx_error_count = 0
        self.tx_drop_disconnected_count = 0
        self.reconnect_count = 0

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

    def tx_callback(self, msg: String):
        self.enqueue_payload(
            payload_text=msg.data,
            payload_type=1,
            source="/serial_messenger/tx",
        )

    def robot_command_callback(self, msg: String):
        self.enqueue_payload(
            payload_text=msg.data,
            payload_type=2,
            source="/robot/command",
        )

    def heartbeat_callback(self):
        # Important: do not fill the TX queue when V5 is disconnected.
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

            # Heartbeats are expected to be skipped while disconnected.
            # For real commands, warn so we know something tried to command the robot.
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

                        # Put newline back for logging consistency.
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
                f"payload={packet.payload_text[:120]}"
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

        # Current V5 format:
        # soutV5,ODOM,x,y,heading,frame
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

        self.rx_odom_count += 1

        packet = {
            "stamp_ns": self.get_clock().now().nanoseconds,
            "source": "v5",
            "message_type": "ODOM",
            "x": pos_x,
            "y": pos_y,
            "heading": heading,
            "frame": frame,
            "raw": raw_text,
        }

        msg = String()
        msg.data = json.dumps(packet)
        self.odom_pub.publish(msg)

        self.publish_parsed_message(packet)

    def publish_parsed_message(self, packet: dict):
        msg = String()
        msg.data = json.dumps(packet)
        self.rx_parsed_pub.publish(msg)

    def publish_status(self):
        now = time.time()

        packet = {
            "stamp_ns": self.get_clock().now().nanoseconds,
            "connected": self.connected,
            "current_port": self.current_port,
            "baud_rate": self.baud_rate,
            "rx_line_count": self.rx_line_count,
            "rx_odom_count": self.rx_odom_count,
            "rx_parse_error_count": self.rx_parse_error_count,
            "tx_count": self.tx_count,
            "tx_error_count": self.tx_error_count,
            "tx_drop_disconnected_count": self.tx_drop_disconnected_count,
            "tx_queue_size": self.tx_queue.qsize(),
            "reconnect_count": self.reconnect_count,
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