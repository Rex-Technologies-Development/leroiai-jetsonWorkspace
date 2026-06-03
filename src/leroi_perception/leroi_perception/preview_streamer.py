import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np


class MjpegPreviewStreamer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        jpeg_quality: int = 70,
        stream_fps: float = 15.0,
        title: str = "Leroi Perception Preview",
        show_map: bool = False,
        show_goal: bool = False,
        map_port: int = 0,
        goal_port: int = 0,
    ):
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.stream_fps = max(stream_fps, 1.0)
        self.title = title

        # Backward-compatible: if old code passes map_port/goal_port, show panels.
        self.show_map = show_map or map_port > 0
        self.show_goal = show_goal or goal_port > 0

        self._frames = {
            "camera": {
                "jpeg": None,
                "frame_id": 0,
            },
            "map": {
                "jpeg": None,
                "frame_id": 0,
            },
            "goal": {
                "jpeg": None,
                "frame_id": 0,
            },
        }

        self._lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def _build_index_html(self) -> bytes:
        extra_panels = ""

        if self.show_map:
            extra_panels += """
            <div class="panel">
                <h2>Top-Down Map</h2>
                <img src="/map.mjpg" />
            </div>
            """

        if self.show_goal:
            extra_panels += """
            <div class="panel">
                <h2>Goal View</h2>
                <img src="/goal.mjpg" />
            </div>
            """

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background: #111;
            color: #eee;
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 16px;
            gap: 12px;
        }}

        h1 {{
            font-size: 1.25rem;
            color: #ddd;
        }}

        h2 {{
            font-size: 0.95rem;
            color: #aaa;
            margin-bottom: 6px;
            text-align: center;
        }}

        .streams {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            justify-content: center;
            width: 100%;
        }}

        .panel {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        img {{
            max-width: 95vw;
            max-height: 75vh;
            border: 2px solid #444;
            background: #222;
        }}

        .info {{
            font-size: 0.75rem;
            color: #777;
            margin-top: 8px;
        }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>

    <div class="streams">
        <div class="panel">
            <h2>Camera Feed</h2>
            <img src="/stream.mjpg" />
        </div>

        {extra_panels}
    </div>

    <div class="info">
        Port {self.port} |
        Camera: /stream.mjpg |
        Map: /map.mjpg |
        Goal: /goal.mjpg
    </div>
</body>
</html>
"""
        return html.encode("utf-8")

    def start(self):
        streamer = self
        index_html = self._build_index_html()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header(
                        "Cache-Control",
                        "no-store, no-cache, must-revalidate, max-age=0",
                    )
                    self.send_header("Pragma", "no-cache")
                    self.end_headers()
                    self.wfile.write(index_html)
                    return

                if self.path == "/stream.mjpg":
                    streamer._serve_mjpeg(self, "camera")
                    return

                if self.path == "/map.mjpg":
                    streamer._serve_mjpeg(self, "map")
                    return

                if self.path == "/goal.mjpg":
                    streamer._serve_mjpeg(self, "goal")
                    return

                self.send_error(404)

            def log_message(self, format, *args):
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._thread.start()

    def _serve_mjpeg(self, handler: BaseHTTPRequestHandler, channel: str):
        handler.send_response(200)
        handler.send_header("Age", "0")
        handler.send_header(
            "Cache-Control",
            "no-store, no-cache, must-revalidate, max-age=0",
        )
        handler.send_header("Pragma", "no-cache")
        handler.send_header("Connection", "close")
        handler.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary=frame",
        )
        handler.end_headers()

        frame_period = 1.0 / self.stream_fps
        last_sent_id = -1

        while True:
            loop_start = time.time()

            with self._lock:
                frame = self._frames[channel]["jpeg"]
                frame_id = self._frames[channel]["frame_id"]

            if frame is None or frame_id == last_sent_id:
                time.sleep(0.005)
                continue

            last_sent_id = frame_id

            try:
                handler.wfile.write(b"--frame\r\n")
                handler.wfile.write(b"Content-Type: image/jpeg\r\n")
                handler.wfile.write(
                    f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8")
                )
                handler.wfile.write(frame)
                handler.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                break

            elapsed = time.time() - loop_start
            sleep_time = frame_period - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    def _update_channel(self, channel: str, frame: np.ndarray, frame_id: int = 0):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        ok, jpeg = cv2.imencode(".jpg", frame, encode_params)

        if not ok:
            return

        with self._lock:
            self._frames[channel]["jpeg"] = jpeg.tobytes()
            self._frames[channel]["frame_id"] = frame_id

    def update_frame(self, frame: np.ndarray, frame_id: int = 0):
        self._update_channel("camera", frame, frame_id)

    def update_map_frame(self, frame: np.ndarray, frame_id: int = 0):
        self._update_channel("map", frame, frame_id)

    def update_goal_frame(self, frame: np.ndarray, frame_id: int = 0):
        self._update_channel("goal", frame, frame_id)

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None