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
    ):
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.stream_fps = max(stream_fps, 1.0)

        self._latest_jpeg: Optional[bytes] = None
        self._latest_frame_id = 0
        self._lock = threading.Lock()
        self._server = None
        self._thread = None

    def start(self):
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/" or self.path == "/index.html":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                    self.send_header("Pragma", "no-cache")
                    self.end_headers()
                    self.wfile.write(
                        b"""
                        <html>
                        <head>
                            <title>Leroi Perception Preview</title>
                            <style>
                                body {
                                    background: #111;
                                    color: #eee;
                                    font-family: Arial;
                                    text-align: center;
                                }
                                img {
                                    max-width: 95vw;
                                    max-height: 85vh;
                                    border: 2px solid #444;
                                }
                            </style>
                        </head>
                        <body>
                            <h1>Leroi Perception Preview</h1>
                            <img src="/stream.mjpg" />
                        </body>
                        </html>
                        """
                    )
                    return

                if self.path != "/stream.mjpg":
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                frame_period = 1.0 / streamer.stream_fps
                last_sent_id = -1

                while True:
                    loop_start = time.time()

                    with streamer._lock:
                        frame = streamer._latest_jpeg
                        frame_id = streamer._latest_frame_id

                    if frame is None or frame_id == last_sent_id:
                        time.sleep(0.005)
                        continue

                    last_sent_id = frame_id

                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break

                    elapsed = time.time() - loop_start
                    sleep_time = frame_period - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            def log_message(self, format, *args):
                return

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def update_frame(self, frame: np.ndarray, frame_id: int = 0):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        ok, jpeg = cv2.imencode(".jpg", frame, encode_params)

        if not ok:
            return

        with self._lock:
            self._latest_jpeg = jpeg.tobytes()
            self._latest_frame_id = frame_id

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None