import time
from typing import Optional

import cv2
import numpy as np


class CameraSource:
    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class FakeCameraSource(CameraSource):
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.frame_id = 0

    def read(self) -> Optional[np.ndarray]:
        self.frame_id += 1

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        cv2.putText(
            frame,
            f"Fake Camera Frame {self.frame_id}",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        return frame


class USBCameraSource(CameraSource):
    def __init__(
        self,
        camera_index: int,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ):
        self.camera_index = camera_index

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open USB camera index {camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()

        if not ok or frame is None:
            return None

        return frame

    def close(self) -> None:
        self.cap.release()


class OakDCameraSource(CameraSource):
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        mxid: str = "",
    ):
        try:
            import depthai as dai
        except ImportError as exc:
            raise RuntimeError(
                "depthai is not installed. Install it with: "
                "python3 -m pip install --user depthai"
            ) from exc

        self.dai = dai

        pipeline = dai.Pipeline()

        cam_rgb = pipeline.create(dai.node.ColorCamera)

        # Newer DepthAI uses CAM_A. Older versions may use RGB.
        if hasattr(dai.CameraBoardSocket, "CAM_A"):
            cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        else:
            cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

        cam_rgb.setPreviewSize(width, height)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(fps)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        if mxid:
            self.device = dai.Device(pipeline, mxid)
        else:
            self.device = dai.Device(pipeline)

        self.rgb_queue = self.device.getOutputQueue(
            name="rgb",
            maxSize=4,
            blocking=False,
        )

        time.sleep(0.5)

    def read(self) -> Optional[np.ndarray]:
        packet = self.rgb_queue.tryGet()

        if packet is None:
            return None

        return packet.getCvFrame()

    def close(self) -> None:
        self.device.close()


def create_camera_source(
    camera_type: str,
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    oak_mxid: str = "",
) -> CameraSource:
    camera_type = camera_type.lower().strip()

    if camera_type == "oak":
        return OakDCameraSource(
            width=width,
            height=height,
            fps=fps,
            mxid=oak_mxid,
        )

    if camera_type == "usb":
        return USBCameraSource(
            camera_index=camera_index,
            width=width,
            height=height,
            fps=fps,
        )

    if camera_type == "fake":
        return FakeCameraSource(
            width=width,
            height=height,
        )

    raise ValueError(f"Unsupported camera_type: {camera_type}")