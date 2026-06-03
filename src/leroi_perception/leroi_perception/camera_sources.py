import time
from typing import Optional, Tuple

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
        with_depth: bool = False,
    ):
        try:
            import depthai as dai
        except ImportError as exc:
            raise RuntimeError(
                "depthai is not installed. Install it with: "
                "python3 -m pip install --user depthai"
            ) from exc

        self.dai = dai
        self.with_depth = with_depth
        self._fx = self._fy = self._cx = self._cy = None

        pipeline = dai.Pipeline()

        cam_rgb = pipeline.create(dai.node.ColorCamera)

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

        if with_depth:
            mono_left = pipeline.create(dai.node.MonoCamera)
            mono_right = pipeline.create(dai.node.MonoCamera)

            mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono_left.setFps(fps)
            mono_right.setFps(fps)

            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setLeftRightCheck(True)
            stereo.setExtendedDisparity(False)
            stereo.setSubpixel(False)

            rgb_socket = cam_rgb.getBoardSocket()
            stereo.setDepthAlign(rgb_socket)
            stereo.setOutputSize(width, height)

            mono_left.out.link(stereo.left)
            mono_right.out.link(stereo.right)

            xout_depth = pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName("depth")
            stereo.depth.link(xout_depth.input)

        if mxid:
            self.device = dai.Device(pipeline, mxid)
        else:
            self.device = dai.Device(pipeline)

        self.rgb_queue = self.device.getOutputQueue(
            name="rgb",
            maxSize=1,
            blocking=False,
        )

        self.depth_queue = None
        if with_depth:
            self.depth_queue = self.device.getOutputQueue(
                name="depth",
                maxSize=1,
                blocking=False,
            )
            self._load_intrinsics(width, height)

        time.sleep(0.5)

    def _load_intrinsics(self, width: int, height: int) -> None:
        calib = self.device.readCalibration2()
        dai = self.dai

        rgb_socket = (
            dai.CameraBoardSocket.CAM_A
            if hasattr(dai.CameraBoardSocket, "CAM_A")
            else dai.CameraBoardSocket.RGB
        )

        intrinsics = calib.getCameraIntrinsics(rgb_socket, width, height)
        self._fx = intrinsics[0][0]
        self._fy = intrinsics[1][1]
        self._cx = intrinsics[0][2]
        self._cy = intrinsics[1][2]

    @property
    def intrinsics(self) -> Optional[Tuple[float, float, float, float]]:
        """Returns (fx, fy, cx, cy) when with_depth=True, else None."""
        if self._fx is None:
            return None
        return (self._fx, self._fy, self._cx, self._cy)

    def read(self) -> Optional[np.ndarray]:
        packet = self.rgb_queue.tryGet()

        if packet is None:
            return None

        return packet.getCvFrame()

    def read_depth(self) -> Optional[np.ndarray]:
        """Returns the latest stereo depth frame (uint16, mm), or None if unavailable."""
        if self.depth_queue is None:
            return None

        packet = self.depth_queue.tryGet()

        if packet is None:
            return None

        return packet.getFrame()

    def close(self) -> None:
        self.device.close()


def create_camera_source(
    camera_type: str,
    camera_index: int,
    width: int,
    height: int,
    fps: int,
    oak_mxid: str = "",
    oak_with_depth: bool = False,
) -> CameraSource:
    camera_type = camera_type.lower().strip()

    if camera_type == "oak":
        return OakDCameraSource(
            width=width,
            height=height,
            fps=fps,
            mxid=oak_mxid,
            with_depth=oak_with_depth,
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
