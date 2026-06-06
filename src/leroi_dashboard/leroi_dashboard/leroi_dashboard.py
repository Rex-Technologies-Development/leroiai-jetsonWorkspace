"""Sync helpers for connecting a bot to the VexAI dashboard.

  from module import connect_robot, send_location, connect_blocks, send_blocks
  from module import connect_camera, send_frame

  robot, _ = connect_robot("67goonbot")
  send_location(robot, x=182.88, y=182.88, heading_deg=0)
  robot.close()
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import quote

from websockets.sync.client import ClientConnection, connect

DEFAULT_HOST = os.environ.get("VEXAI_HOST", "162.248.101.133:6741")
RECV_TIMEOUT_SEC = 10
COMMAND_RECV_TIMEOUT_SEC = 30


def resolve_host(host: str | None = None) -> str:
    return (host or os.environ.get("VEXAI_HOST") or DEFAULT_HOST).strip()


def _ws_url(path: str, bot_id: str, host: str | None = None, **extra: str) -> str:
    params = f"role=bot&bot_id={quote(bot_id)}"
    for key, value in extra.items():
        params += f"&{key}={quote(value)}"
    return f"ws://{resolve_host(host)}{path}?{params}"


def _connect(url: str, channel: str) -> tuple[ClientConnection, dict[str, Any]]:
    try:
        ws = connect(url)
    except ConnectionRefusedError as exc:
        raise ConnectionError(
            f"Could not connect {channel} to {url}. Is the server running?"
        ) from exc
    hello = json.loads(ws.recv(timeout=RECV_TIMEOUT_SEC))
    return ws, hello


def connect_robot(
    bot_id: str,
    host: str | None = None,
) -> tuple[ClientConnection, dict[str, Any]]:
    """Open /ws/robot. Close the socket when done."""
    return _connect(_ws_url("/ws/robot", bot_id, host=host), "robot")


def connect_blocks(
    bot_id: str,
    host: str | None = None,
) -> tuple[ClientConnection, dict[str, Any]]:
    """Open /ws/blocks. Close the socket when done."""
    return _connect(_ws_url("/ws/blocks", bot_id, host=host), "blocks")


def connect_camera(
    bot_id: str,
    camera_id: str = "cam-1",
    label: str | None = None,
    host: str | None = None,
) -> tuple[ClientConnection, dict[str, Any]]:
    """Open /ws/video for one camera. Close the socket when done."""
    return _connect(
        _ws_url(
            "/ws/video",
            bot_id,
            host=host,
            camera_id=camera_id,
            label=label or camera_id,
        ),
        f"camera:{camera_id}",
    )


def connect_command(
    bot_id: str,
    host: str | None = None,
) -> tuple[ClientConnection, dict[str, Any]]:
    """Open /ws/command as role=bot.

    Matches the dashboard mock bot handshake:
      1. recv hello JSON from server
      2. optionally recv one immediate status push (if server sends it)
      3. then enter recv loop for action_request messages
    """
    url = _ws_url("/ws/command", bot_id, host=host)
    try:
        ws = connect(url)
    except ConnectionRefusedError as exc:
        raise ConnectionError(
            f"Could not connect command to {url}. Is the server running?"
        ) from exc

    hello = json.loads(ws.recv(timeout=COMMAND_RECV_TIMEOUT_SEC))
    session: dict[str, Any] = {"hello": hello}

    try:
        initial_status = json.loads(ws.recv(timeout=2.0))
        session["initial_status"] = initial_status
    except TimeoutError:
        pass

    return ws, session


def _send_json(ws: ClientConnection, payload: dict[str, Any]) -> dict[str, Any]:
    ws.send(json.dumps(payload))
    return json.loads(ws.recv(timeout=RECV_TIMEOUT_SEC))


def send_location(
    ws: ClientConnection,
    *,
    x: float,
    y: float,
    heading_deg: float,
    patch: bool = False,
) -> dict[str, Any]:
    """Push robot pose. Returns server ack."""
    return _send_json(
        ws,
        {
            "type": "patch" if patch else "state",
            "data": {"x": x, "y": y, "heading_deg": heading_deg},
        },
    )


def send_blocks(
    ws: ClientConnection,
    *,
    blocks: list[dict[str, Any]] | None = None,
    bot_holding: list[str] | None = None,
    patch: bool = False,
) -> dict[str, Any]:
    """Push field blocks and/or collected list. Returns server ack."""
    data: dict[str, Any] = {}
    if blocks is not None:
        data["blocks"] = blocks
    if bot_holding is not None:
        data["bot_holding"] = bot_holding
    return _send_json(
        ws,
        {"type": "patch" if patch else "state", "data": data},
    )


def send_frame(ws: ClientConnection, jpeg: bytes) -> None:
    """Push one JPEG frame (binary). No ack from server."""
    ws.send(jpeg)


def send_frame_b64(ws: ClientConnection, jpeg_b64: str) -> None:
    """Push one JPEG frame as JSON base64. No ack from server."""
    ws.send(json.dumps({"type": "frame", "jpeg": jpeg_b64}))


def send_command_message(ws: ClientConnection, payload: dict[str, Any]) -> None:
    """Push one command-channel JSON message. No ack expected from server."""
    ws.send(json.dumps(payload))
