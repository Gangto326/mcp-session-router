"""
Unix Domain Socket client used by the MCP server to talk with the wrapper.

Connects to the wrapper's AF_UNIX SOCK_STREAM socket and exchanges
line-delimited JSON messages — the mirror image of ``socket_server.py``.
The client is intentionally synchronous (blocking) because MCP tool
handlers only need short fire-and-forget sends, and the single
handshake at startup is a one-shot exchange that completes before any
tool call can arrive.

래퍼와 통신하기 위해 MCP 서버가 사용하는 Unix Domain Socket 클라이언트.

래퍼의 AF_UNIX SOCK_STREAM 소켓에 연결하여 라인 구분 JSON 메시지를
교환한다 — ``socket_server.py``의 반대. 클라이언트는 의도적으로
동기(블로킹)로 구현되었다. MCP 도구 핸들러는 짧은 fire-and-forget 송신만
필요하고, 시작 시 단 한 번의 핸드셰이크도 도구 호출이 들어오기 전에
완료되는 일회성 교환이기 때문이다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


class WrapperSocketClient:
    """Synchronous AF_UNIX client that talks to the wrapper socket server.

    래퍼 소켓 서버와 통신하는 동기 AF_UNIX 클라이언트.
    """

    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path
        self._sock: socket.socket | None = None
        self._read_buffer: bytes = b""

    # ------------------------------------------------------------ lifecycle
    # 생명주기 -------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to the wrapper's Unix socket.

        래퍼의 Unix 소켓에 연결한다. 연결 실패 시 예외를 그대로 전파한다.
        """
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.socket_path)
        self._sock = sock
        self._read_buffer = b""

    def close(self) -> None:
        """Close the socket connection.

        소켓 연결을 닫는다.
        """
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        self._read_buffer = b""

    # ----------------------------------------------------------- handshake
    # 핸드셰이크 -----------------------------------------------------------------

    def request_handshake(self) -> str | None:
        """Send a handshake request and return the current session name.

        핸드셰이크 요청을 보내고 래퍼가 응답한 현재 세션 이름을 반환한다.
        래퍼에 활성 세션이 없으면 None을 반환한다.
        """
        self._send({"type": "handshake_request"})
        response = self._recv_one()
        if response is None:
            return None
        return response.get("current_session_name")

    # -------------------------------------------------------- signal sender
    # 신호 송신 ------------------------------------------------------------------

    def send_signal(self, message: dict[str, Any]) -> None:
        """Send a signal message (switch / new / session_end_completed) to the wrapper.

        래퍼에 신호 메시지(switch / new / session_end_completed)를 전송한다.
        """
        self._send(message)

    # --------------------------------------------------- low-level helpers
    # 저수준 헬퍼 ----------------------------------------------------------------

    def _send(self, message: dict[str, Any]) -> None:
        """Encode *message* as line-delimited JSON and send.

        message를 라인 구분 JSON으로 직렬화하여 전송한다.
        """
        if self._sock is None:
            raise RuntimeError("Not connected")
        payload = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        self._sock.sendall(payload)

    def _recv_one(self) -> dict[str, Any] | None:
        """Block until one complete JSON line arrives and return it.

        완전한 JSON 라인 하나가 도착할 때까지 블로킹한 뒤 파싱하여 반환한다.
        연결이 끊기면 None을 반환한다.
        """
        if self._sock is None:
            return None
        while b"\n" not in self._read_buffer:
            chunk = self._sock.recv(4096)
            if not chunk:
                return None
            self._read_buffer += chunk
        line, self._read_buffer = self._read_buffer.split(b"\n", 1)
        return json.loads(line.decode("utf-8"))

    # ------------------------------------------------------ async receive
    # 비동기 수신 ----------------------------------------------------------------

    async def recv_loop(
        self,
        on_message: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """Background async loop that receives push messages from the wrapper.

        래퍼가 push하는 메시지를 받는 백그라운드 비동기 루프. handshake가
        끝난 뒤 호출하면 socket을 non-blocking 모드로 전환하고, 매 라인
        구분 JSON 메시지를 ``on_message`` 콜백으로 전달한다.

        연결이 끊기거나 task가 취소되면 루프 종료.
        """
        if self._sock is None:
            return
        self._sock.setblocking(False)
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    chunk = await loop.sock_recv(self._sock, 4096)
                except (OSError, asyncio.CancelledError):
                    return
                if not chunk:
                    # EOF — wrapper closed its end.
                    # EOF — 래퍼가 자기 쪽을 닫음.
                    return
                self._read_buffer += chunk
                while b"\n" in self._read_buffer:
                    line, self._read_buffer = self._read_buffer.split(b"\n", 1)
                    try:
                        msg = json.loads(line.decode("utf-8"))
                    except (ValueError, UnicodeDecodeError):
                        # Malformed line — skip silently to keep the loop alive.
                        # 깨진 라인은 무시하고 루프 유지.
                        continue
                    if not isinstance(msg, dict):
                        continue
                    try:
                        await on_message(msg)
                    except Exception:
                        # Don't let a callback error kill the receive loop.
                        # 콜백 에러로 receive 루프가 죽지 않도록.
                        logger.exception("on_message callback raised")
        finally:
            # Restore blocking mode for any subsequent synchronous send.
            # 이후 동기 송신을 위해 blocking 모드 복원.
            try:
                self._sock.setblocking(True)
            except OSError:
                pass
