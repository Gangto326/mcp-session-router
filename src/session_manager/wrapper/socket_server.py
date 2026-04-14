"""
Unix Domain Socket server used by the wrapper to talk with the MCP process.

Hosts a single-client AF_UNIX SOCK_STREAM socket. The wrapper exposes the
listening fd (and, once connected, the client fd) to its main select() loop
so socket I/O is multiplexed alongside PTY and stdin without an extra
thread. Messages are line-delimited JSON: each line on the wire is one
JSON object.

PTY 래퍼가 MCP 프로세스와 통신하기 위한 Unix Domain Socket 서버 모듈.

단일 클라이언트만 허용하는 AF_UNIX SOCK_STREAM 소켓을 호스팅한다. 래퍼는
listen fd와 (연결된 후의) client fd를 자기 메인 select() 루프에 노출하므로,
별도 스레드 없이 PTY·stdin과 함께 다중화된다. 메시지 프레이밍은 라인 기반
JSON이다 — 와이어상 한 줄이 곧 하나의 JSON 객체에 대응한다.

지원하는 메시지 종류:
- MCP → 래퍼: handshake_request, action=switch, action=new, session_end_completed
- 래퍼 → MCP: handshake_response(current_session_name), user_action
실제 메시지 라우팅은 호출자(SessionManagerWrapper)가 `on_message` 콜백
안에서 처리하며, 본 모듈은 프레이밍·연결·전송 책임만 진다.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any


class WrapperSocketServer:
    def __init__(
        self,
        socket_path: str,
        on_message: Callable[[dict[str, Any]], None],
    ) -> None:
        self.socket_path = socket_path
        self._on_message = on_message
        self._listen_sock: socket.socket | None = None
        self._client_sock: socket.socket | None = None
        self._read_buffer: bytes = b""

    def start(self) -> None:
        """
        Bind and listen on the configured socket path.

        지정 경로에 Unix 소켓을 바인딩하고 listen 상태로 진입한다.
        """
        # Remove a stale socket file left over from a prior crashed run.
        # 이전 실행이 비정상 종료되며 남긴 소켓 파일 정리.
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

        Path(self.socket_path).parent.mkdir(parents=True, exist_ok=True)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(self.socket_path)
        sock.listen(1)
        sock.setblocking(False)
        self._listen_sock = sock

    def stop(self) -> None:
        """
        Close the client and listen sockets and unlink the socket file.

        클라이언트와 listen 소켓을 닫고 소켓 파일을 제거한다.
        """
        self._close_client()
        if self._listen_sock is not None:
            try:
                self._listen_sock.close()
            except OSError:
                pass
            self._listen_sock = None
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

    # ----------------------------------------------------------- fd accessors
    # fd 접근자 -----------------------------------------------------------------

    @property
    def listen_fileno(self) -> int:
        if self._listen_sock is None:
            return -1
        return self._listen_sock.fileno()

    @property
    def client_fileno(self) -> int:
        if self._client_sock is None:
            return -1
        return self._client_sock.fileno()

    def has_client(self) -> bool:
        return self._client_sock is not None

    # ---------------------------------------------------- Readable event hooks
    # readable 이벤트 핸들러 ---------------------------------------------------

    def handle_listen_readable(self) -> None:
        """
        Accept a pending connection. Reject extra connections.

        listen 소켓에 대기 중인 연결을 수락한다. 이미 클라이언트가 있으면
        새 연결을 거부 (단일 클라이언트 정책).
        """
        if self._listen_sock is None:
            return
        try:
            client, _ = self._listen_sock.accept()
        except (BlockingIOError, OSError):
            return

        if self._client_sock is not None:
            # Single-client policy: drop the new connection so the existing
            # MCP-wrapper session isn't disturbed.
            # 단일 클라이언트 정책 — 기존 MCP-래퍼 세션을 흔들지 않도록 새
            # 연결을 즉시 닫는다.
            try:
                client.close()
            except OSError:
                pass
            return

        client.setblocking(False)
        self._client_sock = client
        self._read_buffer = b""

    def handle_client_readable(self) -> None:
        """
        Read framed messages from the client and dispatch to on_message.

        클라이언트 소켓에서 라인 단위 JSON을 읽어 on_message로 디스패치한다.
        EOF나 오류 시에는 클라이언트 연결을 닫는다.
        """
        if self._client_sock is None:
            return
        try:
            chunk = self._client_sock.recv(4096)
        except BlockingIOError:
            return
        except OSError:
            self._close_client()
            return
        if not chunk:
            self._close_client()
            return

        self._read_buffer += chunk
        while b"\n" in self._read_buffer:
            line, self._read_buffer = self._read_buffer.split(b"\n", 1)
            if not line:
                continue
            try:
                message = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Drop malformed frames; a misbehaving peer shouldn't crash
                # the wrapper, and there's nothing useful to do with garbage.
                # 잘못된 프레임은 무시. 잘못 동작하는 피어가 래퍼를 죽이지 않도록.
                continue
            self._on_message(message)

    # ----------------------------------------------------------------- Sender
    # 송신 ----------------------------------------------------------------------

    def send(self, message: dict[str, Any]) -> bool:
        """
        Encode `message` as JSON, append newline, and send.

        message를 JSON으로 직렬화한 뒤 개행을 붙여 클라이언트로 전송한다.
        성공 여부를 반환하며, 실패 시 클라이언트 연결을 닫는다.
        """
        if self._client_sock is None:
            return False
        payload = (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            self._client_sock.sendall(payload)
        except OSError:
            self._close_client()
            return False
        return True

    # ----------------------------------------------------------------- Internal

    def _close_client(self) -> None:
        if self._client_sock is None:
            return
        try:
            self._client_sock.close()
        except OSError:
            pass
        self._client_sock = None
        self._read_buffer = b""
