"""
Unix Domain Socket server used by the wrapper to talk with the MCP process
and short-lived hook processes.

Hosts an AF_UNIX SOCK_STREAM socket with two connection types:

- Resident connection (MCP process): exactly one at a time. Long-lived,
  bidirectional. A second resident attempt is rejected (SOCKET_REJECT).
- Short-lived connection (hook process, spawned per prompt): connect,
  send one message carrying ``"client": "hook"``, receive an ack, close.
  Any number may come and go without disturbing the resident slot.

A newly accepted connection is held as *pending* until its first complete
message reveals which type it is — the connection itself carries no
identity. The wrapper exposes the listening fd, the resident client fd,
and all pending fds to its main select() loop so socket I/O is
multiplexed alongside PTY and stdin without an extra thread. Messages
are line-delimited JSON: each line on the wire is one JSON object.

PTY 래퍼가 MCP 프로세스·단발 hook 프로세스와 통신하기 위한 Unix Domain
Socket 서버 모듈.

두 가지 연결 유형을 지원한다:

- 상주 연결 (MCP 프로세스): 동시에 정확히 1개. 장수명·양방향.
  두 번째 상주 시도는 거부된다 (SOCKET_REJECT).
- 단발 연결 (hook 프로세스, 매 프롬프트마다 새로 뜸): 접속 → ``"client":
  "hook"`` 필드를 담은 메시지 1건 송신 → ack 수신 → 종료. 상주 자리를
  건드리지 않고 얼마든지 드나들 수 있다.

접속 자체에는 신원 정보가 없으므로, 새로 수락된 연결은 첫 완전한
메시지가 유형을 드러낼 때까지 *pending* 상태로 보관한다. 래퍼는 listen
fd·상주 client fd·모든 pending fd를 자기 메인 select() 루프에 노출하므로,
별도 스레드 없이 PTY·stdin과 함께 다중화된다. 메시지 프레이밍은 라인 기반
JSON이다 — 와이어상 한 줄이 곧 하나의 JSON 객체에 대응한다.

Hook messages support two reply shapes:

- Fire-and-forget (default): the server dispatches to ``on_message``,
  sends the ack, and closes.
- Deferred reply: when the ``on_hook_message`` callback is configured
  and returns True, connection ownership transfers to the callback's
  side (e.g. a judge worker thread). The server forgets the fd — the
  new owner must eventually send its reply and close the socket. This
  exists for requests whose answer takes seconds (routing judgment):
  an immediate ack could not carry the result.

hook 메시지의 회신 형태는 두 가지다:

- 즉발형 (기본): 서버가 ``on_message``로 디스패치하고 ack를 보낸 뒤
  닫는다.
- 지연 회신형: ``on_hook_message`` 콜백이 설정되어 있고 True를 반환하면
  연결 소유권이 콜백 측(예: 판정 워커 스레드)으로 이관된다. 서버는 해당
  fd를 잊는다 — 새 소유자가 회신 송신과 소켓 닫기를 책임진다. 응답에
  수 초가 걸리는 요청(라우팅 판정)을 위한 유형이다: 즉시 ack로는 결과를
  실어 보낼 수 없다.

지원하는 메시지 종류:
- MCP → 래퍼: handshake_request, action=switch, action=new, session_end_completed
- hook → 래퍼: client="hook"을 담은 단발 메시지 (라우팅 판정 요청 등)
- 래퍼 → MCP: handshake_response(current_session_name), user_action
- 래퍼 → hook: {"type": "ack", "ok": true} 또는 지연 회신 (소유자가 송신)
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

from session_manager import debug_log

# Listen backlog. Engineering parameter — theoretical max simultaneous
# connectors is 2 (one resident MCP + one hook; hooks are serialized by the
# TUI prompt loop), ×4 safety margin for restart races.
# listen 백로그. 공학 파라미터 — 동시 접속 이론 최대치는 2 (상주 MCP 1 +
# hook 1; hook은 TUI 프롬프트 루프에 의해 직렬화됨), 재시작 경합 대비 4배 여유.
_LISTEN_BACKLOG = 8

# Ack payload sent back to a short-lived (hook) client after its message
# has been dispatched.
# 단발(hook) 클라이언트의 메시지를 디스패치한 뒤 돌려주는 ack 페이로드.
_ACK_MESSAGE: dict[str, Any] = {"type": "ack", "ok": True}


class WrapperSocketServer:
    def __init__(
        self,
        socket_path: str,
        on_message: Callable[[dict[str, Any]], None],
        on_hook_message: Callable[[dict[str, Any], socket.socket], bool]
        | None = None,
    ) -> None:
        self.socket_path = socket_path
        self._on_message = on_message
        # Optional deferred-reply dispatcher for hook messages. Returning
        # True means "I took ownership of the socket — I will reply and
        # close it"; the socket is handed over in non-blocking mode, so
        # the new owner should call settimeout() before writing.
        # hook 메시지용 지연 회신 디스패처 (선택). True 반환은 "소켓
        # 소유권을 가져갔다 — 회신과 닫기를 내가 한다"는 뜻이다. 소켓은
        # non-blocking 상태로 이관되므로 새 소유자는 쓰기 전에
        # settimeout()을 호출해야 한다.
        self._on_hook_message = on_hook_message
        self._listen_sock: socket.socket | None = None
        self._client_sock: socket.socket | None = None
        self._read_buffer: bytes = b""
        # Accepted connections whose first message hasn't arrived yet,
        # keyed by fd. The first message decides: hook → ack and close,
        # otherwise → promote to the resident slot.
        # 첫 메시지가 아직 도착하지 않은 수락된 연결 (fd 키). 첫 메시지가
        # 유형을 결정한다: hook → ack 후 종료, 그 외 → 상주 자리로 승격.
        self._pending: dict[int, socket.socket] = {}
        self._pending_buffers: dict[int, bytes] = {}

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
        # Owner-only, regardless of umask (F17): on multi-user machines a
        # world-connectable socket would let any user drive the wrapper.
        # umask 와 무관하게 소유자 전용 (F17) — 다중 사용자 머신에서 아무나
        # connect 가능한 소켓은 래퍼 조종을 허용하게 된다.
        os.chmod(self.socket_path, 0o600)
        sock.listen(_LISTEN_BACKLOG)
        sock.setblocking(False)
        self._listen_sock = sock

    def stop(self) -> None:
        """
        Close the client, pending, and listen sockets and unlink the
        socket file.

        클라이언트·pending·listen 소켓을 닫고 소켓 파일을 제거한다.
        """
        self._close_client()
        for fd in list(self._pending):
            self._close_pending(fd)
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

    @property
    def pending_filenos(self) -> list[int]:
        """
        fds of accepted connections awaiting their first message.

        첫 메시지를 기다리는 수락된 연결들의 fd 목록.
        """
        return list(self._pending)

    def has_client(self) -> bool:
        return self._client_sock is not None

    # ---------------------------------------------------- Readable event hooks
    # readable 이벤트 핸들러 ---------------------------------------------------

    def handle_listen_readable(self) -> None:
        """
        Accept a connection and hold it as pending until its first message
        reveals whether it is a resident (MCP) or short-lived (hook) client.

        listen 소켓의 연결을 수락해 pending 으로 보관한다. 첫 메시지가
        상주(MCP)인지 단발(hook)인지 드러낼 때까지 유형을 확정하지 않는다.
        """
        if self._listen_sock is None:
            return
        try:
            client, _ = self._listen_sock.accept()
        except (BlockingIOError, OSError):
            return

        client.setblocking(False)
        fd = client.fileno()
        self._pending[fd] = client
        self._pending_buffers[fd] = b""
        debug_log.log(
            "SOCKET_ACCEPT",
            "SYSTEM",
            {"client_fd": fd, "state": "pending"},
        )

    def handle_pending_readable(self, fd: int) -> None:
        """
        Read from a pending connection and settle its type on the first
        complete message: ``client == "hook"`` → dispatch, ack, close;
        otherwise → promote to the resident slot (or reject if occupied).

        pending 연결에서 읽어 첫 완전한 메시지로 유형을 확정한다.
        ``client == "hook"`` → 디스패치·ack·종료, 그 외 → 상주 자리로
        승격 (자리가 차 있으면 거부).
        """
        sock = self._pending.get(fd)
        if sock is None:
            return
        try:
            chunk = sock.recv(4096)
        except BlockingIOError:
            return
        except OSError:
            self._close_pending(fd)
            return
        if not chunk:
            self._close_pending(fd)
            return

        self._pending_buffers[fd] += chunk
        while b"\n" in self._pending_buffers[fd]:
            line, self._pending_buffers[fd] = self._pending_buffers[fd].split(
                b"\n", 1
            )
            if not line:
                continue
            try:
                message = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Same policy as the resident path: drop malformed frames.
                # 상주 경로와 동일 정책 — 잘못된 프레임은 버린다.
                debug_log.log(
                    "SOCKET_RECV",
                    "SYSTEM",
                    {
                        "direction": "wrapper<-pending",
                        "dropped": True,
                        "reason": "malformed_frame",
                        "len": len(line),
                    },
                )
                continue

            if isinstance(message, dict) and message.get("client") == "hook":
                self._settle_hook(fd, sock, message)
            else:
                self._settle_resident(fd, sock, message)
            return

    def _settle_hook(
        self, fd: int, sock: socket.socket, message: dict[str, Any]
    ) -> None:
        """
        Complete a short-lived exchange. Default path: dispatch the
        message, send the ack, close the connection. Deferred path: if
        ``on_hook_message`` takes ownership, forget the fd and let the
        new owner reply and close. Anything buffered past the first
        message is discarded — the protocol is one message per
        connection.

        단발 왕복을 완결한다. 기본 경로: 메시지 디스패치 → ack 송신 →
        연결 종료. 지연 경로: ``on_hook_message``가 소유권을 가져가면
        fd를 잊고 회신·닫기를 새 소유자에게 맡긴다. 첫 메시지 이후
        버퍼에 남은 데이터는 버린다 — 프로토콜은 연결당 메시지 1건이다.
        """
        debug_log.log(
            "SOCKET_RECV",
            "SYSTEM",
            {
                "direction": "wrapper<-hook",
                "type": message.get("type"),
                "action": message.get("action"),
                "payload": message,
            },
        )
        if self._on_hook_message is not None:
            try:
                taken = bool(self._on_hook_message(message, sock))
            except Exception:
                # A broken dispatcher must not crash the wrapper's I/O
                # loop. Close without ack — the hook side times out and
                # passes the prompt through (graceful degradation).
                # 디스패처의 예외가 래퍼 I/O 루프를 죽여선 안 된다. ack
                # 없이 닫는다 — hook 측은 타임아웃 후 프롬프트를
                # 통과시킨다 (graceful degradation).
                debug_log.log(
                    "SOCKET_DETACH",
                    "SYSTEM",
                    {"client_fd": fd, "error": "hook_dispatcher_raised"},
                )
                self._close_pending(fd)
                return
            if taken:
                # Ownership transferred — forget the fd without closing.
                # 소유권 이관 — 닫지 않고 fd만 잊는다.
                self._pending.pop(fd, None)
                self._pending_buffers.pop(fd, None)
                debug_log.log(
                    "SOCKET_DETACH",
                    "SYSTEM",
                    {"client_fd": fd},
                )
                return
        self._on_message(message)
        try:
            payload = (
                json.dumps(_ACK_MESSAGE, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            sock.sendall(payload)
        except OSError:
            # The hook may have given up (its own timeout); the message was
            # already dispatched, so there is nothing to roll back.
            # hook 이 자체 타임아웃으로 먼저 떠났을 수 있다. 메시지는 이미
            # 디스패치되었으므로 되돌릴 것이 없다.
            pass
        self._close_pending(fd)

    def _settle_resident(
        self, fd: int, sock: socket.socket, message: dict[str, Any]
    ) -> None:
        """
        Promote a pending connection to the resident slot, or reject it if
        the slot is occupied (single-resident policy).

        pending 연결을 상주 자리로 승격하거나, 자리가 차 있으면 거부한다
        (단일 상주 정책).
        """
        if self._client_sock is not None:
            # Single-resident policy: drop the new connection so the
            # existing MCP-wrapper session isn't disturbed.
            # 단일 상주 정책 — 기존 MCP-래퍼 세션을 흔들지 않도록 새
            # 연결을 닫는다.
            debug_log.log(
                "SOCKET_REJECT",
                "SYSTEM",
                {"reason": "second_client_attempted"},
            )
            self._close_pending(fd)
            return

        # Carry over any bytes buffered past the first message so frames
        # arriving in the same chunk aren't lost.
        # 첫 메시지 뒤에 버퍼링된 바이트를 승계 — 같은 chunk 로 도착한
        # 후속 프레임이 유실되지 않도록.
        remainder = self._pending_buffers.pop(fd, b"")
        del self._pending[fd]
        self._client_sock = sock
        self._read_buffer = remainder
        debug_log.log(
            "SOCKET_ACCEPT",
            "SYSTEM",
            {"client_fd": fd, "state": "resident"},
        )
        self._dispatch_resident_message(message)
        self._drain_resident_buffer()

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
        self._drain_resident_buffer()

    def _drain_resident_buffer(self) -> None:
        """
        Parse and dispatch every complete line in the resident buffer.

        상주 버퍼에 쌓인 완전한 라인을 모두 파싱해 디스패치한다.
        """
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
                debug_log.log(
                    "SOCKET_RECV",
                    "SYSTEM",
                    {
                        "direction": "wrapper<-mcp",
                        "dropped": True,
                        "reason": "malformed_frame",
                        "len": len(line),
                    },
                )
                continue
            self._dispatch_resident_message(message)

    def _dispatch_resident_message(self, message: Any) -> None:
        debug_log.log(
            "SOCKET_RECV",
            "MCP_TOOL",
            {
                "direction": "wrapper<-mcp",
                "type": message.get("type") if isinstance(message, dict) else None,
                "action": message.get("action")
                if isinstance(message, dict)
                else None,
                "payload": message,
            },
        )
        self._on_message(message)

    # ----------------------------------------------------------------- Sender
    # 송신 ----------------------------------------------------------------------

    def send(self, message: dict[str, Any]) -> bool:
        """
        Encode `message` as JSON, append newline, and send.

        message를 JSON으로 직렬화한 뒤 개행을 붙여 클라이언트로 전송한다.
        성공 여부를 반환하며, 실패 시 클라이언트 연결을 닫는다.
        """
        # Single SOCKET_SEND checkpoint on the wrapper→MCP direction.
        # 단일 SOCKET_SEND 체크포인트 (wrapper → MCP 방향).
        debug_log.log(
            "SOCKET_SEND",
            "WRAPPER",
            {
                "direction": "wrapper->mcp",
                "type": message.get("type"),
                "action": message.get("action"),
                "payload": message,
                "has_client": self._client_sock is not None,
            },
        )
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

    def _close_pending(self, fd: int) -> None:
        sock = self._pending.pop(fd, None)
        self._pending_buffers.pop(fd, None)
        if sock is None:
            return
        try:
            sock.close()
        except OSError:
            pass
