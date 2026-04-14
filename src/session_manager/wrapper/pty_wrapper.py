"""
PTY wrapper that mediates between the user terminal and Claude Code.

Provides the I/O loop skeleton on which additional handlers hang
SWITCH/NEW logic, MCP socket integration, and stdin slash-command
interception. By itself it is a transparent passthrough: spawns Claude
Code on a PTY, forwards stdin to the PTY master, forwards PTY output to
stdout, and detects the input prompt so the rest of the wrapper can pick
a safe moment to inject text.

사용자 터미널과 Claude Code 프로세스 사이에 끼어들어 양방향 I/O를
중계하는 PTY 래퍼 모듈이다.

이 모듈은 I/O 루프의 골격만 제공한다. 세션 전환(SWITCH/NEW) 처리,
MCP 소켓 통합, stdin 슬래시 커맨드 가로채기 같은 상위 로직은 별도의
핸들러로 이 골격 위에 얹어 확장한다.

단독으로 사용할 경우 투명한 패스스루로 동작한다. Claude Code를 PTY에
띄운 뒤, 사용자가 입력한 바이트는 PTY master로 그대로 전달하고, PTY가
출력하는 바이트는 stdout으로 흘려보낸다. 동시에 출력 스트림에서 입력
프롬프트(figures.pointer "❯" 직후의 반전 커서 시퀀스)를 감지해, 후속
로직이 텍스트를 안전하게 주입할 수 있는 시점을 파악할 수 있게 한다.
"""

from __future__ import annotations

import os
import select
import signal
import sys
import termios
import tty
from typing import Any, Literal

import pexpect

from session_manager.wrapper.socket_server import WrapperSocketServer

# figures.pointer "❯" (UTF-8 E2 9D AF) followed by chalk.inverse "\x1b[7m"
# is what ink-text-input renders for the prompt cursor. Verified against
# the Claude Code binary (strings shows both \u276F and inverse: [7, 27]).
#
# ink-text-input이 프롬프트 커서를 그릴 때 쓰는 패턴. "❯"(UTF-8 E2 9D AF)
# 뒤에 chalk.inverse "\x1b[7m"이 이어지는 형태. Claude Code 바이너리에서
# \u276F와 inverse: [7, 27] 정의를 strings로 확인.
PROMPT_POINTER = b"\xe2\x9d\xaf"
INVERSE_VIDEO_START = b"\x1b[7m"

# Cap the prompt-detection buffer to prevent unbounded growth in long
# sessions. When truncating, keep enough tail so an in-flight multi-byte
# sequence at the boundary isn't sliced apart.
#
# 장시간 세션에서 프롬프트 감지 버퍼 무한 증가 방지. 잘라낼 때는
# 끝부분을 충분히 남겨 경계에 걸친 멀티바이트 시퀀스 보존.
OUTPUT_BUFFER_CAP = 16 * 1024
OUTPUT_BUFFER_TAIL_KEEP = 256

Mode = Literal["passthrough", "filtering"]


class SessionManagerWrapper:
    def __init__(self, socket_path: str, claude_args: list[str]) -> None:
        self.socket_path = socket_path
        self.claude_args = list(claude_args)

        self.child: pexpect.spawn | None = None
        self.pty_fd: int = -1

        self.mode: Mode = "passthrough"
        self.output_buffer: bytes = b""
        self.input_queue: bytes = b""
        self.stdin_line_buffer: bytes = b""

        self._stdin_fd: int = sys.stdin.fileno()
        self._stdout_fd: int = sys.stdout.fileno()
        self._original_stdin_attrs: list[Any] | None = None
        self._previous_winch_handler: Any = None

        self.socket_server = WrapperSocketServer(
            socket_path=socket_path,
            on_message=self._handle_mcp_signal,
        )

    def start(self) -> None:
        """
        Spawn Claude Code on a PTY and run the I/O loop until it exits.

        Claude Code를 PTY에 띄우고 종료될 때까지 I/O 루프를 실행한다.
        """
        # Listen on the Unix socket before spawning the child so the MCP
        # process can connect at any point after we hand off control.
        # 자식 spawn 전에 Unix 소켓을 listen 상태로 만들어, 이후 어느 시점에
        # MCP가 접속해도 받을 수 있도록 한다.
        self.socket_server.start()

        self.child = pexpect.spawn(
            "claude",
            self.claude_args,
            encoding=None,
            echo=False,
        )
        self.pty_fd = self.child.fileno()

        self._enter_raw_mode()
        self._install_winch_handler()
        self._sync_winsize()

        try:
            self._io_loop()
        finally:
            self._restore_terminal()
            self.socket_server.stop()

    # ------------------------------------------------------------------ I/O loop
    # I/O 루프 ------------------------------------------------------------------

    def _io_loop(self) -> None:
        assert self.child is not None
        while self.child.isalive():
            # Build the watch list each tick: socket fds appear/disappear
            # as MCP connects and disconnects.
            # 매 틱마다 watch 대상을 새로 구성. 소켓 fd는 MCP의 연결·해제에
            # 따라 등장하거나 사라진다.
            watch_fds: list[int] = [self.pty_fd, self._stdin_fd]
            listen_fd = self.socket_server.listen_fileno
            client_fd = self.socket_server.client_fileno
            if listen_fd >= 0:
                watch_fds.append(listen_fd)
            if client_fd >= 0:
                watch_fds.append(client_fd)

            try:
                # 100 ms timeout polls child liveness without burning CPU.
                # 100ms 타임아웃으로 자식 생존 여부를 폴링 (CPU 낭비 방지).
                readable, _, _ = select.select(watch_fds, [], [], 0.1)
            except InterruptedError:
                # A signal (e.g. SIGWINCH) interrupted select; just retry.
                # 시그널(예: SIGWINCH)로 select가 중단된 경우 단순 재시도.
                continue
            except OSError:
                break

            if self.pty_fd in readable:
                if not self._handle_pty_readable():
                    break

            if self._stdin_fd in readable:
                self._handle_stdin_readable()

            if listen_fd >= 0 and listen_fd in readable:
                self.socket_server.handle_listen_readable()

            if client_fd >= 0 and client_fd in readable:
                self.socket_server.handle_client_readable()

        self._drain_pty()

    def _handle_pty_readable(self) -> bool:
        try:
            chunk = os.read(self.pty_fd, 4096)
        except OSError:
            return False
        if not chunk:
            # EOF on PTY master means the child closed its end.
            # PTY master에서의 EOF — 자식 프로세스가 자기 쪽을 닫음.
            return False

        self.output_buffer += chunk
        if self._detect_prompt(self.output_buffer):
            self._handle_prompt_detected()
        self._truncate_output_buffer()

        if self.mode == "passthrough":
            os.write(self._stdout_fd, chunk)
        return True

    def _handle_stdin_readable(self) -> None:
        try:
            chunk = os.read(self._stdin_fd, 4096)
        except OSError:
            return
        if not chunk:
            return

        if self.mode == "filtering":
            # Buffer keystrokes during injection; drained back to the PTY
            # when filtering ends.
            # 주입 중 들어온 키 입력은 큐에 보관, 필터링 종료 시 PTY로 일괄 반영.
            self.input_queue += chunk
            return

        self.stdin_line_buffer += chunk
        while b"\n" in self.stdin_line_buffer:
            line, self.stdin_line_buffer = self.stdin_line_buffer.split(b"\n", 1)
            self._handle_user_line(line + b"\n")

        # Forward keystrokes to the PTY so Ink can render them in real time.
        # Slash-command interception will later gate this path differently.
        #
        # Ink가 실시간으로 렌더링할 수 있도록 키 입력을 PTY로 즉시 전달.
        # 슬래시 커맨드 가로채기 도입 시 이 경로를 분기 처리할 예정.
        os.write(self.pty_fd, chunk)

    # --------------------------------------------------- Detection & injection
    # 프롬프트 감지 / 텍스트 주입 -----------------------------------------------

    def _detect_prompt(self, buffer: bytes) -> bool:
        idx = buffer.rfind(PROMPT_POINTER)
        if idx == -1:
            return False
        # Only count an inverse-video sequence in a small window after the
        # pointer; a stale "❯" elsewhere in the buffer must not match.
        #
        # 표지자 직후 좁은 윈도우 안의 반전 시퀀스만 인정. 버퍼 다른 위치에
        # 남아 있는 오래된 "❯"가 잘못 매칭되지 않도록 함.
        return INVERSE_VIDEO_START in buffer[idx : idx + 64]

    def _inject_text(self, text: str) -> None:
        os.write(self.pty_fd, text.encode("utf-8"))

    # ------------------------------------------------------------ Extension hooks
    # 확장 지점 -------------------------------------------------------------------

    def _handle_prompt_detected(self) -> None:
        return

    def _handle_user_line(self, line: bytes) -> None:
        return

    def _handle_mcp_signal(self, message: dict) -> None:
        return

    # ------------------------------------------------------ Buffer management
    # 버퍼 관리 -----------------------------------------------------------------

    def _truncate_output_buffer(self) -> None:
        if len(self.output_buffer) <= OUTPUT_BUFFER_CAP:
            return
        # Drop the front but keep the tail so a partial multi-byte prompt
        # sequence at the boundary survives across truncation.
        #
        # 앞부분은 버리고 끝부분만 유지. 경계에 걸친 멀티바이트 프롬프트
        # 시퀀스가 다음 매칭에서도 살아남도록 함.
        self.output_buffer = self.output_buffer[-OUTPUT_BUFFER_TAIL_KEEP:]

    def _drain_pty(self) -> None:
        try:
            while True:
                chunk = os.read(self.pty_fd, 4096)
                if not chunk:
                    return
                if self.mode == "passthrough":
                    os.write(self._stdout_fd, chunk)
        except OSError:
            return

    # --------------------------------------------------------- Terminal state
    # 터미널 상태 관리 ---------------------------------------------------------

    def _enter_raw_mode(self) -> None:
        if not os.isatty(self._stdin_fd):
            return
        self._original_stdin_attrs = termios.tcgetattr(self._stdin_fd)
        tty.setraw(self._stdin_fd)

    def _restore_terminal(self) -> None:
        if self._original_stdin_attrs is not None:
            termios.tcsetattr(
                self._stdin_fd, termios.TCSADRAIN, self._original_stdin_attrs
            )
            self._original_stdin_attrs = None
        if self._previous_winch_handler is not None:
            signal.signal(signal.SIGWINCH, self._previous_winch_handler)
            self._previous_winch_handler = None

    def _install_winch_handler(self) -> None:
        self._previous_winch_handler = signal.signal(
            signal.SIGWINCH, self._on_resize
        )

    def _on_resize(self, signum: int, frame: Any) -> None:
        self._sync_winsize()

    def _sync_winsize(self) -> None:
        if self.pty_fd < 0 or not os.isatty(self._stdout_fd):
            return
        try:
            rows, cols = termios.tcgetwinsize(self._stdout_fd)
        except OSError:
            return
        try:
            termios.tcsetwinsize(self.pty_fd, (rows, cols))
        except OSError:
            return
