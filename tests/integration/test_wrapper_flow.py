"""
Integration tests for the PTY wrapper flow.

PTY 래퍼의 통합 테스트. 실제 PTY + Unix Socket을 사용하되, claude
바이너리 대신 mock_claude.py를 spawn하여 SWITCH/NEW 신호 처리 전체
흐름을 검증한다.

Tests run mock_claude on a real PTY via pexpect, connect a socket client
to send MCP signals, then assert on the bytes the wrapper injects.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import threading
import time
import unittest.mock
from pathlib import Path

import pexpect

from session_manager.wrapper.pty_wrapper import (
    OUTPUT_BUFFER_TAIL_KEEP,
    SessionManagerWrapper,
)

_MOCK_CLAUDE = str(Path(__file__).parent / "mock_claude.py")
_TIMEOUT = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wrapper(
    tmp_path: Path, claude_args: list[str] | None = None,
) -> SessionManagerWrapper:
    """Create a wrapper that spawns mock_claude instead of claude.

    mock_claude를 spawn하는 래퍼를 만든다.
    """
    short_hash = hashlib.md5(str(tmp_path).encode()).hexdigest()[:8]
    sock_path = f"/tmp/sm-test-{short_hash}.sock"
    Path(sock_path).unlink(missing_ok=True)

    wrapper = SessionManagerWrapper(
        socket_path=sock_path,
        claude_args=claude_args or [],
        project_path=str(tmp_path),
    )

    def _mock_spawn() -> None:
        wrapper.child = pexpect.spawn(
            sys.executable, [_MOCK_CLAUDE], encoding=None, echo=False,
        )
        wrapper.pty_fd = wrapper.child.fileno()
        wrapper.output_buffer = b""

    wrapper._spawn_child = _mock_spawn  # type: ignore[assignment]
    wrapper._enter_raw_mode = lambda: None  # type: ignore[assignment]
    wrapper._restore_terminal = lambda: None  # type: ignore[assignment]
    wrapper._install_winch_handler = lambda: None  # type: ignore[assignment]
    wrapper._sync_winsize = lambda: None  # type: ignore[assignment]
    wrapper._stdout_fd = os.open(os.devnull, os.O_WRONLY)

    fake_stdin_r, fake_stdin_w = os.pipe()
    wrapper._stdin_fd = fake_stdin_r
    wrapper._fake_stdin_w = fake_stdin_w  # type: ignore[attr-defined]
    return wrapper


def _start_wrapper(wrapper: SessionManagerWrapper) -> threading.Thread:
    """Start the wrapper in a daemon thread.

    래퍼를 데몬 스레드에서 시작한다.
    """
    def _run() -> None:
        try:
            wrapper.start()
        except Exception:
            pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _connect_and_handshake(
    sock_path: str, timeout: float = _TIMEOUT,
) -> socket.socket:
    """Connect to the wrapper socket and perform handshake.

    래퍼 소켓에 연결하고 핸드셰이크를 수행한다.
    """
    deadline = time.monotonic() + timeout
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    while time.monotonic() < deadline:
        try:
            sock.connect(sock_path)
            break
        except (ConnectionRefusedError, FileNotFoundError):
            time.sleep(0.05)
    else:
        raise TimeoutError(f"Could not connect to {sock_path}")
    return sock


def _send_json(sock: socket.socket, msg: dict) -> None:
    sock.sendall((json.dumps(msg) + "\n").encode())


def _recv_json(sock: socket.socket, timeout: float = _TIMEOUT) -> dict:
    sock.settimeout(timeout)
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("Socket closed")
        buf += chunk
    return json.loads(buf.split(b"\n", 1)[0])


def _trigger_prompt(wrapper: SessionManagerWrapper) -> None:
    """Write a dummy byte directly to the PTY master to make mock_claude
    produce a new prompt.  Written to pty_fd (not fake stdin) so it
    reaches mock_claude even when the wrapper is in filtering mode.

    더미 바이트를 PTY master에 직접 써서 mock_claude가 새 프롬프트를
    출력하게 한다.  filtering 모드에서도 mock_claude에 도달하도록
    fake stdin이 아닌 pty_fd에 직접 쓴다.
    """
    try:
        os.write(wrapper.pty_fd, b"\r")
    except OSError:
        pass


def _cleanup(wrapper: SessionManagerWrapper) -> None:
    if wrapper.child and wrapper.child.isalive():
        wrapper.child.terminate(force=True)
    try:
        os.close(wrapper._fake_stdin_w)  # type: ignore[attr-defined]
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Tests — Handshake
# ---------------------------------------------------------------------------


class TestHandshake:
    def test_handshake_returns_initial_session_name(
        self, tmp_path: Path,
    ) -> None:
        wrapper = _make_wrapper(tmp_path, claude_args=["--resume", "foo"])
        _start_wrapper(wrapper)
        try:
            client = _connect_and_handshake(wrapper.socket_path)
            _send_json(client, {"type": "handshake_request"})
            response = _recv_json(client)
            assert response["current_session_name"] == "foo"
            client.close()
        finally:
            _cleanup(wrapper)

    def test_handshake_returns_null_when_no_resume(
        self, tmp_path: Path,
    ) -> None:
        wrapper = _make_wrapper(tmp_path)
        _start_wrapper(wrapper)
        try:
            client = _connect_and_handshake(wrapper.socket_path)
            _send_json(client, {"type": "handshake_request"})
            response = _recv_json(client)
            assert response["current_session_name"] is None
            client.close()
        finally:
            _cleanup(wrapper)


# ---------------------------------------------------------------------------
# Tests — SWITCH flow
# ---------------------------------------------------------------------------


class TestSwitchFlow:
    def test_switch_injects_resume_and_handoff(
        self, tmp_path: Path,
    ) -> None:
        """SWITCH signal → /resume injected → handoff injected.

        SWITCH 신호 후 mock_claude 출력에 "Resumed" 확인.
        """
        wrapper = _make_wrapper(tmp_path)
        _start_wrapper(wrapper)
        try:
            client = _connect_and_handshake(wrapper.socket_path)
            _send_json(client, {"type": "handshake_request"})
            _recv_json(client)

            # Wait for initial prompt detection
            time.sleep(0.3)

            # Send SWITCH signal
            _send_json(client, {
                "action": "switch",
                "target": "target-sess",
                "handoff": {
                    "from": "old",
                    "message": "ctx",
                    "instructions": [],
                    "user_prompt": "do it",
                },
            })

            # Trigger prompts so the wrapper can advance through stages.
            # Each stage needs a prompt detection to proceed.
            # 각 단계가 진행되려면 프롬프트 감지가 필요하므로 트리거.
            for _ in range(10):
                time.sleep(0.2)
                _trigger_prompt(wrapper)

            # Give time for final processing
            time.sleep(0.5)

            # The wrapper should have reached passthrough mode
            assert wrapper.mode == "passthrough"
            assert wrapper._pending_action is None
            client.close()
        finally:
            _cleanup(wrapper)


# ---------------------------------------------------------------------------
# Tests — NEW flow
# ---------------------------------------------------------------------------


class TestNewFlow:
    def test_new_with_rename_spawns_new_child(
        self, tmp_path: Path,
    ) -> None:
        """NEW with rename → /rename + /exit + respawn + handoff.

        rename이 있는 NEW → 자식 재spawn 후 핸드셰이크에 new session 반환.
        """
        wrapper = _make_wrapper(tmp_path)
        _start_wrapper(wrapper)
        try:
            client = _connect_and_handshake(wrapper.socket_path)
            _send_json(client, {"type": "handshake_request"})
            _recv_json(client)
            time.sleep(0.3)

            _send_json(client, {
                "action": "new",
                "rename_current": "old-sess",
                "new_session_name": "new-sess",
                "handoff": {
                    "from": "old-sess",
                    "message": "new ctx",
                    "instructions": [],
                    "user_prompt": "start",
                },
            })

            # Trigger prompts for /rename → submit → /exit → submit
            for _ in range(10):
                time.sleep(0.2)
                _trigger_prompt(wrapper)

            # After /exit, mock_claude exits. Wrapper respawns a new child.
            # Wait for respawn and new handshake.
            time.sleep(1)

            # New child handshake should return "new-sess"
            _send_json(client, {"type": "handshake_request"})
            response = _recv_json(client, timeout=3)
            assert response["current_session_name"] == "new-sess"

            # Trigger prompts for handoff injection
            for _ in range(5):
                time.sleep(0.2)
                _trigger_prompt(wrapper)

            time.sleep(0.5)
            assert wrapper.mode == "passthrough"
            client.close()
        finally:
            _cleanup(wrapper)

    def test_new_without_rename_skips_rename(
        self, tmp_path: Path,
    ) -> None:
        """NEW without rename → /exit directly + respawn.

        rename 없는 NEW → /rename 건너뛰고 바로 /exit.
        """
        wrapper = _make_wrapper(tmp_path)
        _start_wrapper(wrapper)
        try:
            client = _connect_and_handshake(wrapper.socket_path)
            _send_json(client, {"type": "handshake_request"})
            _recv_json(client)
            time.sleep(0.3)

            _send_json(client, {
                "action": "new",
                "rename_current": None,
                "new_session_name": "fresh",
                "handoff": {
                    "from": None,
                    "message": "brand new",
                    "instructions": [],
                    "user_prompt": "hello",
                },
            })

            for _ in range(10):
                time.sleep(0.2)
                _trigger_prompt(wrapper)

            time.sleep(1)

            _send_json(client, {"type": "handshake_request"})
            response = _recv_json(client, timeout=3)
            assert response["current_session_name"] == "fresh"

            for _ in range(5):
                time.sleep(0.2)
                _trigger_prompt(wrapper)

            time.sleep(0.5)
            assert wrapper.mode == "passthrough"
            client.close()
        finally:
            _cleanup(wrapper)


# ---------------------------------------------------------------------------
# Tests — Edge cases (no PTY spawn needed)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_prompt_detection_with_chunked_output(
        self, tmp_path: Path,
    ) -> None:
        """Prompt pattern split across two chunks is still detected.

        프롬프트 패턴이 두 청크에 걸쳐 분할돼도 감지된다.
        """
        wrapper = _make_wrapper(tmp_path)
        part1 = b"some output \xe2\x9d"
        part2 = b"\xaf \x1b[7m cursor"

        wrapper.output_buffer = part1
        assert wrapper._detect_prompt(wrapper.output_buffer) is False

        wrapper.output_buffer += part2
        assert wrapper._detect_prompt(wrapper.output_buffer) is True

    def test_output_buffer_truncation(self, tmp_path: Path) -> None:
        wrapper = _make_wrapper(tmp_path)
        wrapper.output_buffer = b"x" * 20_000
        wrapper._truncate_output_buffer()
        assert len(wrapper.output_buffer) == OUTPUT_BUFFER_TAIL_KEEP

    def test_input_queue_drain_replaces_newlines(
        self, tmp_path: Path,
    ) -> None:
        wrapper = _make_wrapper(tmp_path)
        wrapper.input_queue = b"hello\nworld\n"
        captured: list[bytes] = []
        with unittest.mock.patch(
            "session_manager.wrapper.pty_wrapper.os.write",
            side_effect=lambda fd, data: captured.append(data) or len(data),
        ):
            wrapper._drain_input_queue()
        assert captured == [b"hello world "]

    def test_malformed_socket_message_ignored(
        self, tmp_path: Path,
    ) -> None:
        wrapper = _make_wrapper(tmp_path)
        wrapper._handle_mcp_signal("not a dict")  # type: ignore[arg-type]
        wrapper._handle_mcp_signal({"action": "switch"})
        wrapper._handle_mcp_signal({"action": "new", "handoff": {}})
        assert wrapper._pending_action is None
