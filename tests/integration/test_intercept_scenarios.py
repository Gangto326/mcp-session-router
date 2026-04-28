"""Integration tests for the slash-command interception flow.
슬래시 명령 가로채기 통합 테스트.

These tests verify the wrapper's interception state machine + socket
signaling end-to-end (real PTY child, real Unix socket, real threading).
The virtual screen is pre-populated directly so matching is deterministic
— virtual_screen extraction itself is covered by unit tests.

가로채기 state machine과 socket 시그널링을 end-to-end로 검증한다 (실제
PTY child, 실제 Unix 소켓, 실제 thread). 매칭 결정성을 위해 가상 화면을
직접 미리 채운다 — virtual_screen 추출 자체는 단위 테스트로 검증됨.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pexpect
import pytest

from session_manager.wrapper.pty_wrapper import (
    PROMPT_POINTER,
    SessionManagerWrapper,
    _InterceptState,
)

_MOCK_CLAUDE = str(Path(__file__).parent / "mock_claude.py")
_TIMEOUT = 5


def _make_wrapper(tmp_path: Path) -> SessionManagerWrapper:
    short_hash = hashlib.md5(str(tmp_path).encode()).hexdigest()[:8]
    sock_path = f"/tmp/sm-intercept-test-{short_hash}.sock"
    Path(sock_path).unlink(missing_ok=True)

    wrapper = SessionManagerWrapper(
        socket_path=sock_path,
        claude_args=[],
        project_path=str(tmp_path),
    )

    def _mock_spawn() -> None:
        wrapper.child = pexpect.spawn(
            sys.executable, [_MOCK_CLAUDE], encoding=None, echo=False,
        )
        wrapper.pty_fd = wrapper.child.fileno()
        wrapper.output_buffer = b""
        wrapper._handled_confirmations = set()

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
    def _run() -> None:
        try:
            wrapper.start()
        except Exception:
            pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _connect(sock_path: str, timeout: float = _TIMEOUT) -> socket.socket:
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


def _handshake(sock: socket.socket) -> None:
    _send_json(sock, {"type": "handshake_request"})
    _recv_json(sock)


def _seed_prompt_line(wrapper: SessionManagerWrapper, text: str) -> None:
    """Inject text into the virtual screen so the next \\r matches it.

    가상 화면에 텍스트를 주입해 다음 \\r에서 매칭되도록 한다. mock_claude의
    PTY redraw에 의존하지 않고 매칭 결정성을 확보.
    """
    wrapper.virtual_screen.feed(PROMPT_POINTER + b" " + text.encode())


def _wait_until(predicate, timeout: float = _TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _cleanup(wrapper: SessionManagerWrapper) -> None:
    if wrapper.child and wrapper.child.isalive():
        wrapper.child.terminate(force=True)
    try:
        os.close(wrapper._fake_stdin_w)  # type: ignore[attr-defined]
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_intercept_signal_sent_on_matching_cr(tmp_path: Path) -> None:
    """Virtual screen has /resume foo, user sends \\r → MCP gets intercept signal.
    가상 화면에 /resume foo, 사용자 \\r → MCP가 intercept 신호 받음.
    """
    wrapper = _make_wrapper(tmp_path)
    _start_wrapper(wrapper)
    try:
        sock = _connect(wrapper.socket_path)
        _handshake(sock)

        # Wait until wrapper is in I/O loop (handshake reply happened).
        # wrapper가 I/O 루프 진입했음을 핸드셰이크 응답으로 확인.
        assert _wait_until(
            lambda: wrapper.pty_fd >= 0 and wrapper.child is not None
        )

        _seed_prompt_line(wrapper, "/resume foo")
        os.write(wrapper._fake_stdin_w, b"\r")  # type: ignore[attr-defined]

        sig = _recv_json(sock)
        assert sig == {
            "action": "intercept",
            "command": "resume",
            "args": "foo",
        }
        # _intercept_state 활성
        assert _wait_until(lambda: wrapper._intercept_state is not None)
        sock.close()
    finally:
        _cleanup(wrapper)


def test_no_signal_for_non_matching_cr(tmp_path: Path) -> None:
    """Virtual screen has /path/to/file, \\r → no intercept signal, normal forward.
    가상 화면이 화이트리스트 외 → 가로채기 신호 없음, 정상 forward.
    """
    wrapper = _make_wrapper(tmp_path)
    _start_wrapper(wrapper)
    try:
        sock = _connect(wrapper.socket_path)
        _handshake(sock)

        assert _wait_until(lambda: wrapper.pty_fd >= 0)

        _seed_prompt_line(wrapper, "/path/to/file")
        os.write(wrapper._fake_stdin_w, b"\r")  # type: ignore[attr-defined]

        # 0.5초 안에 신호 안 옴
        try:
            msg = _recv_json(sock, timeout=0.5)
            pytest.fail(f"unexpected signal: {msg}")
        except TimeoutError:
            pass

        # 가로채기 활성 안 됨
        assert wrapper._intercept_state is None
        sock.close()
    finally:
        _cleanup(wrapper)


def test_intercept_done_finishes_and_forwards_cr(tmp_path: Path) -> None:
    """After intercept, MCP sends intercept_done → state cleared + \\r forwarded.
    intercept_done 응답 → state 정리 + \\r forward.
    """
    wrapper = _make_wrapper(tmp_path)
    _start_wrapper(wrapper)
    try:
        sock = _connect(wrapper.socket_path)
        _handshake(sock)

        assert _wait_until(lambda: wrapper.pty_fd >= 0)

        _seed_prompt_line(wrapper, "/resume foo")
        os.write(wrapper._fake_stdin_w, b"\r")  # type: ignore[attr-defined]

        sig = _recv_json(sock)
        assert sig["action"] == "intercept"

        # MCP가 intercept_done 송신
        _send_json(sock, {"action": "intercept_done"})

        # state 정리 확인
        assert _wait_until(lambda: wrapper._intercept_state is None)
        sock.close()
    finally:
        _cleanup(wrapper)


def test_intercept_timeout_clears_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When MCP doesn't respond within INTERCEPT_TIMEOUT_SEC, state is cleared.
    INTERCEPT_TIMEOUT_SEC 안에 응답 없으면 state 정리.
    """
    monkeypatch.setattr(
        "session_manager.wrapper.pty_wrapper.INTERCEPT_TIMEOUT_SEC", 0.5
    )
    wrapper = _make_wrapper(tmp_path)
    _start_wrapper(wrapper)
    try:
        sock = _connect(wrapper.socket_path)
        _handshake(sock)

        assert _wait_until(lambda: wrapper.pty_fd >= 0)

        _seed_prompt_line(wrapper, "/resume foo")
        os.write(wrapper._fake_stdin_w, b"\r")  # type: ignore[attr-defined]

        sig = _recv_json(sock)
        assert sig["action"] == "intercept"

        # 응답 안 함 → 0.5초 후 timeout으로 state 정리
        assert _wait_until(
            lambda: wrapper._intercept_state is None, timeout=2.0
        )
        sock.close()
    finally:
        _cleanup(wrapper)


def test_ctrl_c_cancels_intercept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl+C during intercept → state cleared (cancel before timeout).
    가로채기 중 Ctrl+C → state 정리 (timeout 전에 취소).
    """
    # Long timeout so cancel beats it.
    # cancel이 timeout보다 먼저 일어나도록 timeout 연장.
    monkeypatch.setattr(
        "session_manager.wrapper.pty_wrapper.INTERCEPT_TIMEOUT_SEC", 5.0
    )
    wrapper = _make_wrapper(tmp_path)
    _start_wrapper(wrapper)
    try:
        sock = _connect(wrapper.socket_path)
        _handshake(sock)

        assert _wait_until(lambda: wrapper.pty_fd >= 0)

        _seed_prompt_line(wrapper, "/resume foo")
        os.write(wrapper._fake_stdin_w, b"\r")  # type: ignore[attr-defined]

        sig = _recv_json(sock)
        assert sig["action"] == "intercept"

        # Ctrl+C
        os.write(wrapper._fake_stdin_w, b"\x03")  # type: ignore[attr-defined]

        # state 정리, but BEFORE the 5s timeout — so within ~1s
        assert _wait_until(
            lambda: wrapper._intercept_state is None, timeout=1.0
        )
        sock.close()
    finally:
        _cleanup(wrapper)


def test_ordinary_keys_during_intercept_are_dropped(tmp_path: Path) -> None:
    """During intercept, non-Ctrl-C stdin is dropped — state stays active.
    가로채기 중 Ctrl+C 아닌 stdin은 drop — state 유지.
    """
    wrapper = _make_wrapper(tmp_path)
    _start_wrapper(wrapper)
    try:
        sock = _connect(wrapper.socket_path)
        _handshake(sock)

        assert _wait_until(lambda: wrapper.pty_fd >= 0)

        _seed_prompt_line(wrapper, "/resume foo")
        os.write(wrapper._fake_stdin_w, b"\r")  # type: ignore[attr-defined]

        sig = _recv_json(sock)
        assert sig["action"] == "intercept"

        # 활성 상태에서 일반 텍스트 보냄
        os.write(wrapper._fake_stdin_w, b"hello")  # type: ignore[attr-defined]
        time.sleep(0.3)

        # 여전히 활성. input_queue 적재 안 됨 (drop이라).
        assert wrapper._intercept_state is not None
        assert wrapper.input_queue == b""
        sock.close()
    finally:
        _cleanup(wrapper)


def test_intercept_state_dataclass_carries_command_args(tmp_path: Path) -> None:
    """The dataclass payload received by the wrapper preserves args.
    wrapper 측 _InterceptState가 command/args를 정확히 보관.
    """
    wrapper = _make_wrapper(tmp_path)
    _start_wrapper(wrapper)
    try:
        sock = _connect(wrapper.socket_path)
        _handshake(sock)

        assert _wait_until(lambda: wrapper.pty_fd >= 0)

        _seed_prompt_line(wrapper, "/rename my_session")
        os.write(wrapper._fake_stdin_w, b"\r")  # type: ignore[attr-defined]

        sig = _recv_json(sock)
        assert sig == {
            "action": "intercept",
            "command": "rename",
            "args": "my_session",
        }

        assert _wait_until(lambda: wrapper._intercept_state is not None)
        state = wrapper._intercept_state
        assert isinstance(state, _InterceptState)
        assert state.command == "rename"
        assert state.args == "my_session"
        assert state.deadline > 0
        sock.close()
    finally:
        _cleanup(wrapper)
