"""Integration tests for slash-command observation.

Verifies end-to-end (real PTY child, real Unix socket, real threading)
that a session-changing command is reported to the MCP server *and* runs
immediately — the wrapper no longer holds the keystroke.

슬래시 명령 관찰 통합 테스트.

세션 변경 명령이 MCP 서버에 보고되면서 *동시에* 즉시 실행되는 것을
end-to-end 로 검증한다 (실제 PTY child, 실제 Unix 소켓, 실제 thread) —
래퍼는 더 이상 키 입력을 붙잡지 않는다.
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

from session_manager.wrapper.pty_wrapper import SessionManagerWrapper

# Ink 입력란 포인터 (관찰용 가상 화면 렌더링에 사용)
PROMPT_POINTER = "❯".encode()

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

    Settles first: the child redraws its prompt on startup, and a redraw
    landing after the seed would overwrite it and break the match.
    먼저 출력이 안정되기를 기다린다 — 자식은 시작 시 프롬프트를 다시 그리며,
    시드 이후에 도착한 redraw 는 시드를 덮어써 매칭을 깨뜨린다.
    """
    _settle(wrapper)
    # Carriage return + erase-line first: the child has already drawn its
    # own prompt on this line, and appending would produce "❯ ❯ /resume foo".
    # 먼저 캐리지 리턴 + 줄 지우기 — 자식이 이미 이 줄에 자기 프롬프트를 그려
    # 두었으므로, 그냥 덧붙이면 "❯ ❯ /resume foo" 가 된다.
    wrapper.virtual_screen.feed(b"\r\x1b[2K" + PROMPT_POINTER + b" " + text.encode())


def _settle(wrapper: SessionManagerWrapper, quiet_for: float = 0.4) -> None:
    """Wait until the child's output has been quiet for *quiet_for* seconds.

    자식 출력이 *quiet_for* 초간 잠잠해질 때까지 기다린다.
    """
    deadline = time.monotonic() + _TIMEOUT
    last = None
    quiet_since = time.monotonic()
    while time.monotonic() < deadline:
        snapshot = tuple(wrapper.virtual_screen._safe_display())
        if snapshot != last:
            last = snapshot
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since >= quiet_for:
            return
        time.sleep(0.05)


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


def test_session_command_signal_sent_and_key_forwarded(tmp_path: Path) -> None:
    """/resume observed: MCP is notified and the keystroke goes straight through.

    /resume 관찰 — MCP 에 통보하고 키 입력은 곧바로 통과한다.

    The old flow held the \\r here while the in-session LLM was asked for a
    summary. Nothing is held now, so the command executes with no delay.
    옛 흐름은 여기서 \\r 을 보관한 채 세션 안 LLM 에게 요약을 부탁했다. 이제는
    아무것도 붙잡지 않으므로 명령이 지연 없이 실행된다.
    """
    wrapper = _make_wrapper(tmp_path)
    _start_wrapper(wrapper)
    try:
        sock = _connect(wrapper.socket_path)
        _handshake(sock)
        assert _wait_until(
            lambda: wrapper.pty_fd >= 0 and wrapper.child is not None
        )

        _seed_prompt_line(wrapper, "/resume foo")
        os.write(wrapper._fake_stdin_w, b"\r")  # type: ignore[attr-defined]

        sig = _recv_json(sock)
        assert sig == {
            "action": "session_command",
            "command": "resume",
            "args": "foo",
        }
        # Nothing is held back: a follow-up command still reaches the child.
        # The old flow dropped every keystroke until the LLM replied.
        # 아무것도 붙잡지 않는다 — 뒤이은 명령이 자식에게 도달한다. 옛 흐름은
        # LLM 응답이 올 때까지 모든 키 입력을 버렸다.
        os.write(wrapper._fake_stdin_w, b"hello")  # type: ignore[attr-defined]
        os.write(wrapper._fake_stdin_w, b"\r")  # type: ignore[attr-defined]
        assert _wait_until(
            lambda: wrapper.virtual_screen.contains("Echo: hello"),
            timeout=3.0,
        )
        sock.close()
    finally:
        _cleanup(wrapper)


def test_no_signal_for_non_matching_cr(tmp_path: Path) -> None:
    """A non-session command is forwarded with no signal at all.

    세션 명령이 아니면 신호 없이 그대로 forward 된다.
    """
    wrapper = _make_wrapper(tmp_path)
    _start_wrapper(wrapper)
    try:
        sock = _connect(wrapper.socket_path)
        _handshake(sock)
        assert _wait_until(
            lambda: wrapper.pty_fd >= 0 and wrapper.child is not None
        )

        _seed_prompt_line(wrapper, "/path/to/file")
        os.write(wrapper._fake_stdin_w, b"\r")  # type: ignore[attr-defined]

        sock.settimeout(1.0)
        with pytest.raises((TimeoutError, socket.timeout)):
            _recv_json(sock)
        sock.close()
    finally:
        _cleanup(wrapper)


def test_ordinary_keys_are_never_dropped(tmp_path: Path) -> None:
    """Typing after a session command still reaches the child.

    세션 명령 이후에 친 키도 자식에게 그대로 도달한다 — 옛 흐름은 가로채기
    중 사용자 입력을 통째로 버렸다.
    """
    wrapper = _make_wrapper(tmp_path)
    _start_wrapper(wrapper)
    try:
        sock = _connect(wrapper.socket_path)
        _handshake(sock)
        assert _wait_until(
            lambda: wrapper.pty_fd >= 0 and wrapper.child is not None
        )

        _seed_prompt_line(wrapper, "/resume foo")
        os.write(wrapper._fake_stdin_w, b"\r")  # type: ignore[attr-defined]
        _recv_json(sock)

        os.write(wrapper._fake_stdin_w, b"hello")  # type: ignore[attr-defined]
        os.write(wrapper._fake_stdin_w, b"\r")  # type: ignore[attr-defined]
        assert _wait_until(
            lambda: wrapper.virtual_screen.contains("Echo: hello"),
            timeout=3.0,
        )
        sock.close()
    finally:
        _cleanup(wrapper)
