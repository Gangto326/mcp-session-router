"""
Integration tests for the PTY wrapper flow (respawn transition model).

PTY 래퍼 통합 테스트 (respawn 전환 모델). 실제 PTY + Unix Socket 을
사용하되 claude 대신 mock_claude.py 를 spawn 한다. 전환은 더 이상 TUI
주입이 아니라 자식 교체이므로, 검증 대상은 "신호 수신 → 자식 종료 →
새 자식 spawn (재개 인자·트리거 포함) → pending handoff 파일" 이다.

pexpect.spawn 자체를 가로채 (진짜 _spawn_child 로직은 그대로 실행)
스폰 인자를 기록한다.
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

from session_manager import handoff_store
from session_manager.wrapper import pty_wrapper as pty_module
from session_manager.wrapper.pty_wrapper import SessionManagerWrapper

_MOCK_CLAUDE = str(Path(__file__).parent / "mock_claude.py")
_TIMEOUT = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wrapper(
    tmp_path: Path,
    claude_args: list[str] | None = None,
) -> tuple[SessionManagerWrapper, list[list[str]]]:
    """Create a wrapper whose pexpect.spawn launches mock_claude.

    pexpect.spawn 이 mock_claude 를 띄우게 한 래퍼를 만든다. 진짜
    ``_spawn_child`` 가 그대로 실행되므로 전환 부기·인자 조립까지
    통합 검증된다. 반환된 리스트에 스폰 인자가 기록된다.
    """
    short_hash = hashlib.md5(str(tmp_path).encode()).hexdigest()[:8]
    sock_path = f"/tmp/sm-test-{short_hash}.sock"
    Path(sock_path).unlink(missing_ok=True)

    wrapper = SessionManagerWrapper(
        socket_path=sock_path,
        claude_args=claude_args or [],
        project_path=str(tmp_path),
    )

    spawned_args: list[list[str]] = []
    real_spawn = pexpect.spawn

    def fake_spawn(cmd, args, **kwargs):
        spawned_args.append(list(args))
        return real_spawn(
            sys.executable, [_MOCK_CLAUDE], encoding=None, echo=False
        )

    # Patch the module-level reference the wrapper uses.
    # 래퍼가 쓰는 모듈 수준 참조를 패치한다.
    wrapper._pexpect_spawn_patch = (pty_module.pexpect, "spawn", fake_spawn)  # type: ignore[attr-defined]
    pty_module.pexpect.spawn = fake_spawn  # type: ignore[assignment]

    wrapper._enter_raw_mode = lambda: None  # type: ignore[assignment]
    wrapper._restore_terminal = lambda: None  # type: ignore[assignment]
    wrapper._install_winch_handler = lambda: None  # type: ignore[assignment]
    wrapper._sync_winsize = lambda: None  # type: ignore[assignment]
    wrapper._stdout_fd = os.open(os.devnull, os.O_WRONLY)

    fake_stdin_r, fake_stdin_w = os.pipe()
    wrapper._stdin_fd = fake_stdin_r
    wrapper._fake_stdin_w = fake_stdin_w  # type: ignore[attr-defined]
    return wrapper, spawned_args


def _start_wrapper(wrapper: SessionManagerWrapper) -> threading.Thread:
    def _run() -> None:
        try:
            wrapper.start()
        except Exception:
            pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _connect_and_handshake(
    sock_path: str, timeout: float = _TIMEOUT
) -> socket.socket:
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


def _wait_until(predicate, timeout: float = _TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _cleanup(wrapper: SessionManagerWrapper) -> None:
    patch = getattr(wrapper, "_pexpect_spawn_patch", None)
    if patch is not None:
        module, name, _fake = patch
        module.spawn = pexpect.spawn if module.spawn is not _fake else module.spawn
    # Restore the real pexpect.spawn on the module.
    # 모듈의 진짜 pexpect.spawn 을 복원한다.
    import importlib

    pty_module.pexpect = importlib.import_module("pexpect")
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
        self, tmp_path: Path
    ) -> None:
        wrapper, _spawned = _make_wrapper(
            tmp_path, claude_args=["--resume", "foo"]
        )
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
        self, tmp_path: Path
    ) -> None:
        wrapper, _spawned = _make_wrapper(tmp_path)
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
# Tests — respawn transition flow
# ---------------------------------------------------------------------------


class TestSwitchRespawnFlow:
    def test_switch_signal_swaps_child_with_trigger(
        self, tmp_path: Path
    ) -> None:
        """SWITCH signal → old child dies → new child spawned with the
        trigger prompt → pending handoff file awaits the hook.

        SWITCH 신호 → 기존 자식 종료 → 트리거 프롬프트를 단 새 자식
        spawn → pending handoff 파일이 hook 을 기다린다.
        """
        wrapper, spawned = _make_wrapper(tmp_path)
        _start_wrapper(wrapper)
        try:
            client = _connect_and_handshake(wrapper.socket_path)
            _send_json(client, {"type": "handshake_request"})
            _recv_json(client)
            assert _wait_until(lambda: len(spawned) == 1)
            first_child = wrapper.child

            _send_json(
                client,
                {
                    "action": "switch",
                    "target": "backend",
                    "handoff": {"from": "frontend", "message": "요약"},
                    "user_prompt": "옮겨갈 프롬프트",
                },
            )

            # The swap: a second spawn happens and the first child dies.
            # 교체 — 두 번째 spawn 이 일어나고 첫 자식은 죽는다.
            assert _wait_until(lambda: len(spawned) == 2, timeout=10)
            assert _wait_until(lambda: not first_child.isalive(), timeout=10)

            # Respawn args: trigger prompt last; no --resume (the target
            # session has no recorded conversation in this project).
            # 재spawn 인자 — 트리거가 마지막. 대상 세션에 기록된
            # conversation 이 없으므로 --resume 없음.
            assert spawned[1][-1] == handoff_store.TRIGGER_PROMPT
            assert any(
                a.startswith("--append-system-prompt=") for a in spawned[1]
            )

            # The handoff waits in the pending file for the hook.
            # handoff 는 pending 파일에서 hook 을 기다린다.
            assert _wait_until(
                lambda: wrapper._pending_respawn is None, timeout=5
            )
            pending = handoff_store.take_pending(tmp_path)
            assert pending is not None
            assert pending["target"] == "backend"
            assert pending["user_prompt"] == "옮겨갈 프롬프트"
            assert wrapper._current_session_name == "backend"
            client.close()
        finally:
            _cleanup(wrapper)

    def test_new_signal_swaps_child_without_resume(
        self, tmp_path: Path
    ) -> None:
        wrapper, spawned = _make_wrapper(tmp_path)
        _start_wrapper(wrapper)
        try:
            client = _connect_and_handshake(wrapper.socket_path)
            assert _wait_until(lambda: len(spawned) == 1)

            _send_json(
                client,
                {
                    "action": "new",
                    "rename_current": "old-name",
                    "new_session_name": "fresh",
                    "handoff": {"from": "old-name"},
                    "user_prompt": "새 주제",
                },
            )
            assert _wait_until(lambda: len(spawned) == 2, timeout=10)
            assert not any(a.startswith("--resume") for a in spawned[1])
            assert spawned[1][-1] == handoff_store.TRIGGER_PROMPT
            assert _wait_until(
                lambda: wrapper._current_session_name == "fresh", timeout=5
            )
            client.close()
        finally:
            _cleanup(wrapper)


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_malformed_socket_message_ignored(self, tmp_path: Path) -> None:
        wrapper, _spawned = _make_wrapper(tmp_path)
        try:
            wrapper._handle_mcp_signal("not a dict")  # type: ignore[arg-type]
            wrapper._handle_mcp_signal({"action": "switch"})
            wrapper._handle_mcp_signal({"action": "new", "handoff": {}})
            assert wrapper._pending_respawn is None
        finally:
            _cleanup(wrapper)
