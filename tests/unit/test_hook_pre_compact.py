"""
Unit tests for the PreCompact auto-compact-blocking hook.

Focus: only auto-compact under a wrapper is blocked, the block stands
even when the signal fails, and every failure passes the compact
through.

PreCompact auto-compact 차단 hook 단위 테스트.

초점: 래퍼 아래의 auto-compact 만 차단하고, 신호 실패에도 차단은
유지되며, 모든 실패가 compact 를 통과시키는지.
"""

from __future__ import annotations

import io
import json
import os
import socket
import threading
import uuid
from pathlib import Path

import pytest

from session_manager.hooks import pre_compact


def _run(
    monkeypatch: pytest.MonkeyPatch,
    payload: object,
    socket_env: str | None,
) -> None:
    if socket_env is None:
        monkeypatch.delenv("SESSION_MANAGER_SOCKET", raising=False)
    else:
        monkeypatch.setenv("SESSION_MANAGER_SOCKET", socket_env)
    text = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    pre_compact.main()


def _block_output(capsys: pytest.CaptureFixture[str]) -> dict | None:
    out = capsys.readouterr().out.strip()
    return json.loads(out) if out else None


class TestMain:
    def test_auto_in_wrapper_blocks_and_signals(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        sent: list[str | None] = []
        monkeypatch.setattr(
            pre_compact,
            "_send_rollover_signal",
            lambda conv: sent.append(conv) or True,
        )
        _run(
            monkeypatch,
            {"trigger": "auto", "session_id": "conv-9"},
            socket_env="/tmp/s.sock",
        )
        output = _block_output(capsys)
        assert output["decision"] == "block"
        assert output["reason"] == pre_compact.BLOCK_REASON
        assert sent == ["conv-9"]

    def test_block_stands_when_signal_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # A lost signal is recoverable (R4-C1 re-marks next turn); a
        # slipped-through auto-compact is not.
        # 유실 신호는 복구된다 (R4-C1 이 다음 턴 재마킹). 통과한
        # auto-compact 는 복구 불가.
        monkeypatch.setattr(
            pre_compact, "_send_rollover_signal", lambda _c: False
        )
        _run(monkeypatch, {"trigger": "auto"}, socket_env="/tmp/s.sock")
        assert _block_output(capsys)["decision"] == "block"

    def test_manual_passes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            pre_compact,
            "_send_rollover_signal",
            lambda _c: pytest.fail("signal must not be sent"),
        )
        _run(monkeypatch, {"trigger": "manual"}, socket_env="/tmp/s.sock")
        assert _block_output(capsys) is None

    def test_auto_without_wrapper_passes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _run(monkeypatch, {"trigger": "auto"}, socket_env=None)
        assert _block_output(capsys) is None

    def test_broken_stdin_passes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _run(monkeypatch, "{broken", socket_env="/tmp/s.sock")
        assert _block_output(capsys) is None


class TestSendRolloverSignal:
    def test_roundtrip_with_ack(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        received: list[dict] = []
        # macOS pytest tmp_path can exceed the AF_UNIX 104B path limit —
        # bind a short /tmp path instead (same as test_socket_server).
        # macOS pytest tmp_path 는 AF_UNIX 104B 한계를 넘을 수 있다 —
        # /tmp 짧은 경로 사용 (test_socket_server 와 동일).
        path = f"/tmp/test-precompact-{uuid.uuid4().hex[:8]}.sock"
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(path)
        server.listen(1)

        def serve() -> None:
            conn, _ = server.accept()
            with conn:
                data = b""
                while b"\n" not in data:
                    data += conn.recv(4096)
                received.append(json.loads(data.split(b"\n", 1)[0]))
                conn.sendall(b'{"type": "ack"}\n')

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        monkeypatch.setenv("SESSION_MANAGER_SOCKET", path)
        try:
            assert pre_compact._send_rollover_signal("conv-9") is True
        finally:
            thread.join(timeout=5)
            server.close()
            os.unlink(path)
        assert received == [
            {
                "client": "hook",
                "action": "rollover_signal",
                "conversation_id": "conv-9",
            }
        ]

    def test_no_socket_env_is_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SESSION_MANAGER_SOCKET", raising=False)
        assert pre_compact._send_rollover_signal("c") is False

    def test_unreachable_socket_is_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "SESSION_MANAGER_SOCKET", str(tmp_path / "nowhere.sock")
        )
        assert pre_compact._send_rollover_signal("c") is False
