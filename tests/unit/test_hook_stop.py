"""
Unit tests for the Stop turn-end hook.

Focus: signals fire only in a wrapper context, carry the conversation id
and response body, and every failure stays silent.

Stop 턴 종료 hook 단위 테스트.

초점: 래퍼 문맥에서만 신호를 보내고, conversation id 와 응답 본문을
실어 나르며, 모든 실패가 침묵하는지.
"""

from __future__ import annotations

import io
import json
import os
import socket
import threading
import uuid

import pytest

from session_manager.hooks import stop


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
    stop.main()


class TestMain:
    def test_sends_signal_in_wrapper_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent: list[tuple[str, str]] = []
        monkeypatch.setattr(
            stop,
            "_send_turn_end",
            lambda conv, msg: sent.append((conv, msg)) or True,
        )
        _run(
            monkeypatch,
            {"session_id": "conv-9", "last_assistant_message": "# Handoff"},
            socket_env="/tmp/s.sock",
        )
        assert sent == [("conv-9", "# Handoff")]

    def test_missing_message_sends_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sent: list[tuple[str, str]] = []
        monkeypatch.setattr(
            stop,
            "_send_turn_end",
            lambda conv, msg: sent.append((conv, msg)) or True,
        )
        _run(monkeypatch, {"session_id": "conv-9"}, socket_env="/tmp/s.sock")
        assert sent == [("conv-9", "")]

    def test_outside_wrapper_is_silent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            stop,
            "_send_turn_end",
            lambda *_a: pytest.fail("signal must not be sent"),
        )
        _run(monkeypatch, {"session_id": "conv-9"}, socket_env=None)

    def test_missing_session_id_is_silent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            stop,
            "_send_turn_end",
            lambda *_a: pytest.fail("signal must not be sent"),
        )
        _run(monkeypatch, {"last_assistant_message": "x"}, socket_env="/tmp/s")

    def test_broken_stdin_is_silent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _run(monkeypatch, "{broken", socket_env="/tmp/s.sock")


class TestSendTurnEnd:
    def test_roundtrip_with_ack(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received: list[dict] = []
        # macOS pytest tmp_path can exceed the AF_UNIX 104B path limit —
        # bind a short /tmp path instead (same as test_socket_server).
        # macOS pytest tmp_path 는 AF_UNIX 104B 한계를 넘을 수 있다 —
        # /tmp 짧은 경로 사용 (test_socket_server 와 동일).
        path = f"/tmp/test-stop-{uuid.uuid4().hex[:8]}.sock"
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
            assert stop._send_turn_end("conv-9", "본문") is True
        finally:
            thread.join(timeout=5)
            server.close()
            os.unlink(path)
        assert received == [
            {
                "client": "hook",
                "action": "turn_end",
                "conversation_id": "conv-9",
                "last_assistant_message": "본문",
            }
        ]

    def test_unreachable_socket_is_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SESSION_MANAGER_SOCKET", "/tmp/nowhere-stop.sock")
        assert stop._send_turn_end("c", "") is False
