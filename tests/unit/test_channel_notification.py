"""Tests for ChannelFastMCP capability + channel notification + intercept handler.
ChannelFastMCP capability + channel notification + intercept handler 테스트.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from session_manager.models.session import SessionMetadata
from session_manager.server import (
    AppContext,
    ChannelFastMCP,
    _make_intercept_handler,
    send_channel_notification,
)
from session_manager.state import SessionManagerState
from session_manager.storage import FieldStore, ProjectContextStore, SessionStore


@pytest.fixture
def app(tmp_path: Path) -> AppContext:
    return AppContext(
        state=SessionManagerState(),
        session_store=SessionStore(tmp_path),
        field_store=FieldStore(tmp_path),
        project_context_store=ProjectContextStore(tmp_path),
        socket_client=MagicMock(),
        project_path=tmp_path,
    )


class _FakeStream:
    """Captures messages instead of writing to a real stream.
    실제 stream 대신 메시지를 캡처.
    """

    def __init__(self) -> None:
        self.sent: list = []

    async def send(self, message) -> None:
        self.sent.append(message)


class TestChannelFastMCP:
    """Verify the subclass exposes the channel write_stream slot.
    서브클래스가 channel write_stream slot을 가지는지.
    """

    def test_subclass_has_write_stream_attribute(self) -> None:
        mcp = ChannelFastMCP("test")
        assert hasattr(mcp, "_channel_write_stream")
        assert mcp._channel_write_stream is None


class TestSendChannelNotification:
    """JSONRPCNotification serialization sent through write_stream.
    write_stream으로 보낸 JSONRPCNotification 직렬화 검증.
    """

    @pytest.mark.asyncio
    async def test_sends_well_formed_notification(self) -> None:
        stream = _FakeStream()
        await send_channel_notification(
            stream, "hello", {"command": "resume", "args": "foo"}
        )

        assert len(stream.sent) == 1
        session_msg = stream.sent[0]
        # SessionMessage wraps a JSONRPCMessage; the inner notification
        # carries our custom method + params.
        # SessionMessage가 JSONRPCMessage를 감싸고, notification에 우리 method/params.
        notif = session_msg.message.root
        assert notif.method == "notifications/claude/channel"
        assert notif.params["content"] == "hello"
        assert notif.params["meta"] == {"command": "resume", "args": "foo"}

    @pytest.mark.asyncio
    async def test_serializes_to_jsonrpc(self) -> None:
        """End-to-end check: notification serializes to expected JSON shape.
        직렬화 결과가 기대한 JSON 형태인지 end-to-end 확인.
        """
        stream = _FakeStream()
        await send_channel_notification(stream, "x", {"k": "v"})
        notif = stream.sent[0].message.root
        payload = notif.model_dump(by_alias=True, exclude_none=True)
        assert payload["method"] == "notifications/claude/channel"
        assert payload["jsonrpc"] == "2.0"
        # ensure content/meta survive a JSON roundtrip
        # content/meta가 JSON roundtrip 후에도 보존되는지
        round_tripped = json.loads(json.dumps(payload))
        assert round_tripped["params"]["content"] == "x"
        assert round_tripped["params"]["meta"] == {"k": "v"}


class TestInterceptHandler:
    """The handler returned by _make_intercept_handler converts wrapper signals
    into channel notifications.

    _make_intercept_handler가 반환한 핸들러가 래퍼 신호를 channel notification으로 변환.
    """

    @pytest.mark.asyncio
    async def test_intercept_signal_pushes_channel_notification(
        self, app: AppContext
    ) -> None:
        app.session_store.save_session(SessionMetadata.new(name="cur", title="Cur"))
        app.state.set_current_session("cur")

        stream = _FakeStream()
        server = MagicMock()
        server._channel_write_stream = stream

        handler = _make_intercept_handler(server, app)
        await handler({"action": "intercept", "command": "resume", "args": "foo"})

        assert len(stream.sent) == 1
        notif = stream.sent[0].message.root
        assert notif.method == "notifications/claude/channel"
        # 명령이 본문과 meta 양쪽에 들어가는지
        assert "/resume foo" in notif.params["content"]
        assert "cur" in notif.params["content"]
        assert notif.params["meta"]["command"] == "resume"
        assert notif.params["meta"]["args"] == "foo"
        assert notif.params["meta"]["current_session"] == "cur"
        # intercept_active 플래그가 True로 set
        assert app.intercept_active["value"] is True

    @pytest.mark.asyncio
    async def test_non_intercept_action_ignored(self, app: AppContext) -> None:
        stream = _FakeStream()
        server = MagicMock()
        server._channel_write_stream = stream

        handler = _make_intercept_handler(server, app)
        await handler({"action": "something_else"})

        assert stream.sent == []
        assert app.intercept_active["value"] is False

    @pytest.mark.asyncio
    async def test_missing_write_stream_logs_and_returns(
        self, app: AppContext
    ) -> None:
        """If the channel write_stream is not yet wired, handler is a no-op.
        write_stream 미설정 시 handler는 no-op (warning만).
        """
        server = MagicMock()
        server._channel_write_stream = None

        handler = _make_intercept_handler(server, app)
        # Should not raise
        # 예외 없이 정상 반환해야
        await handler({"action": "intercept", "command": "exit", "args": ""})
        assert app.intercept_active["value"] is False

    @pytest.mark.asyncio
    async def test_invalid_command_field_ignored(self, app: AppContext) -> None:
        """Non-string command/args are dropped silently.
        command/args가 문자열이 아니면 무시.
        """
        stream = _FakeStream()
        server = MagicMock()
        server._channel_write_stream = stream

        handler = _make_intercept_handler(server, app)
        await handler({"action": "intercept", "command": 123, "args": "foo"})

        assert stream.sent == []
        assert app.intercept_active["value"] is False

    @pytest.mark.asyncio
    async def test_no_args_for_exit_command(self, app: AppContext) -> None:
        """/exit (no args) renders without trailing space.
        /exit (인자 없음) → 끝에 공백 없이 렌더링.
        """
        stream = _FakeStream()
        server = MagicMock()
        server._channel_write_stream = stream

        handler = _make_intercept_handler(server, app)
        await handler({"action": "intercept", "command": "exit", "args": ""})

        assert len(stream.sent) == 1
        notif = stream.sent[0].message.root
        assert "/exit" in notif.params["content"]
        # /exit가 trailing 공백 없이
        assert "/exit." in notif.params["content"] or "/exit " in notif.params["content"]
