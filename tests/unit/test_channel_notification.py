"""Tests for the ChannelFastMCP capability and channel notification.

The intercept flow that used these was removed — slash commands are now
observed, not held (see test_wrapper_signal_handler.py). The channel
plumbing itself is removed in the next commit.

ChannelFastMCP capability 와 channel notification 테스트.

이를 사용하던 가로채기 흐름은 제거되었다 — 슬래시 명령은 이제 붙잡지 않고
관찰만 한다 (test_wrapper_signal_handler.py 참조). channel 배관 자체는 다음
커밋에서 제거된다.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from session_manager.server import (
    AppContext,
    ChannelFastMCP,
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
