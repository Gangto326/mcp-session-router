"""
Tests for the wrapper→MCP signal handler.

The wrapper reports session-changing slash commands the user typed
directly. Nothing is asked of the LLM (the background summariser owns the
summary), so the handler's whole job is to drop the cached session
pointer the user just invalidated.

래퍼 → MCP 신호 핸들러 테스트.

래퍼는 사용자가 직접 입력한 세션 변경 슬래시 명령을 보고한다. LLM 에게
요구하는 것은 없으므로 (요약은 백그라운드 요약기 담당), 핸들러가 하는 일은
사용자가 방금 무효화한 캐시된 세션 포인터를 버리는 것뿐이다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from session_manager.server import AppContext, _make_wrapper_signal_handler
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


class TestSessionCommandSignal:
    async def test_invalidates_current_session(self, app: AppContext) -> None:
        app.state.set_current_session("frontend")
        handler = _make_wrapper_signal_handler(app)

        await handler({"action": "session_command", "command": "resume", "args": "x"})

        # The next tool call re-resolves from the active conversation.
        # 다음 도구 호출이 활성 conversation 으로부터 다시 해석한다.
        assert app.state.get_current_session() is None

    async def test_exit_without_args(self, app: AppContext) -> None:
        app.state.set_current_session("frontend")
        handler = _make_wrapper_signal_handler(app)

        await handler({"action": "session_command", "command": "exit", "args": ""})

        assert app.state.get_current_session() is None

    async def test_other_actions_ignored(self, app: AppContext) -> None:
        app.state.set_current_session("frontend")
        handler = _make_wrapper_signal_handler(app)

        await handler({"action": "handshake_request"})
        await handler({})

        assert app.state.get_current_session() == "frontend"

    async def test_malformed_command_ignored(self, app: AppContext) -> None:
        app.state.set_current_session("frontend")
        handler = _make_wrapper_signal_handler(app)

        await handler({"action": "session_command", "command": 42})

        assert app.state.get_current_session() == "frontend"

    async def test_no_llm_involvement(self, app: AppContext) -> None:
        """The handler must not send anything back or call the LLM.

        핸들러는 아무것도 되돌려 보내지 않고 LLM 도 부르지 않는다 — 예전
        흐름은 여기서 channel 로 session_end 를 요청했고, 그 응답을 기다리는
        동안 사용자 입력란이 얼었다.
        """
        handler = _make_wrapper_signal_handler(app)

        await handler({"action": "session_command", "command": "resume", "args": ""})

        app.socket_client.send_signal.assert_not_called()


class TestActiveConversationSignal:
    """F18: the wrapper reports the conversation id; the server adopts it.

    F18: 래퍼가 대화 id 를 보고하고 서버는 그대로 채택한다.
    """

    async def test_adopts_reported_id(self, app: AppContext) -> None:
        handler = _make_wrapper_signal_handler(app)
        await handler({"action": "active_conversation", "conversation_id": "c1"})
        assert app.state.get_active_conversation_id() == "c1"

    async def test_none_clears_it(self, app: AppContext) -> None:
        app.state.set_active_conversation_id("c1")
        handler = _make_wrapper_signal_handler(app)
        await handler({"action": "active_conversation", "conversation_id": None})
        assert app.state.get_active_conversation_id() is None

    async def test_garbage_is_treated_as_unknown(self, app: AppContext) -> None:
        app.state.set_active_conversation_id("c1")
        handler = _make_wrapper_signal_handler(app)
        await handler({"action": "active_conversation", "conversation_id": 42})
        assert app.state.get_active_conversation_id() is None

    async def test_does_not_touch_current_session(self, app: AppContext) -> None:
        app.state.set_current_session("frontend")
        handler = _make_wrapper_signal_handler(app)
        await handler({"action": "active_conversation", "conversation_id": "c1"})
        assert app.state.get_current_session() == "frontend"
