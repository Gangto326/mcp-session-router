"""
Tests for MCP tool handlers in server.py.

server.py의 MCP 도구 핸들러 단위 테스트.
도구 함수에 mock Context를 주입하여 state·store 변화와 소켓 메시지를 검증한다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from session_manager.models.session import SessionMetadata, SessionStatus
from session_manager.server import (
    AppContext,
    check_session,
    init_project,
    reinit_project,
    session_create,
    session_end,
    session_register,
    session_switch,
    update_project_context,
    update_static,
)
from session_manager.state import SessionManagerState
from session_manager.storage import FieldStore, ProjectContextStore, SessionStore


def _make_ctx(app: AppContext) -> MagicMock:
    """Build a mock Context whose request_context.lifespan_context is *app*.

    request_context.lifespan_context가 app인 mock Context를 생성한다.
    """
    ctx = MagicMock()
    ctx.request_context.lifespan_context = app
    return ctx


@pytest.fixture
def app(tmp_path: Path) -> AppContext:
    """AppContext backed by real stores under a temp directory.

    임시 디렉토리 기반 실제 store를 사용하는 AppContext.
    """
    client = MagicMock()
    return AppContext(
        state=SessionManagerState(),
        session_store=SessionStore(tmp_path),
        field_store=FieldStore(tmp_path),
        project_context_store=ProjectContextStore(tmp_path),
        socket_client=client,
        project_path=tmp_path,
    )


# ---------------------------------------------------------------- check_session


class TestCheckSession:
    def test_empty_store_returns_null_current_and_empty_list(
        self, app: AppContext
    ) -> None:
        result = check_session(_make_ctx(app))
        assert result["current"] is None
        assert result["sessions"] == []

    def test_returns_registered_sessions(self, app: AppContext) -> None:
        app.session_store.save_session(
            SessionMetadata.new(name="a", title="A", summary="about a")
        )
        app.state.set_current_session("a")

        result = check_session(_make_ctx(app))
        assert result["current"] == "a"
        assert len(result["sessions"]) == 1
        s = result["sessions"][0]
        assert s["name"] == "a"
        assert s["title"] == "A"
        assert s["summary"] == "about a"
        assert s["status"] == "active"

    def test_multiple_sessions(self, app: AppContext) -> None:
        app.session_store.save_session(SessionMetadata.new(name="x", title="X"))
        app.session_store.save_session(SessionMetadata.new(name="y", title="Y"))
        result = check_session(_make_ctx(app))
        assert len(result["sessions"]) == 2


# -------------------------------------------------------------- session_register


class TestSessionRegister:
    def test_registers_new_session(self, app: AppContext) -> None:
        result = session_register(
            name="dev", title="Dev Session", ctx=_make_ctx(app)
        )
        assert result["registered"] == "dev"
        assert "session_id" in result
        assert app.state.get_current_session() == "dev"

        stored = app.session_store.load_session_by_name("dev")
        assert stored is not None
        assert stored.title == "Dev Session"
        assert stored.summary is None

    def test_registers_with_summary(self, app: AppContext) -> None:
        session_register(
            name="ops",
            title="Ops",
            ctx=_make_ctx(app),
            summary="deployment tasks",
        )
        stored = app.session_store.load_session_by_name("ops")
        assert stored is not None
        assert stored.summary == "deployment tasks"


# --------------------------------------------------------------- session_switch


class TestSessionSwitch:
    def test_updates_outgoing_session_and_sends_signal(
        self, app: AppContext
    ) -> None:
        # Set up: register current session.
        # 준비: 현재 세션을 등록한다.
        app.session_store.save_session(
            SessionMetadata.new(name="src", title="Source")
        )
        app.state.set_current_session("src")

        result = session_switch(
            target="dst",
            summary="done with src",
            user_prompt="work on dst",
            ctx=_make_ctx(app),
        )

        assert result["switched_to"] == "dst"
        assert app.state.get_current_session() == "dst"

        # Outgoing session metadata should be updated.
        # 나간 세션의 메타데이터가 갱신되어야 한다.
        src = app.session_store.load_session_by_name("src")
        assert src is not None
        assert src.summary == "done with src"
        assert len(src.transitions) == 1
        assert src.transitions[0].to_session == "dst"

        # Socket signal should have been sent.
        # 소켓 신호가 전송되어야 한다.
        app.socket_client.send_signal.assert_called_once()
        signal = app.socket_client.send_signal.call_args[0][0]
        assert signal["action"] == "switch"
        assert signal["target"] == "dst"
        assert signal["user_prompt"] == "work on dst"

    def test_updates_title_when_provided(self, app: AppContext) -> None:
        app.session_store.save_session(
            SessionMetadata.new(name="s", title="Old")
        )
        app.state.set_current_session("s")

        session_switch(
            target="t",
            summary="bye",
            user_prompt="hi",
            ctx=_make_ctx(app),
            updated_title="New Title",
        )

        s = app.session_store.load_session_by_name("s")
        assert s is not None
        assert s.title == "New Title"

    def test_switch_without_registered_current(self, app: AppContext) -> None:
        # Current session name set but no metadata on disk.
        # 현재 세션 이름은 설정되었지만 디스크에 메타데이터가 없는 경우.
        app.state.set_current_session("ghost")

        result = session_switch(
            target="real",
            summary="n/a",
            user_prompt="go",
            ctx=_make_ctx(app),
        )

        assert result["switched_to"] == "real"
        app.socket_client.send_signal.assert_called_once()

    def test_switch_from_null_current(self, app: AppContext) -> None:
        result = session_switch(
            target="first",
            summary="",
            user_prompt="hi",
            ctx=_make_ctx(app),
        )
        assert result["switched_to"] == "first"
        assert app.state.get_current_session() == "first"


# --------------------------------------------------------------- session_create


class TestSessionCreate:
    def test_creates_new_session_with_registered_current(
        self, app: AppContext
    ) -> None:
        # Set up: registered current session.
        # 준비: 등록된 현재 세션.
        app.session_store.save_session(
            SessionMetadata.new(name="old", title="Old")
        )
        app.state.set_current_session("old")

        result = session_create(
            new_session_name="fresh",
            title="Fresh Session",
            handoff_summary="wrapping up old",
            user_prompt="start fresh",
            ctx=_make_ctx(app),
        )

        assert result["created"] == "fresh"
        assert result["rename_current"] == "old"
        assert app.state.get_current_session() == "fresh"

        # Outgoing session's summary should be updated.
        # 나간 세션의 summary가 갱신되어야 한다.
        old = app.session_store.load_session_by_name("old")
        assert old is not None
        assert old.summary == "wrapping up old"

        # Socket signal should carry rename_current.
        # 소켓 신호에 rename_current가 포함되어야 한다.
        signal = app.socket_client.send_signal.call_args[0][0]
        assert signal["action"] == "new"
        assert signal["rename_current"] == "old"
        assert signal["new_session_name"] == "fresh"

    def test_creates_new_session_without_registered_current(
        self, app: AppContext
    ) -> None:
        # Current session not registered — rename_current should be null.
        # 현재 세션이 미등록 — rename_current는 null이어야 한다.
        app.state.set_current_session("unregistered")

        result = session_create(
            new_session_name="brand-new",
            title="Brand New",
            handoff_summary="",
            user_prompt="go",
            ctx=_make_ctx(app),
        )

        assert result["rename_current"] is None

        signal = app.socket_client.send_signal.call_args[0][0]
        assert signal["rename_current"] is None

    def test_creates_from_null_current(self, app: AppContext) -> None:
        result = session_create(
            new_session_name="first",
            title="First",
            handoff_summary="",
            user_prompt="hi",
            ctx=_make_ctx(app),
        )

        assert result["created"] == "first"
        assert result["rename_current"] is None

    def test_handoff_includes_title(self, app: AppContext) -> None:
        session_create(
            new_session_name="n",
            title="New Title",
            handoff_summary="s",
            user_prompt="p",
            ctx=_make_ctx(app),
        )
        signal = app.socket_client.send_signal.call_args[0][0]
        assert signal["handoff"]["new_session_title"] == "New Title"


# ----------------------------------------------------------------- session_end


class TestSessionEnd:
    def test_archives_current_session(self, app: AppContext) -> None:
        app.session_store.save_session(
            SessionMetadata.new(name="done", title="Done")
        )
        app.state.set_current_session("done")

        result = session_end(summary="all done", ctx=_make_ctx(app))

        assert result["ended"] == "done"
        assert app.state.get_current_session() is None

        stored = app.session_store.load_session_by_name("done")
        assert stored is not None
        assert stored.summary == "all done"
        assert stored.status == SessionStatus.ARCHIVED

    def test_end_with_null_current(self, app: AppContext) -> None:
        result = session_end(summary="n/a", ctx=_make_ctx(app))
        assert result["ended"] is None
        assert app.state.get_current_session() is None

    def test_end_with_unregistered_current(self, app: AppContext) -> None:
        app.state.set_current_session("ghost")
        result = session_end(summary="bye", ctx=_make_ctx(app))
        assert result["ended"] == "ghost"
        assert app.state.get_current_session() is None

    def test_sends_intercept_done_when_active(self, app: AppContext) -> None:
        """intercept_active=True 일 때 session_end → 래퍼에 intercept_done 송신.
        intercept_active=True → session_end notifies wrapper.
        """
        app.session_store.save_session(SessionMetadata.new(name="s", title="S"))
        app.state.set_current_session("s")
        app.intercept_active["value"] = True

        session_end(summary="bye", ctx=_make_ctx(app))

        app.socket_client.send_signal.assert_called_once_with(
            {"action": "intercept_done"}
        )
        assert app.intercept_active["value"] is False

    def test_no_intercept_done_when_inactive(self, app: AppContext) -> None:
        """intercept_active=False 일 때 session_end → 래퍼 통보 없음.
        intercept_active=False → no notify.
        """
        app.session_store.save_session(SessionMetadata.new(name="s", title="S"))
        app.state.set_current_session("s")
        app.intercept_active["value"] = False

        session_end(summary="bye", ctx=_make_ctx(app))

        app.socket_client.send_signal.assert_not_called()


# --------------------------------------------------------------- update_static


class TestUpdateStatic:
    def test_partial_update_preserves_other_fields(
        self, app: AppContext
    ) -> None:
        # First set some initial values.
        # 먼저 초기값을 설정한다.
        update_static(
            ctx=_make_ctx(app),
            project_context="ctx",
            conventions="conv",
        )
        # Then update only conventions.
        # 그다음 conventions만 갱신한다.
        result = update_static(ctx=_make_ctx(app), conventions="new conv")

        assert "updated_at" in result
        static = app.field_store.load_static()
        assert static.project_context == "ctx"
        assert static.conventions == "new conv"

    def test_update_variables(self, app: AppContext) -> None:
        update_static(
            ctx=_make_ctx(app),
            variables={"db_host": "localhost", "port": 5432},
        )
        static = app.field_store.load_static()
        assert static.variables["db_host"] == "localhost"
        assert static.variables["port"] == 5432

    def test_no_args_only_touches_timestamp(self, app: AppContext) -> None:
        update_static(ctx=_make_ctx(app))
        static = app.field_store.load_static()
        assert static.updated_at != ""


# ----------------------------------------- init_project / reinit / update


class TestProjectContextTools:
    def test_init_creates_when_absent(self, app: AppContext) -> None:
        result = init_project(content="# My Project", ctx=_make_ctx(app))
        assert result["created"] is True
        assert app.project_context_store.read() == "# My Project"

    def test_init_noop_when_exists(self, app: AppContext) -> None:
        app.project_context_store.write("existing")
        result = init_project(content="overwrite?", ctx=_make_ctx(app))
        assert result["created"] is False
        assert result["content"] == "existing"
        assert app.project_context_store.read() == "existing"

    def test_reinit_overwrites(self, app: AppContext) -> None:
        app.project_context_store.write("old")
        result = reinit_project(content="brand new", ctx=_make_ctx(app))
        assert result["reinitialized"] is True
        assert app.project_context_store.read() == "brand new"

    def test_update_replaces_content(self, app: AppContext) -> None:
        app.project_context_store.write("v1")
        result = update_project_context(content="v2", ctx=_make_ctx(app))
        assert result["updated"] is True
        assert app.project_context_store.read() == "v2"
