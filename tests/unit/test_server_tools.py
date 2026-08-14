"""
Tests for MCP tool handlers in server.py.

server.py의 MCP 도구 핸들러 단위 테스트.
도구 함수에 mock Context를 주입하여 state·store 변화와 소켓 메시지를 검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from session_manager import server as server_module
from session_manager import summarizer
from session_manager.models.session import (
    PrecedentRecord,
    SessionMetadata,
    SessionStatus,
)
from session_manager.routing import decision_log
from session_manager.server import (
    AppContext,
    check_session,
    get_routing_status,
    init_project,
    reinit_project,
    reject_switch,
    session_create,
    session_end,
    session_register,
    session_switch,
    set_routing_mode,
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


def _signals(app, action: str) -> list[dict]:
    """All socket signals of *action* sent during the call.

    호출 중 전송된 *action* 신호 전부. 도구는 세션 변경 시 current_session
    통보를 함께 보내므로, 특정 신호는 마지막 호출이 아니라 action 으로 찾는다.
    """
    return [
        call[0][0]
        for call in app.socket_client.send_signal.call_args_list
        if call[0] and isinstance(call[0][0], dict) and call[0][0].get("action") == action
    ]


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

    def test_retired_sessions_are_excluded(self, app: AppContext) -> None:
        # R4-C5: a retired session leaves the routing candidate set — the
        # in-session LLM must not be able to propose it.
        # R4-C5: retired 세션은 라우팅 후보에서 빠진다 — 세션 안의 LLM 이
        # 그 세션을 제안할 수 없어야 한다.
        app.session_store.save_session(SessionMetadata.new(name="alive", title="A"))
        gone = SessionMetadata.new(name="gone", title="G")
        gone.retire("manual")
        app.session_store.save_session(gone)
        result = check_session(_make_ctx(app))
        names = [s["name"] for s in result["sessions"]]
        assert names == ["alive"]

    def test_archived_sessions_still_listed(self, app: AppContext) -> None:
        # Only RETIRED is filtered; ARCHIVED (session_end) keeps its
        # existing visibility.
        # 필터 대상은 RETIRED 뿐 — ARCHIVED (session_end) 의 기존 노출은
        # 유지된다.
        archived = SessionMetadata.new(name="done", title="D")
        archived.status = SessionStatus.ARCHIVED
        app.session_store.save_session(archived)
        result = check_session(_make_ctx(app))
        assert [s["name"] for s in result["sessions"]] == ["done"]


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
        switch_signals = _signals(app, "switch")
        assert len(switch_signals) == 1
        signal = switch_signals[0]
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
        assert len(_signals(app, "switch")) == 1

    def test_switch_from_null_current(self, app: AppContext) -> None:
        result = session_switch(
            target="first",
            summary="",
            user_prompt="hi",
            ctx=_make_ctx(app),
        )
        assert result["switched_to"] == "first"
        assert app.state.get_current_session() == "first"

    def test_switch_appends_accept_label(self, app: AppContext) -> None:
        # Calibration label (R3-C4): an executed switch = accept.
        # 보정 라벨 (R3-C4) — 실행된 전환은 수용이다.
        app.session_store.save_session(
            SessionMetadata.new(name="src", title="Source")
        )
        app.state.set_current_session("src")
        session_switch(
            target="dst", summary="s", user_prompt="p", ctx=_make_ctx(app)
        )
        events = decision_log.load_events(app.project_path)
        assert len(events) == 1
        assert events[0]["type"] == "label"
        assert events[0]["label"] == "accept"
        assert events[0]["target"] == "dst"
        assert events[0]["source"] == "session_switch"

    def test_accepted_switch_drops_precedents_for_target_only(
        self, app: AppContext
    ) -> None:
        # Invalidation (b): accepting a switch to a previously rejected
        # target overturns only the precedents against that target.
        # 무효화 (b): 이전에 거부됐던 대상으로의 전환 수용은 그 대상의
        # 판례만 뒤집는다.
        src = SessionMetadata.new(name="src", title="Source")
        src.precedents = [
            PrecedentRecord.new(
                prompt_gist="API 오류", kept_in="src", rejected="dst"
            ),
            PrecedentRecord.new(
                prompt_gist="배포 설정", kept_in="src", rejected="infra"
            ),
        ]
        app.session_store.save_session(src)
        app.state.set_current_session("src")

        session_switch(
            target="dst",
            summary="moving on",
            user_prompt="dst 작업",
            ctx=_make_ctx(app),
        )

        stored = app.session_store.load_session_by_name("src")
        assert stored is not None
        assert [p.rejected for p in stored.precedents] == ["infra"]


class TestSessionSwitchRetiredPreResolution:
    """Retirement pre-resolution in session_switch (R4-C6 prep).

    session_switch 의 만료 선해석 (R4-C6 선행 수정).

    A retired target is resolved to its living successor before any
    bookkeeping, so the transition record, calibration label, wrapper
    signal and links all name the real destination. Without this the
    wrapper redirected alone and the links landed on the retired
    session (observed in the C5 e2e).

    만료된 target 은 어떤 부기보다 먼저 후계로 해석된다 — 전환 기록·
    보정 라벨·래퍼 신호·링크가 전부 실제 목적지를 가리키도록. 이 단계가
    없으면 래퍼만 재지향해 링크가 만료 세션에 남는다 (C5 e2e 실관측).
    """

    def test_retired_target_resolves_to_successor(self, app: AppContext) -> None:
        app.session_store.save_session(
            SessionMetadata.new(name="src", title="Source")
        )
        app.state.set_current_session("src")
        app.session_store.save_session(SessionMetadata.new(name="heir", title="H"))
        gone = SessionMetadata.new(name="gone", title="G")
        gone.retire("manual", successor="heir")
        app.session_store.save_session(gone)

        result = session_switch(
            target="gone",
            summary="moving",
            user_prompt="go",
            ctx=_make_ctx(app),
        )

        assert result["switched_to"] == "heir"
        assert app.state.get_current_session() == "heir"
        assert _signals(app, "switch")[0]["target"] == "heir"
        src = app.session_store.load_session_by_name("src")
        assert src is not None
        assert src.transitions[0].to_session == "heir"
        events = decision_log.load_events(app.project_path)
        assert [e["target"] for e in events if e["type"] == "label"] == ["heir"]

    def test_retired_chain_resolves_to_living_end(self, app: AppContext) -> None:
        app.state.set_current_session(None)
        first = SessionMetadata.new(name="first", title="1")
        first.retire("manual", successor="second")
        app.session_store.save_session(first)
        second = SessionMetadata.new(name="second", title="2")
        second.retire("manual", successor="third")
        app.session_store.save_session(second)
        app.session_store.save_session(SessionMetadata.new(name="third", title="3"))

        result = session_switch(
            target="first",
            summary="",
            user_prompt="go",
            ctx=_make_ctx(app),
        )

        assert result["switched_to"] == "third"
        assert _signals(app, "switch")[0]["target"] == "third"

    def test_dead_end_refuses_without_bookkeeping(self, app: AppContext) -> None:
        # The wrapper would abort the swap anyway, but by then the
        # outgoing summary, label and links would already be written —
        # so the tool must refuse before touching any state.
        # 래퍼도 교체를 중단하겠지만 그때는 요약·라벨·링크가 이미 기록된
        # 뒤다 — 도구가 상태를 건드리기 전에 거절해야 한다.
        app.session_store.save_session(
            SessionMetadata.new(name="src", title="Source")
        )
        app.state.set_current_session("src")
        gone = SessionMetadata.new(name="gone", title="G")
        gone.retire("manual")
        app.session_store.save_session(gone)

        result = session_switch(
            target="gone",
            summary="should not be written",
            user_prompt="go",
            ctx=_make_ctx(app),
        )

        assert result["ok"] is False
        assert result["error"] == "target_retired_no_successor"
        assert app.state.get_current_session() == "src"
        assert _signals(app, "switch") == []
        src = app.session_store.load_session_by_name("src")
        assert src is not None
        assert src.summary is None
        assert src.transitions == []
        assert decision_log.load_events(app.project_path) == []


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
        signal = _signals(app, "new")[0]
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

        signal = _signals(app, "new")[0]
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
        signal = _signals(app, "new")[0]
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

    def test_does_not_ask_the_wrapper_to_forward_anything(
        self, app: AppContext
    ) -> None:
        """session_end no longer participates in a slash-command handshake.

        session_end 는 더 이상 슬래시 명령 핸드셰이크에 참여하지 않는다 —
        옛 흐름에서는 래퍼가 보관한 키 입력을 흘려보내라는 신호를 보냈다.
        """
        app.session_store.save_session(SessionMetadata.new(name="s", title="S"))
        app.state.set_current_session("s")

        session_end(summary="bye", ctx=_make_ctx(app))

        assert _signals(app, "intercept_done") == []


# --------------------------------------------------------------- update_static


class TestRejectSwitch:
    """reject_switch: precedent append under the F15 lock path.

    reject_switch — F15 잠금 경로를 통한 판례 append.
    """

    def test_appends_precedent_to_current_session(
        self, app: AppContext
    ) -> None:
        app.session_store.save_session(
            SessionMetadata.new(name="frontend", title="차트")
        )
        app.state.set_current_session("frontend")

        result = reject_switch(
            rejected_target="backend",
            prompt_gist="로그인 API 500 조사",
            ctx=_make_ctx(app),
        )

        assert result["recorded"] is True
        assert result["kept_in"] == "frontend"
        stored = app.session_store.load_session_by_name("frontend")
        assert stored is not None
        assert len(stored.precedents) == 1
        record = stored.precedents[0]
        assert record.prompt_gist == "로그인 API 500 조사"
        assert record.kept_in == "frontend"
        assert record.rejected == "backend"
        assert record.at.endswith("+00:00")

    def test_repeated_rejections_accumulate(self, app: AppContext) -> None:
        app.session_store.save_session(
            SessionMetadata.new(name="frontend", title="차트")
        )
        app.state.set_current_session("frontend")

        reject_switch(
            rejected_target="backend", prompt_gist="첫 거부", ctx=_make_ctx(app)
        )
        reject_switch(
            rejected_target="infra", prompt_gist="둘째 거부", ctx=_make_ctx(app)
        )

        stored = app.session_store.load_session_by_name("frontend")
        assert stored is not None
        assert [p.rejected for p in stored.precedents] == ["backend", "infra"]

    def test_uses_locked_mutate_path(self, app: AppContext) -> None:
        # The append must go through the F15 locked load-modify-save —
        # a direct load/save would race the wrapper's summarizer.
        # append 는 F15 잠금 load-modify-save 를 거쳐야 한다 — 직접
        # load/save 는 래퍼 요약기와 경합한다.
        app.session_store.save_session(
            SessionMetadata.new(name="frontend", title="차트")
        )
        app.state.set_current_session("frontend")

        calls: list[str] = []
        original = app.session_store.mutate_session_by_name

        def spying_mutate(name, mutator):
            calls.append(name)
            return original(name, mutator)

        app.session_store.mutate_session_by_name = spying_mutate

        reject_switch(
            rejected_target="backend", prompt_gist="요지", ctx=_make_ctx(app)
        )
        assert calls == ["frontend"]

    def test_enqueues_refresh_and_rooting_check(
        self, app: AppContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # R3-C2: a rejection queues (a) an immediate refresh flagged
        # from_reject and (b) a rooting check carrying the prompt gist.
        # R3-C2 — 거부는 (a) from_reject 표시가 붙은 즉시 갱신과 (b)
        # 프롬프트 요지를 담은 정착 확인을 함께 적재한다.
        app.session_store.save_session(
            SessionMetadata.new(name="frontend", title="차트")
        )
        app.state.set_current_session("frontend")
        monkeypatch.setattr(
            server_module, "get_active_conversation_id", lambda _p: "conv-1"
        )

        result = reject_switch(
            rejected_target="backend",
            prompt_gist="로그인 API 500 조사",
            ctx=_make_ctx(app),
        )

        assert result["refresh_enqueued"] is True
        tasks = [t for _, t in summarizer.load_pending_tasks(app.project_path)]
        by_kind = {t.kind: t for t in tasks}
        assert set(by_kind) == {
            summarizer.KIND_ACTIVE,
            summarizer.KIND_ROOTING_CHECK,
        }
        active = by_kind[summarizer.KIND_ACTIVE]
        assert active.session_name == "frontend"
        assert active.conversation_id == "conv-1"
        assert active.extra.get(summarizer.EXTRA_FROM_REJECT) is True
        rooting = by_kind[summarizer.KIND_ROOTING_CHECK]
        assert (
            rooting.extra.get(summarizer.EXTRA_REJECTED_TOPIC)
            == "로그인 API 500 조사"
        )

    def test_without_conversation_id_skips_enqueue(
        self, app: AppContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app.session_store.save_session(
            SessionMetadata.new(name="frontend", title="차트")
        )
        app.state.set_current_session("frontend")
        monkeypatch.setattr(
            server_module, "get_active_conversation_id", lambda _p: None
        )

        result = reject_switch(
            rejected_target="backend", prompt_gist="요지", ctx=_make_ctx(app)
        )

        # The precedent still records; only the refresh is skipped.
        # 판례는 기록되고 갱신 적재만 생략된다.
        assert result["recorded"] is True
        assert result["refresh_enqueued"] is False
        assert summarizer.load_pending_tasks(app.project_path) == []

    def test_reject_appends_reject_label(self, app: AppContext) -> None:
        # Calibration label (R3-C4): the explicit keep choice = reject.
        # 보정 라벨 (R3-C4) — 명시적 유지 선택은 거부다.
        app.session_store.save_session(
            SessionMetadata.new(name="frontend", title="차트")
        )
        app.state.set_current_session("frontend")
        reject_switch(
            rejected_target="backend", prompt_gist="요지", ctx=_make_ctx(app)
        )
        events = decision_log.load_events(app.project_path)
        labels = [e for e in events if e.get("type") == "label"]
        assert len(labels) == 1
        assert labels[0]["label"] == "reject"
        assert labels[0]["target"] == "backend"
        assert labels[0]["source"] == "reject_switch"

    def test_no_current_session_is_noop(self, app: AppContext) -> None:
        result = reject_switch(
            rejected_target="backend",
            prompt_gist="요지",
            ctx=_make_ctx(app),
        )
        assert result["recorded"] is False
        assert result["reason"] == "no_current_session"
        assert decision_log.load_events(app.project_path) == []

    def test_unregistered_current_session_reports_not_found(
        self, app: AppContext
    ) -> None:
        app.state.set_current_session("ghost")
        result = reject_switch(
            rejected_target="backend",
            prompt_gist="요지",
            ctx=_make_ctx(app),
        )
        assert result["recorded"] is False
        assert result["reason"] == "session_not_found"


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


class TestCurrentSessionNotification:
    """The wrapper is told the current session on every change (F4).

    현재 세션이 바뀔 때마다 래퍼에 통보한다 (F4).

    Without this the wrapper never learns the session name on a plain
    `ccode` start — the handshake only flows wrapper→MCP — and its
    session-scoped triggers (/clear summary, periodic refresh) no-op.
    이 통보가 없으면 인자 없는 `ccode` 시작에서 래퍼는 세션 이름을 알지 못해
    세션 단위 트리거가 무효화된다.
    """

    def test_register_notifies(self, app: AppContext) -> None:
        session_register(name="alpha", title="첫 세션", ctx=_make_ctx(app))
        assert _signals(app, "current_session") == [
            {"action": "current_session", "name": "alpha"}
        ]

    def test_switch_notifies_target(self, app: AppContext) -> None:
        app.session_store.save_session(SessionMetadata.new(name="src", title="s"))
        app.session_store.save_session(SessionMetadata.new(name="dst", title="d"))
        app.state.set_current_session("src")
        session_switch(
            target="dst",
            summary="요약",
            user_prompt="다음 작업",
            ctx=_make_ctx(app),
        )
        assert _signals(app, "current_session") == [
            {"action": "current_session", "name": "dst"}
        ]

    def test_create_notifies_new_session(self, app: AppContext) -> None:
        session_create(
            new_session_name="fresh",
            title="새 세션",
            handoff_summary="요약",
            user_prompt="시작",
            ctx=_make_ctx(app),
        )
        assert _signals(app, "current_session") == [
            {"action": "current_session", "name": "fresh"}
        ]

    def test_end_notifies_none(self, app: AppContext) -> None:
        app.session_store.save_session(SessionMetadata.new(name="solo", title="s"))
        app.state.set_current_session("solo")
        session_end(summary="마무리", ctx=_make_ctx(app))
        assert _signals(app, "current_session") == [
            {"action": "current_session", "name": None}
        ]

    def test_send_failure_does_not_break_the_tool(self, app: AppContext) -> None:
        """A dead socket must not fail the tool call — triggers just stay off.

        소켓이 죽어도 도구 호출은 실패하지 않아야 한다 — 트리거만 꺼질 뿐.
        """
        app.socket_client.send_signal.side_effect = OSError("socket gone")
        result = session_register(name="alpha", title="첫 세션", ctx=_make_ctx(app))
        assert result["registered"] == "alpha"
        assert app.state.get_current_session() == "alpha"


# ------------------------------------------------------- routing mode tools


class TestSetRoutingMode:
    """set_routing_mode: config.json update preserving foreign keys.

    set_routing_mode — 타 키를 보존하는 config.json 갱신.
    """

    def _config_path(self, app: AppContext) -> Path:
        return app.project_path / ".session-manager" / "config.json"

    def test_updates_mode_preserving_other_keys(self, app: AppContext) -> None:
        path = self._config_path(app)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"socket_path": "/tmp/x.sock", "routing_mode": "confirm"}),
            encoding="utf-8",
        )
        result = set_routing_mode(mode="auto", ctx=_make_ctx(app))
        assert result == {"ok": True, "mode": "auto"}
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["routing_mode"] == "auto"
        assert data["socket_path"] == "/tmp/x.sock"

    def test_creates_config_when_missing(self, app: AppContext) -> None:
        result = set_routing_mode(mode="off", ctx=_make_ctx(app))
        assert result["ok"] is True
        data = json.loads(self._config_path(app).read_text(encoding="utf-8"))
        assert data == {"routing_mode": "off"}

    def test_corrupt_config_replaced(self, app: AppContext) -> None:
        path = self._config_path(app)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{깨진", encoding="utf-8")
        assert set_routing_mode(mode="confirm", ctx=_make_ctx(app))["ok"] is True
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["routing_mode"] == "confirm"

    def test_invalid_mode_rejected(self, app: AppContext) -> None:
        result = set_routing_mode(mode="turbo", ctx=_make_ctx(app))
        assert result["ok"] is False
        assert result["error"] == "invalid_mode"
        assert "auto" in result["valid"]
        assert not self._config_path(app).exists()


class TestGetRoutingStatus:
    """get_routing_status: mode + acceptance stats + auto availability.

    get_routing_status — 모드 + 수용률 통계 + auto 가능 여부.
    """

    def _seed(self, app: AppContext, count: int, accept: bool = True) -> None:
        for _ in range(count):
            decision_log.append_proposal(
                app.project_path, "backend", 0.9, mode="confirm"
            )
            decision_log.append_label(
                app.project_path,
                "backend",
                decision_log.LABEL_ACCEPT if accept else decision_log.LABEL_REJECT,
                source="test",
            )

    def test_empty_project_defaults(self, app: AppContext) -> None:
        status = get_routing_status(ctx=_make_ctx(app))
        assert status["mode"] == "confirm"
        assert status["acceptance"] == {
            "accepted": 0,
            "rejected": 0,
            "unlabeled": 0,
        }
        assert status["overall_acceptance_rate"] is None
        assert status["recent_acceptance_rate"] is None
        assert status["auto_available"] is False
        assert status["auto_threshold"] is None
        assert status["auto_error_tolerance"] == 0.05

    def test_stats_and_auto_available_with_sufficient_data(
        self, app: AppContext
    ) -> None:
        # 60/60 accepts — beyond the one-sided 95% Wilson bar (52).
        # 60/60 수용 — 단측 95% Wilson 기준선 (52건) 초과.
        self._seed(app, 60)
        status = get_routing_status(ctx=_make_ctx(app))
        assert status["acceptance"]["accepted"] == 60
        assert status["overall_acceptance_rate"] == 1.0
        assert status["recent_acceptance_rate"] == 1.0
        assert status["auto_available"] is True
        assert status["auto_threshold"] == 0.9

    def test_recent_trend_reflects_latest_half(self, app: AppContext) -> None:
        # Old accepts then recent rejects: overall 0.5, recent 0.0 —
        # the trend must expose the deterioration.
        # 과거 수용 후 최근 거부 — 전체 0.5, 최근 0.0. 추세가 악화를
        # 드러내야 한다.
        self._seed(app, 10, accept=True)
        self._seed(app, 10, accept=False)
        status = get_routing_status(ctx=_make_ctx(app))
        assert status["overall_acceptance_rate"] == 0.5
        assert status["recent_acceptance_rate"] == 0.0
        assert status["auto_available"] is False

    def test_mode_read_from_config(self, app: AppContext) -> None:
        path = app.project_path / ".session-manager" / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"routing_mode": "off"}), encoding="utf-8")
        assert get_routing_status(ctx=_make_ctx(app))["mode"] == "off"
