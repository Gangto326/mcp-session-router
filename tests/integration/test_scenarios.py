"""
Scenario-level integration tests for session management workflows.

세션 관리 워크플로우 시나리오 통합 테스트.
실제 디스크 스토어를 사용하여 MCP 도구 호출 시퀀스가 올바른 상태 변화를
만들어내는지 검증한다. 서브 에이전트의 판단은 테스트 코드가 대신하고,
그 판단 결과에 따른 도구 호출 시퀀스가 정상 동작하는지 테스트한다.

Uses real SessionStore / FieldStore / ProjectContextStore backed by
tmp_path and a MagicMock socket client.  Lifespan tests exercise
``app_lifespan`` with a mock FastMCP server to verify startup
initialisation paths (--resume handshake, --continue fallback, and
first-run auto-registration).
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from session_manager.models.session import SessionMetadata, SessionStatus
from session_manager.server import (
    AppContext,
    app_lifespan,
    check_session,
    session_create,
    session_register,
    session_switch,
)
from session_manager.state import SessionManagerState
from session_manager.storage import FieldStore, ProjectContextStore, SessionStore

# ---------------------------------------------------------------------------
# Fixtures
# 픽스처
# ---------------------------------------------------------------------------


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
    소켓 클라이언트는 mock이다.
    """
    return AppContext(
        state=SessionManagerState(),
        session_store=SessionStore(tmp_path),
        field_store=FieldStore(tmp_path),
        project_context_store=ProjectContextStore(tmp_path),
        socket_client=MagicMock(),
        project_path=tmp_path,
    )


# ---------------------------------------------------------------------------
# 1. Clear topic switch — sub-agent determines SWITCH
# 1. 명확한 주제 전환 — 서브 에이전트가 SWITCH 판정
# ---------------------------------------------------------------------------


class TestClearTopicSwitch:
    def test_switch_updates_outgoing_and_transitions_to_target(
        self, app: AppContext
    ) -> None:
        """Session A (frontend) is active. User asks about backend API.
        Sub-agent says SWITCH to session B → session_switch called.

        세션 A(프론트엔드) 활성 상태에서 사용자가 백엔드 API를 질문.
        서브 에이전트가 세션 B로 SWITCH 판정 → session_switch 호출.
        결과: A의 summary 갱신, transition 기록, 상태가 B로 이동.
        """
        ctx = _make_ctx(app)

        # Prepare: two sessions already registered.
        # 준비: 세션 두 개가 이미 등록됨.
        session_register(name="frontend", title="Frontend UI", ctx=ctx)
        app.session_store.save_session(
            SessionMetadata.new(
                name="backend",
                title="Backend API",
                summary="REST endpoint design for /api/users",
            )
        )

        # Sub-agent decision: SWITCH to backend.
        # 서브 에이전트 판정: backend으로 SWITCH.
        result = session_switch(
            target="backend",
            summary="Built login form with validation",
            user_prompt="How does the /api/users endpoint handle pagination?",
            ctx=ctx,
        )

        assert result["switched_to"] == "backend"
        assert app.state.get_current_session() == "backend"

        # Outgoing session (frontend) should have updated summary + transition.
        # 나간 세션(frontend)의 summary가 갱신되고 transition이 기록되어야 한다.
        fe = app.session_store.load_session_by_name("frontend")
        assert fe is not None
        assert fe.summary == "Built login form with validation"
        assert len(fe.transitions) == 1
        assert fe.transitions[0].from_session == "frontend"
        assert fe.transitions[0].to_session == "backend"

        # Socket signal sent to wrapper.
        # 래퍼에 소켓 신호가 전송되어야 한다.
        signal = app.socket_client.send_signal.call_args[0][0]
        assert signal["action"] == "switch"
        assert signal["target"] == "backend"


# ---------------------------------------------------------------------------
# 2. Ambiguous reference resolved by summary comparison
# 2. 모호한 참조를 summary로 해소
# ---------------------------------------------------------------------------


class TestAmbiguousReferenceResolvedBySummary:
    def test_check_session_summaries_disambiguate_target(
        self, app: AppContext
    ) -> None:
        """User says "fix the test". Two sessions exist — sub-agent reads
        summaries via check_session to decide which one.

        사용자가 "테스트 고쳐줘"라고 할 때, 세션이 두 개 있으면
        check_session으로 summary를 읽어 올바른 대상을 판별한다.
        """
        ctx = _make_ctx(app)

        # Session A: UI work (no tests).
        # 세션 A: UI 작업 (테스트 없음).
        session_register(name="ui-styling", title="UI Styling", ctx=ctx)
        ui = app.session_store.load_session_by_name("ui-styling")
        assert ui is not None
        ui.summary = "Adjusted CSS grid layout for dashboard cards"
        app.session_store.save_session(ui)

        # Session B: test refactoring.
        # 세션 B: 테스트 리팩토링.
        app.session_store.save_session(
            SessionMetadata.new(
                name="test-refactor",
                title="Test Refactoring",
                summary="Migrating pytest fixtures from conftest to per-module, 3 files remain",
            )
        )

        # Sub-agent calls check_session to read summaries.
        # 서브 에이전트가 check_session으로 summary를 읽는다.
        info = check_session(ctx)
        assert info["current"] == "ui-styling"
        summaries = {s["name"]: s["summary"] for s in info["sessions"]}

        # "test" keyword matches test-refactor's summary, not ui-styling's.
        # "test" 키워드는 test-refactor의 summary와 매칭된다.
        assert "test" not in summaries["ui-styling"].lower()
        assert "pytest" in summaries["test-refactor"].lower()

        # Sub-agent decides: SWITCH to test-refactor.
        # 서브 에이전트 판정: test-refactor로 SWITCH.
        result = session_switch(
            target="test-refactor",
            summary="Adjusted CSS grid layout for dashboard cards",
            user_prompt="fix the test",
            ctx=ctx,
        )
        assert result["switched_to"] == "test-refactor"
        assert app.state.get_current_session() == "test-refactor"


# ---------------------------------------------------------------------------
# 3. New topic creates a session
# 3. 새 주제 → 세션 생성
# ---------------------------------------------------------------------------


class TestNewTopicCreatesSession:
    def test_session_create_when_no_existing_session_matches(
        self, app: AppContext
    ) -> None:
        """User asks about CI/CD pipeline — no existing session covers it.
        Sub-agent says NEW → session_create called.

        사용자가 CI/CD를 질문하는데 기존 세션이 커버하지 않는 주제.
        서브 에이전트가 NEW 판정 → session_create 호출.
        """
        ctx = _make_ctx(app)

        # One session exists, covering frontend work.
        # 프론트엔드 작업을 다루는 세션 하나가 존재.
        session_register(name="frontend", title="Frontend UI", ctx=ctx)

        # Sub-agent calls check_session.
        # 서브 에이전트가 check_session 호출.
        info = check_session(ctx)
        assert len(info["sessions"]) == 1

        # No summary mentions CI/CD → sub-agent says NEW.
        # CI/CD를 언급하는 summary가 없음 → 서브 에이전트가 NEW 판정.
        result = session_create(
            new_session_name="ci-pipeline",
            title="CI/CD Pipeline Setup",
            handoff_summary="Built login form with validation",
            user_prompt="Set up GitHub Actions for the project",
            ctx=ctx,
        )

        assert result["created"] == "ci-pipeline"
        assert app.state.get_current_session() == "ci-pipeline"

        # Outgoing session's summary should be saved.
        # 나간 세션의 summary가 저장되어야 한다.
        fe = app.session_store.load_session_by_name("frontend")
        assert fe is not None
        assert fe.summary == "Built login form with validation"

        # Socket signal should carry NEW action.
        # 소켓 신호에 NEW action이 포함되어야 한다.
        signal = app.socket_client.send_signal.call_args[0][0]
        assert signal["action"] == "new"
        assert signal["new_session_name"] == "ci-pipeline"
        assert signal["rename_current"] == "frontend"


# ---------------------------------------------------------------------------
# 4. Stay in current session
# 4. 현재 세션 유지 (STAY)
# ---------------------------------------------------------------------------


class TestStayInCurrentSession:
    def test_no_tool_call_needed_when_topic_matches_current(
        self, app: AppContext
    ) -> None:
        """User asks about login form — current session is frontend UI.
        Sub-agent says STAY → no switch/create tool called.

        사용자가 로그인 폼을 질문 — 현재 세션이 프론트엔드 UI.
        서브 에이전트가 STAY 판정 → switch/create 도구 호출 없음.
        check_session으로 판단만 하고 상태 변경 없음을 검증.
        """
        ctx = _make_ctx(app)

        session_register(
            name="frontend",
            title="Frontend UI",
            ctx=ctx,
            summary="Building login form with React",
        )

        # Sub-agent calls check_session to evaluate.
        # 서브 에이전트가 check_session으로 평가한다.
        info = check_session(ctx)
        assert info["current"] == "frontend"
        current_summary = next(
            s["summary"] for s in info["sessions"] if s["name"] == "frontend"
        )
        assert "login" in current_summary.lower()

        # Sub-agent says STAY — no further tool calls.
        # 서브 에이전트가 STAY 판정 — 추가 도구 호출 없음.
        # Verify state is unchanged.
        # 상태가 변하지 않았는지 검증.
        assert app.state.get_current_session() == "frontend"
        app.socket_client.send_signal.assert_not_called()


# ---------------------------------------------------------------------------
# 5. --resume handshake
# 5. --resume 핸드셰이크
# ---------------------------------------------------------------------------


class TestResumeViaHandshake:
    async def test_lifespan_sets_current_from_handshake(
        self, tmp_path: Path
    ) -> None:
        """ccode --resume backend → wrapper reports "backend" in handshake
        → lifespan sets current session to "backend".

        ccode --resume backend → 래퍼가 핸드셰이크에서 "backend" 응답
        → lifespan이 현재 세션을 "backend"으로 설정.
        """
        # Pre-populate a session on disk so the store is not empty.
        # 디스크에 세션을 미리 넣어 store가 비어 있지 않게 한다.
        store = SessionStore(tmp_path)
        store.save_session(
            SessionMetadata.new(name="backend", title="Backend API")
        )

        mock_client_instance = MagicMock()
        mock_client_instance.request_handshake.return_value = "backend"

        mock_server = MagicMock()

        with (
            patch("session_manager.server.os.getcwd", return_value=str(tmp_path)),
            patch.dict("os.environ", {"SESSION_MANAGER_SOCKET": "/tmp/fake.sock"}),
            patch(
                "session_manager.server.WrapperSocketClient",
                return_value=mock_client_instance,
            ),
        ):
            async with app_lifespan(mock_server) as ctx:
                assert ctx.state.get_current_session() == "backend"


# ---------------------------------------------------------------------------
# 6. --continue resolves latest by last_accessed
# 6. --continue (last_accessed 기준으로 최신 세션 선택)
# ---------------------------------------------------------------------------


class TestContinueResolvesLatest:
    async def test_lifespan_resolves_from_store_when_handshake_null(
        self, tmp_path: Path
    ) -> None:
        """No --resume → handshake returns null → resolve_from_store picks
        the session with the most recent last_accessed.

        --resume 없이 시작 → 핸드셰이크 null → resolve_from_store가
        last_accessed가 가장 최신인 세션을 선택.
        """
        store = SessionStore(tmp_path)

        # Create two sessions with different last_accessed times.
        # last_accessed가 다른 세션 두 개를 생성한다.
        old = SessionMetadata.new(name="old-sess", title="Old")
        store.save_session(old)

        # Small delay to ensure different timestamps.
        # 타임스탬프가 달라지도록 짧은 대기.
        time.sleep(0.01)

        recent = SessionMetadata.new(name="recent-sess", title="Recent")
        store.save_session(recent)

        mock_client_instance = MagicMock()
        mock_client_instance.request_handshake.return_value = None

        mock_server = MagicMock()

        with (
            patch("session_manager.server.os.getcwd", return_value=str(tmp_path)),
            patch.dict("os.environ", {"SESSION_MANAGER_SOCKET": "/tmp/fake.sock"}),
            patch(
                "session_manager.server.WrapperSocketClient",
                return_value=mock_client_instance,
            ),
        ):
            async with app_lifespan(mock_server) as ctx:
                assert ctx.state.get_current_session() == "recent-sess"


# ---------------------------------------------------------------------------
# 7. First ccode run — auto-registers default session
# 7. 첫 ccode 실행 → 기본 세션 자동 등록
# ---------------------------------------------------------------------------


class TestFirstRunAutoRegistersDefault:
    async def test_lifespan_creates_default_when_no_sessions(
        self, tmp_path: Path
    ) -> None:
        """No sessions on disk, no --resume → lifespan auto-registers
        a "default" session.

        디스크에 세션이 없고 --resume도 없으면 lifespan이
        "default" 세션을 자동 등록한다.
        """
        mock_client_instance = MagicMock()
        mock_client_instance.request_handshake.return_value = None

        mock_server = MagicMock()

        with (
            patch("session_manager.server.os.getcwd", return_value=str(tmp_path)),
            patch.dict("os.environ", {"SESSION_MANAGER_SOCKET": "/tmp/fake.sock"}),
            patch(
                "session_manager.server.WrapperSocketClient",
                return_value=mock_client_instance,
            ),
        ):
            async with app_lifespan(mock_server) as ctx:
                assert ctx.state.get_current_session() == "default"

                # Default session should be persisted on disk.
                # 기본 세션이 디스크에 저장되어야 한다.
                stored = ctx.session_store.load_session_by_name("default")
                assert stored is not None
                assert stored.title == "Default session"
                assert stored.status == SessionStatus.ACTIVE


# ---------------------------------------------------------------------------
# 8. Multi-session round-trip switching
# 8. 다중 세션 순환 전환 (A→B→C→A)
# ---------------------------------------------------------------------------


class TestMultiSessionRoundTrip:
    def test_a_b_c_a_switching_accumulates_transitions(
        self, app: AppContext
    ) -> None:
        """A→B→C→A round-trip: each switch updates the outgoing summary
        and records a transition. Final state: back at A with all
        transitions recorded.

        A→B→C→A 순환 전환: 매번 나가는 세션의 summary가 갱신되고
        transition이 기록된다. 최종 상태: A에 복귀, 모든 transition 보존.
        """
        ctx = _make_ctx(app)

        # Register three sessions.
        # 세션 세 개를 등록한다.
        session_register(name="auth", title="Auth Module", ctx=ctx)
        app.session_store.save_session(
            SessionMetadata.new(name="db", title="Database Layer")
        )
        app.session_store.save_session(
            SessionMetadata.new(name="api", title="API Endpoints")
        )

        # Step 1: auth → db
        # 1단계: auth → db
        session_switch(
            target="db",
            summary="Implemented JWT token validation",
            user_prompt="Check the user table schema",
            ctx=ctx,
        )
        assert app.state.get_current_session() == "db"

        # Step 2: db → api
        # 2단계: db → api
        session_switch(
            target="api",
            summary="Added migration for user_roles table",
            user_prompt="Add the /api/roles endpoint",
            ctx=ctx,
        )
        assert app.state.get_current_session() == "api"

        # Step 3: api → auth (back to start)
        # 3단계: api → auth (시작점 복귀)
        session_switch(
            target="auth",
            summary="Created GET/POST /api/roles with pagination",
            user_prompt="Add refresh token rotation",
            ctx=ctx,
        )
        assert app.state.get_current_session() == "auth"

        # Verify accumulated state across all sessions.
        # 모든 세션에 걸친 누적 상태를 검증한다.
        auth = app.session_store.load_session_by_name("auth")
        assert auth is not None
        assert auth.summary == "Implemented JWT token validation"
        assert len(auth.transitions) == 1
        assert auth.transitions[0].to_session == "db"

        db = app.session_store.load_session_by_name("db")
        assert db is not None
        assert db.summary == "Added migration for user_roles table"
        assert len(db.transitions) == 1
        assert db.transitions[0].to_session == "api"

        api = app.session_store.load_session_by_name("api")
        assert api is not None
        assert api.summary == "Created GET/POST /api/roles with pagination"
        assert len(api.transitions) == 1
        assert api.transitions[0].to_session == "auth"

        # Three socket signals should have been sent.
        # 소켓 신호가 3번 전송되어야 한다.
        assert app.socket_client.send_signal.call_count == 3


# ---------------------------------------------------------------------------
# 9. NEW from first session — rename_current is null
# 9. 첫 세션에서 NEW → rename_current=null
# ---------------------------------------------------------------------------


class TestNewFromFirstSessionRenameNull:
    def test_session_create_from_unregistered_first_session(
        self, app: AppContext
    ) -> None:
        """First session was not registered (or is the auto-created
        "default" that has no meaningful content). Sub-agent decides
        NEW → rename_current should be null.

        첫 세션이 미등록(또는 내용이 없는 자동 생성 "default").
        서브 에이전트가 NEW 판정 → rename_current가 null이어야 한다.
        """
        ctx = _make_ctx(app)

        # State has no current session — simulating a wrapper that
        # just started without --resume.
        # 현재 세션이 없는 상태 — --resume 없이 시작된 래퍼를 시뮬레이션.
        assert app.state.get_current_session() is None

        result = session_create(
            new_session_name="first-real",
            title="First Real Session",
            handoff_summary="",
            user_prompt="Hello, let's start working on the project",
            ctx=ctx,
        )

        assert result["created"] == "first-real"
        assert result["rename_current"] is None
        assert app.state.get_current_session() == "first-real"

        signal = app.socket_client.send_signal.call_args[0][0]
        assert signal["action"] == "new"
        assert signal["rename_current"] is None
        assert signal["new_session_name"] == "first-real"
