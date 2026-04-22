"""
Context Session Manager MCP server entry point.

Hosts the MCP tools that Claude Code's sub-agent uses to inspect and
manage sessions.  At startup it connects to the PTY wrapper via a Unix
Domain Socket (path from ``SESSION_MANAGER_SOCKET`` env var), performs a
handshake to learn the current session name, and initialises the
in-memory state together with the on-disk stores.  All tool handlers
share this state through the FastMCP *lifespan* context.

Context Session Manager MCP 서버 진입점.

Claude Code의 서브 에이전트가 세션을 조회·관리하는 MCP 도구를 호스팅한다.
시작 시 ``SESSION_MANAGER_SOCKET`` 환경변수에 지정된 경로로 PTY 래퍼의
Unix Domain Socket에 연결하고, 핸드셰이크를 거쳐 현재 세션 이름을 파악한
뒤, 인메모리 상태와 디스크 스토어를 초기화한다. 모든 도구 핸들러는 FastMCP
의 *lifespan* 컨텍스트를 통해 이 상태를 공유한다.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from session_manager.lifecycle import cleanup_expired_sessions, get_cleanup_period_days
from session_manager.models.session import (
    SessionMetadata,
    SessionStatus,
    TransitionRecord,
)
from session_manager.state import SessionManagerState
from session_manager.storage import FieldStore, ProjectContextStore, SessionStore
from session_manager.wrapper.socket_client import WrapperSocketClient

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    """
    Shared state accessible from every tool handler via lifespan context.

    lifespan 컨텍스트를 통해 모든 도구 핸들러에서 접근 가능한 공유 상태.
    """

    state: SessionManagerState
    session_store: SessionStore
    field_store: FieldStore
    project_context_store: ProjectContextStore
    socket_client: WrapperSocketClient
    project_path: Path


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Initialise shared resources before the server accepts tool calls.

    서버가 도구 호출을 받기 전에 공유 자원을 초기화한다.
    핸드셰이크로 래퍼에서 현재 세션 이름을 받고, 실패 시 스토어에서 추론한다.
    """
    project_path = Path(os.getcwd())
    socket_path = os.environ.get("SESSION_MANAGER_SOCKET", "")

    # -- stores
    session_store = SessionStore(project_path)
    field_store = FieldStore(project_path)
    project_context_store = ProjectContextStore(project_path)

    # -- state
    state = SessionManagerState()

    # -- socket client + handshake
    client = WrapperSocketClient(socket_path)
    if socket_path:
        try:
            client.connect()
            current = client.request_handshake()
            if current is not None:
                state.set_current_session(current)
                logger.info("Handshake OK — current session: %s", current)
            else:
                resolved = state.resolve_from_store(session_store)
                if resolved is not None:
                    state.set_current_session(resolved)
                logger.info(
                    "Handshake returned null — resolved from store: %s", resolved
                )
        except OSError:
            logger.warning(
                "Could not connect to wrapper socket at %s — "
                "running without wrapper connection",
                socket_path,
            )
    else:
        logger.warning(
            "SESSION_MANAGER_SOCKET not set — running without wrapper connection"
        )

    # Clean up expired sessions at startup.
    # 서버 시작 시 만료된 세션을 정리한다.
    period = get_cleanup_period_days()
    deleted = cleanup_expired_sessions(session_store, period)
    if deleted:
        logger.info("Startup cleanup: removed %d expired session(s)", len(deleted))

    ctx = AppContext(
        state=state,
        session_store=session_store,
        field_store=field_store,
        project_context_store=project_context_store,
        socket_client=client,
        project_path=project_path,
    )

    try:
        yield ctx
    finally:
        client.close()


_SERVER_INSTRUCTIONS = """\
You manage multiple conversation sessions within a single Claude Code process.

## Context Switch Detection
When the user's message concerns a different topic/code area from the current \
session, spawn a sub-agent with this prompt:
  "User prompt: '{prompt}'. Current session: {name}. \
Call check_session to get the session list. Compare each summary against \
the prompt. Respond as ACTION:SESSION_NAME:REASON. \
ACTION = STAY | SWITCH | NEW | ASK_USER."
Then:
- STAY: process normally.
- SWITCH: confirm with user, then switch.
- NEW: confirm with user, then create a new session.
- ASK_USER: present candidates and let user choose.
If context clearly matches the current session, skip the sub-agent.

## Handoff Block
When input contains [handoff]...[/handoff]:
1. Parse the JSON inside. Follow the instructions list (read the listed files).
2. Read the message field for previous session context.
3. Text after [/handoff] is the user's actual prompt.

## Auto-Init
When input contains [자동 초기화], the wrapper detected that bootstrapping \
is needed. Follow each instruction in the message before handling the user's \
request.

## Summary Format
When switching or ending a session, write a 2-3 sentence summary: \
where (files/areas touched), what (work performed), status \
(done / in-progress / remaining). Update the title if it has evolved.\
"""

mcp_server = FastMCP(
    "session-manager",
    instructions=_SERVER_INSTRUCTIONS,
    lifespan=app_lifespan,
)

def _get_app_ctx(ctx: Context) -> AppContext:
    """
    Extract AppContext from the MCP request context.

    MCP 요청 컨텍스트에서 AppContext를 꺼낸다.
    """
    return ctx.request_context.lifespan_context  # type: ignore[return-value]


# ------------------------------------------------------------------ tools
# 도구 등록 -------------------------------------------------------------------


@mcp_server.tool()
def check_session(ctx: Context) -> dict:
    """
    Return the current session and a list of all registered sessions.

    현재 세션 이름과 등록된 전체 세션 목록을 반환한다.
    서브 에이전트가 사용자의 메시지를 어느 세션으로 보낼지 판단할 때 사용한다.
    """
    app = _get_app_ctx(ctx)
    sessions = app.session_store.list_sessions()
    return {
        "current": app.state.get_current_session(),
        "sessions": [
            {
                "name": s.name,
                "title": s.title,
                "summary": s.summary,
                "last_accessed": s.last_accessed,
                "status": s.status.value,
            }
            for s in sessions
        ],
    }


@mcp_server.tool()
def session_register(name: str, title: str, ctx: Context, summary: str | None = None) -> dict:
    """
    Register a new session with the given name and title.

    새 세션을 등록한다. 첫 대화 시작(부트스트랩)이나 새 세션 생성 직후에
    호출되어, 세션에 이름·제목을 부여하고 현재 세션으로 설정한다.
    """
    app = _get_app_ctx(ctx)
    session = SessionMetadata.new(name=name, title=title, summary=summary)
    app.session_store.save_session(session)
    app.state.set_current_session(name)
    return {
        "registered": name,
        "session_id": session.session_id,
    }


_HANDOFF_INSTRUCTIONS = [
    ".session-manager/static-field.json 읽기 — 다른 세션이 환경/서버 정보를 변경했을 수 있음",
    ".session-manager/project-context.md 읽기"
    " — 다른 세션이 프로젝트 구조/의존성을 변경했을 수 있음",
]


@mcp_server.tool()
def session_switch(
    target: str,
    summary: str,
    user_prompt: str,
    ctx: Context,
    updated_title: str | None = None,
) -> dict:
    """
    Switch from the current session to *target*.

    현재 세션을 마무리(요약 저장)하고, 래퍼에 SWITCH 신호를 보내
    대상 세션으로 전환한다. 서브 에이전트가 사용자의 메시지가 다른
    세션에 속한다고 판단했을 때 호출한다.
    """
    app = _get_app_ctx(ctx)
    current_name = app.state.get_current_session()

    # Update the outgoing session's metadata.
    # 나가는 세션의 메타데이터를 갱신한다.
    if current_name is not None:
        current = app.session_store.load_session_by_name(current_name)
        if current is not None:
            current.summary = summary
            if updated_title is not None:
                current.title = updated_title
            current.transitions.append(
                TransitionRecord.new(from_session=current_name, to_session=target)
            )
            current.touch()
            app.session_store.save_session(current)

    # Send SWITCH signal to the wrapper.
    # 래퍼에 SWITCH 신호를 전송한다.
    handoff = {
        "from": current_name,
        "message": summary,
        "instructions": _HANDOFF_INSTRUCTIONS,
    }
    app.socket_client.send_signal({
        "action": "switch",
        "target": target,
        "handoff": handoff,
        "user_prompt": user_prompt,
    })

    app.state.set_current_session(target)
    return {"switched_to": target}


@mcp_server.tool()
def session_create(
    new_session_name: str,
    title: str,
    handoff_summary: str,
    user_prompt: str,
    ctx: Context,
) -> dict:
    """
    Create a brand-new session and restart Claude Code into it.

    새 세션을 만들어 Claude Code를 재시작시킨다. 서브 에이전트가 사용자의
    메시지가 기존 세션 어디에도 해당하지 않는다고 판단했을 때 호출한다.
    현재 세션을 마무리하고 래퍼에 NEW 신호를 보낸다.
    """
    app = _get_app_ctx(ctx)

    # Clean up expired sessions when creating a new one.
    # 새 세션 생성 시 만료된 세션을 정리한다.
    period = get_cleanup_period_days()
    deleted = cleanup_expired_sessions(app.session_store, period)
    if deleted:
        logger.info("Pre-create cleanup: removed %d expired session(s)", len(deleted))

    current_name = app.state.get_current_session()

    # Update the outgoing session's metadata (if registered).
    # 나가는 세션의 메타데이터를 갱신한다 (등록된 경우에만).
    rename_current: str | None = None
    if current_name is not None:
        current = app.session_store.load_session_by_name(current_name)
        if current is not None:
            current.summary = handoff_summary
            current.touch()
            app.session_store.save_session(current)
            rename_current = current_name

    # Send NEW signal to the wrapper.
    # 래퍼에 NEW 신호를 전송한다.
    handoff = {
        "from": current_name,
        "message": handoff_summary,
        "instructions": _HANDOFF_INSTRUCTIONS,
        "new_session_title": title,
    }
    app.socket_client.send_signal({
        "action": "new",
        "rename_current": rename_current,
        "new_session_name": new_session_name,
        "handoff": handoff,
        "user_prompt": user_prompt,
    })

    app.state.set_current_session(new_session_name)
    return {
        "created": new_session_name,
        "rename_current": rename_current,
    }


# NOTE: No natural trigger exists in the current usage flow.  The
# original plan was to intercept /exit via stdin and call this tool
# before the session actually ends, but stdin interception is suspended
# due to low matching reliability.  /clear does NOT trigger this either
# — it only resets LLM context while the session ID and MCP server
# stay alive, so summaries may go stale after /clear.  Kept for future
# use when an alternative trigger (e.g. UserPromptSubmit hook) is
# confirmed.
#
# NOTE: 현재 사용 플로우에서 자연스러운 호출 시점이 없다.  원래는
# stdin에서 /exit을 가로채 이 도구를 먼저 호출할 계획이었으나, 매칭
# 신뢰도 부족으로 가로채기 자체가 보류됨.  /clear 시에도 호출되지
# 않음 — /clear는 LLM 컨텍스트만 리셋하고 세션 ID와 MCP 서버는
# 유지되므로, /clear 후 summary가 오염됨.  대체 트리거
# (예: UserPromptSubmit hook) 확정 시 활용 예정.
@mcp_server.tool()
def session_end(summary: str, ctx: Context) -> dict:
    """
    Archive the current session with a final summary.

    현재 세션을 종료한다. 최종 요약을 저장하고 상태를 ARCHIVED로 변경하여,
    이후 세션 매칭 대상에서 제외되도록 한다.
    """
    app = _get_app_ctx(ctx)
    current_name = app.state.get_current_session()

    if current_name is not None:
        current = app.session_store.load_session_by_name(current_name)
        if current is not None:
            current.summary = summary
            current.status = SessionStatus.ARCHIVED
            current.touch()
            app.session_store.save_session(current)

    app.state.set_current_session(None)
    return {"ended": current_name}


@mcp_server.tool()
def update_static(
    ctx: Context,
    project_context: str | None = None,
    conventions: str | None = None,
    project_map: dict[str, str] | None = None,
    variables: dict | None = None,
) -> dict:
    """
    Partially update the project-wide shared static field.

    프로젝트 전역 공유 정보(환경, 컨벤션, 변수 등)를 부분 갱신한다.
    제공된 필드만 덮어쓰고 나머지는 기존 값을 유지한다. 어떤 세션에서든
    갱신하면 다른 세션에서 최신 값을 읽을 수 있다.
    """
    app = _get_app_ctx(ctx)
    static = app.field_store.load_static()

    if project_context is not None:
        static.project_context = project_context
    if conventions is not None:
        static.conventions = conventions
    if project_map is not None:
        static.project_map = project_map
    if variables is not None:
        static.variables = variables

    static.touch()
    app.field_store.save_static(static)
    return {"updated_at": static.updated_at}


@mcp_server.tool()
def init_project(content: str, ctx: Context) -> dict:
    """
    Create project-context.md if it does not exist yet.

    project-context.md가 아직 없을 때 새로 생성한다. 프로젝트 구조와
    의존성을 설명하는 문서로, 세션 전환 시 새 LLM이 맥락을 파악하는 데 쓰인다.
    이미 존재하면 덮어쓰지 않고 기존 내용을 그대로 반환한다.
    """
    app = _get_app_ctx(ctx)
    if app.project_context_store.exists():
        return {
            "created": False,
            "content": app.project_context_store.read(),
        }
    app.project_context_store.write(content)
    return {"created": True}


@mcp_server.tool()
def reinit_project(content: str, ctx: Context) -> dict:
    """
    Overwrite project-context.md with fresh content.

    project-context.md를 처음부터 다시 작성한다. 사용자가 명시적으로
    프로젝트 맥락 문서를 새로 쓰고 싶을 때 호출한다.
    """
    app = _get_app_ctx(ctx)
    app.project_context_store.write(content)
    return {"reinitialized": True}


@mcp_server.tool()
def update_project_context(content: str, ctx: Context) -> dict:
    """
    Replace project-context.md with updated content.

    project-context.md를 새 내용으로 교체한다. 프로젝트 구조나 의존성이
    변경되었을 때 호출하여 문서를 최신 상태로 유지한다.
    """
    app = _get_app_ctx(ctx)
    app.project_context_store.write(content)
    return {"updated": True}


def main() -> None:
    """
    Entry point invoked by Claude Code when spawning this MCP server.

    Claude Code가 이 MCP 서버를 spawn할 때 호출하는 진입점.
    """
    mcp_server.run()


if __name__ == "__main__":
    main()
