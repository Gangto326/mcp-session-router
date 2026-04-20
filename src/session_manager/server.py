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

from session_manager.models.session import SessionMetadata, TransitionRecord
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


mcp_server = FastMCP(
    "session-manager",
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


def main() -> None:
    """
    Entry point invoked by Claude Code when spawning this MCP server.

    Claude Code가 이 MCP 서버를 spawn할 때 호출하는 진입점.
    """
    mcp_server.run()


if __name__ == "__main__":
    main()
