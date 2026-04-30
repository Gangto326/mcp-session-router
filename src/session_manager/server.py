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

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.stdio import stdio_server
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage, JSONRPCNotification

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


class ChannelFastMCP(FastMCP):
    """FastMCP that declares the experimental claude/channel capability.

    실험적 claude/channel capability를 선언하는 FastMCP 서브클래스.
    Stdio 모드의 write_stream을 인스턴스 멤버로 보존해, lifespan 안에서
    백그라운드 task가 channel notification을 push할 수 있도록 한다.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._channel_write_stream: Any = None

    async def run_stdio_async(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            self._channel_write_stream = write_stream
            try:
                opts = self._mcp_server.create_initialization_options(
                    experimental_capabilities={"claude/channel": {}},
                )
                await self._mcp_server.run(read_stream, write_stream, opts)
            finally:
                self._channel_write_stream = None


async def send_channel_notification(
    write_stream: Any, content: str, meta: dict[str, str]
) -> None:
    """Push a notifications/claude/channel JSON-RPC message to the LLM context.

    LLM 컨텍스트로 ``notifications/claude/channel`` JSON-RPC 메시지를 push.
    The message wraps as ``<channel source="session-manager" ...>content</channel>``
    in Claude's input.
    """
    notif = JSONRPCNotification(
        jsonrpc="2.0",
        method="notifications/claude/channel",
        params={"content": content, "meta": meta},
    )
    await write_stream.send(SessionMessage(message=JSONRPCMessage(notif)))

_DEFAULT_SESSION_NAME = "default"
_DEFAULT_SESSION_TITLE = "Default session"

# Mark every session-manager tool as always-loaded so Claude Code skips its
# deferred-tool / ToolSearch step. Required because:
#   1) The routing harness (AGENT_GUIDE.md) needs these tools callable from
#      the *first* user message, before any ToolSearch round-trip.
#   2) Sub-agents spawned by the harness do not inherit the parent's
#      ToolSearch results (anthropics/claude-code Issue #25200), so deferred
#      tools are unreachable from a sub-agent without this opt-out.
# Each tool's `_meta` carries `"anthropic/alwaysLoad": true`, which Claude
# Code v2.1.121+ honours per-tool regardless of `ENABLE_TOOL_SEARCH` setting.
# 모든 session-manager 도구를 always-loaded로 표시 — Claude Code의 deferred /
# ToolSearch 단계를 건너뛰게 함. 이유: (1) 라우팅 하네스가 첫 사용자 메시지에서
# 도구를 즉시 호출해야 하고, (2) sub-agent가 parent의 ToolSearch 결과를
# 상속하지 않아 deferred 도구를 호출할 수 없기 때문 (Issue #25200).
_ALWAYS_LOAD_META: dict[str, bool] = {"anthropic/alwaysLoad": True}


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
    # Tracks whether session_end was the response to an active intercept.
    # session_end가 활성 가로채기에 대한 응답인지 추적.
    intercept_active: dict[str, bool] = field(default_factory=lambda: {"value": False})


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

    # Auto-register a default session if none exists and no --resume
    # was given.  The LLM can update the name/title later via
    # session_switch or session_create once it understands the context.
    #
    # 세션이 없고 --resume 인자도 없으면 기본 세션을 자동 등록한다.
    # LLM이 맥락을 파악한 뒤 session_switch/session_create로 이름을
    # 갱신할 수 있다.
    if state.get_current_session() is None and not session_store.list_sessions():
        default = SessionMetadata.new(
            name=_DEFAULT_SESSION_NAME, title=_DEFAULT_SESSION_TITLE
        )
        session_store.save_session(default)
        state.set_current_session(_DEFAULT_SESSION_NAME)
        logger.info("Auto-registered default session")

    # Build instructions dynamically — add a project-context.md hint
    # when the file does not exist yet so the LLM creates it.
    #
    # instructions를 동적으로 구성한다 — project-context.md가 없으면
    # LLM에게 생성하라는 힌트를 추가한다.
    instructions = _SERVER_INSTRUCTIONS
    if not project_context_store.exists():
        instructions += _INIT_PROJECT_HINT
    server._mcp_server.instructions = instructions  # type: ignore[attr-defined]

    ctx = AppContext(
        state=state,
        session_store=session_store,
        field_store=field_store,
        project_context_store=project_context_store,
        socket_client=client,
        project_path=project_path,
    )

    # Start the wrapper → MCP receive loop so slash-command intercepts get
    # forwarded to the LLM as <channel> messages. Only spawn when the socket
    # is actually connected (no-op for tool tests / standalone runs).
    # 슬래시 명령 가로채기 신호를 LLM에게 <channel> 메시지로 forward하기 위해
    # 래퍼 → MCP receive 루프 시작. 소켓이 연결된 경우에만 spawn.
    recv_task: asyncio.Task[None] | None = None
    if socket_path and client._sock is not None:
        recv_task = asyncio.create_task(
            client.recv_loop(_make_intercept_handler(server, ctx))
        )

    try:
        yield ctx
    finally:
        if recv_task is not None:
            recv_task.cancel()
            try:
                await recv_task
            except (asyncio.CancelledError, Exception):
                pass
        client.close()


def _make_intercept_handler(
    server: FastMCP, app: AppContext
) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """Build the on_message callback that converts wrapper signals into
    channel notifications.

    래퍼가 보낸 가로채기 신호를 LLM의 <channel> 메시지로 변환하는
    on_message 콜백을 만든다. server는 ChannelFastMCP 인스턴스 — 그 안의
    _channel_write_stream을 이용해 stdio write_stream에 직접 push한다.
    """

    async def handler(msg: dict[str, Any]) -> None:
        action = msg.get("action")
        if action != "intercept":
            return
        command = msg.get("command")
        args = msg.get("args", "")
        if not isinstance(command, str) or not isinstance(args, str):
            return

        write_stream = getattr(server, "_channel_write_stream", None)
        if write_stream is None:
            logger.warning("intercept signal received but no channel write_stream")
            return

        current = app.state.get_current_session() or "(none)"
        full_cmd = f"/{command}" + (f" {args}" if args else "")
        content = (
            f"User typed slash command directly: {full_cmd}. "
            f"This bypasses the LLM-mediated session switch flow, so the "
            f"current session ('{current}') summary will NOT be updated "
            f"unless you act now. Call session_end with a 2-3 sentence "
            f"summary of work done in '{current}'. After session_end "
            f"completes, the original command will be auto-forwarded to "
            f"Claude Code — do not call any other tool."
        )
        meta = {
            "command": command,
            "args": args,
            "current_session": current,
        }
        # Mark intercept active so session_end knows to notify the wrapper.
        # 가로채기 활성 표시 — session_end가 래퍼 통보 여부 판단에 사용.
        app.intercept_active["value"] = True
        await send_channel_notification(write_stream, content, meta)
        logger.info("Pushed intercept channel notification: %s", full_cmd)

    return handler


_SERVER_INSTRUCTIONS = """\
You manage multiple conversation sessions within a single Claude Code process.

## Context Switch Detection
When the user's message could shift topic, code area, file/directory scope, \
or domain from the current session, spawn a sub-agent with this prompt:
  "User prompt: '{prompt}'. Current session: {name}.

  Call check_session to get all sessions and their summaries.

  Decision rules (apply in order):
  - If exactly one session matches → SWITCH:that_name
  - If multiple sessions plausibly match → ASK_USER (list candidates)
  - If the current session matches and others don't → STAY
  - If NO session matches OR all summaries are null/missing → NEW
  - A default-named session with null summary represents an unclaimed \
  conversation; treat it as no match unless the prompt is a literal \
  continuation of the current turn.

  Prefer SWITCH or NEW over STAY when in doubt — keeping each session \
  focused is cheaper than a polluted multi-topic session.

  Respond as exactly: ACTION:SESSION_NAME:REASON
  ACTION = STAY | SWITCH | NEW | ASK_USER"
Then:
- STAY: process normally.
- SWITCH: confirm with user, then switch.
- NEW: confirm with user, then create a new session.
- ASK_USER: present candidates and let user choose.

**Why dispatch eagerly**: a sub-agent call costs ~1-2s and a few hundred \
tokens. A polluted single session costs much more — longer context every \
turn, degraded answers, and far more total tokens over time. Skip the \
sub-agent ONLY when the message is an obvious follow-up to the immediately \
previous turn within the same session.

## Handoff Block
When input contains [handoff]...[/handoff]:
1. Parse the JSON inside. Follow the instructions list (read the listed files).
2. Read the message field for previous session context.
3. Text after [/handoff] is the user's actual prompt.

## Summary Format
When switching or ending a session, write a 2-3 sentence summary: \
where (files/areas touched), what (work performed), status \
(done / in-progress / remaining). Update the title if it has evolved.

## Channel Intercept
If you receive a <channel source="session-manager"> tag asking you to call \
session_end, that's a slash-command intercept (the user typed /resume, \
/exit, /rename, or /new directly). Call session_end immediately with a \
2-3 sentence summary of the current session's work. The wrapper will then \
auto-forward the original command. Do NOT call any other tool, do NOT \
ask the user for confirmation, do NOT respond with text — just call \
session_end and stop.\
"""

_INIT_PROJECT_HINT = """

## Project Initialization Required
.session-manager/project-context.md does not exist yet. Before handling the \
user's first request, briefly explore the project structure and call \
init_project with a concise overview.\
"""

mcp_server = ChannelFastMCP(
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


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
def check_session(ctx: Context) -> dict:
    """Return all registered sessions so you can route the user message to the right one.

    **Call this tool BEFORE responding to any user message** that could shift
    the topic, code area, file/directory scope, or domain from the current
    session. The returned summaries let you decide:

    - STAY in the current session (topic clearly matches)
    - SWITCH to a different existing session (topic matches another one)
    - CREATE a new session via session_create (topic matches none)

    **Trigger conditions — call this when ANY apply**:
    - User mentions a different file path, directory, component, or domain
    - User shifts from one area (e.g. frontend) to another (e.g. backend)
    - User starts a topic the current conversation does not naturally cover
    - You are uncertain which session this message belongs to

    Skip ONLY when the message is a direct continuation of the immediately
    previous turn AND clearly within the current session's scope.

    **Why route proactively** — the cost of running a sub-agent (one extra
    call, 1-2 seconds, a few hundred tokens) is **much smaller** than the
    cost of answering multi-topic questions inside a polluted single session
    (longer context window every turn, degraded answer quality, far more
    total tokens spent over many turns). Switching sessions early keeps each
    session's context focused — better answers, fewer total tokens, easier
    to resume later. **When in doubt, prefer SWITCH or NEW over STAY.**

    현재 세션과 등록된 전체 세션 목록을 반환한다. 사용자 메시지가 현재 세션의
    주제·코드 영역·파일/디렉토리 범위·도메인과 조금이라도 다를 가능성이 있으면
    응답 전에 먼저 이 도구를 호출해 어느 세션에서 처리할지 판단한다.

    **적극적으로 호출해야 이득인 이유**:
    서브 에이전트 1회 호출 (1-2초, 수백 토큰)의 비용은 한 세션에 여러 주제가
    누적되어 컨텍스트가 오염될 때의 손실 (긴 context window, 답변 품질 저하,
    누적 토큰 폭증) 보다 훨씬 작다. 세션을 빨리 분리하면 각 세션이 초점을
    유지해 답변 품질이 올라가고 총 토큰 소비가 줄며 나중에 복귀하기도 쉽다.
    의심스러우면 STAY 보다 SWITCH/NEW 를 선호한다.
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


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
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


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
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


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
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


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
def session_end(summary: str, ctx: Context) -> dict:
    """
    Archive the current session with a final summary.

    현재 세션을 종료한다. 최종 요약을 저장하고 상태를 ARCHIVED로 변경한다.
    가로채기 흐름 (사용자가 /resume·/exit·/rename·/new를 직접 입력)에서 LLM이
    호출하면 래퍼에 intercept_done 신호를 보내 큐잉된 사용자 명령을 forward
    하게 한다.
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

    # If this session_end is the response to an active intercept, notify
    # the wrapper so it drains the queued user command. The wrapper itself
    # ignores intercept_done when no intercept is active, so an extra send
    # is harmless.
    # 활성 가로채기 응답이면 래퍼에 intercept_done 신호를 보내 큐잉된 사용자
    # 명령을 forward하게 한다. 래퍼는 비활성 시 무시하므로 안전.
    if app.intercept_active.get("value"):
        try:
            app.socket_client.send_signal({"action": "intercept_done"})
        except (OSError, RuntimeError) as exc:
            logger.warning("Failed to send intercept_done: %s", exc)
        app.intercept_active["value"] = False

    return {"ended": current_name}


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
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


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
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


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
def reinit_project(content: str, ctx: Context) -> dict:
    """
    Overwrite project-context.md with fresh content.

    project-context.md를 처음부터 다시 작성한다. 사용자가 명시적으로
    프로젝트 맥락 문서를 새로 쓰고 싶을 때 호출한다.
    """
    app = _get_app_ctx(ctx)
    app.project_context_store.write(content)
    return {"reinitialized": True}


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
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
