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
import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from session_manager import debug_log, summarizer
from session_manager.claude_conversation import get_active_conversation_id
from session_manager.hooks.user_prompt_submit import (
    _calibrated_auto_threshold,
    _load_auto_error_tolerance,
    _load_routing_mode,
)
from session_manager.lifecycle import cleanup_expired_sessions, get_cleanup_period_days
from session_manager.models.config import ROUTING_MODES
from session_manager.models.fields import (
    VALID_SOURCES,
    UnsupportedStaticSchemaError,
)
from session_manager.models.session import (
    PrecedentRecord,
    SessionMetadata,
    SessionStatus,
    TransitionRecord,
)
from session_manager.routing import decision_log
from session_manager.state import SessionManagerState
from session_manager.storage import FieldStore, ProjectContextStore, SessionStore
from session_manager.storage.file_store import (
    _CONFIG_FILENAME,
    _SESSION_MANAGER_DIRNAME,
    _atomic_write_text,
    _dump_json,
)
from session_manager.wrapper.socket_client import WrapperSocketClient

logger = logging.getLogger(__name__)


def _set_current_session(app: AppContext, name: str | None) -> None:
    """Set the current session and tell the wrapper about it.

    현재 세션을 설정하고 래퍼에 통보한다.

    The handshake only flows wrapper→MCP, so without this push the wrapper
    never learns the session name on a plain ``ccode`` start — and every
    wrapper-side trigger scoped to a session (/clear summary, periodic
    refresh) silently does nothing. Send failures are non-fatal: the
    wrapper degrades to skipping those triggers, exactly as before.

    핸드셰이크는 래퍼→MCP 단방향이므로, 이 push 가 없으면 래퍼는 인자 없는
    ``ccode`` 시작에서 세션 이름을 끝내 알지 못한다 — 세션 단위 래퍼 트리거
    (/clear 요약, 주기 갱신) 가 조용히 무효화된다. 전송 실패는 치명적이지
    않다: 래퍼는 그 트리거들을 건너뛰는 기존 동작으로 degrade 한다.
    """
    app.state.set_current_session(name)
    try:
        app.socket_client.send_signal({"action": "current_session", "name": name})
    except (OSError, RuntimeError) as exc:
        logger.warning("Failed to notify wrapper of current session: %s", exc)


def _log_tool_call(
    tool: str, app: AppContext | None, args: dict[str, Any]
) -> str:
    """Emit MCP_TOOL_CALL with a fresh event id; pair with _log_tool_return.

    MCP_TOOL_CALL 이벤트를 기록하고 새 event_id 반환. 같은 id 로
    _log_tool_return 과 짝지어 호출 ↔ 반환을 묶는다.
    """
    event_id = debug_log.new_event_id()
    debug_log.log(
        "MCP_TOOL_CALL",
        "LLM",
        {"tool": tool, "event_id": event_id, "args": args},
        session=app.state.get_current_session() if app else None,
    )
    return event_id


def _log_tool_return(
    tool: str,
    event_id: str,
    app: AppContext | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Emit MCP_TOOL_RETURN sharing event_id with the matching _log_tool_call.

    같은 event_id 로 _log_tool_call 과 짝지어 MCP_TOOL_RETURN 을 기록한다.
    """
    debug_log.log(
        "MCP_TOOL_RETURN",
        "MCP_TOOL",
        {"tool": tool, "event_id": event_id, "result": result},
        session=app.state.get_current_session() if app else None,
    )
    return result

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


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """
    Initialise shared resources before the server accepts tool calls.

    서버가 도구 호출을 받기 전에 공유 자원을 초기화한다.
    핸드셰이크로 래퍼에서 현재 세션 이름을 받고, 실패 시 스토어에서 추론한다.
    """
    project_path = Path(os.getcwd())
    socket_path = os.environ.get("SESSION_MANAGER_SOCKET", "")
    debug_log.log(
        "MCP_BOOT",
        "SYSTEM",
        {
            "project_path": str(project_path),
            "socket_path": socket_path,
            "env": debug_log.mask_env(),
        },
    )

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
            current, handshake_conv = client.request_handshake()
            state.set_active_conversation_id(handshake_conv)
            if current is not None:
                state.set_current_session(current)
                logger.info("Handshake OK — current session: %s", current)
            else:
                # Which conversation should the store resolution key on?
                # A wrapper-named NEW conversation owns no session yet, so
                # keying on it would skip the owner match and drop to the
                # last_accessed scan — worse than before F18. Use the
                # handshake id only when some session already owns it;
                # otherwise the previous conversation (mtime) is the lead.
                # 스토어 해석의 키를 어느 대화로 잡을까? 래퍼가 새로 이름
                # 붙인 대화는 아직 아무 세션도 소유하지 않아, 그것으로
                # 키를 잡으면 소유자 매칭을 건너뛰고 last_accessed 스캔으로
                # 떨어진다 — F18 이전보다 나쁘다. 핸드셰이크 id 는 어떤
                # 세션이 이미 소유할 때만 쓰고, 아니면 직전 대화 (mtime)
                # 가 단서다.
                owned = handshake_conv is not None and any(
                    handshake_conv in s.claude_conversation_ids
                    for s in session_store.list_sessions()
                )
                resolved = state.resolve_from_store(
                    session_store,
                    handshake_conv
                    if owned
                    else get_active_conversation_id(project_path),
                )
                if resolved is not None:
                    state.set_current_session(resolved)
                logger.info(
                    "Handshake returned null — resolved from store: %s", resolved
                )
            # Link the child's conversation to the session we settled on,
            # right now (F18): both facts are in hand, and waiting for a
            # session tool call leaves a tool-less conversation unowned
            # forever — the next boot then cannot key on it (measured:
            # e2e 2026-08-22, docs/poc/R5-conversation-id.md).
            # Idempotent (link_conversation skips duplicates).
            # 자식의 대화를 지금 정해진 세션에 바로 연결한다 (F18): 두
            # 사실이 모두 손에 있고, 세션 도구 호출을 기다리면 도구 호출이
            # 없는 대화는 영영 소유자 없이 남아 다음 부팅이 그 대화로
            # 해석하지 못한다 (실측: e2e 2026-08-22,
            # docs/poc/R5-conversation-id.md). 멱등 (link_conversation 이
            # 중복을 건너뜀).
            settled = state.get_current_session()
            if settled is not None and handshake_conv is not None:
                try:
                    session_store.mutate_session_by_name(
                        settled, lambda s: s.link_conversation(handshake_conv)
                    )
                except Exception as exc:
                    logger.warning("Boot-time conversation link failed: %s", exc)
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
    deleted = cleanup_expired_sessions(session_store, period, project_path)
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
        # Link the conversation the wrapper named for this child (F18):
        # the transcript does not exist yet at boot, so only the wrapper
        # can tell — the heuristic would return None here.
        # 래퍼가 이 자식에 붙인 대화를 연결한다 (F18): 부팅 시점엔
        # transcript 가 아직 없어 래퍼만 알 수 있다 — 휴리스틱은 여기서
        # None 을 돌려준다.
        boot_conv = state.get_active_conversation_id()
        if boot_conv is not None:
            default.link_conversation(boot_conv)
        session_store.save_session(default)
        state.set_current_session(_DEFAULT_SESSION_NAME)
        logger.info("Auto-registered default session")

    # Tell the wrapper which session we settled on. Without this the
    # wrapper's mirror stays None on a plain `ccode` start and its
    # session-scoped triggers never fire (see _set_current_session).
    # 확정된 세션을 래퍼에 알린다. 이 통보가 없으면 인자 없는 `ccode` 시작에서
    # 래퍼 미러가 None 으로 남아 세션 단위 트리거가 발동하지 않는다.
    settled = state.get_current_session()
    if settled is not None:
        try:
            client.send_signal({"action": "current_session", "name": settled})
        except (OSError, RuntimeError) as exc:
            logger.warning("Failed to notify wrapper of current session: %s", exc)

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

    # Start the wrapper → MCP receive loop so observed slash commands can
    # invalidate the cached session pointer. Only spawn when the socket is
    # actually connected (no-op for tool tests / standalone runs).
    # 관찰된 슬래시 명령이 캐시된 세션 포인터를 무효화할 수 있도록 래퍼 →
    # MCP receive 루프 시작. 소켓이 연결된 경우에만 spawn.
    recv_task: asyncio.Task[None] | None = None
    if socket_path and client._sock is not None:
        recv_task = asyncio.create_task(
            client.recv_loop(_make_wrapper_signal_handler(ctx))
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


def _active_conversation_id(app: AppContext) -> str | None:
    """The current conversation id: wrapper-reported first, guessed last (F18).

    현재 대화 id — 래퍼가 알려 준 값 먼저, 추측은 마지막 (F18).

    The wrapper names the conversation it spawns (``--session-id``) or
    learns it from the Stop hook, and reports it over the socket
    (handshake + ``active_conversation`` pushes). Only when it does not
    know — a user-driven ``--continue``/``--resume`` before its first
    turn ends — does the transcript-mtime heuristic run.
    래퍼는 자신이 띄우는 대화에 이름을 붙이거나 (``--session-id``) Stop
    hook 으로 알아내, 소켓 (핸드셰이크 + ``active_conversation`` push) 으로
    알려 준다. 래퍼도 모를 때만 — 사용자 주도 ``--continue``/``--resume``
    의 첫 턴 종료 전 — transcript mtime 휴리스틱이 돈다.
    """
    known = app.state.get_active_conversation_id()
    if known is not None:
        return known
    return get_active_conversation_id(app.project_path)


def _make_wrapper_signal_handler(
    app: AppContext,
) -> Callable[[dict[str, Any]], Awaitable[None]]:
    """Build the on_message callback for signals the wrapper observes.

    래퍼가 관찰한 신호를 처리하는 on_message 콜백을 만든다.

    The wrapper reports session-changing slash commands the user typed
    directly (``/resume``, ``/exit``, ``/rename``, ``/new``). Nothing is
    asked of the LLM — the background summariser handles the summary — so
    the only thing to do here is drop the cached session pointer, which the
    user has just invalidated by moving elsewhere by hand. The next tool
    call re-resolves it from the active conversation.

    래퍼는 사용자가 직접 입력한 세션 변경 슬래시 명령을 보고한다. LLM 에게
    요구하는 것은 없다 — 요약은 백그라운드 요약기가 맡는다 — 따라서 여기서
    할 일은 캐시된 세션 포인터를 버리는 것뿐이다. 사용자가 손수 다른 곳으로
    이동해 그 포인터를 무효화했기 때문이며, 다음 도구 호출이 활성
    conversation 으로부터 다시 해석한다.
    """

    async def handler(msg: dict[str, Any]) -> None:
        if msg.get("action") == "active_conversation":
            # Wrapper's conversation-id report (F18): adopt as-is, None
            # included (the wrapper forgot it → back to the heuristic).
            # 래퍼의 대화 id 보고 (F18): 그대로 채택, None 포함 (래퍼가
            # 잊었다 → 휴리스틱으로 복귀).
            conv_id = msg.get("conversation_id")
            app.state.set_active_conversation_id(
                conv_id if isinstance(conv_id, str) and conv_id else None
            )
            return
        if msg.get("action") != "session_command":
            return
        command = msg.get("command")
        if not isinstance(command, str):
            return
        debug_log.log(
            "SESSION_COMMAND_RECEIVED",
            "USER",
            {
                "command": command,
                "args": msg.get("args", ""),
                "invalidated_session": app.state.get_current_session(),
            },
            session=app.state.get_current_session(),
        )
        app.state.set_current_session(None)

    return handler


_SERVER_INSTRUCTIONS = """\
You manage multiple conversation sessions within a single Claude Code process.

## Routing Is Decided Outside You
Session routing runs in a deterministic hook on every prompt submission. \
When a switch looks right, an instruction marked [session-manager 라우터] \
arrives in your context: follow it exactly — ask the user with \
AskUserQuestion, and on acceptance call session_switch / session_create as \
instructed. Without such an instruction, answer in the current session and \
do NOT proactively route, spawn routing sub-agents, or call \
session_switch / session_create on your own topic-shift suspicion.

## Handoff Block
When input contains [handoff]...[/handoff]:
1. Parse the JSON inside. Follow the instructions list (read the listed files).
2. Read the message field for previous session context.
3. Text after [/handoff] is the user's actual prompt.

## Summary Format
When switching or ending a session, write a 2-3 sentence summary: \
where (files/areas touched), what (work performed), status \
(done / in-progress / remaining). Update the title if it has evolved.
\
"""

_INIT_PROJECT_HINT = """

## Project Initialization Required
.session-manager/project-context.md does not exist yet. Before handling the \
user's first request, briefly explore the project structure and call \
init_project with a concise overview.\
"""

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
    event_id = _log_tool_call("check_session", app, {})
    sessions = app.session_store.list_sessions()
    # active_conversation_id is the Claude Code conversation id whose
    # jsonl was most recently appended to under this cwd. Combined with
    # each session's claude_conversation_ids, the routing harness can
    # match picker-driven conversation transitions to the correct
    # tracked session even when current_session is None.
    # active_conversation_id 와 각 세션의 claude_conversation_ids 를 결합하면
    # current 가 None 인 picker 후 상태에서도 라우팅 하네스가 사용자가 들어간
    # conversation 을 정확히 어느 세션에 매칭할지 결정할 수 있다.
    result = {
        "current": app.state.get_current_session(),
        "active_conversation_id": _active_conversation_id(app),
        "sessions": [
            {
                "name": s.name,
                "title": s.title,
                "summary": s.summary,
                "last_accessed": s.last_accessed,
                "status": s.status.value,
                "claude_conversation_ids": list(s.claude_conversation_ids),
            }
            for s in sessions
            # Only ACTIVE sessions are candidates — the same rule the
            # judge and the boot-time inference apply, so the list the
            # LLM sees never contradicts them. An ARCHIVED session
            # comes back on its own when a transition targets it.
            # ACTIVE 세션만 후보 — 판정기·부팅 추론과 같은 규칙이라 LLM
            # 이 보는 목록이 그들과 모순되지 않는다. ARCHIVED 세션은
            # 전환 대상이 되면 스스로 돌아온다.
            if s.status == SessionStatus.ACTIVE
        ],
    }
    return _log_tool_return("check_session", event_id, app, result)


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
def session_register(name: str, title: str, ctx: Context, summary: str | None = None) -> dict:
    """
    Register a new session with the given name and title.

    새 세션을 등록한다. 첫 대화 시작(부트스트랩)이나 새 세션 생성 직후에
    호출되어, 세션에 이름·제목을 부여하고 현재 세션으로 설정한다.
    """
    app = _get_app_ctx(ctx)
    event_id = _log_tool_call(
        "session_register",
        app,
        {
            "name": name,
            "title": title,
            "summary": debug_log.mask_text(summary),
        },
    )
    session = SessionMetadata.new(name=name, title=title, summary=summary)
    # Link the active Claude Code conversation so picker-driven returns
    # to this conversation can be matched back to this session.
    # 활성 Claude Code conversation 을 연결 — 이후 사용자가 picker 로 같은
    # conversation 에 돌아왔을 때 라우팅이 정확히 이 세션으로 매칭되게 한다.
    conv_id = _active_conversation_id(app)
    if conv_id is not None:
        session.link_conversation(conv_id)
    app.session_store.save_session(session)
    _set_current_session(app, name)
    return _log_tool_return(
        "session_register",
        event_id,
        app,
        {
            "registered": name,
            "session_id": session.session_id,
            "claude_conversation_ids": list(session.claude_conversation_ids),
        },
    )


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
    event_id = _log_tool_call(
        "session_switch",
        app,
        {
            "target": target,
            "summary": debug_log.mask_text(summary),
            "user_prompt": debug_log.mask_text(user_prompt),
            "updated_title": updated_title,
        },
    )
    current_name = app.state.get_current_session()

    # Compute active conversation once — used for both outgoing-session
    # link and target-session link below.
    # 활성 conversation 을 한 번만 계산해 outgoing/target 세션 양쪽에 연결한다.
    active_conv_id = _active_conversation_id(app)

    # Update the outgoing session's metadata under the F15 lock — the
    # wrapper's summarizer may be saving this same session concurrently.
    # 나가는 세션의 메타데이터를 F15 잠금 하에 갱신한다 — 래퍼의 요약기가
    # 같은 세션을 동시에 저장하고 있을 수 있다.
    if current_name is not None:

        def apply_outgoing(current: SessionMetadata) -> None:
            current.summary = summary
            if updated_title is not None:
                current.title = updated_title
            current.transitions.append(
                TransitionRecord.new(from_session=current_name, to_session=target)
            )
            # Precedent invalidation (b): an accepted switch to *target*
            # overturns any recorded rejection of that same target.
            # 판례 무효화 (b): *target* 으로의 전환 수용은 같은 대상에 대한
            # 기록된 거부(선례)를 뒤집는다.
            current.drop_precedents_for(target)
            if active_conv_id is not None:
                current.link_conversation(active_conv_id)
            current.touch()

        app.session_store.mutate_session_by_name(current_name, apply_outgoing)

    # Calibration label (R3-C4): an executed switch is the "accept" side
    # of the confirm flow. Pairing with the originating proposal happens
    # at read time — a voluntary switch with no proposal stays orphan.
    # 보정 라벨 (R3-C4) — 실행된 전환은 confirm 흐름의 "수용" 쪽이다.
    # 제안과의 짝짓기는 읽기 시점에 일어난다 — 제안 없는 자발 전환은
    # 무연고로 남는다.
    decision_log.append_label(
        app.project_path,
        target,
        decision_log.LABEL_ACCEPT,
        source="session_switch",
        kept_in=current_name,
    )

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

    # The wrapper already mirrors the target from the switch signal above;
    # this keeps both processes in sync through one path.
    # 래퍼는 위 switch 신호로 이미 target 을 미러링하지만, 두 프로세스가 한
    # 경로로 동기화되도록 여기서도 통보한다.
    _set_current_session(app, target)

    # The departing conversation is deliberately NOT linked to the target.
    # The R2-era injection flow bridged its long switch-to-conversation
    # gap with that link; the R3-FIX1 respawn flow resumes a conversation
    # that is already in the target's lineage, so the link only polluted
    # every lineage consumer (wrong resume pick, judge current-session
    # ambiguity, false stale-conversation warnings — e2e run 006e7dfe).
    # 떠나는 conversation 을 target 에 링크하지 않는다 (의도적). R2 주입
    # 흐름은 전환~대화 변경 사이의 긴 틈을 그 링크로 메웠지만, R3-FIX1
    # respawn 흐름은 target 계보에 이미 있는 conversation 을 재개하므로
    # 링크는 계보 소비처 전부를 오염시키기만 했다 (잘못된 resume 선택·
    # 판정기 현재 세션 모호·만료 대화 오경고 — e2e run 006e7dfe 실측).

    return _log_tool_return(
        "session_switch", event_id, app, {"switched_to": target}
    )


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
    event_id = _log_tool_call(
        "session_create",
        app,
        {
            "new_session_name": new_session_name,
            "title": title,
            "handoff_summary": debug_log.mask_text(handoff_summary),
            "user_prompt": debug_log.mask_text(user_prompt),
        },
    )

    # Clean up expired sessions when creating a new one.
    # 새 세션 생성 시 만료된 세션을 정리한다.
    period = get_cleanup_period_days()
    deleted = cleanup_expired_sessions(app.session_store, period, app.project_path)
    if deleted:
        logger.info("Pre-create cleanup: removed %d expired session(s)", len(deleted))

    current_name = app.state.get_current_session()

    # Update the outgoing session's metadata (if registered), under the
    # F15 lock (concurrent summarizer saves).
    # 나가는 세션의 메타데이터를 갱신한다 (등록된 경우에만) — F15 잠금
    # 하에 (요약기 동시 저장 대비).
    rename_current: str | None = None
    if current_name is not None:
        conv_id = _active_conversation_id(app)

        def apply_outgoing(current: SessionMetadata) -> None:
            current.summary = handoff_summary
            # Link the conversation we're leaving before the wrapper
            # respawns into a brand-new one.
            # 자식 재spawn 으로 새 conversation 으로 가기 전, 떠나는 conversation
            # 을 outgoing 세션에 연결해둔다.
            if conv_id is not None:
                current.link_conversation(conv_id)
            current.touch()

        saved = app.session_store.mutate_session_by_name(
            current_name, apply_outgoing
        )
        if saved is not None:
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

    _set_current_session(app, new_session_name)
    return _log_tool_return(
        "session_create",
        event_id,
        app,
        {
            "created": new_session_name,
            "rename_current": rename_current,
        },
    )


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
def session_end(summary: str, ctx: Context) -> dict:
    """
    Archive the current session with a final summary.

    현재 세션을 종료한다. 최종 요약을 저장하고 상태를 ARCHIVED로 변경한다.
    LLM 이 자발적으로 세션을 마감할 때 호출한다 — 사용자가 /resume·/exit 등을
    직접 입력하는 경우의 요약은 백그라운드 요약기가 담당하므로 이 도구를
    거치지 않는다.
    """
    app = _get_app_ctx(ctx)
    event_id = _log_tool_call(
        "session_end",
        app,
        {"summary": debug_log.mask_text(summary)},
    )
    current_name = app.state.get_current_session()

    if current_name is not None:
        conv_id = _active_conversation_id(app)

        def apply_end(current: SessionMetadata) -> None:
            current.summary = summary
            current.status = SessionStatus.ARCHIVED
            # Link the conversation being archived so the metadata
            # remembers which conversation this session was inside.
            # 마감되는 시점의 conversation 을 연결해 메타데이터가 마지막으로
            # 어느 conversation 안에 있었는지 기억한다.
            if conv_id is not None:
                current.link_conversation(conv_id)
            current.touch()

        # F15 lock — see session_switch.
        # F15 잠금 — session_switch 참조.
        app.session_store.mutate_session_by_name(current_name, apply_end)

    _set_current_session(app, None)
    return _log_tool_return("session_end", event_id, app, {"ended": current_name})


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
def reject_switch(rejected_target: str, prompt_gist: str, ctx: Context) -> dict:
    """
    Record that the user rejected a switch proposal and stayed here.

    Called by the LLM when the user picks "keep current session" in the
    router's confirm flow. Appends a precedent to the current session so
    the judge stops repeating the same proposal (R3-C1).

    사용자가 전환 제안을 거부하고 현재 세션에 머물렀음을 기록한다.

    라우터 confirm 흐름에서 사용자가 "현재 세션 유지"를 선택하면 LLM 이
    호출한다. 현재 세션에 판례를 추가해 판정기가 같은 제안을 반복하지
    않게 한다 (R3-C1).
    """
    app = _get_app_ctx(ctx)
    event_id = _log_tool_call(
        "reject_switch",
        app,
        {
            "rejected_target": rejected_target,
            "prompt_gist": debug_log.mask_text(prompt_gist),
        },
    )
    current_name = app.state.get_current_session()
    if current_name is None:
        # Harmless no-op: without a current session there is no kept_in
        # side to attach the precedent to.
        # 무해한 no-op — 현재 세션이 없으면 판례를 붙일 kept_in 쪽이 없다.
        return _log_tool_return(
            "reject_switch",
            event_id,
            app,
            {"recorded": False, "reason": "no_current_session"},
        )

    # Calibration label (R3-C4): the explicit "keep" choice is the
    # reject side of the confirm flow.
    # 보정 라벨 (R3-C4) — 명시적 "유지" 선택은 confirm 흐름의 거부 쪽이다.
    decision_log.append_label(
        app.project_path,
        rejected_target,
        decision_log.LABEL_REJECT,
        source="reject_switch",
        kept_in=current_name,
    )

    record = PrecedentRecord.new(
        prompt_gist=prompt_gist,
        kept_in=current_name,
        rejected=rejected_target,
    )

    def apply_reject(current: SessionMetadata) -> None:
        current.precedents.append(record)
        current.touch()

    # F15 lock — the wrapper's summarizer may be saving this same
    # session concurrently (see session_switch).
    # F15 잠금 — 래퍼의 요약기가 같은 세션을 동시에 저장하고 있을 수
    # 있다 (session_switch 참조).
    saved = app.session_store.mutate_session_by_name(current_name, apply_reject)
    if saved is None:
        return _log_tool_return(
            "reject_switch",
            event_id,
            app,
            {"recorded": False, "reason": "session_not_found"},
        )

    # R3-C2: a rejection means the current session just absorbed a topic
    # its summary doesn't cover yet — refresh it now (EXTRA_FROM_REJECT
    # keeps the rooting check off this too-early refresh), and queue a
    # rooting check that rides the *next* refresh. The queue is
    # file-based, so the wrapper's worker picks both up on its poll.
    # R3-C2 — 거부는 현재 세션에 summary 가 아직 모르는 주제가 방금
    # 들어왔다는 뜻이다. 즉시 갱신을 적재하고 (EXTRA_FROM_REJECT 로 이
    # 너무 이른 갱신에는 정착 확인이 편승하지 못하게 표시), *다음* 갱신에
    # 편승할 정착 확인을 함께 적재한다. 큐는 파일 기반이라 래퍼 워커가
    # 폴링으로 둘 다 집어간다.
    conv_id = _active_conversation_id(app)
    if conv_id is not None:
        summarizer.enqueue(
            app.project_path,
            summarizer.SummaryTask(
                session_name=current_name,
                conversation_id=conv_id,
                kind=summarizer.KIND_ACTIVE,
                extra={summarizer.EXTRA_FROM_REJECT: True},
            ),
        )
        summarizer.enqueue(
            app.project_path,
            summarizer.SummaryTask(
                session_name=current_name,
                conversation_id=conv_id,
                kind=summarizer.KIND_ROOTING_CHECK,
                extra={summarizer.EXTRA_REJECTED_TOPIC: prompt_gist},
            ),
        )
    return _log_tool_return(
        "reject_switch",
        event_id,
        app,
        {
            "recorded": True,
            "kept_in": current_name,
            "rejected": rejected_target,
            "refresh_enqueued": conv_id is not None,
        },
    )


def _acceptance_rate(pairs: list[tuple[float, bool]]) -> float | None:
    """Acceptance rate of calibration pairs; None without samples.

    보정 쌍의 수용률. 표본이 없으면 None.
    """
    if not pairs:
        return None
    return sum(1 for _, accepted in pairs if accepted) / len(pairs)


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
def get_routing_status(ctx: Context) -> dict:
    """
    Report the routing mode and acceptance statistics (R3-C5).

    라우팅 모드와 수용률 통계를 보고한다 (R3-C5). /router 스킬이 사용자
    에게 보여줄 값들이다: 현재 모드, 전체 수용/거부/미라벨 집계, 전체·
    최근 추세 수용률, 그리고 auto 권장 여부 (보정 임계 산출 가능 여부).
    """
    app = _get_app_ctx(ctx)
    event_id = _log_tool_call("get_routing_status", app, {})
    root = app.project_path / _SESSION_MANAGER_DIRNAME
    pairs = decision_log.labeled_pairs(decision_log.load_events(app.project_path))
    # Recent trend without a fixed-N window constant (rule 8): the
    # chronologically most recent HALF of the labeled samples — a
    # proportional window that grows with the data.
    # 고정 N 창 상수 없는 최근 추세 (규칙 8) — 라벨 표본의 시간순 **최근
    # 절반**. 데이터와 함께 커지는 비례 창이다.
    recent_pairs = pairs[len(pairs) // 2 :]
    threshold = _calibrated_auto_threshold(root)
    return _log_tool_return(
        "get_routing_status",
        event_id,
        app,
        {
            "mode": _load_routing_mode(root),
            "acceptance": decision_log.acceptance_stats(app.project_path),
            "overall_acceptance_rate": _acceptance_rate(pairs),
            "recent_acceptance_rate": _acceptance_rate(recent_pairs),
            # auto is "recommended" exactly when the calibration gate can
            # produce a threshold — i.e. auto would actually engage.
            # auto 권장 = 보정 게이트가 임계를 산출할 수 있을 때 — 즉
            # auto 가 실제로 발동 가능한 상태일 때다.
            "auto_available": threshold is not None,
            "auto_threshold": threshold,
            "auto_error_tolerance": _load_auto_error_tolerance(root),
        },
    )


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
def set_routing_mode(mode: str, ctx: Context) -> dict:
    """
    Switch the routing mode (auto / confirm / off) in config.json.

    config.json 의 라우팅 모드를 변경한다 (auto / confirm / off). hook 이
    매 실행 시 config 를 읽으므로 즉시 반영된다. /router 스킬이 사용자
    선택에 따라 호출한다.
    """
    app = _get_app_ctx(ctx)
    event_id = _log_tool_call("set_routing_mode", app, {"mode": mode})
    if mode not in ROUTING_MODES:
        return _log_tool_return(
            "set_routing_mode",
            event_id,
            app,
            {"ok": False, "error": "invalid_mode", "valid": list(ROUTING_MODES)},
        )
    path = app.project_path / _SESSION_MANAGER_DIRNAME / _CONFIG_FILENAME
    # Raw read-modify-write preserving foreign keys — the Config model
    # requires socket_path, which this tool must not invent or drop.
    # 타 키를 보존하는 raw read-modify-write — Config 모델은 socket_path
    # 를 요구하는데 이 도구가 그것을 지어내거나 잃어버리면 안 된다.
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data["routing_mode"] = mode
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, _dump_json(data))
    return _log_tool_return(
        "set_routing_mode", event_id, app, {"ok": True, "mode": mode}
    )


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
def update_static(
    ctx: Context,
    project_context: str | None = None,
    conventions: str | None = None,
    project_map: dict[str, str] | None = None,
    variables: dict | None = None,
    source: str = "auto",
) -> dict:
    """
    Partially update the project-wide shared static field.

    프로젝트 전역 공유 정보(환경, 컨벤션, 변수 등)를 부분 갱신한다.
    project_map·variables 는 전달된 키만 갱신·추가하는 **키별 병합**이다
    — 나머지 키는 유지된다 (R5-C4). 항목마다 출처(source)·시각·직전 값이
    보존된다. source 는 "auto"(LLM 자체 판단, 기본) 또는 "user"(사용자의
    명시 지시). 어떤 세션에서든 갱신하면 다른 세션에서 최신 값을 읽을 수
    있다. 반환의 changed 는 실제로 값이 바뀐 항목 목록이다 — 자동 갱신
    확인 한 줄("static 갱신: ...")의 재료.
    """
    app = _get_app_ctx(ctx)
    event_id = _log_tool_call(
        "update_static",
        app,
        {
            "project_context": debug_log.mask_text(project_context),
            "conventions": debug_log.mask_text(conventions),
            "project_map": project_map,
            # variables can hold secrets — log only key names + value lengths.
            # variables 는 비밀이 들어올 수 있으므로 key 이름과 값 길이만 기록.
            "variables": debug_log.mask_dict_keys_only(variables),
            "source": source,
        },
    )
    if source not in VALID_SOURCES:
        return _log_tool_return(
            "update_static",
            event_id,
            app,
            {"updated": False, "reason": "invalid_source", "valid": VALID_SOURCES},
        )

    changed: list[str] = []

    def apply(static) -> bool:
        if project_context is not None:
            changed.extend(static.set_project_context(project_context, source))
        if conventions is not None:
            changed.extend(static.set_conventions(conventions, source))
        if project_map is not None:
            changed.extend(static.merge_project_map(project_map, source))
        if variables is not None:
            changed.extend(static.merge_variables(variables, source))
        if not changed:
            # Nothing changed — skip the write so updated_at keeps
            # meaning "content changed" (R5-C4 verification).
            # 바뀐 것 없음 — 쓰기를 생략해 updated_at 이 "내용이 바뀜"
            # 을 계속 의미하게 한다 (R5-C4 검증).
            return False
        static.touch()
        return True

    try:
        # Cross-process lock: two sessions updating concurrently must not
        # lose one side's keys (F15 twin — see FieldStore.mutate_static).
        # 프로세스 간 잠금 — 두 세션의 동시 갱신이 한쪽 키를 잃으면 안
        # 된다 (F15 쌍둥이, FieldStore.mutate_static 참조).
        static = app.field_store.mutate_static(apply)
    except UnsupportedStaticSchemaError as exc:
        # A newer/corrupt schema marker: leave the file untouched rather
        # than destroy its provenance (fields module docstring).
        # 더 새롭거나 오염된 schema 마커 — 출처를 파괴하는 대신 파일을
        # 건드리지 않는다 (fields 모듈 docstring).
        return _log_tool_return(
            "update_static",
            event_id,
            app,
            {
                "updated": False,
                "reason": "unsupported_schema",
                "schema": repr(exc.schema),
            },
        )
    return _log_tool_return(
        "update_static",
        event_id,
        app,
        {"updated_at": static.updated_at, "changed": changed},
    )


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
def init_project(content: str, ctx: Context) -> dict:
    """
    Create project-context.md if it does not exist yet.

    project-context.md가 아직 없을 때 새로 생성한다. 프로젝트 구조와
    의존성을 설명하는 문서로, 세션 전환 시 새 LLM이 맥락을 파악하는 데 쓰인다.
    이미 존재하면 덮어쓰지 않고 기존 내용을 그대로 반환한다.
    """
    app = _get_app_ctx(ctx)
    event_id = _log_tool_call(
        "init_project", app, {"content": debug_log.mask_text(content)}
    )
    if app.project_context_store.exists():
        return _log_tool_return(
            "init_project",
            event_id,
            app,
            {
                "created": False,
                "content": app.project_context_store.read(),
            },
        )
    app.project_context_store.write(content)
    return _log_tool_return("init_project", event_id, app, {"created": True})


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
def reinit_project(content: str, ctx: Context) -> dict:
    """
    Overwrite project-context.md with fresh content.

    project-context.md를 처음부터 다시 작성한다. 사용자가 명시적으로
    프로젝트 맥락 문서를 새로 쓰고 싶을 때 호출한다.
    """
    app = _get_app_ctx(ctx)
    event_id = _log_tool_call(
        "reinit_project", app, {"content": debug_log.mask_text(content)}
    )
    app.project_context_store.write(content)
    return _log_tool_return(
        "reinit_project", event_id, app, {"reinitialized": True}
    )


@mcp_server.tool(meta=_ALWAYS_LOAD_META)
def update_project_context(content: str, ctx: Context) -> dict:
    """
    Replace project-context.md with updated content.

    project-context.md를 새 내용으로 교체한다. 프로젝트 구조나 의존성이
    변경되었을 때 호출하여 문서를 최신 상태로 유지한다.
    """
    app = _get_app_ctx(ctx)
    event_id = _log_tool_call(
        "update_project_context", app, {"content": debug_log.mask_text(content)}
    )
    app.project_context_store.write(content)
    return _log_tool_return(
        "update_project_context", event_id, app, {"updated": True}
    )


def main() -> None:
    """
    Entry point invoked by Claude Code when spawning this MCP server.

    Claude Code가 이 MCP 서버를 spawn할 때 호출하는 진입점.
    """
    # Tag this process so log records distinguish MCP server events from
    # wrapper events. The run id is inherited from the wrapper via the
    # SESSION_MANAGER_RUN_ID env var, so all events land in one file.
    # 이 프로세스를 태깅 — 로그 레코드에서 MCP 서버 이벤트와 wrapper 이벤트
    # 가 구분된다. run id는 SESSION_MANAGER_RUN_ID 환경 변수를 통해 wrapper
    # 로부터 상속되므로 모든 이벤트가 한 파일에 모인다.
    debug_log.set_proc_label("mcp")

    mcp_server.run()


if __name__ == "__main__":
    main()
