"""
PTY wrapper that mediates between the user terminal and Claude Code.

Provides the I/O loop skeleton on which additional handlers hang
SWITCH/NEW logic, MCP socket integration, and stdin slash-command
observation. By itself it is a transparent passthrough: spawns Claude
Code on a PTY, forwards stdin to the PTY master, forwards PTY output to
stdout, and detects the input prompt so the rest of the wrapper can pick
a safe moment to inject text.

사용자 터미널과 Claude Code 프로세스 사이에 끼어들어 양방향 I/O를
중계하는 PTY 래퍼 모듈이다.

이 모듈은 I/O 루프의 골격만 제공한다. 세션 전환(SWITCH/NEW) 처리,
MCP 소켓 통합, stdin 슬래시 커맨드 관찰 같은 상위 로직은 별도의
핸들러로 이 골격 위에 얹어 확장한다.

단독으로 사용할 경우 투명한 패스스루로 동작한다. Claude Code를 PTY에
띄운 뒤, 사용자가 입력한 바이트는 PTY master로 그대로 전달하고, PTY가
출력하는 바이트는 stdout으로 흘려보낸다. 동시에 출력 스트림에서 입력
프롬프트(figures.pointer "❯" 직후의 반전 커서 시퀀스)를 감지해, 후속
로직이 텍스트를 안전하게 주입할 수 있는 시점을 파악할 수 있게 한다.
"""

from __future__ import annotations

import json
import os
import re
import select
import signal
import sys
import termios
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pexpect

from session_manager import debug_log, handoff_store, summarizer
from session_manager.claude_conversation import (
    encode_cwd,
    get_active_conversation_id,
    get_conversation_activity,
)
from session_manager.hooks.user_prompt_submit import (
    _count_active_sessions,
    _load_routing_mode,
)
from session_manager.lifecycle import get_cleanup_period_days
from session_manager.models.session import PrecedentRecord
from session_manager.routing import decision_log
from session_manager.storage.file_store import _SESSION_MANAGER_DIRNAME, SessionStore
from session_manager.summarizer import SummarizerWorker, SummaryTask
from session_manager.transcript_excerpt import EXCERPT_MAX_CHARS, scan_dialogue_growth
from session_manager.wrapper import context_monitor, wrapper_state
from session_manager.wrapper.command_matcher import (
    InterceptedCommand,
    match_back_command,
    match_intercept_command,
)
from session_manager.wrapper.judge_host import JudgeHost
from session_manager.wrapper.socket_server import WrapperSocketServer
from session_manager.wrapper.virtual_screen import VirtualScreen

# NOTE (R3-FIX1 redesign, docs/poc/R3-respawn.md): the wrapper no longer
# scans PTY output for renderer-internal patterns (pointer+inverse
# cursor) and never types into the TUI to execute transitions. Measured
# 2026-08-09: the current Claude Code renderer emits no such pattern in
# ANY situation (boot/idle/typing/hook-block - 144 chunks, 0 hits),
# which had silently killed every injection-driven feature. Transitions
# are now executed by swapping the child process (SIGTERM ->
# `claude --resume=<conv>` + trigger prompt), with context delivered via
# the pending-handoff file + UserPromptSubmit hook. Only official
# interfaces remain.
#
# 참고 (R3-FIX1 재설계, docs/poc/R3-respawn.md): 래퍼는 더 이상 PTY 출력의
# 렌더러 내부 패턴 (포인터+inverse 커서) 을 스캔하지 않고, 전환 실행을
# 위해 TUI 에 타이핑하지 않는다. 2026-08-09 실측 — 현 렌더러는 어떤
# 상황에서도 그 패턴을 출력하지 않으며 (부팅·유휴·타이핑·hook block,
# 144 chunk 0회), 주입 기반 기능 전부가 조용히 죽어 있었다. 전환은 자식
# 프로세스 교체 (SIGTERM → `claude --resume=<conv>` + 트리거 프롬프트) 로
# 실행하고, 컨텍스트는 pending handoff 파일 + UserPromptSubmit hook 으로
# 전달한다. 공식 인터페이스만 남는다.

# Confirmation prompts that ccode auto-accepts on every spawn.
#
# All three default to option 1 in Claude Code, so a single \r is enough.
# Patterns must be unique enough that they only match the prompt screen,
# not normal LLM output.
#
# ccode가 매 spawn 마다 자동 승인하는 confirmation prompt 텍스트.
# 셋 다 default가 1번이라 \r 한 번으로 OK. 일반 LLM 출력에는 잘 나오지
# 않을 만큼 고유한 문자열로 골랐다.
AUTO_CONFIRM_PATTERNS: tuple[str, ...] = (
    "Use this and all future MCP servers",  # MCP server 등록, 옵션 1
    "Use this MCP server",  # MCP server 등록, 옵션 2 (1번과 별도 매칭)
)

# AGENT_GUIDE.md sits in the package root (one level above wrapper/). The
# wrapper @-attaches this manual on session start so the LLM gets the full
# operational rules in conversation history without relying on initialize
# instructions (which have a 2KB cap).
# AGENT_GUIDE.md는 패키지 루트에 위치. wrapper가 세션 시작 시 @-attachment로
# 주입해, 2KB 제한이 있는 initialize instructions에 의존하지 않고 운영 매뉴얼
# 전체를 LLM 컨텍스트에 박는다.
AGENT_GUIDE_PATH = (Path(__file__).parent.parent / "AGENT_GUIDE.md").resolve()

# /clear observation: the conversation content is about
# to be wiped, so this is the last chance to summarise the active session.
# The command itself is forwarded untouched.
# /clear 관찰 — 대화 내용이 곧 지워지므로 활성 세션을 요약할 마지막
# 기회다. 명령 자체는 손대지 않고 그대로 통과시킨다.
CLEAR_COMMAND_RE = re.compile(r"^/clear(\s|$)")

# Ctrl+U — clears Claude Code's input line, slash-command popup included
# (measured 3/3, docs/poc/R3-back.md). Used to erase the typed /back
# before the wrapper handles it, since the \r is never forwarded.
# Ctrl+U — Claude Code 입력란을 지운다. 슬래시 명령 팝업이 열려 있어도
# 동작 (실측 3/3, docs/poc/R3-back.md). /back 은 \r 을 forward 하지
# 않으므로, 래퍼가 처리하기 전에 타이핑된 텍스트를 이것으로 지운다.
ERASE_INPUT_LINE = b"\x15"

# Busy marker Ink shows next to the input box while a turn is running
# (measured: present during generation, gone after — docs/poc/R3-back.md
# follow-up PoC). Searched only near the prompt row so conversation text
# QUOTING the phrase cannot false-positive; radius covers the input box
# + status line block. Two uses: (1) a transition waits for the turn to
# end before swapping the child, (2) the falling edge (busy → idle) is
# the turn-end event that triggers the periodic summary-refresh check.
# 턴 실행 중 Ink 가 입력란 곁에 표시하는 바쁨 마커 (실측: 생성 중 표시,
# 종료 후 소멸). 대화 본문의 **인용** 오탐을 막기 위해 입력란 주변 행만
# 검색한다. radius 는 입력 박스+상태 줄 블록을 덮는 값. 용도 둘:
# (1) 전환이 자식 교체 전 턴 종료를 기다리는 가드, (2) 하강 에지
# (busy → idle) 가 턴 종료 이벤트로서 주기 요약 갱신 검사를 발동한다.
BUSY_MARKER = "esc to interrupt"
BUSY_MARKER_RADIUS_ROWS = 4


def _safe_fileno(stream: Any) -> int:
    try:
        return stream.fileno()
    except (OSError, AttributeError, ValueError):
        return -1


def _debug_log(msg: str) -> None:
    """Forward legacy free-form debug strings to the unified debug_log.

    Original stub was a no-op; existing call sites carry useful free-form
    text (prompt-detect stage transitions, auto-confirm hits, chunk flow
    diagnostics). Forwarding them as ``DEBUG_TRACE`` events keeps that
    text searchable in the unified NDJSON without rewriting every call.

    기존 자유 형식 디버그 문자열을 통합 debug_log 로 전달.

    원래 stub 은 no-op 이었으나, 호출 사이트에는 유용한 자유 형식 텍스트가
    들어 있다 (prompt-detect stage 전환, auto-confirm 발동, chunk 흐름
    진단 등). 이를 ``DEBUG_TRACE`` 이벤트로 전달하면 호출 지점을 다 다시
    쓰지 않아도 통합 NDJSON 에서 텍스트 검색이 된다.
    """
    debug_log.log("DEBUG_TRACE", "WRAPPER", {"msg": msg})


@dataclass
class _PendingRespawn:
    """
    A transition waiting for the current turn to end, then a child swap.

    턴 종료를 기다렸다가 자식 교체로 실행되는 전환 1건.

    The whole former stage machine collapses into this: the handoff
    content already sits in the pending file (handoff_store), so the
    only remaining steps are "wait until not busy → SIGTERM → respawn
    with --resume=<conv> + trigger".

    이전의 단계 머신 전체가 이것으로 축약된다 — handoff 내용은 이미
    pending 파일 (handoff_store) 에 있으므로, 남은 절차는 "바쁨 해제
    대기 → SIGTERM → --resume=<conv> + 트리거로 respawn" 뿐이다.
    """

    # Session name being switched to (mirror/logging).
    # 전환 대상 세션 이름 (미러·로그용).
    target: str
    # Conversation to resume; None boots a fresh conversation (NEW).
    # 재개할 conversation. None 이면 새 conversation 부팅 (NEW).
    resume_conv: str | None
    # Departing session name (last-transition bookkeeping).
    # 떠나는 세션 이름 (last_transition 부기용).
    from_name: str | None = None
    # User prompt travelling with the transition (also in the pending
    # file) — kept here for last-transition bookkeeping so /back can
    # re-deliver it.
    # 전환과 함께 이동하는 사용자 프롬프트 (pending 파일에도 있음) —
    # /back 이 재전달할 수 있도록 last_transition 부기용으로 보관.
    user_prompt: str = ""
    # /back reverse switch — consumes the undo record on completion.
    # /back 역전환 — 완료 시 undo 기록을 소비한다.
    is_back: bool = False
    # SIGTERM already sent (idempotence across ticks).
    # SIGTERM 전송 여부 (틱 간 멱등).
    terminated: bool = False


class SessionManagerWrapper:
    def __init__(
        self,
        socket_path: str,
        claude_args: list[str],
        project_path: str | None = None,
    ) -> None:
        self.socket_path = socket_path
        self.claude_args = list(claude_args)
        self.project_path = project_path or os.getcwd()

        self.child: pexpect.spawn | None = None
        self.pty_fd: int = -1

        # 테스트 환경이나 stdin/stdout 이 redirect 된 경우 fileno() 가 실패할 수
        # 있으므로 안전하게 -1 로 폴백. 실런타임에서는 isatty/-1 검사로 가드.
        self._stdin_fd: int = _safe_fileno(sys.stdin)
        self._stdout_fd: int = _safe_fileno(sys.stdout)
        self._original_stdin_attrs: list[Any] | None = None
        self._previous_winch_handler: Any = None

        self.socket_server = WrapperSocketServer(
            socket_path=socket_path,
            on_message=self._handle_mcp_signal,
            on_hook_message=self._handle_hook_message,
        )

        # Virtual terminal screen mirroring Claude Code's PTY output. Used to
        # extract the live input prompt text (the line containing ❯) when
        # the user submits a slash command.
        # Claude Code의 PTY 출력을 미러링하는 가상 터미널 화면. 사용자가
        # 슬래시 명령을 submit한 시점의 입력란 텍스트(❯ 라인) 추출에 사용.
        self.virtual_screen = VirtualScreen()

        self._pending_respawn: _PendingRespawn | None = None

        # Busy state from the previous PTY chunk — the falling edge
        # (busy → idle) is the turn-end event (periodic summary check).
        # 직전 PTY chunk 의 바쁨 상태 — 하강 에지 (busy → idle) 가 턴 종료
        # 이벤트다 (주기 요약 검사).
        self._was_busy: bool = False

        # Confirmation patterns already auto-accepted in the current child.
        # Reset on each spawn so a respawned child re-arms the auto-accept.
        # 현재 자식에서 이미 자동 승인한 confirmation 패턴.
        # 새 자식이 spawn될 때마다 초기화해 자동 승인을 재무장.
        self._handled_confirmations: set[str] = set()

        # Auto-confirm matching window (F13). Confirmation dialogs only
        # appear during child boot, before the user has typed anything —
        # but the pattern strings can legitimately show up on screen later
        # (LLM output quoting them, a Read of a file containing them), and
        # matching then would inject a stray Enter. The window is therefore
        # armed at spawn and closed at the FIRST real user keystroke of
        # that child. Trade-off: a user typing before a boot dialog renders
        # disarms it and confirms manually — harmless, unlike a mis-fire.
        # 자동 승인 매칭 윈도우 (F13). confirmation 다이얼로그는 자식 부팅
        # 구간 — 사용자가 아무것도 입력하기 전 — 에만 나타난다. 반면 패턴
        # 문자열 자체는 이후에도 화면에 정상적으로 나타날 수 있고 (LLM 이
        # 인용, 해당 문자열이 든 파일 Read 표시), 그때 매칭되면 엉뚱한
        # Enter 가 주입된다. 따라서 윈도우는 spawn 시 열리고 그 자식의
        # **첫 실제 사용자 키 입력**에서 닫힌다. 트레이드오프: 다이얼로그
        # 표시 전에 타이핑하면 자동 승인이 꺼져 수동 확인하게 되는데,
        # 이는 오발사와 달리 무해하다.
        self._auto_confirm_armed: bool = True

        # Initial current_session_name handed back during the MCP handshake,
        # decided from CLI args:
        # - `--resume foo` → "foo"
        # - `--continue`   → None (Claude Code resolves internally)
        # - no args        → None (fresh session)
        # MCP가 핸드셰이크에서 받아갈 초기 current_session_name. CLI 인자에서
        # 결정한다.
        self._initial_session_name: str | None = self._parse_initial_session_name(
            self.claude_args
        )

        # Wrapper-side mirror of the current session name, used by triggers
        # that fire without MCP involvement (e.g. /clear observation).
        # Updated on SWITCH/NEW signals. May be None (unregistered fresh
        # start) — triggers then skip with a log; the boot-time recovery
        # and R2's hook architecture cover that gap.
        # 현재 세션 이름의 래퍼 측 미러 — MCP 를 거치지 않는 트리거 (/clear
        # 관찰 등) 가 사용한다. SWITCH/NEW 신호에서 갱신. 미등록 신규
        # 시작이면 None 일 수 있고, 그 경우 트리거는 로그만 남기고 skip —
        # 이 빈틈은 부팅 시 복구와 R2 hook 구조가 메운다.
        self._current_session_name: str | None = self._initial_session_name

        # Background summarizer worker (R1). Lives for the wrapper's whole
        # lifetime; drains the file queue that the triggers below fill.
        # 백그라운드 요약기 워커 (R1). 래퍼 전체 lifetime 동안 유지되며,
        # 아래 트리거들이 채우는 파일 큐를 비운다.
        self.summarizer_worker = SummarizerWorker(Path(self.project_path))

        # Resident routing-judge host (R2). Serves judge_request messages
        # arriving over the hook socket path; started lazily by
        # _maybe_start_judge when routing is actually possible.
        # 상주 라우팅 판정 호스트 (R2). hook 소켓 경로로 오는 judge_request
        # 를 처리한다. 라우팅이 실제로 가능할 때 _maybe_start_judge 가
        # 지연 시작한다.
        self.judge_host = JudgeHost(Path(self.project_path))

        # Incremental dialogue-length scan for the periodic refresh trigger.
        # Keeping the file offset means each turn parses only what was
        # appended since the last one.
        # 주기 갱신 트리거용 증분 대화 길이 스캔. 파일 offset 을 유지하므로 매
        # 턴은 직전 이후 append 된 부분만 파싱한다.
        self._dialogue_scan_conv_id: str | None = None
        self._dialogue_scan_offset: int = 0
        self._dialogue_scan_chars: int = 0

        # Conversation currently marked rollover-pending (R4-C1). None =
        # no mark; acting on the mark is R4-C3/C4.
        # 현재 롤오버 pending 으로 마킹된 conversation (R4-C1). None =
        # 마킹 없음. 마킹에 대한 행동은 R4-C3/C4.
        self._rollover_pending_conv_id: str | None = None

        # Most recent wrapper-executed transition, the target of /back
        # (R3-C3). Memory-first with state.json persistence so the undo
        # survives a wrapper restart. Consumed when the reverse switch
        # COMPLETES (R3-FIX1) — a stalled/crashed undo keeps the record
        # so the user can retry.
        # 가장 최근의 래퍼 실행 전환 — /back (R3-C3) 의 대상. 메모리 우선
        # + state.json 영속화로 래퍼 재시작을 견딘다. 소비는 역전환이
        # **완료**될 때 (R3-FIX1) — 정지·크래시 시 기록이 남아 재시도
        # 가능하다.
        self._last_transition: dict[str, Any] | None = (
            wrapper_state.load_last_transition(Path(self.project_path))
        )

    def start(self) -> None:
        """
        Spawn Claude Code on a PTY and run the I/O loop until it exits.

        Claude Code를 PTY에 띄우고 종료될 때까지 I/O 루프를 실행한다.
        NEW 액션으로 자식이 종료된 경우 새 자식을 spawn해 흐름을 이어간다.
        """
        # The socket and terminal state live for the wrapper's whole
        # lifetime — they outlast individual child processes when NEW
        # respawns Claude Code.
        # 소켓과 터미널 상태는 래퍼 전체 lifetime 동안 유지된다 — NEW로
        # Claude Code가 재시작되더라도 동일하게 살아있다.
        self.socket_server.start()
        self._enter_raw_mode()
        self._install_winch_handler()

        # Recover summaries lost to forced exits before the worker starts,
        # so the first queue pass already sees them.
        # 강제 종료로 누락된 요약을 워커 시작 전에 큐에 복구 — 첫 큐 pass
        # 가 바로 집어가도록.
        self._enqueue_stale_summaries()
        self.summarizer_worker.start()
        self._maybe_start_judge()

        # A pending-handoff file surviving to boot belongs to a transition
        # that never reached its trigger (crash) — dispose of it so a
        # future unrelated prompt cannot slurp a stale handoff.
        # 부팅까지 살아남은 pending handoff 파일은 트리거에 도달하지 못한
        # 전환의 잔재 (크래시) — 무관한 미래 프롬프트가 낡은 handoff 를
        # 소비하지 못하게 처분한다.
        handoff_store.clear_stale_pending(Path(self.project_path))

        try:
            self._spawn_child()
            self._sync_winsize()
            self._io_loop()
            while self._should_respawn():
                self._spawn_child()
                self._sync_winsize()
                self._io_loop()
        finally:
            self._restore_terminal()
            self.summarizer_worker.stop()
            self.judge_host.stop()
            self.socket_server.stop()

    @staticmethod
    def _strip_resume_args(args: list[str]) -> list[str]:
        """Drop resume/continue flags (and their values) from user args.

        사용자 인자에서 resume/continue 계열 플래그 (와 값) 를 제거한다.
        respawn 은 자체 `--resume=` 을 붙이므로 원 인자의 것과 충돌하면
        안 된다. 그 외 인자 (--model 등) 는 그대로 유지된다.
        """
        out: list[str] = []
        skip_value = False
        for arg in args:
            if skip_value:
                skip_value = False
                continue
            if arg in ("--continue", "-c"):
                continue
            if arg in ("--resume", "-r"):
                skip_value = True
                continue
            if arg.startswith("--resume=") or arg.startswith("-r="):
                continue
            out.append(arg)
        return out

    def _agent_guide_flag(self) -> list[str]:
        """AGENT_GUIDE delivery: an official CLI flag instead of typing.

        AGENT_GUIDE 전달 — 타이핑 주입 대신 공식 CLI 플래그.

        ``--append-system-prompt=`` puts the manual in the system prompt
        of every child (measured with interactive resume,
        docs/poc/R3-respawn.md). Per-process, so respawns re-carry it;
        never written into the transcript.

        ``--append-system-prompt=`` 이 매뉴얼을 모든 자식의 시스템
        프롬프트에 싣는다 (대화형 재개와의 조합 실측,
        docs/poc/R3-respawn.md). 프로세스 단위라 respawn 마다 자동
        유지되고 transcript 에는 기록되지 않는다.
        """
        try:
            guide = AGENT_GUIDE_PATH.read_text(encoding="utf-8")
        except OSError:
            return []
        return [f"--append-system-prompt={guide}"]

    @staticmethod
    def _mcp_config_flag() -> list[str]:
        """MCP server delivery: CLI injection instead of user-scope registration.

        MCP 서버 전달 — user 스코프 등록 대신 CLI 인자 직접 주입.

        ``--mcp-config=`` with inline JSON loads session-manager only in
        the children ccode spawns, so a bare ``claude`` never starts the
        server and cannot pollute unrelated folders (F4). Measured
        (docs/poc/R3-mcp-config.md): works in headless, TUI and the
        respawn combination (``--resume=`` + trigger prompt); no trust
        prompt fires; on a name collision with a leftover user-scope
        entry the injected config wins. ``--strict-mcp-config`` is
        deliberately absent — the user's other MCP servers keep loading.
        The server command is ``sys.executable`` so it runs from the same
        venv as ccode itself (no hard-coded project path).

        인라인 JSON 의 ``--mcp-config=`` 는 ccode 가 spawn 한 자식에만
        session-manager 를 로드시키므로, 맨몸 ``claude`` 는 서버를 띄우지
        않아 무관 폴더를 오염시킬 수 없다 (F4). 실측
        (docs/poc/R3-mcp-config.md): headless·TUI·respawn 조합
        (``--resume=`` + 트리거 프롬프트) 전부 동작, 신뢰 프롬프트
        미발생, 잔존 user 스코프 동명 등록과 충돌 시 주입 측 승리.
        ``--strict-mcp-config`` 는 의도적으로 제외 — 사용자의 타 MCP
        서버 로드를 유지한다. 서버 커맨드는 ``sys.executable`` 로 ccode
        와 같은 venv 에서 실행된다 (프로젝트 경로 하드코딩 없음).
        """
        config = json.dumps(
            {
                "mcpServers": {
                    "session-manager": {
                        "type": "stdio",
                        "command": sys.executable,
                        "args": ["-m", "session_manager.server"],
                        "env": {},
                    }
                }
            }
        )
        return [f"--mcp-config={config}"]

    def _build_child_args(self) -> list[str]:
        """Assemble argv for the next child from the pending transition.

        pending 전환으로부터 다음 자식의 argv 를 조립한다.

        Flags use the ``--option=value`` form throughout: the space form
        is greedy and would swallow the trailing trigger prompt
        (measured, docs/poc/R3-respawn.md). The trigger prompt is
        content-free — the actual handoff travels in the pending file
        (argv is world-readable via ``ps``).

        플래그는 전부 ``--옵션=값`` 형식 — 공백 형식은 탐욕적이라 뒤의
        트리거 프롬프트를 삼킨다 (실측, docs/poc/R3-respawn.md). 트리거는
        무내용이며 실제 handoff 는 pending 파일로 이동한다 (argv 는
        ``ps`` 로 노출되므로).
        """
        pending = self._pending_respawn
        if pending is None:
            return (
                list(self.claude_args)
                + self._agent_guide_flag()
                + self._mcp_config_flag()
            )
        args = self._strip_resume_args(self.claude_args)
        args += self._agent_guide_flag()
        args += self._mcp_config_flag()
        if pending.resume_conv is not None:
            args.append(f"--resume={pending.resume_conv}")
        args.append(handoff_store.TRIGGER_PROMPT)
        return args

    def _spawn_child(self) -> None:
        pending = self._pending_respawn
        child_args = self._build_child_args()

        # When ccode itself runs inside a Claude Code session, the spawned
        # claude inherits CLAUDE_CODE_CHILD_SESSION and — in interactive
        # mode only — then writes NO transcript JSONL at all, which silently
        # disables everything built on transcripts (excerpts, summaries,
        # boot recovery, activity-based cleanup). Isolated by experiment
        # (2026-08-02): this single variable suppresses the transcript;
        # CLAUDECODE, CLAUDE_CODE_SESSION_ID, SSE_PORT, ENTRYPOINT,
        # EXECPATH, PID, EFFORT are each innocent, and headless `-p` calls
        # are unaffected even with full inheritance. Remove exactly the
        # proven variable, nothing more.
        # ccode 자체가 Claude Code 세션 안에서 실행되면 spawn 된 claude 가
        # CLAUDE_CODE_CHILD_SESSION 을 상속하고 — 대화형 모드에 한해 —
        # transcript JSONL 을 전혀 쓰지 않는다. transcript 위에 세운 모든
        # 기능 (발췌·요약·부팅 복구·활동 기반 정리) 이 조용히 꺼진다.
        # 실험으로 분리 (2026-08-02): 이 변수 하나가 원인이며 CLAUDECODE,
        # CLAUDE_CODE_SESSION_ID, SSE_PORT, ENTRYPOINT, EXECPATH, PID,
        # EFFORT 는 각각 무해, headless `-p` 는 전체 상속에도 무관.
        # 입증된 변수 하나만 제거한다.
        child_env = {
            k: v
            for k, v in os.environ.items()
            if k != "CLAUDE_CODE_CHILD_SESSION"
        }
        self.child = pexpect.spawn(
            "claude",
            child_args,
            encoding=None,
            echo=False,
            env=child_env,
        )
        self.pty_fd = self.child.fileno()
        debug_log.log(
            "SPAWN",
            "SYSTEM",
            {
                "claude_args": debug_log.mask_text(" ".join(child_args)),
                "pid": self.child.pid,
                "respawn_target": pending.target if pending else None,
                "respawn_resume_conv": pending.resume_conv if pending else None,
                "initial_session_name": self._initial_session_name,
            },
        )
        self._reset_child_detection_state()

        # Transition bookkeeping happens at spawn — the moment the swap
        # actually succeeded. A completed transition becomes the /back
        # target; a /back reverse switch instead consumes its undo record
        # here (crash-before-spawn keeps the record for retry).
        # 전환 부기는 교체가 실제로 성공한 spawn 시점에 수행한다. 완료된
        # 전환은 /back 대상이 되고, /back 역전환은 여기서 undo 기록을
        # 소비한다 (spawn 전 크래시면 기록이 남아 재시도 가능).
        if pending is not None:
            if pending.is_back:
                self._last_transition = None
                wrapper_state.clear_last_transition(Path(self.project_path))
            else:
                self._record_last_transition(
                    pending.from_name, pending.target, pending.user_prompt
                )
        self._pending_respawn = None

    def _should_respawn(self) -> bool:
        """
        Decide whether to spawn another child after the current one exits:
        only when a transition requested the swap. A child that exits on
        its own (user /exit, crash) ends the wrapper as before.

        현재 자식 종료 후 새 자식을 spawn 할지 결정한다 — 전환이 교체를
        요청한 경우에만. 자식이 스스로 종료한 경우 (사용자 /exit·크래시)
        는 기존처럼 래퍼도 끝난다.
        """
        # CHILD_EXIT checkpoint — record the previous child's exit state
        # so respawn decisions are auditable.
        # CHILD_EXIT 체크포인트 — 직전 자식의 종료 상태를 기록해 respawn
        # 결정을 사후 추적 가능하게 한다.
        pending = self._pending_respawn
        debug_log.log(
            "CHILD_EXIT",
            "SYSTEM",
            {
                "exit_status": getattr(self.child, "exitstatus", None)
                if self.child is not None
                else None,
                "signal_status": getattr(self.child, "signalstatus", None)
                if self.child is not None
                else None,
                "pending_target": pending.target if pending else None,
            },
        )
        return pending is not None

    # ------------------------------------------------------------------ I/O loop
    # I/O 루프 ------------------------------------------------------------------

    def _io_loop(self) -> None:
        assert self.child is not None
        while self.child.isalive():
            # Build the watch list each tick: socket fds appear/disappear
            # as MCP connects and disconnects.
            # 매 틱마다 watch 대상을 새로 구성. 소켓 fd는 MCP의 연결·해제에
            # 따라 등장하거나 사라진다.
            watch_fds: list[int] = [self.pty_fd, self._stdin_fd]
            listen_fd = self.socket_server.listen_fileno
            client_fd = self.socket_server.client_fileno
            pending_fds = self.socket_server.pending_filenos
            if listen_fd >= 0:
                watch_fds.append(listen_fd)
            if client_fd >= 0:
                watch_fds.append(client_fd)
            watch_fds.extend(pending_fds)

            try:
                # 100 ms timeout polls child liveness without burning CPU.
                # 100ms 타임아웃으로 자식 생존 여부를 폴링 (CPU 낭비 방지).
                readable, _, _ = select.select(watch_fds, [], [], 0.1)
            except InterruptedError:
                # A signal (e.g. SIGWINCH) interrupted select; just retry.
                # 시그널(예: SIGWINCH)로 select가 중단된 경우 단순 재시도.
                continue
            except OSError:
                break

            if self.pty_fd in readable:
                if not self._handle_pty_readable():
                    break

            if self._stdin_fd in readable:
                self._handle_stdin_readable()

            if listen_fd >= 0 and listen_fd in readable:
                self.socket_server.handle_listen_readable()

            if client_fd >= 0 and client_fd in readable:
                self.socket_server.handle_client_readable()

            for fd in pending_fds:
                if fd in readable:
                    self.socket_server.handle_pending_readable(fd)

            # Transition tick: when a swap is pending and the turn has
            # ended, terminate the child (respawn happens in start()).
            # 전환 틱 — 교체 대기 중이고 턴이 끝났으면 자식을 종료한다
            # (respawn 은 start() 가 수행).
            self._maybe_terminate_for_respawn()

        self._drain_pty()

    def _maybe_terminate_for_respawn(self) -> None:
        """Send SIGTERM once the pending transition's turn-end gate opens.

        pending 전환의 턴 종료 가드가 열리면 SIGTERM 을 1회 보낸다.

        The busy marker holds the swap while Claude is generating (e.g.
        /back typed mid-response, or an MCP switch signalled mid-turn) so
        the in-flight reply lands in the transcript first. SIGTERM
        between turns is measured safe (transcript intact, resumable —
        docs/poc/R3-respawn.md).

        바쁨 마커가 생성 중 교체를 보류한다 (응답 중 /back, 턴 중 MCP
        전환 신호) — 진행 중 응답이 transcript 에 먼저 남는다. 턴 사이
        SIGTERM 은 실측으로 안전하다 (무결·재재개, docs/poc/R3-respawn.md).
        """
        pending = self._pending_respawn
        if pending is None or pending.terminated:
            return
        if self.virtual_screen.contains_near_prompt(
            BUSY_MARKER, BUSY_MARKER_RADIUS_ROWS
        ):
            return
        pending.terminated = True
        debug_log.log(
            "TRANSITION",
            "WRAPPER",
            {"op": "terminate_child", "target": pending.target},
            session=pending.target,
        )
        try:
            if self.child is not None:
                os.kill(self.child.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    def _handle_pty_readable(self) -> bool:
        try:
            chunk = os.read(self.pty_fd, 4096)
        except OSError:
            return False
        if not chunk:
            # EOF on PTY master means the child closed its end.
            # PTY master에서의 EOF — 자식 프로세스가 자기 쪽을 닫음.
            return False

        # Mirror every chunk into the virtual screen so prompt-line
        # extraction and the busy marker are always current — the screen
        # is the wrapper's only view of Claude Code's state (observation
        # only; the wrapper never scans for renderer patterns).
        # 모든 chunk 를 가상 화면에 반영 — 입력란 추출과 바쁨 마커가 항상
        # 최신이 되게 한다. 화면은 래퍼가 Claude Code 상태를 보는 유일한
        # 창이다 (관찰 전용 — 렌더러 패턴 스캔은 하지 않는다).
        self.virtual_screen.feed(chunk)

        # Auto-accept any confirmation prompts that just appeared in the
        # virtual screen (MCP server registration).
        # Each pattern is processed at most once per child.
        # 가상 화면에 새로 등장한 confirmation prompt 자동 승인 (MCP server
        # 등록). 자식별로 패턴당 최대 1회 처리.
        self._auto_accept_confirmations()

        # Turn-end event: the busy marker's falling edge. The one safe
        # point to measure dialogue growth for the periodic summary
        # refresh (previously hung off the dead prompt-detect signal).
        # 턴 종료 이벤트 — 바쁨 마커의 하강 에지. 주기 요약 갱신의 대화
        # 증가량을 재기에 안전한 유일한 지점 (이전에는 죽은 프롬프트 감지
        # 신호에 걸려 있었다).
        busy = self.virtual_screen.contains_near_prompt(
            BUSY_MARKER, BUSY_MARKER_RADIUS_ROWS
        )
        if self._was_busy and not busy:
            self._check_summary_refresh()
            self._check_context_usage()
        self._was_busy = busy

        os.write(self._stdout_fd, chunk)
        return True

    def _reset_child_detection_state(self) -> None:
        """
        Reset per-child observation state on every spawn: the busy edge
        tracker restarts, and the confirmation auto-accept re-arms (its
        matching window re-opens until this child's first user keystroke
        — F13).

        spawn 마다 자식별 관찰 상태를 초기화한다: 바쁨 에지 추적을
        재시작하고, confirmation 자동 승인을 재무장한다 (매칭 윈도우는
        이 자식의 첫 사용자 키 입력까지 다시 열린다 — F13).
        """
        self._was_busy = False
        self._handled_confirmations = set()
        self._auto_confirm_armed = True

    def _handle_stdin_readable(self) -> None:
        try:
            chunk = os.read(self._stdin_fd, 4096)
        except OSError:
            return
        if not chunk:
            return

        # First real user keystroke closes the auto-confirm window (F13):
        # boot dialogs are behind us, so any later pattern match on screen
        # would be a quote/file view, not a dialog.
        # 첫 실제 사용자 키 입력이 자동 승인 윈도우를 닫는다 (F13). 부팅
        # 다이얼로그 구간은 지났으므로 이후의 패턴 매칭은 다이얼로그가
        # 아니라 인용·파일 표시다.
        if self._auto_confirm_armed:
            self._auto_confirm_armed = False
            debug_log.log(
                "AUTO_CONFIRM",
                "WRAPPER",
                {"window": "closed", "reason": "first_user_keystroke"},
            )

        # USER_KEY checkpoint — every real user keystroke arrives here.
        # The mask helper redacts content by default; users can opt in to
        # raw logging via SESSION_MANAGER_LOG_RAW_STDIN=1.
        # USER_KEY 체크포인트 — 실제 사용자 키스트로크는 모두 이 지점으로 진입.
        # 기본은 마스킹, SESSION_MANAGER_LOG_RAW_STDIN=1 로 raw opt-in 가능.
        debug_log.log(
            "USER_KEY",
            "USER",
            {
                "chunk": debug_log.mask_stdin_chunk(chunk),
            },
        )

        # Submit detection: Ink's parseKeypress only treats a lone \r as
        # Return (s === '\r'). Multi-byte chunks are typed text, not submit.
        # submit 감지 — Ink parseKeypress는 단독 \r만 Return으로 인정
        # (s === '\r'). 멀티 바이트 chunk는 타이핑 중인 텍스트로 간주.
        if chunk == b"\r":
            prompt_text = self.virtual_screen.get_prompt_line()
            debug_log.log(
                "VSCREEN",
                "SYSTEM",
                {"phase": "submit_detect", "prompt_text": prompt_text},
            )
            matched = match_intercept_command(prompt_text)
            if matched is not None:
                self._observe_session_command(matched)
            elif match_back_command(prompt_text):
                # Wrapper-native command: never forward — Claude Code has
                # no /back and the \r would only submit an unknown command.
                # 래퍼 자체 명령 — forward 금지. Claude Code 에 /back 은
                # 없으므로 \r 은 unknown command 제출만 만든다.
                self._handle_back_command()
                return
            elif prompt_text and CLEAR_COMMAND_RE.match(prompt_text.strip()):
                # /clear wipes the conversation — summarise it while it's
                # still there.
                # /clear 는 대화를 지운다 — 아직 남아 있을 때 요약한다.
                self._enqueue_active_summary()

        # Forward keystrokes to the PTY so Ink can render them in real time.
        # Ink가 실시간으로 렌더링할 수 있도록 키 입력을 PTY로 즉시 전달.
        os.write(self.pty_fd, chunk)

    # --------------------------------------------------- Slash observation
    # 슬래시 명령 관찰 ----------------------------------------------------------

    def _observe_session_command(self, matched: InterceptedCommand) -> None:
        """Record a session-changing slash command and let it through.

        세션을 바꾸는 슬래시 명령을 기록만 하고 그대로 통과시킨다.

        These commands (``/resume``, ``/exit``, ``/rename``, ``/new``) leave
        the current conversation, so the summary must be refreshed — but
        the wrapper no longer holds the keystroke to arrange that. It used
        to: the \r was withheld while the in-session LLM was asked to write
        a summary, freezing the user's input line for up to 15 seconds and
        failing outright whenever the LLM was busy or the reply never came.
        The background summariser removes that dependency entirely, so the
        command runs at once and the summary is produced afterwards from
        the transcript.

        이 명령들 (``/resume``, ``/exit``, ``/rename``, ``/new``) 은 현재
        conversation 을 떠나므로 summary 갱신이 필요하다 — 하지만 래퍼는 더
        이상 그것을 위해 키 입력을 붙잡지 않는다. 예전에는 세션 안의 LLM 에게
        요약을 부탁하는 동안 \r 을 보관해 사용자 입력란이 최대 15초간 얼었고,
        LLM 이 바쁘거나 응답이 오지 않으면 그대로 실패했다. 백그라운드 요약기가
        그 의존을 없앴으므로, 명령은 즉시 실행되고 요약은 그 뒤에 transcript
        로부터 만들어진다.

        A missed observation (the virtual screen not yet updated, say) is
        harmless: the transcript is still on disk, and the boot-time
        recovery pass picks the session up on the next start.

        관찰을 놓쳐도 (가상 화면이 아직 갱신되지 않은 경우 등) 무해하다 —
        transcript 는 디스크에 남아 있고, 다음 시작 시 부팅 복구가 집어간다.
        """
        debug_log.log(
            "SESSION_COMMAND_OBSERVED",
            "USER",
            {"command": matched.command, "args": matched.args},
            session=self._current_session_name,
        )
        self._enqueue_active_summary()
        # Precedent invalidation on a manual move (R3-FIX2): the user
        # resuming a session BY HAND overturns any recorded rejection of
        # it — the deterministic suppression gate would otherwise have
        # no acceptance path and could suppress that target forever.
        # 수동 이동 시 판례 무효화 (R3-FIX2) — 사용자가 그 세션으로 손수
        # 이동하면 기록된 거부가 뒤집힌다. 이 소멸 경로가 없으면 결정적
        # 억제 게이트에는 수락 경로가 없어 그 대상이 영원히 억제될 수 있다.
        if matched.command == "resume" and matched.args:
            self._drop_precedents_on_manual_move(matched.args)
        # Tell the MCP server its current-session pointer is now stale: the
        # user is moving to another conversation by hand.
        # MCP 서버에 현재 세션 포인터가 낡았음을 알린다 — 사용자가 손수 다른
        # conversation 으로 이동하는 중이다.
        self.socket_server.send(
            {
                "action": "session_command",
                "command": matched.command,
                "args": matched.args,
            }
        )

    def _drop_precedents_on_manual_move(self, resume_arg: str) -> None:
        """Drop the current session's precedents against a manually
        resumed session.

        수동 resume 대상 세션에 대한 현재 세션의 판례를 소멸시킨다.

        The /resume argument may be a conversation id or a session name/
        title — match either against the metadata. A bare /resume opens
        the picker (destination unknown) and never reaches here.

        /resume 인자는 conversation id 또는 세션 이름·제목일 수 있다 —
        메타데이터에서 양쪽 다 매칭한다. 인자 없는 /resume 은 picker 라
        목적지를 알 수 없고 이 지점에 오지 않는다.
        """
        kept_in = self._current_session_name
        if kept_in is None:
            return
        arg = resume_arg.strip()
        try:
            sessions = SessionStore(Path(self.project_path)).list_sessions()
        except Exception:
            return
        target_name: str | None = None
        for session in sessions:
            if arg == session.name or arg in session.claude_conversation_ids:
                target_name = session.name
                break
        if target_name is None or target_name == kept_in:
            return

        def apply(session: Any) -> None:
            session.drop_precedents_for(target_name)

        try:
            SessionStore(Path(self.project_path)).mutate_session_by_name(
                kept_in, apply
            )
        except Exception as exc:
            debug_log.log(
                "PRECEDENT",
                "WRAPPER",
                {"op": "manual_move_drop", "result": "error", "error": str(exc)},
            )

    # ------------------------------------------------------------- /back undo
    # /back 되돌리기 (R3-C3) -----------------------------------------------------

    def _notify_user(self, text: str) -> None:
        """Print one wrapper status line to the user's terminal.

        래퍼 상태 한 줄을 사용자 터미널에 출력한다. Ink 의 다음 redraw 가
        덮을 수 있는 일시적 표시로 충분하다.
        """
        if self._stdout_fd < 0:
            return
        try:
            os.write(self._stdout_fd, f"\r\n[session-manager] {text}\r\n".encode())
        except OSError:
            pass

    def _handle_back_command(self) -> None:
        """Undo the most recent wrapper-executed transition.

        가장 최근의 래퍼 실행 전환을 되돌린다.

        The typed ``/back`` is erased with Ctrl+U (measured,
        docs/poc/R3-back.md) since its \\r is never forwarded. The
        rejection is recorded (precedent + calibration label), then the
        transition runs in reverse through the ordinary respawn path —
        the misrouted prompt travels in the pending-handoff file. The
        undo record is consumed at respawn (completion), so a crash
        beforehand keeps it for retry.

        타이핑된 ``/back`` 은 \\r 을 forward 하지 않으므로 Ctrl+U 로
        지운다 (실측, docs/poc/R3-back.md). 거부를 기록 (판례 + 보정
        라벨) 한 뒤 일반 respawn 경로로 역방향 전환한다 — 잘못 이동했던
        프롬프트는 pending handoff 파일로 이동한다. undo 기록 소비는
        respawn (완료) 시점이라 그 전 크래시면 재시도 가능하다.
        """
        try:
            os.write(self.pty_fd, ERASE_INPUT_LINE)
        except OSError:
            pass
        if self._pending_respawn is not None:
            self._notify_user("세션 전환이 진행 중입니다 — /back 은 무시됩니다")
            return
        record = self._last_transition or wrapper_state.load_last_transition(
            Path(self.project_path)
        )
        if record is None:
            self._notify_user("되돌릴 전환이 없습니다")
            debug_log.log(
                "BACK", "USER", {"result": "no_last_transition"},
                session=self._current_session_name,
            )
            return
        origin, wrong = record["from"], record["to"]
        resume_conv = self._resolve_resume_conv(origin)
        if resume_conv is None:
            self._notify_user(
                f"{origin} 세션의 대화 기록을 찾지 못해 되돌릴 수 없습니다"
            )
            debug_log.log(
                "BACK",
                "USER",
                {"result": "unresolvable_origin", "origin": origin},
                session=self._current_session_name,
            )
            return
        self._record_back_precedent(record)
        user_prompt = record.get("user_prompt")
        if not isinstance(user_prompt, str):
            user_prompt = ""
        # Calibration label (R3-C4): /back is a rejection — this is also
        # how auto switches feed calibration data after auto activates.
        # 보정 라벨 (R3-C4) — /back 은 거부다. auto 활성 후에도 auto
        # 전환의 보정 데이터가 갱신되는 경로가 바로 이것이다.
        decision_log.append_label(
            Path(self.project_path), wrong, decision_log.LABEL_REJECT, source="back"
        )
        debug_log.log(
            "BACK",
            "USER",
            {"result": "undo", "origin": origin, "wrong": wrong},
            session=self._current_session_name,
        )
        self._notify_user(f"⇄ {origin} 세션으로 되돌립니다 (직전: {wrong})")
        self._execute_transition(
            target=origin,
            resume_conv=resume_conv,
            handoff={
                "from": wrong,
                "back": True,
                "message": f"/back — {wrong} 로의 직전 전환을 되돌려 복귀",
            },
            user_prompt=user_prompt,
            is_back=True,
        )
        # Same pointer-invalidation path as manual /resume observation:
        # the MCP re-resolves the session from the active conversation.
        # 수동 /resume 관찰과 동일한 포인터 무효화 경로 — MCP 는 활성
        # conversation 으로부터 세션을 재해석한다.
        self.socket_server.send(
            {"action": "session_command", "command": "back", "args": ""}
        )

    def _record_back_precedent(self, record: dict[str, Any]) -> None:
        """Record the /back as a rejection precedent on the origin session.

        /back 을 복귀(원래) 세션의 거부 판례로 기록한다. kept_in = 복귀
        세션, rejected = 잘못 갔던 세션. 기록 경로는 MCP reject_switch 와
        동일한 F15 잠금 mutate 다.
        """
        user_prompt = record.get("user_prompt")
        gist = ""
        if isinstance(user_prompt, str) and user_prompt.strip():
            gist = user_prompt.strip().splitlines()[0]
        if not gist:
            gist = "(재주입 프롬프트 없음)"
        precedent = PrecedentRecord.new(
            prompt_gist=gist, kept_in=record["from"], rejected=record["to"]
        )

        def apply(session: Any) -> None:
            session.precedents.append(precedent)

        try:
            saved = SessionStore(Path(self.project_path)).mutate_session_by_name(
                record["from"], apply
            )
        except Exception as exc:
            saved = None
            debug_log.log(
                "BACK",
                "WRAPPER",
                {"op": "precedent", "result": "error", "error": str(exc)},
            )
        if saved is None:
            debug_log.log(
                "BACK",
                "WRAPPER",
                {
                    "op": "precedent",
                    "result": "session_not_found",
                    "kept_in": record["from"],
                },
            )

    def _record_last_transition(
        self, from_name: object, to_name: object, user_prompt: str
    ) -> None:
        """Remember a completed wrapper-executed transition for /back.

        완료된 래퍼 실행 전환을 /back 용으로 기억한다. 떠난 세션 이름을
        모르면 (미등록 시작) 되돌아갈 곳이 없으므로 기록하지 않는다.
        """
        if not isinstance(from_name, str) or not from_name:
            return
        if not isinstance(to_name, str) or not to_name:
            return
        record = {
            "from": from_name,
            "to": to_name,
            "user_prompt": user_prompt,
            "at": wrapper_state.utc_now_iso(),
        }
        self._last_transition = record
        wrapper_state.save_last_transition(Path(self.project_path), record)

    # ------------------------------------------------------- Summary triggers
    # 요약 트리거 ---------------------------------------------------------------

    def _enqueue_departed_summary(self, session_name: object) -> None:
        """Queue a background summary for the session being left.

        떠나는 세션의 백그라운드 요약을 큐에 넣는다.

        Called on SWITCH/NEW signal receipt, while the active conversation
        is still the departing one. Skips (with a log) when the session
        name or the active conversation cannot be determined — the boot
        recovery pass covers those later.

        SWITCH/NEW 신호 수신 시점에 호출 — 이때 활성 conversation 은 아직
        떠나는 쪽이다. 세션 이름이나 활성 conversation 을 알 수 없으면
        로그만 남기고 skip — 부팅 복구 pass 가 나중에 메운다.
        """
        if not isinstance(session_name, str) or not session_name:
            debug_log.log(
                "SUMMARY_TRIGGER",
                "WRAPPER",
                {"trigger": "departed", "skipped": "no_session_name"},
            )
            return
        conv_id = get_active_conversation_id(Path(self.project_path))
        if conv_id is None:
            debug_log.log(
                "SUMMARY_TRIGGER",
                "WRAPPER",
                {"trigger": "departed", "skipped": "no_active_conversation"},
                session=session_name,
            )
            return
        summarizer.enqueue(
            Path(self.project_path),
            SummaryTask(
                session_name=session_name,
                conversation_id=conv_id,
                kind=summarizer.KIND_DEPARTED,
            ),
        )
        self.summarizer_worker.wake()

    def _enqueue_active_summary(self) -> None:
        """Queue a background summary for the current session (e.g. on /clear).

        현재 세션의 백그라운드 요약을 큐에 넣는다 (/clear 관찰 등).
        """
        session_name = self._current_session_name
        if session_name is None:
            debug_log.log(
                "SUMMARY_TRIGGER",
                "WRAPPER",
                {"trigger": "active", "skipped": "unknown_current_session"},
            )
            return
        conv_id = get_active_conversation_id(Path(self.project_path))
        if conv_id is None:
            debug_log.log(
                "SUMMARY_TRIGGER",
                "WRAPPER",
                {"trigger": "active", "skipped": "no_active_conversation"},
                session=session_name,
            )
            return
        summarizer.enqueue(
            Path(self.project_path),
            SummaryTask(
                session_name=session_name,
                conversation_id=conv_id,
                kind=summarizer.KIND_ACTIVE,
            ),
        )
        self.summarizer_worker.wake()

    def _check_summary_refresh(self) -> None:
        """Refresh the summary once enough new dialogue has accumulated.

        새 대화가 충분히 쌓이면 요약을 갱신한다.

        Without this a session worked in for hours — no switch, no
        ``/clear`` — keeps the summary it had at the start, and the router
        reads a description of work that finished long ago. The trigger
        measures *dialogue*, not tokens or turns: token growth counts tool
        results and thinking, so reading a few large files would fire a
        pointless re-summary, and turns vary in size too much to mean
        anything. The threshold is the excerpt window itself — once more
        new dialogue exists than one summary can read, the summary is
        provably behind.

        이것이 없으면 몇 시간을 작업한 세션이 (전환도 ``/clear`` 도 없었다면)
        시작 시점의 요약을 그대로 갖고 있어, 라우터는 오래전에 끝난 작업의
        설명을 읽는다. 트리거는 토큰이나 턴이 아니라 *대화* 를 잰다: 토큰
        증가는 도구 결과·thinking 을 포함하므로 큰 파일 몇 개만 읽어도 무의미한
        재요약이 발동하고, 턴은 크기 편차가 너무 커서 의미가 없다. 임계값은
        발췌 창 그 자체다 — 요약 한 번이 읽을 수 있는 양보다 새 대화가 많아진
        시점이면 그 요약은 확실히 뒤처져 있다.
        """
        session_name = self._current_session_name
        if session_name is None:
            return
        project = Path(self.project_path)
        conv_id = get_active_conversation_id(project)
        if conv_id is None:
            return
        if conv_id != self._dialogue_scan_conv_id:
            # New conversation (rollover, switch, /clear): restart the scan.
            # 새 conversation (롤오버·전환·/clear): 스캔을 처음부터.
            self._dialogue_scan_conv_id = conv_id
            self._dialogue_scan_offset = 0
            self._dialogue_scan_chars = 0
        jsonl_path = (
            Path.home()
            / ".claude"
            / "projects"
            / encode_cwd(project)
            / f"{conv_id}.jsonl"
        )
        (
            self._dialogue_scan_offset,
            self._dialogue_scan_chars,
        ) = scan_dialogue_growth(
            jsonl_path, self._dialogue_scan_offset, self._dialogue_scan_chars
        )
        session = SessionStore(project).load_session_by_name(session_name)
        if session is None:
            return
        # A baseline measured in another conversation says nothing about
        # this one — treat it as zero rather than comparing across them.
        # 다른 conversation 에서 측정한 기준값은 이 conversation 에 대해 아무
        # 것도 말해주지 않는다 — 교차 비교 대신 0 으로 본다.
        baseline = (
            session.summary_dialogue_chars
            if session.summary_dialogue_conversation_id == conv_id
            else 0
        )
        growth = self._dialogue_scan_chars - baseline
        if growth < EXCERPT_MAX_CHARS:
            return
        debug_log.log(
            "SUMMARY_TRIGGER",
            "WRAPPER",
            {
                "trigger": "growth",
                "growth_chars": growth,
                "threshold": EXCERPT_MAX_CHARS,
            },
            conv_id=conv_id,
            session=session_name,
        )
        # Duplicate enqueues while this one is still pending are dropped by
        # the queue, so re-checking every turn costs nothing.
        # 대기 중 중복 적재는 큐가 버리므로 매 턴 재확인해도 비용이 없다.
        self._enqueue_active_summary()

    def _check_context_usage(self) -> None:
        """Mark the rollover pending once the active conversation is full.

        활성 대화가 발동점에 닿으면 롤오버 pending 을 마킹한다 (R4-C1).

        Detection only — acting on the mark (Handoff request, respawn) is
        R4-C3/C4. The mark is per-conversation: a conversation change
        (switch, /clear, rollover itself) clears it, and the pending →
        marked transition is logged exactly once.

        감지만 한다 — 마킹에 대한 행동 (Handoff 요청·respawn) 은
        R4-C3/C4. 마킹은 conversation 단위다: conversation 이 바뀌면
        (전환·/clear·롤오버 자신) 해제되고, 미마킹 → 마킹 전이는 정확히
        1회만 기록한다.
        """
        project = Path(self.project_path)
        conv_id = get_active_conversation_id(project)
        if conv_id is None:
            return
        if conv_id != self._rollover_pending_conv_id:
            self._rollover_pending_conv_id = None
        jsonl_path = (
            Path.home()
            / ".claude"
            / "projects"
            / encode_cwd(project)
            / f"{conv_id}.jsonl"
        )
        usage = context_monitor.check_context_usage(project, conv_id, jsonl_path)
        if usage is None or not usage.exceeded:
            return
        if self._rollover_pending_conv_id == conv_id:
            return
        self._rollover_pending_conv_id = conv_id
        debug_log.log(
            "ROLLOVER_PENDING",
            "WRAPPER",
            {
                "used_tokens": usage.used_tokens,
                "window_tokens": usage.window_tokens,
                "trigger_tokens": usage.trigger_tokens,
                "numerator_source": usage.numerator_source,
                "denominator_source": usage.denominator_source,
            },
            conv_id=conv_id,
            session=self._current_session_name,
        )

    def _enqueue_stale_summaries(self) -> None:
        """Boot-time recovery: queue sessions whose summary refresh was lost.

        부팅 시 복구 — 요약 갱신이 유실된 세션들을 큐에 넣는다.

        A session is stale when its transcript was written to after the
        summary was last refreshed (or was never summarised). The predicate
        uses transcript mtime, not ``last_accessed``: the latter is written
        only by tool calls that touch the session, so it neither proves use
        nor proves idleness. Duplicate enqueues are prevented by the queue
        itself. Never raises — a recovery failure must not block startup.

        transcript 가 마지막 summary 갱신 이후에 쓰였으면 (또는 한 번도 요약된
        적 없으면) stale. 술어는 ``last_accessed`` 가 아니라 transcript mtime 을
        쓴다 — 전자는 세션을 건드리는 도구 호출 시에만 기록되어 사용도 유휴도
        증명하지 못한다. 중복 적재는 큐 자체가 막는다. 복구 실패가 래퍼 시작을
        막지 않도록 예외를 내지 않는다.
        """
        try:
            project = Path(self.project_path)
            # Failed and crash-orphaned queue files are never removed by the
            # normal path — clear the old ones here (same retention period
            # the project already uses for sessions).
            # 실패·크래시 고아 큐 파일은 정상 경로에서 지워지지 않는다 —
            # 오래된 것을 여기서 정리한다 (세션에 이미 쓰는 보존 기간 재사용).
            summarizer.sweep_stale_queue_files(project, get_cleanup_period_days())
            queued = 0
            for session in SessionStore(project).list_sessions():
                if not session.claude_conversation_ids:
                    continue
                activity = get_conversation_activity(
                    project, session.claude_conversation_ids
                )
                if activity is None:
                    # Transcripts gone (Claude Code's own cleanup) — nothing
                    # left to summarise.
                    # transcript 가 사라짐 (Claude Code 자체 정리) — 요약할
                    # 대상이 없다.
                    continue
                if session.summary_updated_at is not None and activity <= (
                    datetime.fromisoformat(session.summary_updated_at)
                ):
                    continue
                summarizer.enqueue(
                    project,
                    SummaryTask(
                        session_name=session.name,
                        conversation_id=session.claude_conversation_ids[-1],
                        kind=summarizer.KIND_DEPARTED,
                    ),
                )
                queued += 1
            debug_log.log(
                "SUMMARY_TRIGGER",
                "WRAPPER",
                {"trigger": "boot_recovery", "queued": queued},
            )
        except Exception as exc:
            debug_log.log(
                "SUMMARY_TRIGGER",
                "WRAPPER",
                {"trigger": "boot_recovery", "error": str(exc)},
            )

    # -------------------------------------------- Observation & auto-confirm
    # 관찰·자동 승인 ------------------------------------------------------------

    def _auto_accept_confirmations(self) -> None:
        """Send \\r whenever a known confirmation prompt appears on screen.

        가상 화면에 알려진 confirmation prompt 텍스트가 나타나면 \\r 주입.
        모든 prompt의 default가 1번이라 단순 Enter로 승인된다. 한 번 처리한
        패턴은 ``_handled_confirmations``에 기록해 같은 자식에서 다시 매칭
        되지 않는다. 매칭은 시작 윈도우(첫 사용자 키 입력 전)에서만
        동작한다 (F13 — 이후의 화면 노출은 인용·파일 표시일 수 있다).
        """
        if not self._auto_confirm_armed:
            return
        for pattern in AUTO_CONFIRM_PATTERNS:
            if pattern in self._handled_confirmations:
                continue
            if self.virtual_screen.contains(pattern):
                _debug_log(f"auto-accept: detected '{pattern}', sending \\r")
                debug_log.log(
                    "AUTO_CONFIRM",
                    "WRAPPER",
                    {"pattern": pattern},
                )
                try:
                    debug_log.log(
                        "WRAPPER_INJECT",
                        "WRAPPER",
                        {"caller": "_auto_accept_confirmations", "raw": "\\r"},
                    )
                    os.write(self.pty_fd, b"\r")
                except OSError:
                    return
                self._handled_confirmations.add(pattern)

    def _maybe_start_judge(self) -> None:
        """
        Start the judge host when routing is actually possible: mode not
        "off" and at least two active sessions (the same deterministic
        conditions the hook prefilter checks — a judge warmed for an
        unroutable project would only burn one warmup call per boot).
        Idempotent; called at boot and whenever session topology may
        have changed (MCP switch/new/current_session signals).

        라우팅이 실제로 가능할 때 판정 호스트를 시작한다: 모드가 "off"가
        아니고 활성 세션이 2개 이상 (hook 프리필터와 동일한 결정적 조건 —
        라우팅 불가능한 프로젝트에서 웜업하면 부팅마다 웜업 호출만
        낭비된다). 멱등이며 부팅 시와 세션 구성이 바뀔 수 있는 시점
        (MCP switch/new/current_session 신호)마다 호출된다.
        """
        root = Path(self.project_path) / _SESSION_MANAGER_DIRNAME
        if _load_routing_mode(root) == "off":
            return
        if _count_active_sessions(root) < 2:
            return
        self.judge_host.ensure_started()

    def _handle_hook_message(self, message: dict, sock: Any) -> bool:
        """
        Deferred-reply dispatcher for hook messages (socket server
        callback). judge_request transfers the connection to the judge
        host; anything else falls back to the ack path.

        hook 메시지의 지연 회신 디스패처 (소켓 서버 콜백). judge_request
        는 연결을 판정 호스트로 이관하고, 그 외는 ack 경로로 돌려보낸다.
        """
        if message.get("action") == "judge_request":
            return self.judge_host.handle_request(message, sock)
        return False

    def _handle_mcp_signal(self, message: dict) -> None:
        if not isinstance(message, dict):
            return
        # Any MCP signal may follow a session-topology change (register,
        # switch, create) — cheap idempotent re-check of the judge
        # start conditions.
        # 모든 MCP 신호는 세션 구성 변화(register·switch·create) 뒤에 올
        # 수 있다 — 판정기 시작 조건의 저렴한 멱등 재검사.
        self._maybe_start_judge()
        msg_type = message.get("type")
        if msg_type == "handshake_request":
            self._handle_handshake_request()
            return
        action = message.get("action")
        if action == "switch":
            target = message.get("target")
            handoff = message.get("handoff") or {}
            if not isinstance(target, str) or not isinstance(handoff, dict):
                return
            # The MCP puts user_prompt at the top level of the signal;
            # older senders embedded it in the handoff — accept both.
            # (The handoff-only read silently dropped MCP prompts — found
            # by the respawn-flow integration test.)
            # MCP 는 user_prompt 를 신호 최상위에 싣는다. 구 형식은 handoff
            # 안에 넣었다 — 둘 다 수용. (handoff 만 읽던 코드는 MCP 경유
            # 프롬프트를 조용히 유실했다 — respawn 통합 테스트가 발견.)
            user_prompt_val = message.get(
                "user_prompt", handoff.get("user_prompt", "")
            )
            user_prompt = user_prompt_val if isinstance(user_prompt_val, str) else ""
            # Resume the target's recorded conversation; a target with no
            # resolvable conversation gets a fresh one (still that
            # session — the MCP has already moved its pointer there).
            # 대상 세션의 기록된 conversation 을 재개한다. 해석 불가면 새
            # conversation 으로 (여전히 그 세션 — MCP 포인터는 이미 이동).
            self._execute_transition(
                target=target,
                resume_conv=self._resolve_resume_conv(target),
                handoff=handoff,
                user_prompt=user_prompt,
            )
        elif action == "new":
            new_session_name = message.get("new_session_name")
            handoff = message.get("handoff") or {}
            if not isinstance(new_session_name, str) or not isinstance(handoff, dict):
                return
            # Same top-level-first read as the switch branch.
            # switch 분기와 동일한 최상위 우선 읽기.
            user_prompt_val = message.get(
                "user_prompt", handoff.get("user_prompt", "")
            )
            user_prompt = user_prompt_val if isinstance(user_prompt_val, str) else ""
            # NOTE: the old flow /rename'd the departing conversation so
            # title-based resume could find it. Resume is id-based
            # everywhere now (conversation ids live in the metadata), so
            # rename_current is accepted but unused.
            # 참고 — 구 흐름은 title 기반 resume 을 위해 떠나는 conversation
            # 을 /rename 했다. 이제 resume 은 전부 id 기반 (id 는 메타데이터
            # 에 기록) 이므로 rename_current 는 받되 사용하지 않는다.
            self._execute_transition(
                target=new_session_name,
                resume_conv=None,
                handoff=handoff,
                user_prompt=user_prompt,
            )
        elif action == "route_switch":
            # Auto-routing path (R2): the hook blocked the prompt and
            # delegated the switch here. The hook doesn't know session
            # names, so the departing side comes from the wrapper's own
            # current-session mirror.
            # 자동 라우팅 경로 (R2): hook 이 프롬프트를 차단하고 전환을
            # 여기에 위임했다. hook 은 세션 이름을 모르므로, 떠나는 쪽은
            # 래퍼 자신의 현재 세션 미러에서 채운다.
            target = message.get("target")
            if not isinstance(target, str) or not target:
                return
            user_prompt_val = message.get("user_prompt", "")
            user_prompt = user_prompt_val if isinstance(user_prompt_val, str) else ""
            verdict = message.get("verdict")
            origin = self._current_session_name
            handoff = {"from": origin}
            if isinstance(verdict, dict):
                reason = verdict.get("reason")
                if isinstance(reason, str) and reason:
                    handoff["router_reason"] = reason
            self._execute_transition(
                target=target,
                resume_conv=self._resolve_resume_conv(target),
                handoff=handoff,
                user_prompt=user_prompt,
            )
            # Status line for the unattended switch (Plan R3-C4 wording)
            # — the user must learn about it and how to undo it.
            # 무인 전환의 상태 줄 (Plan R3-C4 원문) — 사용자는 전환 사실과
            # 되돌리는 방법을 알아야 한다.
            self._notify_user(
                f"⇄ {target} 세션으로 전환됨 (이전: {origin}) — 되돌리려면 /back"
            )
        elif action == "current_session":
            # MCP resolved or changed the current session. The wrapper has
            # no other way to learn it — the handshake only flows
            # wrapper→MCP, so on a plain `ccode` start (no --resume) the
            # mirror would stay None and every session-scoped trigger
            # (/clear summary, periodic refresh) would silently no-op.
            # MCP 가 현재 세션을 확정·변경했다. 래퍼는 이를 알 방법이 달리
            # 없다 — 핸드셰이크는 래퍼→MCP 단방향이라, 인자 없는 `ccode`
            # 시작에서는 미러가 None 으로 남아 세션 단위 트리거 (/clear 요약,
            # 주기 갱신) 가 조용히 무효화된다.
            name = message.get("name")
            if name is None or isinstance(name, str):
                before = self._current_session_name
                self._current_session_name = name
                debug_log.log(
                    "CURRENT_SESSION",
                    "MCP_TOOL",
                    {"before": before, "after": name},
                    session=name,
                )

    # ----------------------------------------------------------- Action handlers
    # 세션 액션 처리 ------------------------------------------------------------

    def _execute_transition(
        self,
        *,
        target: str,
        resume_conv: str | None,
        handoff: dict[str, Any],
        user_prompt: str,
        is_back: bool = False,
    ) -> None:
        """Register a transition: pending file + child swap request.

        전환을 등록한다 — pending 파일 기록 + 자식 교체 요청.

        No TUI interaction happens here or later: the handoff content
        goes to the pending file (the hook injects it after respawn),
        and the swap itself is process control (_maybe_terminate_for_
        respawn → _should_respawn → _spawn_child).

        여기서도 이후에도 TUI 상호작용은 없다 — handoff 내용은 pending
        파일로 (respawn 후 hook 이 주입), 교체는 프로세스 제어로 실행된다
        (_maybe_terminate_for_respawn → _should_respawn → _spawn_child).
        """
        if self._pending_respawn is not None:
            # One transition at a time; a second signal before the swap
            # completes is dropped with a log (the MCP retries naturally
            # on the next user action).
            # 전환은 한 번에 하나 — 교체 완료 전의 두 번째 신호는 로그만
            # 남기고 버린다 (다음 사용자 행동에서 자연 재시도).
            debug_log.log(
                "TRANSITION",
                "WRAPPER",
                {"op": "register", "result": "busy_dropped", "target": target},
                session=target,
            )
            return
        from_name = handoff.get("from")
        # Queue a background summary for the departing session while its
        # conversation is still the active one, and move the wrapper-side
        # current-session mirror to the target.
        # 떠나는 세션의 conversation 이 아직 활성인 시점에 백그라운드
        # 요약을 큐에 넣고, 래퍼 측 현재 세션 미러를 target 으로 이동.
        self._enqueue_departed_summary(from_name)
        self._current_session_name = target
        handoff_clean = {k: v for k, v in handoff.items() if k != "user_prompt"}
        handoff_store.write_pending(
            Path(self.project_path), target, handoff_clean, user_prompt
        )
        self._pending_respawn = _PendingRespawn(
            target=target,
            resume_conv=resume_conv,
            from_name=from_name if isinstance(from_name, str) else None,
            user_prompt=user_prompt,
            is_back=is_back,
        )
        debug_log.log(
            "TRANSITION",
            "WRAPPER",
            {
                "op": "register",
                "target": target,
                "resume_conv": resume_conv,
                "is_back": is_back,
                "user_prompt": debug_log.mask_text(user_prompt),
            },
            session=target,
        )

    def _resolve_resume_conv(self, target: str) -> str | None:
        """Resolve *target*'s latest conversation id with a live transcript.

        *target* 세션의 최신 conversation id 를 해석한다 — transcript 가
        실제로 존재할 때만. (Claude Code 자체 정리로 지워진 stale id 로
        ``--resume`` 하면 자식이 부팅 실패로 즉사하므로 사전 확인한다.)
        None 이면 호출자가 새 conversation 부팅 또는 중단을 택한다.
        """
        try:
            session = SessionStore(Path(self.project_path)).load_session_by_name(
                target
            )
        except Exception:
            session = None
        if session is None or not session.claude_conversation_ids:
            return None
        conv = session.claude_conversation_ids[-1]
        transcript = (
            Path.home()
            / ".claude"
            / "projects"
            / encode_cwd(Path(self.project_path))
            / f"{conv}.jsonl"
        )
        return conv if transcript.is_file() else None

    def _handle_handshake_request(self) -> None:
        """
        Reply to MCP's handshake with the wrapper's current-session
        mirror (moved at transition registration), falling back to the
        CLI-args decision on a fresh start.

        MCP 핸드셰이크에 래퍼의 현재 세션 미러 (전환 등록 시 이동) 로
        응답한다. 신규 시작이면 CLI 인자에서 결정된 값으로 폴백.
        """
        name = self._current_session_name or self._initial_session_name
        debug_log.log(
            "HANDSHAKE",
            "WRAPPER",
            {"phase": "wrapper_response", "current_session_name": name},
        )
        self.socket_server.send({"current_session_name": name})

    @staticmethod
    def _parse_initial_session_name(args: list[str]) -> str | None:
        for i, arg in enumerate(args):
            if arg == "--resume" and i + 1 < len(args):
                return args[i + 1]
            if arg.startswith("--resume="):
                return arg[len("--resume=") :]
        return None

    # ---------------------------------------------------------- PTY drainage
    # PTY 배수 -------------------------------------------------------------------

    def _drain_pty(self) -> None:
        try:
            while True:
                chunk = os.read(self.pty_fd, 4096)
                if not chunk:
                    return
                self.virtual_screen.feed(chunk)
                os.write(self._stdout_fd, chunk)
        except OSError:
            return

    # --------------------------------------------------------- Terminal state
    # 터미널 상태 관리 ---------------------------------------------------------

    def _enter_raw_mode(self) -> None:
        if not os.isatty(self._stdin_fd):
            return
        self._original_stdin_attrs = termios.tcgetattr(self._stdin_fd)
        tty.setraw(self._stdin_fd)

    def _restore_terminal(self) -> None:
        if self._original_stdin_attrs is not None:
            termios.tcsetattr(
                self._stdin_fd, termios.TCSADRAIN, self._original_stdin_attrs
            )
            self._original_stdin_attrs = None
        if self._previous_winch_handler is not None:
            signal.signal(signal.SIGWINCH, self._previous_winch_handler)
            self._previous_winch_handler = None

    def _install_winch_handler(self) -> None:
        self._previous_winch_handler = signal.signal(
            signal.SIGWINCH, self._on_resize
        )

    def _on_resize(self, signum: int, frame: Any) -> None:
        self._sync_winsize()

    def _sync_winsize(self) -> None:
        if self.pty_fd < 0 or not os.isatty(self._stdout_fd):
            return
        try:
            rows, cols = termios.tcgetwinsize(self._stdout_fd)
        except OSError:
            return
        try:
            termios.tcsetwinsize(self.pty_fd, (rows, cols))
        except OSError:
            return
        # Keep the virtual screen in lockstep with the actual PTY size so
        # Ink's wrap-aware redraws extract correctly.
        # Ink가 wrap을 고려해 그리는 부분 갱신이 정확히 추출되도록 가상
        # 화면을 실제 PTY 크기와 동기화.
        self.virtual_screen.resize(cols, rows)
