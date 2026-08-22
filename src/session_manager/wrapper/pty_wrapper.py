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
import time
import tty
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pexpect

from session_manager import (
    debug_log,
    handoff_store,
    rollover,
    statusline,
    summarizer,
)
from session_manager.claude_conversation import (
    conversation_exists,
    encode_cwd,
    get_active_conversation_id,
    get_conversation_activity,
)
from session_manager.hooks.user_prompt_submit import (
    _count_active_sessions,
    _load_routing_mode,
)
from session_manager.lifecycle import get_cleanup_period_days
from session_manager.models.session import PrecedentRecord, SessionStatus
from session_manager.routing import decision_log
from session_manager.storage.file_store import _SESSION_MANAGER_DIRNAME, SessionStore
from session_manager.summarizer import SummarizerWorker, SummaryTask
from session_manager.transcript_excerpt import (
    EXCERPT_MAX_CHARS,
    extract_full_text,
    scan_dialogue_growth,
)
from session_manager.wrapper import context_monitor, wrapper_state
from session_manager.wrapper.command_matcher import (
    InterceptedCommand,
    match_back_command,
    match_handoff_command,
    match_intercept_command,
    match_sessions_command,
)
from session_manager.wrapper.judge_host import JudgeHost
from session_manager.wrapper.notice import (
    NoticeKind,
    format_notice,
    format_session_list,
)
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

# Second busy marker: the spinner's ellipsis ("Sprouting… ❯"). Measured
# (R4-C3 e2e screen logs, 2026-08-13): short turns (~2s) render the
# spinner verb + "…" but NOT "esc to interrupt" — the footer hint
# rotates and may never reach it, so the marker alone misses short
# turns entirely (no falling edge → no turn-end checks). The ellipsis
# is present during any generation. False-positive direction is safe:
# conversation text near the prompt containing "…" only DELAYS a swap
# or a turn-end check until the next redraw — it can never kill a turn
# mid-generation (that would require a false NEGATIVE).
# 두 번째 바쁨 마커 — 스피너의 말줄임표 ("Sprouting… ❯"). 실측 (R4-C3
# e2e 스크린 로그, 2026-08-13): 짧은 턴 (~2초) 은 스피너 동사+"…" 는
# 그리지만 "esc to interrupt" 는 안 그린다 — 푸터 힌트가 로테이션이라
# 도달하지 못할 수 있고, 그 마커만으로는 짧은 턴을 통째로 놓친다 (하강
# 에지 없음 → 턴 종료 검사 전부 침묵). 말줄임표는 생성 중 항상 표시된다.
# 오탐 방향은 안전하다: 프롬프트 주변 본문의 "…" 는 교체·검사를 다음
# redraw 까지 **지연**시킬 뿐, 생성 중인 턴을 죽이려면 거짓 음성이
# 필요하다.
BUSY_SPINNER_ELLIPSIS = "…"


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
    # Rollover handoff-request respawn (R4-C3): same session, same
    # conversation — no /back bookkeeping, no departed summary.
    # 롤오버 handoff 요청 respawn (R4-C3): 같은 세션·같은 conversation —
    # /back 부기도 departed 요약도 없다.
    is_rollover_request: bool = False
    # Rollover swap respawn (R4-C4): same session, NEW conversation —
    # /back to one's own predecessor would be a pointless self-switch,
    # so no bookkeeping; the session summary refresh happens at finalize.
    # 롤오버 교체 respawn (R4-C4): 같은 세션·새 conversation — 자기
    # 선대로의 /back 은 무의미한 자기 전환이라 부기 없음. 세션 요약
    # 갱신은 finalize 에서.
    is_rollover_swap: bool = False
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
        self._current_session_name: str | None = None
        self._set_current_session(self._initial_session_name)

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

        # Conversation whose rollover is suppressed by the loop guard
        # (birth footprint ≥ trigger — see _check_context_usage). Logged
        # once per conversation.
        # 루프 가드로 롤오버가 억제된 conversation (태생 점유 ≥ 트리거 —
        # _check_context_usage 참조). conversation 당 1회만 로그.
        self._rollover_suppressed_conv_id: str | None = None

        # Active conversation id as delivered by the Stop hook (contract
        # source, F18-safe). Mirror only for now — consumers migrate from
        # the mtime scan incrementally.
        # Stop hook 이 전달한 활성 conversation id (계약 소스, F18 안전).
        # 지금은 미러만 — 소비처는 mtime 스캔에서 점진 전환.
        self._active_conv_from_hook: str | None = None
        # Conversation id the wrapper itself assigned to the current
        # child (F18): a fresh uuid passed as ``--session-id=`` for a new
        # conversation, or the ``--resume=`` target for a wrapper-driven
        # resume. None when the user drove the resume (``--continue``,
        # ``--resume <x>``) — then only the hook can tell.
        # 래퍼가 현재 자식에 스스로 지정한 대화 id (F18): 새 대화면
        # ``--session-id=`` 로 넘긴 새 uuid, 래퍼 주도 재개면 ``--resume=``
        # 대상. 사용자가 재개를 주도했으면 (``--continue``, ``--resume <x>``)
        # None — 그때는 hook 만이 알려 준다.
        self._assigned_conv_id: str | None = None
        # Last conversation id pushed to the MCP server, to push only on
        # change. / MCP 서버에 마지막으로 push 한 대화 id — 변경 시에만 push.
        self._pushed_conv_id: str | None = None

        # monotonic time of the last user submit (\r) — turn-duration
        # measurement's start mark (end mark = the Stop-hook signal).
        # 마지막 사용자 제출 (\r) 의 monotonic 시각 — 턴 지속시간 계측의
        # 시작점 (끝점 = Stop hook 신호).
        self._last_submit_monotonic: float | None = None

        # Rollover handoff request in flight (R4-C3). Keys: session, n,
        # conv_id, attempts, phase ("requested" until the dedicated turn
        # spawns, then "writing"). None = no request running.
        # 진행 중인 롤오버 handoff 요청 (R4-C3). 키: session, n, conv_id,
        # attempts, phase (전용 턴 spawn 전 "requested", 이후 "writing").
        # None = 요청 없음.
        self._rollover_request_state: dict[str, Any] | None = None

        # Validated handoff waiting for the actual swap (R4-C4). Keys:
        # session, n, conv_id, path.
        # 실제 교체 (R4-C4) 를 기다리는 검증된 handoff. 키: session, n,
        # conv_id, path.
        self._rollover_ready: dict[str, Any] | None = None

        # Swap in flight, awaiting the successor conversation (R4-C4).
        # Keys: session, n, path, predecessor_conv. Finalize (link +
        # precedent clearing + summary) runs when a NEW active
        # conversation id is observed.
        # 진행 중인 교체 — 후계 conversation 대기 (R4-C4). 키: session,
        # n, path, predecessor_conv. 새 활성 conversation id 가 관찰되면
        # finalize (link + 판례 소멸 + 요약) 가 돈다.
        self._rollover_swap_state: dict[str, Any] | None = None

        # Context percentage at marking time (status-line wording).
        # 마킹 시점의 컨텍스트 퍼센트 (상태 줄 문구용).
        self._rollover_pending_pct: int | None = None

        # Conversation just arrived at via a transition respawn (R4-C6 A):
        # armed at spawn when the transition resumed an existing
        # conversation, disarmed at the first conclusive context reading.
        # If that reading marks the rollover, the mark gets the entry
        # notice — the user should know the session they just landed in
        # is about to roll over.
        # 전환 respawn 으로 방금 도착한 conversation (R4-C6 A) — 기존
        # conversation 을 resume 한 전환의 spawn 시점에 무장하고, 첫 확정
        # 컨텍스트 판독에서 해제한다. 그 판독이 롤오버를 마킹하면 진입
        # 안내를 붙인다 — 방금 도착한 세션이 곧 롤오버됨을 사용자가
        # 알아야 한다.
        self._entry_check_conv_id: str | None = None

        # Conversations already warned about being a rolled-over
        # predecessor (R4-C6 B) — one warning per conversation per
        # wrapper lifetime; a chosen "그대로 진행" must not be nagged
        # every turn.
        # 롤오버된 선대 대화라고 이미 경고한 conversation 들 (R4-C6 B) —
        # 래퍼 수명당 대화별 1회. "그대로 진행" 을 택한 사용자를 매 턴
        # 잔소리하면 안 된다.
        self._stale_conv_warned: set[str] = set()

        # Conversation whose move-or-stay question is still in flight
        # (R4-C6 B): rollover marking is deferred for it until the hook
        # consumes the notice — measured race: a full predecessor
        # re-entered via /resume was re-rolled-over by the SAME turn-end
        # tick that should have warned, forking the lineage (two
        # successors, the new one blind to the old one's progress).
        # 이동/계속 질문이 아직 답을 기다리는 conversation (R4-C6 B) —
        # hook 이 notice 를 소비할 때까지 롤오버 마킹을 유예한다. 실측된
        # race: /resume 으로 재진입한 꽉 찬 선대가 경고해야 할 바로 그
        # 턴 종료 틱에서 재롤오버되어 계보가 분기했다 (후계 둘, 새 후계는
        # 기존 후계의 진행을 모름).
        self._stale_notice_conv: str | None = None

        # context.json observation state (second turn-end signal): file
        # signature (mtime_ns, size) gates the cheap stat-per-tick path;
        # the (conversation_id, used_tokens) key fires _on_turn_end only
        # when the USAGE actually changed — the collector rewrites the
        # file several times per turn with unchanged usage.
        # context.json 관찰 상태 (제2 턴 종료 신호). 파일 서명 (mtime_ns,
        # size) 이 틱당 stat 경로를 gate 하고, (conversation_id,
        # used_tokens) 키가 **usage 가 실제로 바뀐** 때만 _on_turn_end 를
        # 발동한다 — 수집기는 한 턴에 여러 번, usage 무변경으로도 파일을
        # 다시 쓴다.
        self._context_file_sig: tuple[int, int] | None = None
        self._context_usage_key: tuple[Any, Any] | None = None

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
        # Same disposal for a leftover notice — self-correcting: the
        # stale-conversation detector rewrites it if still true (R4-C6 B).
        # 잔류 notice 도 같은 처분 — 조건이 여전히 참이면 만료 대화
        # 감지기가 다시 써 넣으므로 자기 교정적이다 (R4-C6 B).
        handoff_store.clear_stale_notice(Path(self.project_path))

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
            if arg == "--session-id":
                skip_value = True
                continue
            if arg.startswith("--session-id="):
                continue
            out.append(arg)
        return out

    @staticmethod
    def _parse_session_id_arg(args: list[str]) -> str | None:
        """Return a user-supplied ``--session-id`` value, if any.

        사용자가 넘긴 ``--session-id`` 값이 있으면 반환한다.
        """
        for i, arg in enumerate(args):
            if arg == "--session-id" and i + 1 < len(args):
                return args[i + 1]
            if arg.startswith("--session-id="):
                return arg[len("--session-id=") :]
        return None

    @staticmethod
    def _has_resume_args(args: list[str]) -> bool:
        """True if the user asked Claude Code to resume a conversation.

        사용자가 Claude Code 에 대화 재개를 요청했으면 True.
        """
        return any(
            a in ("--continue", "-c", "--resume", "-r")
            or a.startswith("--resume=")
            or a.startswith("-r=")
            for a in args
        )

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
            # Plain boot. A new conversation gets a wrapper-chosen id
            # (F18, measured: ``--session-id`` honoured in headless and
            # TUI, docs/poc/R5-conversation-id.md); a user-driven resume
            # keeps the user's flags and the id stays unknown until the
            # first Stop hook reports it.
            # 맨몸 부팅. 새 대화면 래퍼가 id 를 정한다 (F18, 실측:
            # ``--session-id`` 가 headless·TUI 에서 동작,
            # docs/poc/R5-conversation-id.md); 사용자 주도 재개면 사용자
            # 플래그를 그대로 두고 id 는 첫 Stop hook 이 알려 줄 때까지
            # 미상.
            args = list(self.claude_args)
            self._assigned_conv_id = self._parse_session_id_arg(args)
            if self._assigned_conv_id is None and not self._has_resume_args(args):
                self._assigned_conv_id = str(uuid.uuid4())
                args.append(f"--session-id={self._assigned_conv_id}")
            return args + self._agent_guide_flag() + self._mcp_config_flag()
        args = self._strip_resume_args(self.claude_args)
        args += self._agent_guide_flag()
        args += self._mcp_config_flag()
        if pending.resume_conv is not None:
            args.append(f"--resume={pending.resume_conv}")
            self._assigned_conv_id = pending.resume_conv
        else:
            # NEW session / rollover: the wrapper names the conversation
            # up front, so nothing has to guess it later.
            # NEW 세션 / 롤오버: 래퍼가 대화 이름을 미리 정하므로 나중에
            # 추측할 것이 없다.
            self._assigned_conv_id = str(uuid.uuid4())
            args.append(f"--session-id={self._assigned_conv_id}")
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
                "assigned_conv_id": self._assigned_conv_id,
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
            if pending.is_rollover_request:
                # Same session, same conversation — not a transition, so
                # no /back bookkeeping. The handoff turn starts now:
                # advance the rollover machine to its validation phase.
                # 같은 세션·같은 conversation — 전환이 아니므로 /back
                # 부기 없음. handoff 턴이 지금 시작된다 — 롤오버 머신을
                # 검증 단계로 진행.
                if self._rollover_request_state is not None:
                    self._rollover_request_state["phase"] = "writing"
            elif pending.is_rollover_swap:
                # Successor boot — finalize runs when its conversation
                # is observed (_poll_rollover_finalize); no /back record
                # (a self-switch to one's own predecessor is pointless).
                # 후계 부팅 — finalize 는 conversation 관찰 시
                # (_poll_rollover_finalize). /back 기록 없음 (자기 선대
                # 로의 자기 전환은 무의미).
                pass
            elif pending.is_back:
                self._last_transition = None
                wrapper_state.clear_last_transition(Path(self.project_path))
            else:
                self._record_last_transition(
                    pending.from_name, pending.target, pending.user_prompt
                )
            # Entry-fullness check (R4-C6 A): a switch or /back that
            # resumed an existing conversation may have landed in one
            # that is already over the trigger — arm the one-shot entry
            # check so the first marking after arrival carries the entry
            # notice. Resume keeps the conversation id (R3-respawn PoC;
            # the R4-C3 handoff turn matches its Stop signal by that
            # same id in production). NEW spawns (resume_conv None) and
            # rollover legs are excluded above.
            # 진입 점유 검사 (R4-C6 A) — 기존 conversation 을 resume 한
            # 전환·/back 은 이미 트리거를 넘은 대화에 도착했을 수 있다.
            # 도착 후 첫 마킹에 진입 안내가 붙도록 1회용 검사를 무장한다.
            # resume 은 conversation id 를 유지한다 (R3-respawn PoC —
            # R4-C3 handoff 턴이 같은 id 의 Stop 신호 매칭으로 실동작).
            # NEW spawn (resume_conv 없음) 과 롤오버 단계는 위에서 제외.
            if (
                not pending.is_rollover_request
                and not pending.is_rollover_swap
                and pending.resume_conv is not None
            ):
                self._entry_check_conv_id = pending.resume_conv
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

            # Second turn-end signal: a usage change in context.json.
            # 제2 턴 종료 신호 — context.json 의 usage 변화.
            self._observe_context_update()

            # Handoff-turn completion can outrun the transcript flush —
            # watch the transcript itself while a handoff is pending.
            # handoff 턴 종료 신호가 transcript flush 를 앞지를 수 있다 —
            # handoff 대기 중에는 transcript 자체를 관찰한다.
            self._poll_handoff_transcript()

            # Swap in flight: finalize once the successor conversation
            # is observed (R4-C4).
            # 교체 진행 중 — 후계 conversation 관찰 시 finalize (R4-C4).
            self._poll_rollover_finalize()

        self._drain_pty()

    def _poll_handoff_transcript(self) -> None:
        """Re-validate the handoff turn whenever its transcript changes.

        handoff 턴의 transcript 가 바뀔 때마다 재검증한다.

        The dedicated turn's LAST turn-end signal can arrive milliseconds
        BEFORE the response event is flushed to the JSONL (measured —
        R4-C3 e2e: busy edge at .397, body event stamped .337 but not
        yet readable; no further signal ever came and the machine hung
        in "waiting"). The condition we actually await is "the
        transcript gained the reply", so watch exactly that resource:
        one stat(2) per idle tick while a handoff turn is in flight,
        re-running validation on any (mtime, size) change.

        전용 턴의 마지막 턴 종료 신호가 응답 이벤트의 JSONL flush 보다
        수 ms 먼저 도착할 수 있다 (실측 — R4-C3 e2e: busy 에지 .397,
        본문 이벤트 스탬프 .337 이지만 아직 읽히지 않음. 이후 신호가
        더는 오지 않아 머신이 "waiting" 에 매달렸다). 실제로 기다리는
        조건은 "transcript 에 응답이 실렸다" 이므로 정확히 그 자원을
        관찰한다 — handoff 턴 진행 중에만 유휴 틱당 stat(2) 1회,
        (mtime, size) 변화마다 재검증.
        """
        state = self._rollover_request_state
        if state is None or state.get("phase") != "writing":
            return
        path = (
            Path.home()
            / ".claude"
            / "projects"
            / encode_cwd(Path(self.project_path))
            / f"{state['conv_id']}.jsonl"
        )
        try:
            stat = path.stat()
        except OSError:
            return
        sig = (stat.st_mtime_ns, stat.st_size)
        if sig == state.get("jsonl_sig"):
            return
        state["jsonl_sig"] = sig
        self._finish_handoff_turn(state)

    def _observe_context_update(self) -> None:
        """Fire _on_turn_end when the statusline collector saw new usage.

        statusline 수집기가 새 usage 를 봤으면 _on_turn_end 를 발동한다.

        Why a second signal exists: short turns (~2s) may never render
        the busy footer hint, so the falling edge misses them entirely
        (measured — R4-C3 e2e, 2026-08-13); a session full of short
        turns would never run its context checks. The collector writes
        context.json after every API completion, which IS the turn-end
        fact, delivered through an official interface. One stat(2) per
        idle tick (≤10/s) is the entire cost. The first observation
        after boot only sets the baseline — a stale pre-existing file
        must not fire checks for a turn that ended in a previous run.

        제2 신호가 필요한 이유: 짧은 턴 (~2초) 은 바쁨 푸터 힌트를 아예
        안 그릴 수 있어 하강 에지가 통째로 놓친다 (실측 — R4-C3 e2e,
        2026-08-13). 짧은 턴만 이어지는 세션은 컨텍스트 검사가 영영 안
        돈다. 수집기는 매 API 완결 후 context.json 을 쓰며, 그것이 곧
        공식 인터페이스로 전달되는 턴 종료 사실이다. 비용은 유휴 틱당
        stat(2) 1회 (초당 ≤10회) 가 전부. 부팅 후 첫 관찰은 기준선만
        잡는다 — 이전 실행에서 끝난 턴의 잔존 파일이 검사를 발동하면
        안 된다.
        """
        path = (
            Path(self.project_path)
            / _SESSION_MANAGER_DIRNAME
            / statusline.CONTEXT_FILENAME
        )
        try:
            stat = path.stat()
        except OSError:
            return
        sig = (stat.st_mtime_ns, stat.st_size)
        if sig == self._context_file_sig:
            return
        first_observation = self._context_file_sig is None
        self._context_file_sig = sig
        record = statusline.read_context(Path(self.project_path))
        if not isinstance(record, dict):
            return
        key = (record.get("conversation_id"), record.get("used_tokens"))
        if key == self._context_usage_key:
            return
        self._context_usage_key = key
        if first_observation:
            return
        self._on_turn_end("context_update")

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
        if self._screen_busy():
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
        busy = self._screen_busy()
        if self._was_busy and not busy:
            self._on_turn_end("busy_edge")
        self._was_busy = busy

        os.write(self._stdout_fd, chunk)
        return True

    def _screen_busy(self) -> bool:
        """Is a turn running, judged from the virtual screen?

        가상 화면 기준으로 턴이 실행 중인가?

        Either marker counts — the footer hint rotates, so short turns
        may show only the spinner ellipsis (see BUSY_SPINNER_ELLIPSIS).
        어느 마커든 인정 — 푸터 힌트는 로테이션이라 짧은 턴은 스피너
        말줄임표만 보일 수 있다 (BUSY_SPINNER_ELLIPSIS 참조).
        """
        return self.virtual_screen.contains_near_prompt(
            BUSY_MARKER, BUSY_MARKER_RADIUS_ROWS
        ) or self.virtual_screen.contains_near_prompt(
            BUSY_SPINNER_ELLIPSIS, BUSY_MARKER_RADIUS_ROWS
        )

    def _handle_turn_end_signal(self, message: dict[str, Any]) -> None:
        """Consume one Stop-hook turn-end signal (primary path).

        Stop hook 턴 종료 신호 1건을 소비한다 (주 경로).

        Routing by rollover phase / 롤오버 단계별 분기:
        - handoff turn in flight and the signal is its conversation →
          conclude the attempt from the payload text (no transcript
          read — measured: the transcript is not yet flushed at Stop
          time, but the payload carries the response).
          handoff 턴 진행 중 + 그 conversation 의 신호 → payload 본문으로
          시도 판정 (transcript 무읽기 — 실측: Stop 시점에 transcript 는
          아직 flush 전이지만 payload 에 응답이 실려 있다).
        - swap in flight and the signal is NOT the predecessor → it is
          the successor's first turn: finalize directly.
          교체 진행 중 + 선대가 아닌 신호 → 후계의 첫 턴: 즉시 finalize.
        - otherwise → the ordinary turn-end battery.
          그 외 → 일반 턴 종료 검사 일괄.
        """
        conv_id = message.get("conversation_id")
        if not isinstance(conv_id, str) or not conv_id:
            return
        # Active-conversation mirror (adoption step 3): hook-delivered id
        # is authoritative; consumers migrate incrementally (F18).
        # 활성 대화 미러 (채택 3번): hook 전달 id 가 신뢰 소스. 소비처는
        # 점진 전환 (F18).
        self._active_conv_from_hook = conv_id
        self._push_active_conversation()

        # Turn duration: submit (\r) → Stop signal. Measurement source
        # for R5 --stats and the C3 backstop derivation.
        # 턴 지속시간 — 제출 (\r) → Stop 신호. R5 --stats 와 C3 백스톱
        # 도출의 계측 원천.
        duration_ms: int | None = None
        if self._last_submit_monotonic is not None:
            duration_ms = int(
                (time.monotonic() - self._last_submit_monotonic) * 1000
            )
            self._last_submit_monotonic = None
        debug_log.log(
            "TURN_END",
            "WRAPPER",
            {"source": "stop_hook", "turn_duration_ms": duration_ms},
            conv_id=conv_id,
        )

        # Move-or-stay deferral release (R4-C6 B): a Stop with the notice
        # already consumed means the question turn truly ended — the user
        # has answered. Moved → a transition is in flight and the marking
        # below self-corrects on the conversation change; stayed → the
        # battery below marks the rollover normally. Only THIS signal may
        # release the deferral (fallback edges fire mid-turn — measured).
        # 이동/계속 유예 해제 (R4-C6 B) — notice 가 이미 소비된 상태의
        # Stop 은 질문 턴이 진짜 끝났다는 뜻 = 사용자가 답했다. 이동이면
        # 전환이 진행 중이고 아래 마킹은 conversation 변경으로 자기
        # 교정된다. 계속이면 아래 일괄이 정상 마킹한다. 유예 해제는 이
        # 신호만 할 수 있다 (폴백 에지는 턴 도중 발화 — 실측).
        if self._stale_notice_conv is not None and not handoff_store.notice_pending(
            Path(self.project_path)
        ):
            self._stale_notice_conv = None

        request_state = self._rollover_request_state
        if (
            request_state is not None
            and request_state.get("phase") == "writing"
            and request_state.get("conv_id") == conv_id
        ):
            text_val = message.get("last_assistant_message")
            text = text_val if isinstance(text_val, str) else ""
            # An empty reply is a settled failure, not "waiting": this
            # signal IS the dedicated turn's end.
            # 빈 응답은 "waiting" 이 아니라 확정 실패다 — 이 신호 자체가
            # 전용 턴의 종료이므로.
            self._conclude_handoff_attempt(
                request_state, "answered" if text.strip() else "missing", text
            )
            return
        swap_state = self._rollover_swap_state
        if swap_state is not None:
            if conv_id != swap_state["predecessor_conv"]:
                self._finalize_rollover(conv_id)
            return
        self._check_stale_conv_entry(conv_id)
        self._check_summary_refresh()
        self._check_context_usage()
        self._advance_rollover()

    def _on_turn_end(self, source: str) -> None:
        """Run the turn-end check battery (summary, context, rollover).

        턴 종료 검사 일괄 실행 (요약·컨텍스트·롤오버).

        Fired by either turn-end signal: the busy falling edge, or an
        observed context.json usage change (statusline collector — the
        only signal short turns reliably produce, see _observe_context_
        update). Every check is idempotent, so double firing for one
        turn is harmless.
        두 턴 종료 신호 중 무엇이든 발동한다 — 바쁨 하강 에지, 또는
        context.json usage 변화 관찰 (statusline 수집기 — 짧은 턴이
        확실히 만드는 유일한 신호, _observe_context_update 참조). 모든
        검사는 멱등이라 한 턴에 이중 발동해도 무해하다.
        """
        debug_log.log("TURN_END", "WRAPPER", {"source": source})
        # Stale-entry detection must precede the context check on THIS
        # path too: a manual /resume into a conversation produces its
        # first turn-end as a busy edge (loading, no Stop signal), and a
        # full predecessor would otherwise be re-rolled-over in the same
        # tick that should have warned (measured — see
        # _check_stale_conv_entry). The mtime conversation id is
        # acceptable here: the sole-ownership guard resolves any
        # misidentification to silence.
        # 이 경로에서도 만료 진입 감지가 컨텍스트 검사보다 먼저여야 한다
        # — 수동 /resume 진입의 첫 턴 종료는 바쁨 에지다 (로드만, Stop
        # 신호 없음). 아니면 꽉 찬 선대가 경고해야 할 바로 그 틱에서
        # 재롤오버된다 (실측 — _check_stale_conv_entry 참조). mtime
        # conversation id 로 충분하다 — 오인은 단독 소유 가드가 침묵으로
        # 해석한다.
        conv_id = self._current_conv_id()
        if conv_id is not None:
            self._check_stale_conv_entry(conv_id)
        self._check_summary_refresh()
        self._check_context_usage()
        self._advance_rollover()

    def _check_stale_conv_entry(self, conv_id: str) -> None:
        """Warn once when the turn ran in a rolled-over predecessor (R4-C6 B).

        롤오버된 선대 대화에서 턴이 돌았으면 1회 경고한다 (R4-C6 B).

        Reached from every turn-end path: the Stop signal passes its
        authoritative conversation id, the fallback battery passes the
        mtime-derived one (a /resume load produces no Stop — measured).
        Judgement, not a guarantee: a missed warning is acceptable, a
        false one is not, so every ambiguous shape resolves to silence.

        모든 턴 종료 경로에서 도달한다 — Stop 신호는 신뢰할 수 있는
        conversation id 를, 폴백 일괄은 mtime 유래 id 를 넘긴다
        (/resume 로드는 Stop 을 만들지 않는다 — 실측). 보장이 아닌
        판단의 영역이다 — 경고 누락은 허용되지만 오경고는 안 되므로,
        모호한 형태는 전부 침묵으로 해석한다.

        Ambiguity guard: a conversation linked to MORE than one session
        (an R2-era bridge-link legacy — the producer was removed, see
        session_switch) is skipped, and "latest" is the last SOLE-owned
        id. Verified against the observed legacy shape (e2e run 006e7dfe:
        stray as ids[-1] would otherwise false-warn on the session's real
        latest conversation). Suppression-only use — resume selection
        must NOT use this rule (it would drop a session's real latest,
        penguin-verify 2026-08-15).

        모호성 가드 — 둘 이상의 세션에 링크된 conversation (R2 다리 링크
        레거시 — 생산자는 제거됨, session_switch 참조) 은 건너뛰고,
        "최신" 은 마지막 **단독 소유** id 로 잡는다. 실관측 레거시 형태
        (e2e run 006e7dfe — 떠돌이가 ids[-1] 이면 진짜 최신 대화에
        오경고) 로 검산됨. 경고 억제 전용 규칙이다 — resume 선택에 쓰면
        세션의 진짜 최신 대화를 버리게 되므로 금지 (penguin-verify
        2026-08-15).
        """
        if (
            self._rollover_request_state is not None
            or self._rollover_swap_state is not None
            or self._rollover_ready is not None
            or self._pending_respawn is not None
        ):
            return
        if conv_id in self._stale_conv_warned:
            return
        session_name = self._current_session_name
        if session_name is None:
            return
        try:
            sessions = SessionStore(Path(self.project_path)).list_sessions()
        except Exception:
            return
        current = next((s for s in sessions if s.name == session_name), None)
        if current is None:
            return
        owner_counts: dict[str, int] = {}
        for session in sessions:
            for cid in session.claude_conversation_ids:
                owner_counts[cid] = owner_counts.get(cid, 0) + 1
        sole_ids = [
            cid
            for cid in current.claude_conversation_ids
            if owner_counts.get(cid, 0) == 1
        ]
        if conv_id not in sole_ids or conv_id == sole_ids[-1]:
            return
        self._stale_conv_warned.add(conv_id)
        self._stale_notice_conv = conv_id
        handoff_store.write_notice(
            Path(self.project_path),
            {
                "type": "stale_conversation",
                "session": session_name,
                "conv_id": conv_id,
                "latest_conv": sole_ids[-1],
            },
        )
        self._notify(
            NoticeKind.ROLLOVER,
            "롤오버된 이전 대화에 있습니다 — 다음 입력 때 최신 대화로 "
            "이동할지 묻습니다",
        )
        debug_log.log(
            "STALE_CONV_ENTRY",
            "WRAPPER",
            {"latest_conv": sole_ids[-1]},
            conv_id=conv_id,
            session=session_name,
        )

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
        # The previous child's hook-reported conversation is gone with it.
        # 이전 자식이 hook 으로 알려 준 대화는 자식과 함께 사라졌다.
        self._active_conv_from_hook = None
        self._push_active_conversation()

    def _reported_conv_id(self) -> str | None:
        """The conversation id the wrapper positively knows, or None (F18).

        래퍼가 확실히 아는 대화 id, 없으면 None (F18).

        Hook-reported id for this child first (authoritative — it is what
        Claude Code itself calls the conversation), else the id the
        wrapper assigned at spawn. No transcript gate: for naming and
        linking, the assigned id is valid from spawn (measured:
        ``--session-id`` is honoured). The one consumer that needs
        "has the successor actually appeared?" — rollover finalize —
        checks ``conversation_exists`` itself. Measured need for no gate
        here: the MCP server boots and registers the default session
        before the first turn writes the transcript (2/2,
        docs/poc/R5-conversation-id.md).

        이 자식에 대해 hook 이 보고한 id (권위 — Claude Code 자신이 부르는
        대화 이름), 없으면 spawn 시 래퍼가 지정한 id. transcript 게이트
        없음: 이름 붙이기·연결 용도로는 지정 id 가 spawn 시점부터 유효하다
        (실측: ``--session-id`` 존중). "후계가 실제로 나타났는가" 가 필요한
        유일한 소비처 — 롤오버 finalize — 는 스스로 ``conversation_exists``
        를 검사한다. 게이트를 두지 않는 실측 근거: MCP 서버는 첫 턴이
        transcript 를 쓰기 전에 부팅해 기본 세션을 등록한다 (2/2,
        docs/poc/R5-conversation-id.md).
        """
        return self._active_conv_from_hook or self._assigned_conv_id

    def _current_conv_id(self) -> str | None:
        """Known conversation id first, mtime heuristic last (F18).

        아는 대화 id 먼저, mtime 휴리스틱은 마지막 (F18). 휴리스틱은
        사용자 주도 재개 (``--continue``/``--resume``) 의 첫 턴 종료 전과
        대화 전환 직후 (``/clear`` 등) 에만 닿는다.
        """
        known = self._reported_conv_id()
        if known is not None:
            return known
        return get_active_conversation_id(Path(self.project_path))

    def _forget_conversation(self) -> None:
        """Drop both conversation-id sources after a hand-made switch.

        손수 대화를 바꾼 뒤 두 대화 id 출처를 버린다. 다음 Stop hook 이
        새 대화를 알려 줄 때까지 mtime 폴백으로 내려간다.
        """
        self._active_conv_from_hook = None
        self._assigned_conv_id = None
        self._push_active_conversation()

    def _push_active_conversation(self) -> None:
        """Tell the MCP server the known conversation id when it changes.

        아는 대화 id 가 바뀌면 MCP 서버에 알린다 (F18). 서버는 이 값을
        1차로 쓰고 None 이면 자체 폴백으로 내려간다.
        """
        conv_id = self._reported_conv_id()
        if conv_id == self._pushed_conv_id:
            return
        self._pushed_conv_id = conv_id
        self.socket_server.send(
            {"action": "active_conversation", "conversation_id": conv_id}
        )

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
            elif match_handoff_command(prompt_text):
                # Wrapper-native manual rollover (R4-C4) — same contract.
                # 래퍼 자체 수동 롤오버 (R4-C4) — 동일 계약.
                self._handle_handoff_command()
                return
            elif match_sessions_command(prompt_text):
                # Wrapper-native instant listing (R5-C2) — same contract.
                # 래퍼 자체 즉시 목록 (R5-C2) — 동일 계약.
                self._handle_sessions_command()
                return
            elif prompt_text and CLEAR_COMMAND_RE.match(prompt_text.strip()):
                # /clear wipes the conversation — summarise it while it's
                # still there.
                # /clear 는 대화를 지운다 — 아직 남아 있을 때 요약한다.
                self._enqueue_active_summary()
                # ...and Claude Code issues a NEW session id for what
                # follows (measured: SessionStart source=clear with a
                # fresh id, docs/poc/R5-conversation-id.md) — the ids the
                # wrapper held are stale from here on.
                # ...그리고 Claude Code 는 이후 대화에 **새** session id 를
                # 발급한다 (실측: SessionStart source=clear 에 새 id,
                # docs/poc/R5-conversation-id.md) — 래퍼가 쥔 id 는 여기서
                # 부터 낡았다.
                self._forget_conversation()

            # Forwarded \r = a turn submit: start mark for the turn
            # duration (end mark = the Stop-hook signal). A \r on an
            # empty input line just moves the mark — harmless for a
            # measurement-only value.
            # 전달되는 \r = 턴 제출 — 턴 지속시간의 시작점 (끝점 = Stop
            # hook 신호). 빈 입력란의 \r 은 시작점만 옮길 뿐이라 계측
            # 전용 값에는 무해하다.
            self._last_submit_monotonic = time.monotonic()

            # Move-or-stay deferral, second release path (R4-C6 B): a
            # submit AFTER the notice was consumed means the question
            # attempt is behind us — normally the Stop released already,
            # but a conversation at its hard context limit cannot run
            # the question turn at all (measured: "Context limit
            # reached", no Stop ever fires) and would defer the rollover
            # forever. The user's next keystroke breaks that deadlock.
            # 이동/계속 유예의 제2 해제 경로 (R4-C6 B) — notice 소비
            # **후** 의 제출은 질문 시도가 이미 지나갔다는 뜻이다.
            # 보통은 Stop 이 먼저 해제하지만, 하드 컨텍스트 한계의
            # 대화는 질문 턴 자체가 돌지 못해 (실측: "Context limit
            # reached", Stop 무발화) 롤오버가 영원히 유예된다. 사용자의
            # 다음 제출이 그 교착을 푼다.
            if self._stale_notice_conv is not None and not (
                handoff_store.notice_pending(Path(self.project_path))
            ):
                self._stale_notice_conv = None

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
        # Leaving the conversation by hand (/resume, /exit, /new): what
        # the wrapper knew about "the current conversation" no longer
        # holds — forget both sources; the next Stop hook names the new
        # one. /rename stays in the same conversation and keeps them.
        # 손수 대화를 떠남 (/resume, /exit, /new): 래퍼가 알던 "현재
        # 대화" 는 더 이상 유효하지 않다 — 두 출처를 잊고 다음 Stop hook
        # 이 새 대화를 알려 주게 한다. /rename 은 같은 대화에 머무르므로
        # 유지한다.
        if matched.command != "rename":
            self._forget_conversation()
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

    def _notify(self, kind: NoticeKind, text: str) -> None:
        """Print one router intervention in the shared notice grammar (R5-C2).

        공통 알림 문법으로 라우터 개입 한 줄을 찍는다 (R5-C2). 기호는
        ``notice.NoticeKind`` 가 단일 출처다.
        """
        self._notify_user(format_notice(kind, text))

    def _notify_block(self, lines: list[str]) -> None:
        """Print a multi-line wrapper block (first line carries the prefix).

        여러 줄 래퍼 블록을 찍는다 (첫 줄에만 접두사).
        """
        if self._stdout_fd < 0 or not lines:
            return
        body = "\r\n".join(lines)
        try:
            os.write(self._stdout_fd, f"\r\n[session-manager] {body}\r\n".encode())
        except OSError:
            pass

    def _handle_sessions_command(self) -> None:
        """``/sessions``: list sessions instantly, no LLM round-trip (R5-C2).

        ``/sessions`` — LLM 왕복 없이 세션 목록을 즉시 찍는다 (R5-C2).
        Rows are clipped to the terminal width when it is known.
        터미널 폭을 알면 행을 그 폭에 맞춰 자른다.
        """
        try:
            os.write(self.pty_fd, ERASE_INPUT_LINE)
        except OSError:
            pass
        try:
            sessions = SessionStore(Path(self.project_path)).list_sessions()
        except Exception as exc:
            self._notify_user(f"세션 목록을 읽지 못했습니다: {exc}")
            return
        width: int | None = None
        try:
            width = os.get_terminal_size(self._stdout_fd).columns
        except (OSError, ValueError):
            pass
        self._notify_block(
            format_session_list(sessions, self._current_session_name, width)
        )
        debug_log.log(
            "SESSIONS_LIST",
            "USER",
            {"count": len(sessions)},
            session=self._current_session_name,
        )

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
        self._notify(
            NoticeKind.BACK, f"{origin} 세션으로 되돌립니다 (직전: {wrong})"
        )
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

    def _set_current_session(self, name: str | None) -> None:
        """Move the wrapper-side session mirror and persist it (R5-C1).

        The persisted copy in ``state.json`` is what the statusline
        process reads — it has no other way to learn the session name.
        Persistence is best-effort: a failed write is logged inside
        ``wrapper_state`` and the statusline just omits that segment.

        래퍼 측 세션 미러를 옮기고 영속화한다 (R5-C1). ``state.json``
        의 사본은 statusline 프로세스가 읽는 값이다 — 세션 이름을 알
        다른 길이 없다. 영속화는 best-effort: 쓰기 실패는
        ``wrapper_state`` 안에서 로그만 남고, 표시줄은 그 세그먼트를
        생략할 뿐이다.
        """
        self._current_session_name = name
        wrapper_state.save_current_session(Path(self.project_path), name)

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
        conv_id = self._current_conv_id()
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
        conv_id = self._current_conv_id()
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
        conv_id = self._current_conv_id()
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

        Entry check (R4-C6 A): when the armed arrival conversation gets
        its first conclusive reading here, a marking carries the entry
        notice; any conclusive non-marking outcome just disarms. An
        inconclusive reading (no usage data, unknown birth) keeps the
        arming so a deferred marking still announces the arrival.

        진입 검사 (R4-C6 A) — 무장된 도착 conversation 의 첫 확정 판독이
        여기서 이뤄지면, 마킹에는 진입 안내가 붙고 마킹 없는 확정 결과는
        무장만 해제한다. 미확정 판독 (usage 없음·태생 미상) 은 무장을
        유지해 연기된 마킹도 도착을 안내한다.
        """
        project = Path(self.project_path)
        conv_id = self._current_conv_id()
        if conv_id is None:
            return
        # Move-or-stay deferral (R4-C6 B): while the stale-entry question
        # for THIS conversation awaits its ANSWER, marking a rollover
        # would fork the lineage before the user could choose to move.
        # Cleared only on a Stop signal (_handle_turn_end_signal):
        # AskUserQuestion is a tool, so the question turn cannot end
        # before the user answers — whereas the fallback edges fire
        # MID-turn (measured: context_update 6s after the warning marked
        # the predecessor and the rollover respawn killed the dialog).
        # 이동/계속 유예 (R4-C6 B) — 이 conversation 의 만료 진입 질문이
        # **답변**을 기다리는 동안 롤오버를 마킹하면 사용자가 이동을
        # 택하기 전에 계보가 분기한다. 해제는 Stop 신호에서만
        # (_handle_turn_end_signal) — AskUserQuestion 은 도구라 질문 턴은
        # 답변 전에 끝날 수 없다. 반면 폴백 에지는 턴 **도중**에 발화한다
        # (실측 — 경고 6초 뒤 context_update 가 선대를 마킹해 롤오버
        # respawn 이 다이얼로그를 죽였다).
        if self._stale_notice_conv == conv_id:
            return
        if conv_id != self._rollover_pending_conv_id:
            self._rollover_pending_conv_id = None
        if (
            self._entry_check_conv_id is not None
            and conv_id != self._entry_check_conv_id
        ):
            # The conversation moved on (another switch, /clear) before
            # a conclusive reading — the arrival is stale.
            # 확정 판독 전에 conversation 이 바뀌었다 (다른 전환·/clear)
            # — 도착 무장이 낡았다.
            self._entry_check_conv_id = None
        jsonl_path = (
            Path.home()
            / ".claude"
            / "projects"
            / encode_cwd(project)
            / f"{conv_id}.jsonl"
        )
        usage = context_monitor.check_context_usage(project, conv_id, jsonl_path)
        if usage is None:
            return
        if not usage.exceeded:
            # Conclusive: the arrived conversation is not full.
            # 확정 — 도착한 conversation 은 안 찼다.
            self._entry_check_conv_id = None
            return
        if self._rollover_pending_conv_id == conv_id:
            self._entry_check_conv_id = None
            return
        # Loop guard: a rollover cannot shrink a conversation below its
        # birth footprint. If that already meets the trigger, a successor
        # would be born equally full and the wrapper would roll over
        # forever (observed — R4-C4 e2e with a too-low budget override:
        # 4 rollovers in one run). Suppress and log instead; the log is
        # R5's measurement source for misconfigured thresholds.
        # 루프 가드 — 롤오버는 대화를 태생 점유량 아래로 줄일 수 없다.
        # 태생 점유가 이미 트리거 이상이면 후계도 똑같이 찬 채 태어나
        # 영원히 반복된다 (실관측 — 과소 budget override 의 R4-C4 e2e 에서
        # 1회 실행에 롤오버 4번). 억제하고 로그만 남긴다 — 이 로그가 R5
        # 의 임계 오설정 계측 원천이다.
        if self._rollover_suppressed_conv_id == conv_id:
            self._entry_check_conv_id = None
            return
        birth = context_monitor.read_first_usage(jsonl_path)
        if birth is None:
            # Unknown birth: the numerator arrived (statusline) but the
            # transcript's first assistant event is not flushed yet (the
            # measured flush race). Marking now would bypass the guard —
            # defer; the next turn-end signal retries with the event
            # present. (Observed: a successor rolled over through this
            # window — 3 conversations from one /handoff.)
            # 태생 미상 — 분자 (statusline) 는 왔는데 transcript 의 첫
            # assistant 이벤트가 아직 flush 전 (실측된 flush 레이스).
            # 지금 마킹하면 가드가 우회된다 — 미룬다. 다음 턴 종료 신호가
            # 이벤트가 실린 뒤 재시도한다. (실관측: 이 창으로 후계가
            # 롤오버돼 /handoff 1회에 대화 3개.)
            return
        if birth >= usage.trigger_tokens:
            # Conclusive for the entry check too: a rollover cannot help
            # this conversation, so no entry notice — log only (approved
            # C6-A design; the suppression log is R5's source).
            # 진입 검사에도 확정 — 이 conversation 은 롤오버로 개선 불가
            # 이므로 진입 안내 없음, 로그만 (승인된 C6-A 설계. 억제
            # 로그는 R5 계측 원천).
            self._entry_check_conv_id = None
            self._rollover_suppressed_conv_id = conv_id
            debug_log.log(
                "ROLLOVER_SUPPRESSED",
                "WRAPPER",
                {
                    "reason": "birth_exceeds_trigger",
                    "birth_tokens": birth,
                    "trigger_tokens": usage.trigger_tokens,
                },
                conv_id=conv_id,
                session=self._current_session_name,
            )
            return
        entry_arrival = self._entry_check_conv_id == conv_id
        self._entry_check_conv_id = None
        self._rollover_pending_conv_id = conv_id
        if usage.window_tokens:
            self._rollover_pending_pct = round(
                usage.used_tokens * 100 / usage.window_tokens
            )
        if entry_arrival:
            # Spec wording (Plan §5 R4-C6) — the user just landed here
            # via a transition and the session is already full.
            # 스펙 원문 (Plan §5 R4-C6) — 방금 전환으로 도착했는데
            # 세션이 이미 차 있다.
            self._notify(
                NoticeKind.ROLLOVER,
                "그 세션은 컨텍스트가 거의 찼습니다 — "
                "Handoff로 이어서 새 대화로 재개합니다",
            )
        debug_log.log(
            "ROLLOVER_PENDING",
            "WRAPPER",
            {
                "used_tokens": usage.used_tokens,
                "window_tokens": usage.window_tokens,
                "trigger_tokens": usage.trigger_tokens,
                "numerator_source": usage.numerator_source,
                "denominator_source": usage.denominator_source,
                "entry_arrival": entry_arrival,
            },
            conv_id=conv_id,
            session=self._current_session_name,
        )

    def _advance_rollover(self) -> None:
        """Rollover machine, driven by turn-end edges (R4-C3).

        턴 종료 에지가 구동하는 롤오버 머신 (R4-C3).

        Two legs / 두 단계:
        1. A conversation is marked pending and nothing is in flight →
           start the dedicated handoff turn (same-conversation respawn;
           the request text rides the pending file as user_prompt, so
           the existing trigger/injection path needs no changes).
           pending 마킹이 있고 진행 중인 것이 없으면 → handoff 전용 턴
           개시 (같은 conversation respawn. 요청문은 pending 파일의
           user_prompt 로 실리므로 기존 트리거·주입 경로 무변경).
        2. The dedicated turn just ended → extract the response from the
           transcript, validate, write the file; one retry, then the
           excerpt fallback. Acting on the ready handoff is R4-C4.
           전용 턴이 방금 끝났으면 → transcript 에서 응답 추출·검증·파일
           기록. 재시도 1회, 그 다음은 발췌 폴백. ready handoff 에 대한
           행동은 R4-C4.
        """
        state = self._rollover_request_state
        if state is not None and state.get("phase") == "writing":
            self._finish_handoff_turn(state)
            return
        if state is not None:
            return  # requested — the dedicated turn has not spawned yet
        if self._rollover_ready is not None:
            # Ready but the swap could not register earlier (another
            # transition was in flight) — retry the swap leg.
            # ready 인데 교체 등록이 밀렸다 (다른 전환 진행 중) — 교체
            # 단계 재시도.
            self._start_rollover_swap()
            return
        if self._rollover_swap_state is not None or self._pending_respawn is not None:
            return
        conv_id = self._rollover_pending_conv_id
        if conv_id is None:
            return
        if conv_id != self._current_conv_id():
            return
        session_name = self._current_session_name
        if session_name is None:
            return
        self._start_handoff_turn(session_name, conv_id, attempts=1)

    def _start_handoff_turn(
        self, session_name: str, conv_id: str, attempts: int
    ) -> None:
        project = Path(self.project_path)
        try:
            session = SessionStore(project).load_session_by_name(session_name)
        except Exception:
            session = None
        requirements = list(session.requirements) if session else []
        n = rollover.next_handoff_number(project, session_name)
        request = rollover.build_request(
            project, session_name, n, conv_id, requirements
        )
        if attempts == 1:
            pct = self._rollover_pending_pct
            self._notify(
                NoticeKind.ROLLOVER,
                f"컨텍스트 {pct}% — 세션을 이어갈 준비를 합니다"
                if pct is not None
                else "컨텍스트 한계 근접 — 세션을 이어갈 준비를 합니다",
            )
        self._rollover_request_state = {
            "session": session_name,
            "n": n,
            "conv_id": conv_id,
            "attempts": attempts,
            "phase": "requested",
            # Validation anchor: only transcript events after this moment
            # belong to the dedicated turn (see rollover.check_trigger_turn).
            # 검증 앵커 — 이 시각 이후의 transcript 이벤트만 전용 턴의
            # 것이다 (rollover.check_trigger_turn 참조).
            "request_at": wrapper_state.utc_now_iso(),
        }
        debug_log.log(
            "ROLLOVER_HANDOFF",
            "WRAPPER",
            {"op": "request", "n": n, "attempts": attempts},
            conv_id=conv_id,
            session=session_name,
        )
        self._execute_transition(
            target=session_name,
            resume_conv=conv_id,
            handoff={"kind": "rollover_handoff_request", "from": session_name},
            user_prompt=request,
            is_rollover_request=True,
        )

    def _finish_handoff_turn(self, state: dict[str, Any]) -> None:
        """Validate the dedicated turn's output; retry once, then fallback.

        전용 턴 산출물을 검증한다 — 재시도 1회, 그 다음 발췌 폴백.

        Anchored on transcript content, not screen edges: the dedicated
        child's boot produces a spurious falling edge before the trigger
        is delivered (measured race — the first e2e burned both attempts
        on the previous conversation's reply). "waiting" keeps the
        attempt alive; only a delivered-and-answered or lost request
        consumes it.

        화면 에지가 아니라 transcript 내용에 앵커한다: 전용 자식의 부팅이
        트리거 전달 전에 가짜 하강 에지를 만든다 (실측 레이스 — 첫 e2e 가
        직전 대화의 응답으로 시도 2회를 전부 태웠다). "waiting" 은 시도를
        보존하고, 전달·응답 완료 또는 유실만 시도를 소진한다.
        """
        jsonl_path = (
            Path.home()
            / ".claude"
            / "projects"
            / encode_cwd(Path(self.project_path))
            / f"{state['conv_id']}.jsonl"
        )
        status, text = rollover.check_trigger_turn(
            jsonl_path,
            handoff_store.TRIGGER_PROMPT,
            state.get("request_at", ""),
        )
        if status == "waiting":
            return
        self._conclude_handoff_attempt(state, status, text)

    def _conclude_handoff_attempt(
        self, state: dict[str, Any], status: str, text: str
    ) -> None:
        """Shared attempt conclusion: write / retry once / excerpt fallback.

        시도 판정 공용부 — 기록 / 재시도 1회 / 발췌 폴백.

        Called with a settled status ("answered" or "missing") from
        either delivery path: the Stop-hook signal (primary — the text
        arrives in the payload) or the transcript anchor (fallback).

        확정 status ("answered"·"missing") 로 두 전달 경로에서 호출된다:
        Stop hook 신호 (주 — 본문이 payload 로 도착) 또는 transcript
        앵커 (폴백).
        """
        project = Path(self.project_path)
        session_name = state["session"]
        conv_id = state["conv_id"]
        if status == "answered" and rollover.validate_handoff_text(text):
            path = rollover.write_handoff(project, session_name, state["n"], text)
            self._rollover_request_state = None
            self._rollover_ready = {**state, "path": str(path)}
            self._notify(
                NoticeKind.ROLLOVER,
                f"Handoff 준비 완료 — {path.relative_to(project)}",
            )
            self._start_rollover_swap()
            return
        if state["attempts"] < 2:
            debug_log.log(
                "ROLLOVER_HANDOFF",
                "WRAPPER",
                {
                    "op": "retry",
                    "reason": "delivery_lost"
                    if status == "missing"
                    else "validation_failed",
                },
                conv_id=conv_id,
                session=session_name,
            )
            self._rollover_request_state = None
            self._start_handoff_turn(session_name, conv_id, attempts=2)
            return
        # Two failures: excerpt fallback — low quality, but the rollover
        # proceeds (Plan R4-C3).
        # 2회 실패: 발췌 폴백 — 품질은 낮아도 롤오버는 진행 (Plan R4-C3).
        excerpt = extract_full_text(
            Path.home()
            / ".claude"
            / "projects"
            / encode_cwd(project)
            / f"{conv_id}.jsonl"
        )
        body = rollover.build_fallback_handoff(
            session_name, state["n"], conv_id, excerpt
        )
        path = rollover.write_handoff(project, session_name, state["n"], body)
        debug_log.log(
            "ROLLOVER_HANDOFF",
            "WRAPPER",
            {"op": "fallback", "path": str(path)},
            conv_id=conv_id,
            session=session_name,
        )
        self._rollover_request_state = None
        self._rollover_ready = {**state, "path": str(path)}
        self._notify(
            NoticeKind.ROLLOVER,
            f"Handoff 준비 완료 (발췌 폴백) — {path.relative_to(project)}",
        )
        self._start_rollover_swap()

    def _start_rollover_swap(self) -> None:
        """Consume the ready handoff: swap to a NEW conversation (R4-C4).

        ready handoff 를 소비해 새 conversation 으로 교체한다 (R4-C4).

        Same respawn path as every transition; ``resume_conv=None`` boots
        the successor. Atomicity (§5.4-g): nothing about the predecessor
        is touched here — linking, precedent clearing and the summary
        refresh all wait for the successor to be OBSERVED (finalize), so
        a failure in between leaves the predecessor fully active.

        모든 전환과 같은 respawn 경로. ``resume_conv=None`` 이 후계를
        부팅한다. 원자성 (§5.4-g): 여기서는 선대를 일절 건드리지 않는다 —
        link·판례 소멸·요약 갱신 전부 후계가 **관찰된** 뒤 (finalize) 로
        미루므로, 중간 실패 시 선대는 온전히 활성으로 남는다.
        """
        ready = self._rollover_ready
        if ready is None or self._pending_respawn is not None:
            return
        project = Path(self.project_path)
        handoff, prompt = rollover.successor_injection(
            project, ready["session"], ready["n"]
        )
        self._rollover_ready = None
        self._rollover_swap_state = {
            "session": ready["session"],
            "n": ready["n"],
            "path": ready["path"],
            "predecessor_conv": ready["conv_id"],
        }
        self._notify(NoticeKind.ROLLOVER, "롤오버 — 새 대화로 이어갑니다")
        self._execute_transition(
            target=ready["session"],
            resume_conv=None,
            handoff=handoff,
            user_prompt=prompt,
            is_rollover_swap=True,
        )

    def _poll_rollover_finalize(self) -> None:
        """Finalize the rollover once the successor conversation appears.

        후계 conversation 이 나타나면 롤오버를 마무리한다.

        Entry confirmation = a new active conversation id (the successor
        writes its transcript on its trigger turn). Deterministic
        bookkeeping, not delegated to the LLM: link the successor to the
        SAME session, clear the session's precedents (invalidation event
        ③ — the rollover changes the session's topical make-up), refresh
        the summary from the new conversation, and drop every rollover
        mark so detection restarts cleanly for the successor.

        진입 확인 = 새 활성 conversation id (후계는 트리거 턴에서
        transcript 를 쓴다). LLM 에 맡기지 않는 결정적 부기 — 후계를
        **같은** 세션에 link, 세션 판례 소멸 (무효화 이벤트 ③ — 롤오버는
        세션의 주제 구성을 바꾼다), 새 conversation 기준 요약 갱신, 롤오버
        마킹 전부 해제 (후계에 대한 감지가 깨끗이 재시작).
        """
        state = self._rollover_swap_state
        if state is None:
            return
        conv_id = self._current_conv_id()
        if conv_id is None or conv_id == state["predecessor_conv"]:
            return
        if (
            conv_id == self._assigned_conv_id
            and not self._active_conv_from_hook
            and not conversation_exists(Path(self.project_path), conv_id)
        ):
            # The wrapper named the successor at spawn, but "appeared"
            # means its transcript exists (§5.4-g atomicity: nothing about
            # the predecessor changes until the successor is observed).
            # 래퍼가 spawn 때 후계 이름을 정했지만 "나타남" 은 transcript
            # 존재를 뜻한다 (§5.4-g 원자성: 후계가 관측되기 전엔 선대에
            # 대해 아무것도 바꾸지 않는다).
            return
        self._finalize_rollover(conv_id)

    def _finalize_rollover(self, conv_id: str) -> None:
        """Run the finalize bookkeeping for observed successor *conv_id*.

        관찰된 후계 *conv_id* 에 대해 finalize 부기를 수행한다.

        Reached from either entry-confirmation path: the successor's own
        Stop-hook signal (primary — carries its conversation id) or the
        mtime-poll fallback above.

        두 진입 확인 경로 어느 쪽에서든 도달한다: 후계 자신의 Stop hook
        신호 (주 — conversation id 를 실어 옴) 또는 위의 mtime 폴링 폴백.
        """
        state = self._rollover_swap_state
        if state is None or conv_id == state["predecessor_conv"]:
            return
        project = Path(self.project_path)
        try:
            # Link the predecessor too: the lineage (§1.4) must not
            # depend on the LLM having called an MCP tool in the old
            # conversation (observed missing in e2e). Idempotent —
            # chronological order predecessor → successor.
            # 선대도 link 한다 — 계보 (§1.4) 가 옛 대화에서의 MCP 도구
            # 호출 여부에 의존하면 안 된다 (e2e 에서 누락 실관측). 멱등,
            # 시간순 선대 → 후계.
            SessionStore(project).mutate_session_by_name(
                state["session"],
                lambda s: (
                    s.link_conversation(state["predecessor_conv"]),
                    s.link_conversation(conv_id),
                    s.clear_precedents(),
                ),
            )
        except Exception as exc:
            # Linking is retried on the next tick only if the state is
            # kept; a persistent storage failure must not loop forever —
            # log and finish. Backstop: the successor gets linked at the
            # session's next departure (session_switch/create/end all
            # link the active conversation to the CURRENT session) —
            # there is no per-tool-call tracking (verified 2026-08-15).
            # link 실패를 영구 재시도하면 안 된다 — 로그 후 마무리.
            # 백스톱: 이 세션의 다음 이탈 시점에 후계가 link 된다
            # (session_switch/create/end 가 활성 conversation 을 현재
            # 세션에 link) — 도구 호출마다의 범용 추적은 존재하지 않는다
            # (2026-08-15 검증).
            debug_log.log(
                "ROLLOVER_COMPLETE",
                "WRAPPER",
                {"result": "link_failed", "error": str(exc)},
                conv_id=conv_id,
                session=state["session"],
            )
        self._enqueue_active_summary()
        self._rollover_swap_state = None
        self._rollover_pending_conv_id = None
        self._rollover_pending_pct = None
        debug_log.log(
            "ROLLOVER_COMPLETE",
            "WRAPPER",
            {
                "predecessor_conv": state["predecessor_conv"],
                "successor_conv": conv_id,
                "handoff_path": state["path"],
                "n": state["n"],
            },
            conv_id=conv_id,
            session=state["session"],
        )

    def _handle_handoff_command(self) -> None:
        """Manual rollover: /handoff marks the active conversation now.

        수동 롤오버 — /handoff 가 활성 conversation 을 즉시 마킹한다.

        Same flow as the threshold trigger (R4-C1) from the mark onward;
        the only difference is who decided. Refused with a notice while
        another rollover step or transition is in flight.

        마킹 이후는 임계 트리거 (R4-C1) 와 동일 흐름 — 다른 점은 결정
        주체뿐. 다른 롤오버 단계·전환이 진행 중이면 안내 후 거절한다.
        """
        try:
            os.write(self.pty_fd, ERASE_INPUT_LINE)
        except OSError:
            pass
        if (
            self._rollover_request_state is not None
            or self._rollover_swap_state is not None
            or self._rollover_ready is not None
            or self._pending_respawn is not None
        ):
            self._notify_user("롤오버가 이미 진행 중입니다")
            return
        conv_id = self._current_conv_id()
        if conv_id is None or self._current_session_name is None:
            self._notify_user("롤오버할 활성 대화가 없습니다")
            return
        self._rollover_pending_conv_id = conv_id
        self._rollover_pending_pct = None
        debug_log.log(
            "ROLLOVER_PENDING",
            "WRAPPER",
            {"trigger": "manual_handoff_command"},
            conv_id=conv_id,
            session=self._current_session_name,
        )
        self._advance_rollover()

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
            self._notify(
                NoticeKind.SWITCH,
                f"{target} 세션으로 전환됨 (이전: {origin}) — 되돌리려면 /back",
            )
        elif action == "turn_end":
            # Stop hook (contract-based turn end): the PRIMARY turn-end
            # signal. Screen edges / context.json observation remain as
            # fallbacks for hook-declined users.
            # Stop hook (계약 기반 턴 종료) — **주** 턴 종료 신호. 화면
            # 에지·context.json 관찰은 hook 미동의 사용자용 폴백으로
            # 유지된다.
            self._handle_turn_end_signal(message)
        elif action == "rollover_signal":
            # PreCompact hook (R4-C2): auto-compact was blocked in this
            # conversation — mark the rollover pending immediately. A
            # second trigger converging with the R4-C1 threshold check;
            # acting on the mark is R4-C3/C4.
            # PreCompact hook (R4-C2): 이 conversation 의 auto-compact 가
            # 차단됐다 — 즉시 롤오버 pending 마킹. R4-C1 임계 검사와
            # 합류하는 제2 트리거이며, 마킹에 대한 행동은 R4-C3/C4.
            conv_id = message.get("conversation_id")
            if not isinstance(conv_id, str) or not conv_id:
                conv_id = self._current_conv_id()
            if conv_id is None or self._rollover_pending_conv_id == conv_id:
                return
            self._rollover_pending_conv_id = conv_id
            debug_log.log(
                "ROLLOVER_PENDING",
                "WRAPPER",
                {"trigger": "pre_compact_hook"},
                conv_id=conv_id,
                session=self._current_session_name,
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
                # While a transition is registered, the wrapper mirror is
                # authoritative: the MCP's pointer may lag behind the
                # registered target (session_switch signals its
                # original target right after the switch signal — R4-C5
                # e2e caught the mirror being clobbered back to a stale
                # session). The post-respawn handshake re-syncs
                # the fresh MCP from this mirror, so dropping the stale
                # update heals both sides.
                # 전환이 등록된 동안은 래퍼 미러가 권위다 — MCP 포인터는
                # 등록된 target 보다 뒤처질 수 있다 (session_switch 는
                # switch 신호 직후 자신의 원래 target 을 통보한다 — R4-C5
                # e2e 가 미러가 낡은 세션으로 되돌려지는 것을 잡았다). respawn
                # 후 핸드셰이크가 새 MCP 를 이 미러로 재동기화하므로,
                # stale 통보를 버리면 양쪽이 치유된다.
                if (
                    self._pending_respawn is not None
                    and name != self._pending_respawn.target
                ):
                    debug_log.log(
                        "CURRENT_SESSION",
                        "MCP_TOOL",
                        {
                            "result": "stale_ignored",
                            "name": name,
                            "pending_target": self._pending_respawn.target,
                        },
                        session=self._current_session_name,
                    )
                    return
                before = self._current_session_name
                self._set_current_session(name)
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
        is_rollover_request: bool = False,
        is_rollover_swap: bool = False,
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
        # Every transition path converges here, so this is the one place
        # an ARCHIVED target comes back to life: using an ended session
        # again IS un-ending it. NEW targets (not in the store yet) and
        # store errors pass through — reactivation is a convenience, not
        # a gate.
        # 모든 전환 경로가 여기로 수렴하므로 ARCHIVED 대상이 되살아나는
        # 곳도 여기 하나다 — 끝낸 세션을 다시 쓰는 것이 곧 끝남 해제다.
        # NEW 대상 (스토어에 아직 없음) 과 스토어 오류는 통과 — 복귀는
        # 편의이지 관문이 아니다.
        self._reactivate_target(target)
        from_name = handoff.get("from")
        # Queue a background summary for the departing session while its
        # conversation is still the active one, and move the wrapper-side
        # current-session mirror to the target. A rollover handoff
        # request departs nothing (same session, same conversation), so
        # no summary is queued for it.
        # 떠나는 세션의 conversation 이 아직 활성인 시점에 백그라운드
        # 요약을 큐에 넣고, 래퍼 측 현재 세션 미러를 target 으로 이동.
        # 롤오버 handoff 요청은 떠나는 것이 없으므로 (같은 세션·같은
        # conversation) 요약을 큐에 넣지 않는다.
        if not (is_rollover_request or is_rollover_swap):
            # A rollover swap departs its own predecessor — its summary
            # refresh happens at finalize from the successor instead.
            # 롤오버 교체는 자기 선대를 떠난다 — 요약 갱신은 finalize 가
            # 후계 기준으로 수행한다.
            self._enqueue_departed_summary(from_name)
        self._set_current_session(target)
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
            is_rollover_request=is_rollover_request,
            is_rollover_swap=is_rollover_swap,
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

    def _reactivate_target(self, target: str) -> None:
        """Return an ARCHIVED *target* to ACTIVE (best-effort).

        ARCHIVED 인 *target* 을 ACTIVE 로 되돌린다 (best-effort).
        """
        try:
            changed = SessionStore(Path(self.project_path)).mutate_session_by_name(
                target, lambda s: s.reactivate()
            )
        except Exception as exc:
            debug_log.log(
                "TRANSITION",
                "WRAPPER",
                {"op": "reactivate", "result": "error", "error": str(exc)},
                session=target,
            )
            return
        if changed is not None and changed.status == SessionStatus.ACTIVE:
            debug_log.log(
                "TRANSITION",
                "WRAPPER",
                {"op": "reactivate", "target": target},
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
        conv_id = self._reported_conv_id()
        self._pushed_conv_id = conv_id
        debug_log.log(
            "HANDSHAKE",
            "WRAPPER",
            {
                "phase": "wrapper_response",
                "current_session_name": name,
                "conversation_id": conv_id,
            },
        )
        self.socket_server.send(
            {"current_session_name": name, "conversation_id": conv_id}
        )

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
