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

import os
import re
import select
import signal
import sys
import termios
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pexpect

from session_manager import debug_log, summarizer
from session_manager.claude_conversation import (
    encode_cwd,
    get_active_conversation_id,
    get_conversation_activity,
)
from session_manager.lifecycle import get_cleanup_period_days
from session_manager.storage.file_store import SessionStore
from session_manager.summarizer import SummarizerWorker, SummaryTask
from session_manager.transcript_excerpt import EXCERPT_MAX_CHARS, scan_dialogue_growth
from session_manager.wrapper.command_matcher import (
    InterceptedCommand,
    match_intercept_command,
)
from session_manager.wrapper.handoff_formatter import format_handoff_injection
from session_manager.wrapper.socket_server import WrapperSocketServer
from session_manager.wrapper.virtual_screen import VirtualScreen

# figures.pointer "❯" (UTF-8 E2 9D AF) followed by chalk.inverse "\x1b[7m"
# is what ink-text-input renders for the prompt cursor. Verified against
# the Claude Code binary (strings shows both \u276F and inverse: [7, 27]).
#
# ink-text-input이 프롬프트 커서를 그릴 때 쓰는 패턴. "❯"(UTF-8 E2 9D AF)
# 뒤에 chalk.inverse "\x1b[7m"이 이어지는 형태. Claude Code 바이너리에서
# \u276F와 inverse: [7, 27] 정의를 strings로 확인.
PROMPT_POINTER = b"\xe2\x9d\xaf"
INVERSE_VIDEO_START = b"\x1b[7m"

# Cap the prompt-detection buffer to prevent unbounded growth in long
# sessions. When truncating, keep enough tail so an in-flight multi-byte
# sequence at the boundary isn't sliced apart.
#
# 장시간 세션에서 프롬프트 감지 버퍼 무한 증가 방지. 잘라낼 때는
# 끝부분을 충분히 남겨 경계에 걸친 멀티바이트 시퀀스 보존.
OUTPUT_BUFFER_CAP = 16 * 1024
OUTPUT_BUFFER_TAIL_KEEP = 256

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
    "I am using this for local development",  # channels dev 경고
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

Mode = Literal["passthrough", "filtering"]


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
class _PendingAction:
    """
    Tracks an in-progress SWITCH or NEW action across multiple prompt events.

    여러 번의 프롬프트 감지 이벤트에 걸쳐 진행되는 SWITCH/NEW 액션 상태를
    보관한다. stage 값은 다음 프롬프트에서 어떤 단계로 진입할지 결정한다.
    """

    action_type: Literal["switch", "new"]
    target: str
    handoff: dict[str, Any]
    user_prompt: str
    stage: str
    # NEW 전용. SWITCH일 때는 기본값 그대로.
    rename_current: str | None = None
    new_session_name: str = ""


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

        self.mode: Mode = "passthrough"
        self.output_buffer: bytes = b""
        self.input_queue: bytes = b""
        self.stdin_line_buffer: bytes = b""

        # 테스트 환경이나 stdin/stdout 이 redirect 된 경우 fileno() 가 실패할 수
        # 있으므로 안전하게 -1 로 폴백. 실런타임에서는 isatty/-1 검사로 가드.
        self._stdin_fd: int = _safe_fileno(sys.stdin)
        self._stdout_fd: int = _safe_fileno(sys.stdout)
        self._original_stdin_attrs: list[Any] | None = None
        self._previous_winch_handler: Any = None

        self.socket_server = WrapperSocketServer(
            socket_path=socket_path,
            on_message=self._handle_mcp_signal,
        )

        # Virtual terminal screen mirroring Claude Code's PTY output. Used to
        # extract the live input prompt text (the line containing ❯) when
        # the user submits a slash command.
        # Claude Code의 PTY 출력을 미러링하는 가상 터미널 화면. 사용자가
        # 슬래시 명령을 submit한 시점의 입력란 텍스트(❯ 라인) 추출에 사용.
        self.virtual_screen = VirtualScreen()

        self._pending_action: _PendingAction | None = None

        # Confirmation patterns already auto-accepted in the current child.
        # Reset on each spawn so a respawned child re-arms the auto-accept.
        # 현재 자식에서 이미 자동 승인한 confirmation 패턴.
        # 새 자식이 spawn될 때마다 초기화해 자동 승인을 재무장.
        self._handled_confirmations: set[str] = set()

        # AGENT_GUIDE.md injection stage machine. (Re)set inside _spawn_child
        # based on whether the new child receives a fresh conversation.
        # Stages must be split across separate prompt-detect cycles because
        # injecting text and sending \r in the same cycle looks like a paste
        # to Ink's TextInput, and Ink treats paste \r as a newline (no submit).
        # AGENT_GUIDE 주입 stage 머신. _spawn_child에서 갱신.
        # inject와 \r submit을 같은 prompt-detect 사이클에 보내면 Ink가 paste로
        # 인식해 \r을 줄바꿈으로 처리(submit 안 됨)하므로, prompt-detect 사이클
        # 사이에 분리 — SWITCH/NEW의 stage 패턴과 동일.
        self._agent_guide_stage: Literal[
            "needed", "injected", "submitted", "done"
        ] = "done"

        # time.monotonic() of the most recent auto-accept fire. Used by the
        # AGENT_GUIDE inject guard to wait a brief cooldown after a
        # confirmation gets accepted, so Claude Code has time to redraw
        # the screen without the confirmation menu before we inject.
        # 마지막 auto-accept 발동 시각. AGENT_GUIDE inject 가드가 자동 승인
        # 직후의 redraw race를 피하기 위해 짧은 cooldown 대기에 사용.
        self._last_auto_accept_at: float = 0.0

        # Tracking for the AGENT_GUIDE submitted -> done transition. We
        # don't want to drop filtering until the LLM has actually finished
        # replying — detected as ❯ disappearing (thinking starts) then
        # reappearing (new input field). Falls back to a hard timeout if
        # neither transition lands.
        # AGENT_GUIDE submitted → done 전환 추적. LLM 응답 종료 전에 filtering
        # 풀리면 응답이 사용자 화면에 노출되므로, ❯ 사라짐→재등장 transition을
        # 기다린다. 30초 hard timeout으로 stuck 방지.
        self._seen_pointer_clear: bool = False
        self._submitted_at: float = 0.0

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

        # Incremental dialogue-length scan for the periodic refresh trigger.
        # Keeping the file offset means each turn parses only what was
        # appended since the last one.
        # 주기 갱신 트리거용 증분 대화 길이 스캔. 파일 offset 을 유지하므로 매
        # 턴은 직전 이후 append 된 부분만 파싱한다.
        self._dialogue_scan_conv_id: str | None = None
        self._dialogue_scan_offset: int = 0
        self._dialogue_scan_chars: int = 0

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

        try:
            self._spawn_child()
            self._sync_winsize()
            self._io_loop()
            while self._should_respawn_for_new():
                self._spawn_child()
                self._sync_winsize()
                self._io_loop()
        finally:
            self._restore_terminal()
            self.summarizer_worker.stop()
            self.socket_server.stop()

    def _spawn_child(self) -> None:
        # Detect a NEW-flow respawn before spawning so we know whether the
        # incoming child receives a brand-new conversation that needs a
        # fresh AGENT_GUIDE injection. (For SWITCH the conversation is
        # reused via /resume, so the prior manual stays in scope.)
        # spawn 전에 NEW 재spawn 여부를 미리 판별. NEW면 새 conversation이라
        # 매뉴얼 재주입이 필요. SWITCH는 /resume으로 같은 conversation을
        # 재사용하므로 기존 매뉴얼이 유지된다.
        is_new_respawn = (
            self._pending_action is not None
            and self._pending_action.action_type == "new"
            and self._pending_action.stage == "await_handshake"
        )

        self.child = pexpect.spawn(
            "claude",
            self.claude_args,
            encoding=None,
            echo=False,
        )
        self.pty_fd = self.child.fileno()
        debug_log.log(
            "SPAWN",
            "SYSTEM",
            {
                "claude_args": list(self.claude_args),
                "pid": self.child.pid,
                "is_new_respawn": is_new_respawn,
                "initial_session_name": self._initial_session_name,
                # input_queue carryover into a new child is a known
                # suspect for "ghost" injection — log explicitly even
                # though we don't reset it here.
                # input_queue 가 새 자식으로 따라 들어가는 동작은 "유령"
                # injection 의 의심 후보 — 여기서 reset 하지 않더라도
                # 잔존 사실을 명시적으로 기록.
                "input_queue_carryover_len": len(self.input_queue),
            },
        )
        # Reset per-child detection state so the previous session's tail
        # bytes can't trigger a false prompt on the new child.
        # 자식별 감지 상태 초기화 — 이전 세션의 잔여 바이트가 새 자식의
        # 첫 프롬프트 감지를 오염시키지 않도록.
        self.output_buffer = b""
        # Re-arm confirmation auto-accept for the new child.
        # 새 자식에 대해 confirmation 자동 승인 재무장.
        self._handled_confirmations = set()

        # Decide whether to inject AGENT_GUIDE on this child's first prompt:
        # - NEW respawn → inject (fresh conversation, no manual yet).
        # - --resume <name> → skip (existing conversation already carries the
        #   manual from the previous start).
        # - fresh start (no --resume) → inject.
        # NOTE: --continue is not detected here, so it currently re-injects
        # the manual every time. Documented limitation; can be tightened
        # later by extending _parse_initial_session_name.
        # AGENT_GUIDE inject 여부 결정. NEW 재spawn / fresh start면 inject,
        # --resume이면 기존 conversation에 이미 박혀 있어 skip.
        # --continue는 별도 감지 안 해 매번 재주입됨 (알려진 limitation).
        if is_new_respawn:
            self._agent_guide_stage = "needed"
        elif self._initial_session_name is not None:
            # --resume — existing conversation already has the manual.
            self._agent_guide_stage = "done"
        else:
            self._agent_guide_stage = "needed"
        self._seen_pointer_clear = False
        self._submitted_at = 0.0
        _debug_log(
            f"spawn: is_new_respawn={is_new_respawn}, "
            f"initial_session={self._initial_session_name!r}, "
            f"agent_guide_stage={self._agent_guide_stage}"
        )

    def _should_respawn_for_new(self) -> bool:
        """
        Decide whether to spawn another child after the current one exits.

        현재 자식 종료 후 새 자식을 spawn할지 결정한다. NEW 흐름이 자식
        종료 단계에 도달한 경우에만 True를 반환하고, 동시에 stage를
        핸드셰이크 대기로 전진시킨다.
        """
        # CHILD_EXIT checkpoint — record the previous child's exit state
        # so respawn decisions are auditable. exitstatus is the OS exit
        # code; signalstatus is the signal number if killed.
        # CHILD_EXIT 체크포인트 — 직전 자식의 종료 상태를 기록해 respawn
        # 결정을 사후 추적 가능하게 한다.
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
                "pending_action_type": self._pending_action.action_type
                if self._pending_action is not None
                else None,
                "pending_stage": self._pending_action.stage
                if self._pending_action is not None
                else None,
            },
        )
        pending = self._pending_action
        if pending is None or pending.action_type != "new":
            return False
        if pending.stage != "await_child_exit":
            return False
        self._set_stage(pending, "await_handshake")
        return True

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
            if listen_fd >= 0:
                watch_fds.append(listen_fd)
            if client_fd >= 0:
                watch_fds.append(client_fd)

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

        self._drain_pty()

    def _handle_pty_readable(self) -> bool:
        try:
            chunk = os.read(self.pty_fd, 4096)
        except OSError:
            return False
        if not chunk:
            # EOF on PTY master means the child closed its end.
            # PTY master에서의 EOF — 자식 프로세스가 자기 쪽을 닫음.
            return False

        # Mirror every chunk into the virtual screen, regardless of mode,
        # so the input prompt line is always up to date for extraction.
        # mode 와 무관하게 모든 chunk를 가상 화면에 반영 — 입력란 추출이
        # 항상 최신 상태에서 가능하도록.
        self.virtual_screen.feed(chunk)

        # Auto-accept any confirmation prompts that just appeared in the
        # virtual screen (channels dev warning, MCP server registration).
        # Each pattern is processed at most once per child.
        # 가상 화면에 새로 등장한 confirmation prompt 자동 승인 (channels
        # dev 경고, MCP server 등록). 자식별로 패턴당 최대 1회 처리.
        self._auto_accept_confirmations()

        # Track ❯ disappearance during the AGENT_GUIDE submitted stage so
        # we can tell when the LLM's reply is actually complete (❯ then
        # reappears). Without this, mode=passthrough flips before the LLM
        # even starts thinking and the reply leaks to the user's screen.
        # AGENT_GUIDE submitted 단계에서 ❯ 사라짐 추적 — LLM 응답 종료 시점
        # (❯ 재등장) 감지에 사용.
        if (
            self._agent_guide_stage == "submitted"
            and PROMPT_POINTER not in chunk
        ):
            self._seen_pointer_clear = True

        self.output_buffer += chunk
        detected = self._detect_prompt(self.output_buffer)
        # TEMP diagnostic: trace every PTY chunk so we can see whether (a)
        # chunks stop arriving during the gap, or (b) chunks arrive but
        # _detect_prompt fails to match. Remove after root cause is found.
        # 임시 진단 — 매 PTY chunk를 로그해 (a) gap 동안 chunk가 끊기는지,
        # (b) chunk는 오는데 _detect_prompt가 못 잡는지 가린다.
        _debug_log(
            f"pty-chunk: len={len(chunk)} "
            f"chunk_has_pointer={PROMPT_POINTER in chunk} "
            f"chunk_has_inverse={INVERSE_VIDEO_START in chunk} "
            f"buffer_len={len(self.output_buffer)} "
            f"buffer_has_pointer={PROMPT_POINTER in self.output_buffer} "
            f"buffer_has_inverse={INVERSE_VIDEO_START in self.output_buffer} "
            f"detected={detected}"
        )
        if detected:
            # The handler decides whether to clear the buffer (it knows
            # which stage we're in). After this call we just truncate to
            # keep size bounded.
            # handler가 stage를 알고 있으므로 buffer 비우기 결정도 handler에
            # 위임. 호출 후엔 size cap만 보장.
            self._handle_prompt_detected()
        self._truncate_output_buffer()

        if self.mode == "passthrough":
            os.write(self._stdout_fd, chunk)
        return True

    def _handle_stdin_readable(self) -> None:
        try:
            chunk = os.read(self._stdin_fd, 4096)
        except OSError:
            return
        if not chunk:
            return

        # USER_KEY checkpoint — every real user keystroke arrives here.
        # The mask helper redacts content by default; users can opt in to
        # raw logging via SESSION_MANAGER_LOG_RAW_STDIN=1.
        # USER_KEY 체크포인트 — 실제 사용자 키스트로크는 모두 이 지점으로 진입.
        # 기본은 마스킹, SESSION_MANAGER_LOG_RAW_STDIN=1 로 raw opt-in 가능.
        debug_log.log(
            "USER_KEY",
            "USER",
            {
                "mode": self.mode,
                "chunk": debug_log.mask_stdin_chunk(chunk),
            },
        )

        if self.mode == "filtering":
            # Buffer keystrokes during SWITCH/NEW injection; drained back to
            # the PTY when filtering ends. Used by SWITCH/NEW only.
            # SWITCH/NEW 주입 중 들어온 키 입력은 큐에 보관, 필터링 종료 시
            # PTY로 일괄 반영. SWITCH/NEW 전용 메커니즘.
            self.input_queue += chunk
            debug_log.log(
                "INPUT_QUEUE",
                "USER",
                {
                    "op": "append",
                    "added_len": len(chunk),
                    "total_len": len(self.input_queue),
                    "reason": "filtering_mode",
                },
            )
            return

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
            elif prompt_text and CLEAR_COMMAND_RE.match(prompt_text.strip()):
                # /clear wipes the conversation — summarise it while it's
                # still there.
                # /clear 는 대화를 지운다 — 아직 남아 있을 때 요약한다.
                self._enqueue_active_summary()

        self.stdin_line_buffer += chunk
        while b"\n" in self.stdin_line_buffer:
            line, self.stdin_line_buffer = self.stdin_line_buffer.split(b"\n", 1)
            self._handle_user_line(line + b"\n")

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

    # --------------------------------------------------- Detection & injection
    # 프롬프트 감지 / 텍스트 주입 -----------------------------------------------

    def _detect_prompt(self, buffer: bytes) -> bool:
        idx = buffer.rfind(PROMPT_POINTER)
        if idx == -1:
            return False
        # Look for an inverse-video sequence anywhere after the latest ❯.
        # Ink's input field can wrap across multiple lines (especially after
        # we inject a long @-attachment), putting the cursor's inverse far
        # from the pointer. rfind() already pinned us to the latest ❯ so
        # earlier stale pointers can't compete.
        # 마지막 ❯ 이후 영역 전체에서 inverse 검색. 입력란이 multi-line wrap
        # 되면 cursor inverse가 ❯에서 멀리 있을 수 있어 좁은 64-byte 윈도우는
        # 누락. rfind()로 마지막 ❯ 위치를 잡으니 stale ❯과는 경합 없음.
        return INVERSE_VIDEO_START in buffer[idx:]

    def _inject_text(self, text: str) -> None:
        # Single WRAPPER_INJECT checkpoint — every wrapper-driven write to
        # the PTY passes through here. The caller frame name disambiguates
        # the inject type (handoff / resume / rename / exit / agent_guide /
        # submit) without changing every call site's signature.
        # 단일 WRAPPER_INJECT 체크포인트 — wrapper 가 PTY 에 쓰는 모든 문장이
        # 이 지점을 통과한다. 호출자 프레임 이름으로 inject type (handoff /
        # resume / rename / exit / agent_guide / submit) 을 구분 — 호출 지점
        # 시그니처를 바꾸지 않고도 분류 가능.
        import sys as _sys

        caller = _sys._getframe(1).f_code.co_name if hasattr(_sys, "_getframe") else "?"
        debug_log.log(
            "WRAPPER_INJECT",
            "WRAPPER",
            {
                "caller": caller,
                "stage": getattr(self._pending_action, "stage", None)
                if self._pending_action is not None
                else None,
                "agent_guide_stage": self._agent_guide_stage,
                "text": debug_log.mask_text(text),
            },
        )
        os.write(self.pty_fd, text.encode("utf-8"))

    def _auto_accept_confirmations(self) -> None:
        """Send \\r whenever a known confirmation prompt appears on screen.

        가상 화면에 알려진 confirmation prompt 텍스트가 나타나면 \\r 주입.
        모든 prompt의 default가 1번이라 단순 Enter로 승인된다. 한 번 처리한
        패턴은 ``_handled_confirmations``에 기록해 같은 자식에서 다시 매칭
        되지 않는다.
        """
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
                self._last_auto_accept_at = time.monotonic()

    def _submit(self) -> None:
        """Send a standalone \\r so Ink recognises it as Return.

        Ink의 parseKeypress가 Return으로 인식하도록 \\r을 단독 전송한다.
        """
        self._inject_text("\r")

    # ------------------------------------------------------------ Extension hooks
    # 확장 지점 -------------------------------------------------------------------

    def _handle_prompt_detected(self) -> None:
        """Process a detected prompt and decide whether to clear the buffer.

        Buffer-clearing rules (in-handler so the stage machine drives them):
        - stage=needed held (cooldown / unhandled confirmation): keep buffer
          so the next chunk can re-trigger this stage.
        - stage=needed advanced to injected: keep buffer so the next chunk
          re-fires detect=True for the injected→submit step (Ink may not
          redraw ❯ after we append text — partial redraw only).
        - stage=injected after submit: clear buffer; we now wait for a brand
          new ❯ that arrives after the LLM finishes its reply.
        - stage=submitted/done: clear buffer.
        - pending SWITCH/NEW step: clear buffer; those stage machines also
          want a brand new ❯ for their next step.

        Buffer 비우기 규칙은 stage 머신이 결정. needed→injected advance 후엔
        buffer를 유지해야 partial redraw 환경(❯이 chunk에 새로 안 들어옴)에서
        다음 stage가 발동된다.
        """
        if self._agent_guide_stage == "needed":
            cooldown_remaining = (
                self._last_auto_accept_at + 0.5 - time.monotonic()
            )
            if self._last_auto_accept_at > 0.0 and cooldown_remaining > 0:
                _debug_log(
                    f"prompt-detect: auto-accept cooldown "
                    f"({cooldown_remaining:.3f}s left), holding inject"
                )
                return  # keep buffer

            for pattern in AUTO_CONFIRM_PATTERNS:
                if pattern in self._handled_confirmations:
                    continue
                if self.virtual_screen.contains(pattern):
                    screen_dump = "|".join(self.virtual_screen._safe_display())
                    _debug_log(
                        f"prompt-detect: unhandled confirmation '{pattern}' "
                        f"on screen, holding AGENT_GUIDE inject. "
                        f"screen[:300]={screen_dump[:300]!r}"
                    )
                    return  # keep buffer

            _debug_log(
                f"prompt-detect: stage=needed -> injecting AGENT_GUIDE "
                f"(handled_confirmations={sorted(self._handled_confirmations)})"
            )
            self.mode = "filtering"
            # Short, directive bootstrap text. The single-line reply
            # signals to the user that the wrapper + MCP layer have
            # finished initialising and the manual is in scope.
            # 짧고 지시적인 부트스트랩. 한 줄 응답이 사용자에게 wrapper + MCP
            # layer 초기화 완료 + 매뉴얼 적용 시작을 알리는 신호.
            self._inject_text(
                f"@{AGENT_GUIDE_PATH} System bootstrap. "
                f"Reply with exactly: \"MCP session-manager ready\""
            )
            self._agent_guide_stage = "injected"
            # Keep buffer: Ink's partial redraw won't re-emit ❯, so we need
            # the existing pointer in the buffer for the next chunk to
            # re-trigger detect=True for the injected→submit step.
            return

        if self._agent_guide_stage == "injected":
            _debug_log("prompt-detect: stage=injected -> submitting \\r")
            self._submit()
            self._agent_guide_stage = "submitted"
            self._submitted_at = time.monotonic()
            self._seen_pointer_clear = False
            self.output_buffer = b""  # wait for LLM-reply-after-submit
            return

        if self._agent_guide_stage == "submitted":
            # Hold filtering until the LLM has actually finished replying.
            # Two stacked guards:
            #   (1) Minimum wait — even if `_seen_pointer_clear` flips True
            #       on the very next chunk, that chunk may just be a
            #       partial redraw that happens not to contain ❯ rather
            #       than a real "thinking-started" signal. A short wait
            #       lets the LLM's brief reply complete first.
            #   (2) Pointer-clear transition — once minimum wait has
            #       elapsed, only drop filtering after we've actually seen
            #       ❯ disappear (LLM thinking) AND now reappear (this
            #       detect=True call = new input field).
            # 30s hard timeout backstops both guards.
            #
            # filtering을 LLM 응답 종료까지 유지. 두 단계 가드:
            #   (1) Minimum wait — submit 직후 들어온 chunk가 partial
            #       redraw일 수 있어 너무 빨리 _seen_pointer_clear가 True가
            #       되는 false-positive 방지.
            #   (2) Pointer-clear transition — minimum wait가 지난 뒤에는
            #       ❯이 실제로 사라졌다 다시 등장한 시점에만 done.
            # 30초 hard timeout이 두 가드 모두를 백스톱.
            elapsed = time.monotonic() - self._submitted_at
            if elapsed < 2.0:
                _debug_log(
                    f"prompt-detect: stage=submitted, minimum wait "
                    f"({elapsed:.1f}s/2.0s), holding filtering"
                )
                return  # buffer keep
            if not self._seen_pointer_clear and elapsed < 30.0:
                _debug_log(
                    f"prompt-detect: stage=submitted, ❯ not yet cleared "
                    f"({elapsed:.1f}s elapsed), holding filtering"
                )
                return  # buffer keep
            _debug_log(
                f"prompt-detect: stage=submitted -> done "
                f"(seen_clear={self._seen_pointer_clear}, "
                f"elapsed={elapsed:.1f}s)"
            )
            self._agent_guide_stage = "done"
            if self._pending_action is None:
                self.mode = "passthrough"
                self._drain_input_queue()
            self.output_buffer = b""
            # fall through to pending-action processing

        pending = self._pending_action
        if pending is not None:
            if pending.action_type == "switch":
                self._advance_switch(pending)
            elif pending.action_type == "new":
                self._advance_new(pending)
            self.output_buffer = b""  # SWITCH/NEW stages await the next ❯
            return

        # stage=done, no pending action — clear so we don't keep matching
        # the same ❯ on every chunk.
        # done 상태에서 pending 없으면 같은 ❯에 매번 매칭되지 않게 비움.
        self.output_buffer = b""

        # A fresh prompt with nothing else in flight means the previous turn
        # finished — the one safe point to measure dialogue growth.
        # 진행 중인 동작 없이 새 프롬프트가 떴다는 것은 직전 턴이 끝났다는
        # 뜻이다 — 대화 증가량을 재기에 안전한 유일한 지점.
        self._check_summary_refresh()

    def _handle_user_line(self, line: bytes) -> None:
        return

    def _handle_mcp_signal(self, message: dict) -> None:
        if not isinstance(message, dict):
            return
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
            user_prompt_val = handoff.get("user_prompt", "")
            user_prompt = user_prompt_val if isinstance(user_prompt_val, str) else ""
            self._handle_switch(target, handoff, user_prompt)
        elif action == "new":
            rename_current = message.get("rename_current")
            new_session_name = message.get("new_session_name")
            handoff = message.get("handoff") or {}
            if not isinstance(new_session_name, str) or not isinstance(handoff, dict):
                return
            if rename_current is not None and not isinstance(rename_current, str):
                return
            user_prompt_val = handoff.get("user_prompt", "")
            user_prompt = user_prompt_val if isinstance(user_prompt_val, str) else ""
            self._handle_new(rename_current, new_session_name, handoff, user_prompt)
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

    def _set_stage(self, pending: _PendingAction, new_stage: str) -> None:
        """Centralised stage transition with logging.

        Every SWITCH/NEW stage move goes through here so the log shows the
        full state machine path for one action.

        중앙화된 stage 전환 + 로깅. 모든 SWITCH/NEW stage 이동이 이 지점을
        통과해 한 액션의 전체 상태 머신 경로가 로그에 남는다.
        """
        debug_log.log(
            "STAGE",
            "WRAPPER",
            {
                "action_type": pending.action_type,
                "target": pending.target,
                "new_session_name": pending.new_session_name,
                "before": pending.stage,
                "after": new_stage,
            },
        )
        pending.stage = new_stage

    def _handle_switch(
        self,
        target: str,
        handoff: dict[str, Any],
        user_prompt: str,
    ) -> None:
        """
        Register a SWITCH action; advanced on subsequent prompt detections.

        SWITCH 액션을 등록한다. 실제 진행은 이후 프롬프트 감지 이벤트마다
        단계적으로 일어난다.
        """
        # Queue a background summary for the departing session while its
        # conversation is still the active one, and move the wrapper-side
        # current-session mirror to the target.
        # 떠나는 세션의 conversation 이 아직 활성인 시점에 백그라운드 요약을
        # 큐에 넣고, 래퍼 측 현재 세션 미러를 target 으로 이동.
        self._enqueue_departed_summary(handoff.get("from"))
        self._current_session_name = target

        # Strip user_prompt from the JSON body so it isn't shown twice
        # (once inside [handoff], once as the prompt text below).
        # JSON 본문에서 user_prompt를 제거 — [handoff] 블록과 그 아래 본문에
        # 같은 텍스트가 두 번 노출되지 않도록 한다.
        handoff_clean = {k: v for k, v in handoff.items() if k != "user_prompt"}
        self._pending_action = _PendingAction(
            action_type="switch",
            target=target,
            handoff=handoff_clean,
            user_prompt=user_prompt,
            stage="await_resume_prompt",
        )
        debug_log.log(
            "STAGE",
            "WRAPPER",
            {
                "action_type": "switch",
                "target": target,
                "before": None,
                "after": "await_resume_prompt",
                "user_prompt": debug_log.mask_text(user_prompt),
            },
        )

    def _advance_switch(self, pending: _PendingAction) -> None:
        if pending.stage == "await_resume_prompt":
            # First prompt after the LLM finished its turn — start filtering
            # so the user doesn't see the raw `/resume` injection, then
            # inject it.  Submit is deferred to await_resume_submit.
            # LLM 응답이 끝난 직후의 첫 프롬프트 — 필터링을 켜서 raw `/resume`
            # 주입이 사용자에게 보이지 않게 한 뒤 주입. 제출은
            # await_resume_submit으로 지연.
            self.mode = "filtering"
            self._inject_text(f"/resume {pending.target}")
            self._set_stage(pending, "await_resume_submit")
        elif pending.stage == "await_resume_submit":
            self._submit()
            self._set_stage(pending, "await_handoff_prompt")
        elif pending.stage == "await_handoff_prompt":
            # The resumed session is ready for input. Inject the handoff
            # block plus the user's prompt.  Submit deferred to next detection.
            # 복귀 세션이 입력 대기 중. handoff 블록과 사용자 프롬프트를
            # 주입한다. 제출은 다음 감지로 지연.
            text = format_handoff_injection(pending.handoff, pending.user_prompt)
            self._inject_text(text)
            self._set_stage(pending, "await_handoff_submit")
        elif pending.stage == "await_handoff_submit":
            self._submit()
            self.mode = "passthrough"
            self._drain_input_queue()
            self._set_stage(pending, "done")
            self._pending_action = None

    def _drain_input_queue(self) -> None:
        if not self.input_queue:
            return
        # Replace newlines with spaces so a buffered Enter doesn't auto-submit
        # text the user typed during filtering; let them press Enter explicitly.
        # 필터링 중 사용자가 친 입력의 개행을 공백으로 치환 — 쌓인 Enter가
        # 자동으로 submit되지 않도록 하고, 사용자가 다시 눌러 보내게 한다.
        cleaned = self.input_queue.replace(b"\n", b" ")
        debug_log.log(
            "INPUT_QUEUE",
            "USER",
            {
                "op": "drain",
                "queue_len": len(self.input_queue),
                "cleaned_len": len(cleaned),
                "raw_chunk": debug_log.mask_stdin_chunk(self.input_queue),
            },
        )
        os.write(self.pty_fd, cleaned)
        self.input_queue = b""

    def _handle_new(
        self,
        rename_current: str | None,
        new_session_name: str,
        handoff: dict[str, Any],
        user_prompt: str,
    ) -> None:
        """
        Register a NEW action; advanced on subsequent prompt detections.

        NEW 액션을 등록한다. 실제 진행은 이후 프롬프트 감지 이벤트마다
        단계적으로 일어난다.
        """
        # Same departing-session summary + mirror handling as SWITCH.
        # SWITCH 와 동일한 떠나는 세션 요약 큐 적재 + 미러 갱신.
        self._enqueue_departed_summary(handoff.get("from"))
        self._current_session_name = new_session_name

        # Same JSON-vs-text de-duplication as SWITCH: keep user_prompt out
        # of the JSON body since it appears as plain text below.
        # SWITCH와 동일하게 JSON 본문에서는 user_prompt 제거 — 아래쪽
        # 평문 본문과 중복으로 노출되지 않도록.
        handoff_clean = {k: v for k, v in handoff.items() if k != "user_prompt"}
        self._pending_action = _PendingAction(
            action_type="new",
            target="",
            handoff=handoff_clean,
            user_prompt=user_prompt,
            stage="await_rename_or_exit_prompt",
            rename_current=rename_current,
            new_session_name=new_session_name,
        )
        debug_log.log(
            "STAGE",
            "WRAPPER",
            {
                "action_type": "new",
                "rename_current": rename_current,
                "new_session_name": new_session_name,
                "before": None,
                "after": "await_rename_or_exit_prompt",
                "user_prompt": debug_log.mask_text(user_prompt),
            },
        )

    def _handle_handshake_request(self) -> None:
        """
        Reply to MCP's handshake. NEW respawns return new_session_name;
        all other startups return whatever was decided from CLI args.

        MCP의 핸드셰이크 요청에 응답한다. NEW로 인한 재시작 흐름이라면
        새 세션 이름을 돌려주고, 그 외 일반 시작에서는 CLI 인자에서
        결정된 값(또는 None)을 돌려준다.
        """
        pending = self._pending_action
        if (
            pending is not None
            and pending.action_type == "new"
            and pending.stage == "await_handshake"
        ):
            debug_log.log(
                "HANDSHAKE",
                "WRAPPER",
                {
                    "phase": "wrapper_response",
                    "branch": "new_respawn",
                    "current_session_name": pending.new_session_name,
                },
            )
            self.socket_server.send(
                {"current_session_name": pending.new_session_name}
            )
            self._set_stage(pending, "await_new_session_prompt")
            return
        debug_log.log(
            "HANDSHAKE",
            "WRAPPER",
            {
                "phase": "wrapper_response",
                "branch": "initial",
                "current_session_name": self._initial_session_name,
            },
        )
        self.socket_server.send(
            {"current_session_name": self._initial_session_name}
        )

    @staticmethod
    def _parse_initial_session_name(args: list[str]) -> str | None:
        for i, arg in enumerate(args):
            if arg == "--resume" and i + 1 < len(args):
                return args[i + 1]
            if arg.startswith("--resume="):
                return arg[len("--resume=") :]
        return None

    def _advance_new(self, pending: _PendingAction) -> None:
        if pending.stage == "await_rename_or_exit_prompt":
            # First prompt after the LLM finished its turn. Start filtering
            # so neither /rename nor /exit is visible to the user.
            # LLM 응답이 끝난 직후의 첫 프롬프트. 필터링을 켜서 /rename·/exit이
            # 사용자에게 보이지 않게 한다.
            self.mode = "filtering"
            if pending.rename_current is not None:
                self._inject_text(f"/rename {pending.rename_current}")
                self._set_stage(pending, "await_rename_submit")
            else:
                self._inject_text("/exit")
                self._set_stage(pending, "await_exit_submit")
        elif pending.stage == "await_rename_submit":
            self._submit()
            self._set_stage(pending, "await_exit_prompt")
        elif pending.stage == "await_exit_prompt":
            # /rename has been processed; now exit the current session.
            # /rename 처리 완료, 이제 현재 세션 종료.
            self._inject_text("/exit")
            self._set_stage(pending, "await_exit_submit")
        elif pending.stage == "await_exit_submit":
            self._submit()
            self._set_stage(pending, "await_child_exit")
        elif pending.stage == "await_new_session_prompt":
            # New child has spawned, MCP handshake completed, and the first
            # prompt of the fresh session is up. Inject the handoff plus the
            # user's prompt. Submit deferred to next detection.
            # 새 자식이 spawn되고 MCP 핸드셰이크가 끝난 뒤 새 세션의 첫
            # 프롬프트가 떴다. handoff와 사용자 프롬프트를 주입. 제출은 다음
            # 감지로 지연.
            text = format_handoff_injection(pending.handoff, pending.user_prompt)
            self._inject_text(text)
            self._set_stage(pending, "await_new_handoff_submit")
        elif pending.stage == "await_new_handoff_submit":
            self._submit()
            self.mode = "passthrough"
            self._drain_input_queue()
            self._set_stage(pending, "done")
            self._pending_action = None
        # `await_child_exit`와 `await_handshake` 단계에서는 프롬프트 감지로
        # 진행하지 않는다 — 자식 종료(outer loop)와 소켓 핸드셰이크(별도
        # 메시지 경로)가 각각 stage를 전진시킨다.

    # ------------------------------------------------------ Buffer management
    # 버퍼 관리 -----------------------------------------------------------------

    def _truncate_output_buffer(self) -> None:
        if len(self.output_buffer) <= OUTPUT_BUFFER_CAP:
            return
        # Drop the front but keep the tail so a partial multi-byte prompt
        # sequence at the boundary survives across truncation.
        #
        # 앞부분은 버리고 끝부분만 유지. 경계에 걸친 멀티바이트 프롬프트
        # 시퀀스가 다음 매칭에서도 살아남도록 함.
        self.output_buffer = self.output_buffer[-OUTPUT_BUFFER_TAIL_KEEP:]

    def _drain_pty(self) -> None:
        try:
            while True:
                chunk = os.read(self.pty_fd, 4096)
                if not chunk:
                    return
                self.virtual_screen.feed(chunk)
                if self.mode == "passthrough":
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
