"""
PTY wrapper that mediates between the user terminal and Claude Code.

Provides the I/O loop skeleton on which additional handlers hang
SWITCH/NEW logic, MCP socket integration, and stdin slash-command
interception. By itself it is a transparent passthrough: spawns Claude
Code on a PTY, forwards stdin to the PTY master, forwards PTY output to
stdout, and detects the input prompt so the rest of the wrapper can pick
a safe moment to inject text.

사용자 터미널과 Claude Code 프로세스 사이에 끼어들어 양방향 I/O를
중계하는 PTY 래퍼 모듈이다.

이 모듈은 I/O 루프의 골격만 제공한다. 세션 전환(SWITCH/NEW) 처리,
MCP 소켓 통합, stdin 슬래시 커맨드 가로채기 같은 상위 로직은 별도의
핸들러로 이 골격 위에 얹어 확장한다.

단독으로 사용할 경우 투명한 패스스루로 동작한다. Claude Code를 PTY에
띄운 뒤, 사용자가 입력한 바이트는 PTY master로 그대로 전달하고, PTY가
출력하는 바이트는 stdout으로 흘려보낸다. 동시에 출력 스트림에서 입력
프롬프트(figures.pointer "❯" 직후의 반전 커서 시퀀스)를 감지해, 후속
로직이 텍스트를 안전하게 주입할 수 있는 시점을 파악할 수 있게 한다.
"""

from __future__ import annotations

import os
import select
import signal
import sys
import termios
import tty
from dataclasses import dataclass
from typing import Any, Literal

import pexpect

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

Mode = Literal["passthrough", "filtering"]


def _safe_fileno(stream: Any) -> int:
    try:
        return stream.fileno()
    except (OSError, AttributeError, ValueError):
        return -1


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


@dataclass
class _InterceptState:
    """
    In-progress slash-command interception waiting for the MCP response.

    사용자가 직접 친 슬래시 명령(`/resume`, `/exit` 등)을 가로채고 MCP에서
    session_end 처리가 끝났다는 응답이 오기를 기다리는 상태를 보관한다.
    응답을 받기 전까지는 filtering 모드로 사용자 입력을 큐잉한다.
    """

    command: str  # one of KNOWN_COMMANDS
    args: str


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
        self._intercept_state: _InterceptState | None = None

        # Confirmation patterns already auto-accepted in the current child.
        # Reset on each spawn so a respawned child re-arms the auto-accept.
        # 현재 자식에서 이미 자동 승인한 confirmation 패턴.
        # 새 자식이 spawn될 때마다 초기화해 자동 승인을 재무장.
        self._handled_confirmations: set[str] = set()

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
            self.socket_server.stop()

    def _spawn_child(self) -> None:
        self.child = pexpect.spawn(
            "claude",
            self.claude_args,
            encoding=None,
            echo=False,
        )
        self.pty_fd = self.child.fileno()
        # Reset per-child detection state so the previous session's tail
        # bytes can't trigger a false prompt on the new child.
        # 자식별 감지 상태 초기화 — 이전 세션의 잔여 바이트가 새 자식의
        # 첫 프롬프트 감지를 오염시키지 않도록.
        self.output_buffer = b""
        # Re-arm confirmation auto-accept for the new child.
        # 새 자식에 대해 confirmation 자동 승인 재무장.
        self._handled_confirmations = set()

    def _should_respawn_for_new(self) -> bool:
        """
        Decide whether to spawn another child after the current one exits.

        현재 자식 종료 후 새 자식을 spawn할지 결정한다. NEW 흐름이 자식
        종료 단계에 도달한 경우에만 True를 반환하고, 동시에 stage를
        핸드셰이크 대기로 전진시킨다.
        """
        pending = self._pending_action
        if pending is None or pending.action_type != "new":
            return False
        if pending.stage != "await_child_exit":
            return False
        pending.stage = "await_handshake"
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

        self.output_buffer += chunk
        if self._detect_prompt(self.output_buffer):
            # Clear after a successful detection so the same prompt isn't
            # matched again as more output trickles in.
            # 감지 직후 버퍼를 비워, 같은 프롬프트가 후속 chunk에서 다시
            # 매칭되는 것을 막는다.
            self.output_buffer = b""
            self._handle_prompt_detected()
        else:
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

        if self.mode == "filtering":
            # Buffer keystrokes during injection; drained back to the PTY
            # when filtering ends.
            # 주입 중 들어온 키 입력은 큐에 보관, 필터링 종료 시 PTY로 일괄 반영.
            self.input_queue += chunk
            return

        # Submit detection: Ink's parseKeypress only treats a lone \r as
        # Return (s === '\r'). Multi-byte chunks are typed text, not submit.
        # submit 감지 — Ink parseKeypress는 단독 \r만 Return으로 인정
        # (s === '\r'). 멀티 바이트 chunk는 타이핑 중인 텍스트로 간주.
        if chunk == b"\r":
            prompt_text = self.virtual_screen.get_prompt_line()
            matched = match_intercept_command(prompt_text)
            if matched is not None:
                self._start_intercept(matched, chunk)
                return

        self.stdin_line_buffer += chunk
        while b"\n" in self.stdin_line_buffer:
            line, self.stdin_line_buffer = self.stdin_line_buffer.split(b"\n", 1)
            self._handle_user_line(line + b"\n")

        # Forward keystrokes to the PTY so Ink can render them in real time.
        # Ink가 실시간으로 렌더링할 수 있도록 키 입력을 PTY로 즉시 전달.
        os.write(self.pty_fd, chunk)

    # ------------------------------------------------------- Slash interception
    # 슬래시 명령 가로채기 ------------------------------------------------------

    def _start_intercept(
        self, matched: InterceptedCommand, submit_chunk: bytes
    ) -> None:
        """Begin a slash-command interception flow.

        슬래시 명령 가로채기 흐름 시작 — filtering 모드로 들어가서 사용자
        입력을 큐잉하고 MCP 측에 가로채기 신호를 보낸다. 이후 사용자 stdin은
        _handle_stdin_readable의 filtering 분기에서 자동으로 input_queue에
        적재된다.
        """
        self.mode = "filtering"
        # Queue the submit chunk so it can be replayed to the PTY when the
        # session_end response arrives.
        # submit chunk를 큐잉 — session_end 응답 도착 시 PTY로 재생되도록.
        self.input_queue += submit_chunk
        self._intercept_state = _InterceptState(
            command=matched.command,
            args=matched.args,
        )
        self.socket_server.send(
            {
                "action": "intercept",
                "command": matched.command,
                "args": matched.args,
            }
        )

    def _finish_intercept(self) -> None:
        """End interception and replay the queued input to the PTY.

        가로채기 종료 — 큐잉된 입력(원래의 submit chunk + filtering 동안
        들어온 추가 입력)을 PTY로 그대로 흘려보낸다. SWITCH/NEW의
        _drain_input_queue와 달리 newline을 공백으로 치환하지 않음 — 사용자가
        의도한 \r submit이 그대로 전달되어야 한다.
        """
        self._intercept_state = None
        self.mode = "passthrough"
        if self.input_queue:
            os.write(self.pty_fd, self.input_queue)
            self.input_queue = b""

    # --------------------------------------------------- Detection & injection
    # 프롬프트 감지 / 텍스트 주입 -----------------------------------------------

    def _detect_prompt(self, buffer: bytes) -> bool:
        idx = buffer.rfind(PROMPT_POINTER)
        if idx == -1:
            return False
        # Only count an inverse-video sequence in a small window after the
        # pointer; a stale "❯" elsewhere in the buffer must not match.
        #
        # 표지자 직후 좁은 윈도우 안의 반전 시퀀스만 인정. 버퍼 다른 위치에
        # 남아 있는 오래된 "❯"가 잘못 매칭되지 않도록 함.
        return INVERSE_VIDEO_START in buffer[idx : idx + 64]

    def _inject_text(self, text: str) -> None:
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
                try:
                    os.write(self.pty_fd, b"\r")
                except OSError:
                    return
                self._handled_confirmations.add(pattern)

    def _submit(self) -> None:
        """Send a standalone \\r so Ink recognises it as Return.

        Ink의 parseKeypress가 Return으로 인식하도록 \\r을 단독 전송한다.
        """
        self._inject_text("\r")

    # ------------------------------------------------------------ Extension hooks
    # 확장 지점 -------------------------------------------------------------------

    def _handle_prompt_detected(self) -> None:
        pending = self._pending_action
        if pending is not None:
            if pending.action_type == "switch":
                self._advance_switch(pending)
            elif pending.action_type == "new":
                self._advance_new(pending)

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
        elif action == "intercept_done":
            # MCP 측에서 session_end 처리가 끝났다는 응답. 가로채기 상태일
            # 때만 종료하고 큐잉된 명령을 흘려보낸다 (그 외 상태에서는 무시).
            if self._intercept_state is not None:
                self._finish_intercept()

    # ----------------------------------------------------------- Action handlers
    # 세션 액션 처리 ------------------------------------------------------------

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
            pending.stage = "await_resume_submit"
        elif pending.stage == "await_resume_submit":
            self._submit()
            pending.stage = "await_handoff_prompt"
        elif pending.stage == "await_handoff_prompt":
            # The resumed session is ready for input. Inject the handoff
            # block plus the user's prompt.  Submit deferred to next detection.
            # 복귀 세션이 입력 대기 중. handoff 블록과 사용자 프롬프트를
            # 주입한다. 제출은 다음 감지로 지연.
            text = format_handoff_injection(pending.handoff, pending.user_prompt)
            self._inject_text(text)
            pending.stage = "await_handoff_submit"
        elif pending.stage == "await_handoff_submit":
            self._submit()
            self.mode = "passthrough"
            self._drain_input_queue()
            self._pending_action = None

    def _drain_input_queue(self) -> None:
        if not self.input_queue:
            return
        # Replace newlines with spaces so a buffered Enter doesn't auto-submit
        # text the user typed during filtering; let them press Enter explicitly.
        # 필터링 중 사용자가 친 입력의 개행을 공백으로 치환 — 쌓인 Enter가
        # 자동으로 submit되지 않도록 하고, 사용자가 다시 눌러 보내게 한다.
        cleaned = self.input_queue.replace(b"\n", b" ")
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
            self.socket_server.send(
                {"current_session_name": pending.new_session_name}
            )
            pending.stage = "await_new_session_prompt"
            return
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
                pending.stage = "await_rename_submit"
            else:
                self._inject_text("/exit")
                pending.stage = "await_exit_submit"
        elif pending.stage == "await_rename_submit":
            self._submit()
            pending.stage = "await_exit_prompt"
        elif pending.stage == "await_exit_prompt":
            # /rename has been processed; now exit the current session.
            # /rename 처리 완료, 이제 현재 세션 종료.
            self._inject_text("/exit")
            pending.stage = "await_exit_submit"
        elif pending.stage == "await_exit_submit":
            self._submit()
            pending.stage = "await_child_exit"
        elif pending.stage == "await_new_session_prompt":
            # New child has spawned, MCP handshake completed, and the first
            # prompt of the fresh session is up. Inject the handoff plus the
            # user's prompt. Submit deferred to next detection.
            # 새 자식이 spawn되고 MCP 핸드셰이크가 끝난 뒤 새 세션의 첫
            # 프롬프트가 떴다. handoff와 사용자 프롬프트를 주입. 제출은 다음
            # 감지로 지연.
            text = format_handoff_injection(pending.handoff, pending.user_prompt)
            self._inject_text(text)
            pending.stage = "await_new_handoff_submit"
        elif pending.stage == "await_new_handoff_submit":
            self._submit()
            self.mode = "passthrough"
            self._drain_input_queue()
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
