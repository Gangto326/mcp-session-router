"""
Unit tests for SessionManagerWrapper internals.

PTY 래퍼의 내부 로직 단위 테스트. PTY 의존 메서드는 monkeypatch 로 mock,
소켓·SIGWINCH·실런타임 동작은 통합 테스트로 이관한다.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from session_manager import summarizer
from session_manager.models import SessionMetadata
from session_manager.storage.file_store import SessionStore
from session_manager.transcript_excerpt import EXCERPT_MAX_CHARS
from session_manager.wrapper.pty_wrapper import (
    AUTO_CONFIRM_PATTERNS,
    CLEAR_COMMAND_RE,
    CTRL_C,
    INTERCEPT_TIMEOUT_SEC,
    INVERSE_VIDEO_START,
    OUTPUT_BUFFER_CAP,
    OUTPUT_BUFFER_TAIL_KEEP,
    PROMPT_POINTER,
    SessionManagerWrapper,
    _InterceptState,
    _PendingAction,
)


@pytest.fixture
def wrapper(tmp_path: Path) -> SessionManagerWrapper:
    return SessionManagerWrapper(
        socket_path=str(tmp_path / "test.sock"),
        claude_args=[],
        project_path=str(tmp_path),
    )


def _capture_injects(
    wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
) -> list[bytes]:
    """Capture all _inject_text calls as bytes."""
    captured: list[bytes] = []

    def fake_inject(text: str) -> None:
        captured.append(text.encode("utf-8"))

    monkeypatch.setattr(wrapper, "_inject_text", fake_inject)
    return captured


class TestDetectPrompt:
    def test_detects_pointer_with_inverse(self, wrapper: SessionManagerWrapper) -> None:
        buffer = (
            b"some output\n"
            + PROMPT_POINTER
            + b" "
            + INVERSE_VIDEO_START
            + b" \x1b[27m"
        )
        assert wrapper._detect_prompt(buffer) is True

    def test_not_detected_pointer_only(self, wrapper: SessionManagerWrapper) -> None:
        buffer = b"output\n" + PROMPT_POINTER + b" no inverse here"
        assert wrapper._detect_prompt(buffer) is False

    def test_not_detected_inverse_only(self, wrapper: SessionManagerWrapper) -> None:
        buffer = INVERSE_VIDEO_START + b"text"
        assert wrapper._detect_prompt(buffer) is False

    def test_detected_inverse_far_from_pointer(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        """Inverse anywhere after the latest ❯ counts as detected.

        The wrapper widened the detect window from a 64-byte slice after
        ❯ to *everything past the latest ❯* to handle multi-line wrapped
        input fields, where ❯ sits on the first line and the cursor
        inverse on the last. A distance-based window misses that case.

        ❯ 이후 거리에 무관하게 inverse 가 있으면 detect 된다.

        Multi-line wrap 입력란에서는 ❯ 마커가 첫 라인에, cursor inverse 가
        마지막 라인에 있을 수 있다. 거리 기반 좁은 윈도우는 이 케이스를
        놓치므로 detect 범위를 마지막 ❯ 이후 buffer 전체로 확대했다.
        """
        buffer = PROMPT_POINTER + b"x" * 100 + INVERSE_VIDEO_START
        assert wrapper._detect_prompt(buffer) is True

    def test_chunk_boundary_detection(self, wrapper: SessionManagerWrapper) -> None:
        # ❯의 첫 2바이트만 도착한 시점에는 매칭 안 됨
        wrapper.output_buffer += PROMPT_POINTER[:2]
        assert wrapper._detect_prompt(wrapper.output_buffer) is False

        # 나머지 1바이트 + inverse 가 따라오면 매칭 성공
        wrapper.output_buffer += PROMPT_POINTER[2:] + b" " + INVERSE_VIDEO_START
        assert wrapper._detect_prompt(wrapper.output_buffer) is True

    def test_uses_rfind_picks_latest_pointer(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        # 오래된 ❯ 는 inverse 와 멀리, 최신 ❯ 는 inverse 와 가까이 — rfind 라
        # 최신 위치만 검사하므로 매칭 성공
        buffer = (
            PROMPT_POINTER
            + b"x" * 200
            + b" newer turn "
            + PROMPT_POINTER
            + b" "
            + INVERSE_VIDEO_START
        )
        assert wrapper._detect_prompt(buffer) is True


class TestTruncateOutputBuffer:
    def test_no_truncation_below_cap(self, wrapper: SessionManagerWrapper) -> None:
        wrapper.output_buffer = b"x" * (OUTPUT_BUFFER_CAP - 1)
        wrapper._truncate_output_buffer()
        assert len(wrapper.output_buffer) == OUTPUT_BUFFER_CAP - 1

    def test_truncates_keeps_tail(self, wrapper: SessionManagerWrapper) -> None:
        head = b"a" * (OUTPUT_BUFFER_CAP // 2)
        tail = b"b" * (OUTPUT_BUFFER_CAP // 2 + 100)
        wrapper.output_buffer = head + tail
        wrapper._truncate_output_buffer()
        assert len(wrapper.output_buffer) == OUTPUT_BUFFER_TAIL_KEEP
        assert wrapper.output_buffer == b"b" * OUTPUT_BUFFER_TAIL_KEEP


class TestParseInitialSessionName:
    def test_resume_with_value(self) -> None:
        assert (
            SessionManagerWrapper._parse_initial_session_name(["--resume", "foo"])
            == "foo"
        )

    def test_resume_with_equals(self) -> None:
        assert (
            SessionManagerWrapper._parse_initial_session_name(["--resume=bar"])
            == "bar"
        )

    def test_continue_returns_none(self) -> None:
        assert (
            SessionManagerWrapper._parse_initial_session_name(["--continue"]) is None
        )

    def test_no_args_returns_none(self) -> None:
        assert SessionManagerWrapper._parse_initial_session_name([]) is None

    def test_resume_at_end_no_value(self) -> None:
        assert (
            SessionManagerWrapper._parse_initial_session_name(["--resume"]) is None
        )

    def test_other_args_ignored(self) -> None:
        assert (
            SessionManagerWrapper._parse_initial_session_name(
                ["--foo", "bar", "--resume", "x", "--baz"]
            )
            == "x"
        )


class TestDrainInputQueue:
    def test_replaces_newlines_with_spaces(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[bytes] = []
        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.os.write",
            lambda fd, data: captured.append(data) or len(data),
        )
        wrapper.pty_fd = 99
        wrapper.input_queue = b"hello\nworld\n"
        wrapper._drain_input_queue()
        assert captured == [b"hello world "]
        assert wrapper.input_queue == b""

    def test_empty_queue_no_write(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[bytes] = []
        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.os.write",
            lambda fd, data: captured.append(data) or len(data),
        )
        wrapper.pty_fd = 99
        wrapper.input_queue = b""
        wrapper._drain_input_queue()
        assert captured == []


class TestSwitchFlow:
    def test_handle_switch_registers_pending(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._handle_switch(
            target="bar",
            handoff={"from": "foo", "user_prompt": "do thing"},
            user_prompt="do thing",
        )
        pending = wrapper._pending_action
        assert pending is not None
        assert pending.action_type == "switch"
        assert pending.target == "bar"
        assert pending.user_prompt == "do thing"
        assert pending.stage == "await_resume_prompt"
        # JSON 본문에서 user_prompt 제거 — 본문 평문과 중복 노출 방지
        assert "user_prompt" not in pending.handoff
        assert pending.handoff == {"from": "foo"}

    def test_advance_switch_stage_one_injects_resume_text_only(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        injected = _capture_injects(wrapper, monkeypatch)
        pending = _PendingAction(
            action_type="switch",
            target="bar",
            handoff={},
            user_prompt="hi",
            stage="await_resume_prompt",
        )
        wrapper._pending_action = pending
        wrapper._advance_switch(pending)
        assert wrapper.mode == "filtering"
        assert injected == [b"/resume bar"]
        assert pending.stage == "await_resume_submit"

    def test_advance_switch_stage_two_submits_resume(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        submitted = _capture_injects(wrapper, monkeypatch)
        pending = _PendingAction(
            action_type="switch",
            target="bar",
            handoff={},
            user_prompt="hi",
            stage="await_resume_submit",
        )
        wrapper._pending_action = pending
        wrapper._advance_switch(pending)
        assert submitted == [b"\r"]
        assert pending.stage == "await_handoff_prompt"

    def test_advance_switch_stage_three_injects_handoff_text_only(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        injected = _capture_injects(wrapper, monkeypatch)
        pending = _PendingAction(
            action_type="switch",
            target="bar",
            handoff={"from": "old"},
            user_prompt="user req",
            stage="await_handoff_prompt",
        )
        wrapper._pending_action = pending
        wrapper.mode = "filtering"
        wrapper._advance_switch(pending)
        assert len(injected) == 1
        text = injected[0].decode("utf-8")
        assert text.startswith("[handoff]\n")
        assert text.endswith("user req")
        assert pending.stage == "await_handoff_submit"

    def test_advance_switch_stage_four_submits_and_unfilters(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _capture_injects(wrapper, monkeypatch)
        monkeypatch.setattr(wrapper, "_drain_input_queue", lambda: None)
        pending = _PendingAction(
            action_type="switch",
            target="bar",
            handoff={},
            user_prompt="hi",
            stage="await_handoff_submit",
        )
        wrapper._pending_action = pending
        wrapper.mode = "filtering"
        wrapper._advance_switch(pending)
        assert wrapper.mode == "passthrough"
        assert wrapper._pending_action is None


class TestNewFlow:
    def test_handle_new_registers_pending(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._handle_new(
            rename_current="old",
            new_session_name="new",
            handoff={"from": "old"},
            user_prompt="hi",
        )
        pending = wrapper._pending_action
        assert pending is not None
        assert pending.action_type == "new"
        assert pending.rename_current == "old"
        assert pending.new_session_name == "new"
        assert pending.stage == "await_rename_or_exit_prompt"

    def test_handle_new_with_null_rename_injects_exit_text_only(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        injected = _capture_injects(wrapper, monkeypatch)
        wrapper._handle_new(
            rename_current=None,
            new_session_name="new",
            handoff={},
            user_prompt="x",
        )
        wrapper._advance_new(wrapper._pending_action)  # type: ignore[arg-type]
        assert injected == [b"/exit"]
        assert wrapper._pending_action is not None
        assert wrapper._pending_action.stage == "await_exit_submit"

    def test_advance_new_with_rename_then_submit_then_exit(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        injected = _capture_injects(wrapper, monkeypatch)
        wrapper._handle_new(
            rename_current="cur",
            new_session_name="new",
            handoff={},
            user_prompt="x",
        )
        # Stage 1: inject /rename text
        wrapper._advance_new(wrapper._pending_action)  # type: ignore[arg-type]
        assert injected == [b"/rename cur"]
        assert wrapper._pending_action.stage == "await_rename_submit"  # type: ignore[union-attr]

        # Stage 2: submit /rename
        wrapper._advance_new(wrapper._pending_action)  # type: ignore[arg-type]
        assert injected == [b"/rename cur", b"\r"]
        assert wrapper._pending_action.stage == "await_exit_prompt"  # type: ignore[union-attr]

        # Stage 3: inject /exit text
        wrapper._advance_new(wrapper._pending_action)  # type: ignore[arg-type]
        assert injected == [b"/rename cur", b"\r", b"/exit"]
        assert wrapper._pending_action.stage == "await_exit_submit"  # type: ignore[union-attr]

        # Stage 4: submit /exit
        wrapper._advance_new(wrapper._pending_action)  # type: ignore[arg-type]
        assert injected == [b"/rename cur", b"\r", b"/exit", b"\r"]
        assert wrapper._pending_action.stage == "await_child_exit"  # type: ignore[union-attr]

    def test_advance_new_handoff_injects_text_only(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        injected = _capture_injects(wrapper, monkeypatch)
        pending = _PendingAction(
            action_type="new",
            target="",
            handoff={"from": "old"},
            user_prompt="user req",
            stage="await_new_session_prompt",
            new_session_name="new",
        )
        wrapper._pending_action = pending
        wrapper.mode = "filtering"
        wrapper._advance_new(pending)
        assert wrapper.mode == "filtering"
        assert pending.stage == "await_new_handoff_submit"
        assert len(injected) == 1
        assert injected[0].decode("utf-8").startswith("[handoff]\n")

    def test_advance_new_handoff_submit_unfilters(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _capture_injects(wrapper, monkeypatch)
        monkeypatch.setattr(wrapper, "_drain_input_queue", lambda: None)
        pending = _PendingAction(
            action_type="new",
            target="",
            handoff={},
            user_prompt="x",
            stage="await_new_handoff_submit",
            new_session_name="new",
        )
        wrapper._pending_action = pending
        wrapper.mode = "filtering"
        wrapper._advance_new(pending)
        assert wrapper.mode == "passthrough"
        assert wrapper._pending_action is None


class TestHandshake:
    def test_replies_with_initial_session_name_on_normal_start(
        self, tmp_path: Path
    ) -> None:
        wrapper = SessionManagerWrapper(
            socket_path=str(tmp_path / "x.sock"),
            claude_args=["--resume", "foo"],
            project_path=str(tmp_path),
        )
        sent: list[dict] = []
        wrapper.socket_server.send = lambda msg: bool(sent.append(msg) or True)  # type: ignore[assignment]
        wrapper._handle_handshake_request()
        assert sent == [{"current_session_name": "foo"}]

    def test_replies_with_new_session_name_during_new_flow(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        sent: list[dict] = []
        wrapper.socket_server.send = lambda msg: bool(sent.append(msg) or True)  # type: ignore[assignment]
        wrapper._pending_action = _PendingAction(
            action_type="new",
            target="",
            handoff={},
            user_prompt="",
            stage="await_handshake",
            new_session_name="new-one",
        )
        wrapper._handle_handshake_request()
        assert sent == [{"current_session_name": "new-one"}]
        assert wrapper._pending_action.stage == "await_new_session_prompt"

    def test_replies_with_none_when_no_initial_and_not_new(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        sent: list[dict] = []
        wrapper.socket_server.send = lambda msg: bool(sent.append(msg) or True)  # type: ignore[assignment]
        wrapper._handle_handshake_request()
        assert sent == [{"current_session_name": None}]


class TestMcpSignalRouting:
    def test_switch_routes_to_handle_switch(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._handle_mcp_signal(
            {"action": "switch", "target": "bar", "handoff": {"user_prompt": "x"}}
        )
        assert wrapper._pending_action is not None
        assert wrapper._pending_action.action_type == "switch"

    def test_new_routes_to_handle_new(self, wrapper: SessionManagerWrapper) -> None:
        wrapper._handle_mcp_signal(
            {
                "action": "new",
                "rename_current": "cur",
                "new_session_name": "new",
                "handoff": {},
            }
        )
        assert wrapper._pending_action is not None
        assert wrapper._pending_action.action_type == "new"

    def test_handshake_request_routes_to_handler(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        sent: list[dict] = []
        wrapper.socket_server.send = lambda msg: bool(sent.append(msg) or True)  # type: ignore[assignment]
        wrapper._handle_mcp_signal({"type": "handshake_request"})
        assert sent == [{"current_session_name": None}]

    def test_invalid_message_ignored(self, wrapper: SessionManagerWrapper) -> None:
        wrapper._handle_mcp_signal("not a dict")  # type: ignore[arg-type]
        wrapper._handle_mcp_signal({})
        assert wrapper._pending_action is None

    def test_switch_missing_target_ignored(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._handle_mcp_signal({"action": "switch", "handoff": {}})
        assert wrapper._pending_action is None

    def test_new_missing_session_name_ignored(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._handle_mcp_signal({"action": "new", "handoff": {}})
        assert wrapper._pending_action is None


class TestVirtualScreenIntegration:
    """Verify PTY chunks reach VirtualScreen and resize stays in sync.
    PTY 청크가 가상 화면에 도달하는지, resize가 동기화되는지 검증.
    """

    def test_init_creates_virtual_screen(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        from session_manager.wrapper.virtual_screen import VirtualScreen

        assert isinstance(wrapper.virtual_screen, VirtualScreen)
        assert wrapper.virtual_screen.get_prompt_line() is None

    def test_handle_pty_readable_feeds_virtual_screen(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chunk = "❯ /test".encode()
        reads = iter([chunk, b""])
        monkeypatch.setattr("os.read", lambda fd, n: next(reads))
        monkeypatch.setattr("os.write", lambda fd, data: len(data))
        wrapper.pty_fd = 0  # any value, os.read is mocked

        assert wrapper._handle_pty_readable() is True
        assert wrapper.virtual_screen.get_prompt_line() == "/test"

    def test_drain_pty_feeds_virtual_screen(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        chunk = "❯ /drained".encode()
        reads = iter([chunk, b""])
        monkeypatch.setattr("os.read", lambda fd, n: next(reads))
        monkeypatch.setattr("os.write", lambda fd, data: len(data))
        wrapper.pty_fd = 0

        wrapper._drain_pty()
        assert wrapper.virtual_screen.get_prompt_line() == "/drained"

    def test_sync_winsize_resizes_virtual_screen(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import termios

        wrapper.pty_fd = 0
        monkeypatch.setattr("os.isatty", lambda fd: True)
        monkeypatch.setattr(termios, "tcgetwinsize", lambda fd: (40, 120))
        monkeypatch.setattr(termios, "tcsetwinsize", lambda fd, size: None)

        wrapper._sync_winsize()
        assert len(wrapper.virtual_screen._screen.display) == 40
        assert len(wrapper.virtual_screen._screen.display[0]) == 120

    def test_sync_winsize_skipped_when_pty_fd_invalid(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Negative pty_fd → early return, virtual screen unchanged.
        pty_fd가 음수면 일찍 return, 가상 화면 변경 없음.
        """
        wrapper.pty_fd = -1
        # Set virtual screen to a known non-default size first
        # 가상 화면을 default가 아닌 크기로 먼저 설정
        wrapper.virtual_screen.resize(120, 40)

        wrapper._sync_winsize()  # should be a no-op
        assert len(wrapper.virtual_screen._screen.display) == 40
        assert len(wrapper.virtual_screen._screen.display[0]) == 120


class TestStdinSubmitInterception:
    """Submit detection (stdin \\r) and intercept entry/exit.
    submit 감지 (stdin \\r) + 가로채기 진입/종료.
    """

    def test_submit_with_match_starts_intercept(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lone \\r with matching prompt text → state set + MCP signal.
        \\r 단독 + 매칭 가능한 prompt → state 설정 + MCP 신호.
        Mode/queue는 변경되지 않음 (큐잉 없이 \\r은 _intercept_state로만 보관).
        """
        wrapper.virtual_screen.feed("❯ /resume foo".encode())
        sent: list[dict] = []
        monkeypatch.setattr(
            wrapper.socket_server, "send",
            lambda msg: sent.append(msg) or True,
        )
        monkeypatch.setattr("os.read", lambda fd, n: b"\r")
        wrapper._stdin_fd = 0

        wrapper._handle_stdin_readable()

        assert wrapper.mode == "passthrough"  # filtering mode 사용 안 함
        assert wrapper.input_queue == b""  # 큐잉 안 함
        assert wrapper._intercept_state is not None
        assert wrapper._intercept_state.command == "resume"
        assert wrapper._intercept_state.args == "foo"
        assert sent == [
            {"action": "intercept", "command": "resume", "args": "foo"}
        ]

    def test_submit_no_match_passes_through(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lone \\r with non-command prompt → forward as normal.
        \\r 단독이지만 prompt가 명령 아님 → 정상 forward.
        """
        wrapper.virtual_screen.feed("❯ hello".encode())
        writes: list[bytes] = []
        monkeypatch.setattr("os.read", lambda fd, n: b"\r")
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )
        wrapper._stdin_fd = 0
        wrapper.pty_fd = 1

        wrapper._handle_stdin_readable()

        assert wrapper.mode == "passthrough"
        assert wrapper._intercept_state is None
        assert writes == [b"\r"]

    def test_non_submit_chunk_skips_match_attempt(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-\\r chunk bypasses match attempt entirely (forwarded as typing).
        \\r 아닌 chunk는 매칭 시도 자체가 없음 (타이핑으로 forward).
        """
        # Even with a matchable prompt, a non-\r chunk must not trigger.
        # 매칭 가능한 prompt가 있어도 \r 아닌 chunk는 trigger되면 안 됨.
        wrapper.virtual_screen.feed("❯ /resume foo".encode())
        writes: list[bytes] = []
        monkeypatch.setattr("os.read", lambda fd, n: b"a")
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )
        wrapper._stdin_fd = 0
        wrapper.pty_fd = 1

        wrapper._handle_stdin_readable()

        assert wrapper.mode == "passthrough"
        assert wrapper._intercept_state is None
        assert writes == [b"a"]

    def test_intercept_active_drops_user_stdin(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Once in intercept, additional stdin is dropped (no PTY write,
        no queue accumulation).
        가로채기 중 추가 stdin은 drop — PTY write 없음, 큐 적재 없음.
        """
        wrapper._intercept_state = _InterceptState(command="resume", args="foo")
        writes: list[bytes] = []
        monkeypatch.setattr("os.read", lambda fd, n: b"hello")
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )
        wrapper._stdin_fd = 0
        wrapper.pty_fd = 1

        wrapper._handle_stdin_readable()

        assert writes == []
        assert wrapper.input_queue == b""
        assert wrapper.mode == "passthrough"

    def test_filtering_mode_still_queues_for_switch_new(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SWITCH/NEW filtering mode keeps the original queueing behaviour
        (separate from interception drop).
        SWITCH/NEW의 filtering mode는 기존 큐잉 동작 유지 (가로채기 drop과 별개).
        """
        wrapper.mode = "filtering"
        wrapper.input_queue = b""
        monkeypatch.setattr("os.read", lambda fd, n: b"hello")
        wrapper._stdin_fd = 0

        wrapper._handle_stdin_readable()

        assert wrapper.input_queue == b"hello"

    def test_intercept_done_signal_finishes_intercept(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`intercept_done` from MCP forwards a single \\r to the PTY.
        MCP의 intercept_done → PTY로 \\r 한 번 forward (큐 없음).
        """
        wrapper._intercept_state = _InterceptState(command="resume", args="foo")
        writes: list[bytes] = []
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )
        wrapper.pty_fd = 1

        wrapper._handle_mcp_signal({"action": "intercept_done"})

        assert wrapper._intercept_state is None
        assert writes == [b"\r"]

    def test_intercept_done_ignored_when_not_active(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        """intercept_done with no active state is a no-op.
        intercept_state가 없을 때 intercept_done은 no-op.
        """
        wrapper._intercept_state = None
        wrapper.mode = "passthrough"

        wrapper._handle_mcp_signal({"action": "intercept_done"})

        assert wrapper.mode == "passthrough"
        assert wrapper._intercept_state is None


class TestInterceptTimeoutAndCancel:
    """Timeout (15s) and Ctrl+C cancellation of an active interception.
    가로채기 timeout (15초) 및 Ctrl+C 취소.
    """

    def test_start_intercept_sets_deadline(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_start_intercept set a deadline INTERCEPT_TIMEOUT_SEC ahead.
        _start_intercept가 INTERCEPT_TIMEOUT_SEC 후로 deadline 설정.
        """
        from session_manager.wrapper.command_matcher import InterceptedCommand

        # Freeze time at a known value
        # 시간 고정
        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.time.monotonic",
            lambda: 1000.0,
        )
        monkeypatch.setattr(
            wrapper.socket_server, "send", lambda msg: True
        )

        wrapper._start_intercept(InterceptedCommand("resume", "foo"))

        assert wrapper._intercept_state is not None
        assert wrapper._intercept_state.deadline == 1000.0 + INTERCEPT_TIMEOUT_SEC

    def test_check_timeout_no_op_when_inactive(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_check_intercept_timeout no-op when no intercept active.
        가로채기 비활성 시 _check_intercept_timeout은 no-op.
        """
        wrapper._intercept_state = None
        writes: list[bytes] = []
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )

        wrapper._check_intercept_timeout()
        assert writes == []

    def test_check_timeout_no_op_before_deadline(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Before deadline, _check_intercept_timeout does nothing.
        deadline 전에는 _check_intercept_timeout이 아무것도 안 함.
        """
        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.time.monotonic",
            lambda: 100.0,
        )
        wrapper._intercept_state = _InterceptState(
            command="resume", args="foo", deadline=200.0
        )
        writes: list[bytes] = []
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )

        wrapper._check_intercept_timeout()

        assert writes == []
        assert wrapper._intercept_state is not None  # 그대로 활성

    def test_check_timeout_forwards_cr_after_deadline(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After deadline: state cleared, notice on stdout, \\r forwarded to PTY.
        deadline 이후: state 정리, stdout 안내, PTY로 \\r forward.
        """
        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.time.monotonic",
            lambda: 300.0,
        )
        wrapper._intercept_state = _InterceptState(
            command="resume", args="foo", deadline=200.0
        )
        wrapper.pty_fd = 1
        wrapper._stdout_fd = 2
        writes: list[tuple[int, bytes]] = []
        monkeypatch.setattr(
            "os.write",
            lambda fd, data: writes.append((fd, data)) or len(data),
        )

        wrapper._check_intercept_timeout()

        assert wrapper._intercept_state is None
        # stdout과 pty 모두 write 발생
        fds_written = [fd for fd, _ in writes]
        assert wrapper._stdout_fd in fds_written
        assert wrapper.pty_fd in fds_written
        # stdout에는 안내 메시지, pty에는 \r
        stdout_data = b"".join(d for fd, d in writes if fd == wrapper._stdout_fd)
        pty_data = b"".join(d for fd, d in writes if fd == wrapper.pty_fd)
        assert b"timeout" in stdout_data
        assert pty_data == b"\r"

    def test_ctrl_c_during_intercept_cancels(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ctrl+C (b\"\\x03\") while intercepting: state cleared, \\x03 forwarded.
        가로채기 중 Ctrl+C: state 정리, PTY로 \\x03 forward (보관 \\r은 폐기).
        """
        wrapper._intercept_state = _InterceptState(
            command="resume", args="foo", deadline=999.0
        )
        wrapper.pty_fd = 1
        writes: list[bytes] = []
        monkeypatch.setattr("os.read", lambda fd, n: CTRL_C)
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )
        wrapper._stdin_fd = 0

        wrapper._handle_stdin_readable()

        assert wrapper._intercept_state is None
        assert writes == [CTRL_C]
        # \r은 forward되지 않음 — 명령 실행 X
        assert b"\r" not in writes

    def test_non_ctrl_c_during_intercept_dropped(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regular keys during intercept are dropped (state stays active).
        가로채기 중 일반 키는 drop (state 그대로 활성 유지).
        """
        wrapper._intercept_state = _InterceptState(
            command="resume", args="foo", deadline=999.0
        )
        wrapper.pty_fd = 1
        writes: list[bytes] = []
        monkeypatch.setattr("os.read", lambda fd, n: b"hello")
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )
        wrapper._stdin_fd = 0

        wrapper._handle_stdin_readable()

        assert wrapper._intercept_state is not None  # 활성 유지
        assert writes == []

    def test_cancel_intercept_helper(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_cancel_intercept clears state and forwards Ctrl+C to PTY.
        _cancel_intercept가 state 정리하고 PTY로 \\x03 forward.
        """
        wrapper._intercept_state = _InterceptState(
            command="exit", args="", deadline=999.0
        )
        wrapper.pty_fd = 1
        writes: list[bytes] = []
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )

        wrapper._cancel_intercept()

        assert wrapper._intercept_state is None
        assert writes == [CTRL_C]


class TestAutoAcceptConfirmations:
    """Auto-accept of channels dev warning + MCP server registration prompts.
    channels dev 경고 + MCP server 등록 prompt 자동 승인.
    """

    def test_injects_cr_when_channels_dev_warning_visible(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Channels dev warning text on screen → \\r injected once.
        가상 화면에 channels dev 경고 텍스트가 있으면 \\r 1회 주입.
        """
        wrapper.virtual_screen.feed(
            b"Some preamble\r\n  I am using this for local development\r\n  Exit"
        )
        wrapper.pty_fd = 1
        writes: list[bytes] = []
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )

        wrapper._auto_accept_confirmations()

        assert b"\r" in writes
        assert "I am using this for local development" in wrapper._handled_confirmations

    def test_each_pattern_handled_at_most_once(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Calling auto-accept twice with same screen → only one \\r.
        같은 화면으로 두 번 호출해도 \\r은 한 번만.
        """
        wrapper.virtual_screen.feed(
            b"  I am using this for local development\r\n"
        )
        wrapper.pty_fd = 1
        writes: list[bytes] = []
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )

        wrapper._auto_accept_confirmations()
        wrapper._auto_accept_confirmations()

        assert writes.count(b"\r") == 1

    def test_no_inject_when_no_pattern_visible(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Random screen content → no injection.
        무관한 화면 → 주입 없음.
        """
        wrapper.virtual_screen.feed(b"Hello world\r\nA normal Claude reply.")
        wrapper.pty_fd = 1
        writes: list[bytes] = []
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )

        wrapper._auto_accept_confirmations()

        assert writes == []
        assert wrapper._handled_confirmations == set()

    def test_handles_multiple_distinct_patterns(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Distinct patterns visible (at different times) → each accepted once.
        서로 다른 패턴 (각각 다른 시점) → 각자 1회씩 승인.
        """
        wrapper.pty_fd = 1
        writes: list[bytes] = []
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )

        # 첫 화면: MCP server 등록
        wrapper.virtual_screen.feed(
            b"  Use this and all future MCP servers in this project\r\n"
        )
        wrapper._auto_accept_confirmations()
        assert writes.count(b"\r") == 1

        # 같은 자식, 두 번째 화면: channels dev 경고
        wrapper.virtual_screen.feed(
            b"  I am using this for local development\r\n"
        )
        wrapper._auto_accept_confirmations()
        assert writes.count(b"\r") == 2

    def test_known_patterns_set(self) -> None:
        """The constant lists exactly the three confirmation prompts we expect.
        상수에 우리가 처리하는 confirmation prompt 3개가 정확히 들어있는지.
        """
        assert AUTO_CONFIRM_PATTERNS == (
            "I am using this for local development",
            "Use this and all future MCP servers",
            "Use this MCP server",
        )

    def test_spawn_resets_handled_set(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A new spawn re-arms confirmations: the set is cleared.
        새 spawn 시 _handled_confirmations 초기화 — 자동 승인 재무장.
        """
        wrapper._handled_confirmations.add("I am using this for local development")
        # _spawn_child calls pexpect.spawn — mock it.
        # _spawn_child가 pexpect.spawn 호출 — mock.
        fake_child = MagicMock()
        fake_child.fileno.return_value = 1
        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.pexpect.spawn",
            lambda *a, **k: fake_child,
        )

        wrapper._spawn_child()

        assert wrapper._handled_confirmations == set()




class TestSummaryTriggers:
    """Background-summary queueing on switch/new signals, /clear, and boot.

    SWITCH/NEW 신호·/clear·부팅 시의 백그라운드 요약 큐 적재.
    """

    @pytest.fixture(autouse=True)
    def _fixed_conversation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pin the active conversation id resolver to a known value.

        활성 conversation id 조회를 고정값으로 고정.
        """
        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.get_active_conversation_id",
            lambda _cwd: "conv-active",
        )

    def _pending(self, wrapper: SessionManagerWrapper) -> list:
        return [
            task
            for _, task in summarizer.load_pending_tasks(Path(wrapper.project_path))
        ]

    def test_switch_signal_enqueues_departed(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._handle_mcp_signal(
            {
                "action": "switch",
                "target": "backend",
                "handoff": {"from": "frontend", "message": "요약"},
            }
        )
        tasks = self._pending(wrapper)
        assert len(tasks) == 1
        assert tasks[0].session_name == "frontend"
        assert tasks[0].conversation_id == "conv-active"
        assert tasks[0].kind == summarizer.KIND_DEPARTED
        assert wrapper._current_session_name == "backend"
        # The worker gets nudged so the task is picked up promptly.
        # 워커가 깨워져 작업을 곧바로 집어가게 된다.
        assert wrapper.summarizer_worker._wakeup.is_set()

    def test_new_signal_enqueues_departed(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._handle_mcp_signal(
            {
                "action": "new",
                "rename_current": "frontend",
                "new_session_name": "payments",
                "handoff": {"from": "frontend", "message": "요약"},
            }
        )
        tasks = self._pending(wrapper)
        assert len(tasks) == 1
        assert tasks[0].session_name == "frontend"
        assert tasks[0].kind == summarizer.KIND_DEPARTED
        assert wrapper._current_session_name == "payments"

    def test_departed_skipped_without_from_session(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        # A fresh unregistered start has handoff.from == None.
        # 미등록 신규 시작은 handoff.from 이 None.
        wrapper._handle_mcp_signal(
            {"action": "switch", "target": "backend", "handoff": {"from": None}}
        )
        assert self._pending(wrapper) == []

    def test_departed_skipped_without_active_conversation(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.get_active_conversation_id",
            lambda _cwd: None,
        )
        wrapper._enqueue_departed_summary("frontend")
        assert self._pending(wrapper) == []

    def test_clear_submit_enqueues_active_and_forwards(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wrapper._current_session_name = "frontend"
        wrapper.virtual_screen.feed("❯ /clear".encode())
        writes: list[bytes] = []
        monkeypatch.setattr("os.read", lambda fd, n: b"\r")
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )
        wrapper._stdin_fd = 0
        wrapper.pty_fd = 1

        wrapper._handle_stdin_readable()

        tasks = self._pending(wrapper)
        assert len(tasks) == 1
        assert tasks[0].session_name == "frontend"
        assert tasks[0].kind == summarizer.KIND_ACTIVE
        # Observation, not interception: the \r is forwarded untouched.
        # 가로채기가 아닌 관찰 — \r 은 그대로 forward 된다.
        assert writes == [b"\r"]
        assert wrapper._intercept_state is None

    def test_clear_with_unknown_session_skips_but_forwards(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wrapper._current_session_name = None
        wrapper.virtual_screen.feed("❯ /clear".encode())
        writes: list[bytes] = []
        monkeypatch.setattr("os.read", lambda fd, n: b"\r")
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )
        wrapper._stdin_fd = 0
        wrapper.pty_fd = 1

        wrapper._handle_stdin_readable()

        assert self._pending(wrapper) == []
        assert writes == [b"\r"]

    def test_clear_regex_requires_exact_command(self) -> None:
        assert CLEAR_COMMAND_RE.match("/clear")
        assert CLEAR_COMMAND_RE.match("/clear  ")
        assert not CLEAR_COMMAND_RE.match("/clearall")
        assert not CLEAR_COMMAND_RE.match("say /clear")

    def _fake_transcripts(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> Path:
        """Redirect transcript lookups to a tmp dir the test can control.

        transcript 조회를 테스트가 제어 가능한 tmp 디렉토리로 우회시킨다.
        """
        transcripts = Path(wrapper.project_path) / "transcripts"
        transcripts.mkdir(parents=True, exist_ok=True)

        def fake_activity(_cwd: Path, conv_ids: object) -> datetime | None:
            newest: float | None = None
            for conv_id in conv_ids:  # type: ignore[union-attr]
                path = transcripts / f"{conv_id}.jsonl"
                if not path.exists():
                    continue
                mtime = path.stat().st_mtime
                if newest is None or mtime > newest:
                    newest = mtime
            return (
                datetime.fromtimestamp(newest, tz=UTC) if newest is not None else None
            )

        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.get_conversation_activity",
            fake_activity,
        )
        return transcripts

    def _write_transcript(self, transcripts: Path, conv_id: str, when: str) -> None:
        path = transcripts / f"{conv_id}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        ts = datetime.fromisoformat(when).timestamp()
        os.utime(path, (ts, ts))

    def test_boot_recovery_enqueues_only_stale_sessions(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Staleness is decided by transcript mtime, not last_accessed.

        stale 판정은 last_accessed 가 아니라 transcript mtime 으로 한다 —
        last_accessed 는 세션을 건드리는 도구 호출 시에만 기록되어 사용 중인
        세션에서도 낡을 수 있다.
        """
        transcripts = self._fake_transcripts(wrapper, monkeypatch)
        project = Path(wrapper.project_path)
        store = SessionStore(project)
        store.init_project()

        # Summarised yesterday, transcript written today → stale.
        # 어제 요약, 오늘 transcript 기록 → stale.
        stale = SessionMetadata.new(name="stale", title="요약 유실")
        stale.claude_conversation_ids = ["conv-stale"]
        stale.summary_updated_at = "2026-07-29T00:00:00+00:00"
        store.save_session(stale)
        self._write_transcript(transcripts, "conv-stale", "2026-07-30T00:00:00+00:00")

        never = SessionMetadata.new(name="never", title="요약 없음")
        never.claude_conversation_ids = ["conv-never"]
        store.save_session(never)
        self._write_transcript(transcripts, "conv-never", "2026-07-30T00:00:00+00:00")

        # last_accessed is stale but the summary postdates the transcript —
        # the old predicate would have re-summarised this needlessly.
        # last_accessed 는 낡았지만 요약이 transcript 보다 최신 — 옛 술어라면
        # 불필요하게 재요약했을 세션.
        fresh = SessionMetadata.new(name="fresh", title="요약 최신")
        fresh.claude_conversation_ids = ["conv-fresh"]
        fresh.last_accessed = "2026-07-01T00:00:00+00:00"
        fresh.summary_updated_at = "2026-07-30T00:00:00+00:00"
        store.save_session(fresh)
        self._write_transcript(transcripts, "conv-fresh", "2026-07-29T00:00:00+00:00")

        # Transcript removed by Claude Code's own cleanup — nothing to summarise.
        # Claude Code 자체 정리로 transcript 소멸 — 요약할 대상 없음.
        gone = SessionMetadata.new(name="gone", title="대화 파일 소멸")
        gone.claude_conversation_ids = ["conv-gone"]
        store.save_session(gone)

        no_conv = SessionMetadata.new(name="no-conv", title="대화 없음")
        store.save_session(no_conv)

        wrapper._enqueue_stale_summaries()

        names = sorted(t.session_name for t in self._pending(wrapper))
        assert names == ["never", "stale"]
        by_name = {t.session_name: t for t in self._pending(wrapper)}
        assert by_name["stale"].conversation_id == "conv-stale"
        assert by_name["stale"].kind == summarizer.KIND_DEPARTED

    def test_boot_recovery_does_not_double_queue(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transcripts = self._fake_transcripts(wrapper, monkeypatch)
        project = Path(wrapper.project_path)
        store = SessionStore(project)
        store.init_project()
        stale = SessionMetadata.new(name="stale", title="요약 유실")
        stale.claude_conversation_ids = ["conv-stale"]
        store.save_session(stale)
        self._write_transcript(transcripts, "conv-stale", "2026-07-30T00:00:00+00:00")

        wrapper._enqueue_stale_summaries()
        wrapper._enqueue_stale_summaries()

        assert len(self._pending(wrapper)) == 1

    def test_boot_recovery_survives_store_errors(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.SessionStore",
            lambda _p: (_ for _ in ()).throw(OSError("disk gone")),
        )
        wrapper._enqueue_stale_summaries()  # must not raise / 예외 없이 통과


class TestPeriodicSummaryRefresh:
    """Growth-based refresh: re-summarise once new dialogue exceeds the window.

    증가량 기반 갱신 — 새 대화가 발췌 창을 넘으면 재요약한다.
    """

    @pytest.fixture
    def transcript(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> Path:
        """Point the wrapper at a controllable transcript for "conv-1".

        래퍼가 "conv-1" 에 대해 제어 가능한 transcript 를 보도록 한다.
        """
        # A per-test fake HOME inside the project dir, so transcripts from
        # one test can't leak into another.
        # 프로젝트 디렉토리 안의 테스트 전용 가짜 HOME — 한 테스트의
        # transcript 가 다른 테스트로 새지 않게 한다.
        fake_home = Path(wrapper.project_path) / "home"
        target = fake_home / ".claude" / "projects" / "enc"
        target.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.get_active_conversation_id",
            lambda _cwd: "conv-1",
        )
        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.encode_cwd", lambda _p: "enc"
        )
        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.Path.home",
            classmethod(lambda _cls: fake_home),
        )
        return target / "conv-1.jsonl"

    def _append_dialogue(self, path: Path, chars: int) -> None:
        event = {"type": "user", "message": {"content": "가" * chars}}
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _session(self, wrapper: SessionManagerWrapper, **kwargs: object) -> None:
        store = SessionStore(Path(wrapper.project_path))
        store.init_project()
        session = SessionMetadata.new(name="work", title="작업")
        for key, value in kwargs.items():
            setattr(session, key, value)
        store.save_session(session)
        wrapper._current_session_name = "work"

    def _pending(self, wrapper: SessionManagerWrapper) -> list:
        return [
            task
            for _, task in summarizer.load_pending_tasks(Path(wrapper.project_path))
        ]

    def test_no_refresh_below_threshold(
        self, wrapper: SessionManagerWrapper, transcript: Path
    ) -> None:
        self._session(wrapper)
        self._append_dialogue(transcript, EXCERPT_MAX_CHARS - 1)
        wrapper._check_summary_refresh()
        assert self._pending(wrapper) == []

    def test_refresh_at_threshold(
        self, wrapper: SessionManagerWrapper, transcript: Path
    ) -> None:
        self._session(wrapper)
        self._append_dialogue(transcript, EXCERPT_MAX_CHARS)
        wrapper._check_summary_refresh()
        tasks = self._pending(wrapper)
        assert len(tasks) == 1
        assert tasks[0].kind == summarizer.KIND_ACTIVE
        assert tasks[0].session_name == "work"

    def test_growth_measured_against_recorded_baseline(
        self, wrapper: SessionManagerWrapper, transcript: Path
    ) -> None:
        """Only dialogue the last summary didn't see counts.

        마지막 요약이 보지 못한 대화만 센다.
        """
        self._append_dialogue(transcript, EXCERPT_MAX_CHARS)
        self._session(
            wrapper,
            summary_dialogue_chars=EXCERPT_MAX_CHARS,
            summary_dialogue_conversation_id="conv-1",
        )
        wrapper._check_summary_refresh()
        assert self._pending(wrapper) == []

        self._append_dialogue(transcript, EXCERPT_MAX_CHARS)
        wrapper._check_summary_refresh()
        assert len(self._pending(wrapper)) == 1

    def test_baseline_from_other_conversation_ignored(
        self, wrapper: SessionManagerWrapper, transcript: Path
    ) -> None:
        """A baseline measured elsewhere must not silence the trigger.

        다른 conversation 에서 측정한 기준값이 트리거를 침묵시키면 안 된다
        (롤오버 후 증가량이 음수가 되는 버그 방지).
        """
        self._append_dialogue(transcript, EXCERPT_MAX_CHARS)
        self._session(
            wrapper,
            summary_dialogue_chars=999_999,
            summary_dialogue_conversation_id="conv-OLD",
        )
        wrapper._check_summary_refresh()
        assert len(self._pending(wrapper)) == 1

    def test_incremental_scan_advances(
        self, wrapper: SessionManagerWrapper, transcript: Path
    ) -> None:
        """Each turn parses only what was appended since the last one.

        매 턴은 직전 이후 append 된 부분만 파싱한다.
        """
        self._session(wrapper)
        self._append_dialogue(transcript, 100)
        wrapper._check_summary_refresh()
        first_offset = wrapper._dialogue_scan_offset
        assert wrapper._dialogue_scan_chars == 100

        self._append_dialogue(transcript, 100)
        wrapper._check_summary_refresh()
        assert wrapper._dialogue_scan_offset > first_offset
        assert wrapper._dialogue_scan_chars == 200

    def test_skipped_without_current_session(
        self, wrapper: SessionManagerWrapper, transcript: Path
    ) -> None:
        self._session(wrapper)
        wrapper._current_session_name = None
        self._append_dialogue(transcript, EXCERPT_MAX_CHARS)
        wrapper._check_summary_refresh()
        assert self._pending(wrapper) == []
