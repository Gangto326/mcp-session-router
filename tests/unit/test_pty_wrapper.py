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
    INVERSE_VIDEO_START,
    OUTPUT_BUFFER_CAP,
    OUTPUT_BUFFER_TAIL_KEEP,
    PROMPT_POINTER,
    SessionManagerWrapper,
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


class TestSessionCommandObservation:
    """Session-changing slash commands are observed, never held.

    세션 변경 슬래시 명령은 관찰만 하고 붙잡지 않는다.

    The old flow withheld the \\r while asking the in-session LLM for a
    summary, freezing the input line for up to 15 seconds. The background
    summariser removed that dependency.
    옛 흐름은 세션 안 LLM 에게 요약을 부탁하는 동안 \\r 을 보관해 입력란을
    최대 15초 얼렸다. 백그라운드 요약기가 그 의존을 없앴다.
    """

    @pytest.fixture(autouse=True)
    def _fixed_conversation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.get_active_conversation_id",
            lambda _cwd: "conv-active",
        )

    def _pending(self, wrapper: SessionManagerWrapper) -> list:
        return [
            task
            for _, task in summarizer.load_pending_tasks(Path(wrapper.project_path))
        ]

    def test_command_forwarded_immediately_and_summarised(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wrapper._current_session_name = "work"
        wrapper.virtual_screen.feed("❯ /resume foo".encode())
        sent: list[dict] = []
        writes: list[bytes] = []
        monkeypatch.setattr(
            wrapper.socket_server, "send", lambda msg: sent.append(msg) or True
        )
        monkeypatch.setattr("os.read", lambda fd, n: b"\r")
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )
        wrapper._stdin_fd = 0
        wrapper.pty_fd = 1

        wrapper._handle_stdin_readable()

        # The keystroke goes through in the same call — no holding, no delay.
        # 키 입력이 같은 호출 안에서 통과한다 — 보관도 지연도 없다.
        assert writes == [b"\r"]
        assert wrapper.mode == "passthrough"
        assert wrapper.input_queue == b""
        # The departing conversation is queued for a background summary.
        # 떠나는 conversation 이 백그라운드 요약 큐에 들어간다.
        tasks = self._pending(wrapper)
        assert len(tasks) == 1
        assert tasks[0].session_name == "work"
        assert tasks[0].kind == summarizer.KIND_ACTIVE
        # The MCP server is told its session pointer is now stale.
        # MCP 서버에 세션 포인터가 낡았음을 알린다.
        assert sent == [
            {"action": "session_command", "command": "resume", "args": "foo"}
        ]

    def test_exit_without_args_observed(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wrapper._current_session_name = "work"
        wrapper.virtual_screen.feed("❯ /exit".encode())
        sent: list[dict] = []
        monkeypatch.setattr(
            wrapper.socket_server, "send", lambda msg: sent.append(msg) or True
        )
        monkeypatch.setattr("os.read", lambda fd, n: b"\r")
        monkeypatch.setattr("os.write", lambda fd, data: len(data))
        wrapper._stdin_fd = 0
        wrapper.pty_fd = 1

        wrapper._handle_stdin_readable()

        assert sent == [
            {"action": "session_command", "command": "exit", "args": ""}
        ]

    def test_unknown_session_still_forwards(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Observation failure must never block the user's command.

        관찰 실패가 사용자 명령을 막으면 안 된다.
        """
        wrapper._current_session_name = None
        wrapper.virtual_screen.feed("❯ /resume foo".encode())
        writes: list[bytes] = []
        monkeypatch.setattr(wrapper.socket_server, "send", lambda msg: True)
        monkeypatch.setattr("os.read", lambda fd, n: b"\r")
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )
        wrapper._stdin_fd = 0
        wrapper.pty_fd = 1

        wrapper._handle_stdin_readable()

        assert writes == [b"\r"]
        assert self._pending(wrapper) == []

    def test_submit_no_match_passes_through(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wrapper._current_session_name = "work"
        wrapper.virtual_screen.feed("❯ hello".encode())
        sent: list[dict] = []
        writes: list[bytes] = []
        monkeypatch.setattr(
            wrapper.socket_server, "send", lambda msg: sent.append(msg) or True
        )
        monkeypatch.setattr("os.read", lambda fd, n: b"\r")
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )
        wrapper._stdin_fd = 0
        wrapper.pty_fd = 1

        wrapper._handle_stdin_readable()

        assert writes == [b"\r"]
        assert sent == []
        assert self._pending(wrapper) == []

    def test_non_submit_chunk_is_not_observed(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Typing the command text is not submitting it.

        명령 텍스트를 타이핑하는 것은 제출이 아니다.
        """
        wrapper._current_session_name = "work"
        wrapper.virtual_screen.feed("❯ /resume foo".encode())
        sent: list[dict] = []
        writes: list[bytes] = []
        monkeypatch.setattr(
            wrapper.socket_server, "send", lambda msg: sent.append(msg) or True
        )
        monkeypatch.setattr("os.read", lambda fd, n: b"o")
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )
        wrapper._stdin_fd = 0
        wrapper.pty_fd = 1

        wrapper._handle_stdin_readable()

        assert writes == [b"o"]
        assert sent == []
        assert self._pending(wrapper) == []

    def test_filtering_mode_still_queues_input(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SWITCH/NEW injection still buffers user keystrokes.

        SWITCH/NEW 주입 중에는 여전히 사용자 키 입력을 큐잉한다.
        """
        wrapper.mode = "filtering"
        monkeypatch.setattr("os.read", lambda fd, n: b"typed")
        wrapper._stdin_fd = 0

        wrapper._handle_stdin_readable()

        assert wrapper.input_queue == b"typed"


class TestAutoAcceptConfirmations:
    """Auto-accept of the MCP server registration prompts.
    MCP server 등록 prompt 자동 승인.
    """

    def test_injects_cr_when_registration_prompt_visible(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Registration prompt on screen → \\r injected once.
        가상 화면에 등록 prompt 텍스트가 있으면 \\r 1회 주입.
        """
        wrapper.virtual_screen.feed(
            b"Some preamble\r\n  Use this and all future MCP servers\r\n  Exit"
        )
        wrapper.pty_fd = 1
        writes: list[bytes] = []
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )

        wrapper._auto_accept_confirmations()

        assert b"\r" in writes
        assert "Use this and all future MCP servers" in wrapper._handled_confirmations

    def test_each_pattern_handled_at_most_once(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Calling auto-accept twice with same screen → only one \\r.
        같은 화면으로 두 번 호출해도 \\r은 한 번만.
        """
        wrapper.virtual_screen.feed(
            b"  Use this and all future MCP servers\r\n"
        )
        wrapper.pty_fd = 1
        writes: list[bytes] = []
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )

        wrapper._auto_accept_confirmations()
        wrapper._auto_accept_confirmations()

        assert writes.count(b"\r") == 1

    def test_window_closed_after_first_user_keystroke(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After the user has typed, pattern on screen must NOT fire (F13).
        사용자 입력 이후에는 화면의 패턴이 발사되면 안 된다 (F13) —
        LLM 인용·파일 표시 오발사 방어.
        """
        wrapper._auto_confirm_armed = False  # 첫 키 입력이 닫은 상태
        wrapper.virtual_screen.feed(
            b"  Use this and all future MCP servers\r\n"
        )
        wrapper.pty_fd = 1
        writes: list[bytes] = []
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )

        wrapper._auto_accept_confirmations()

        assert writes == []
        assert wrapper._handled_confirmations == set()

    def test_first_stdin_keystroke_closes_window(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A real user keystroke through stdin closes the window.
        stdin 을 통한 실제 사용자 키 입력이 윈도우를 닫는다.
        """
        assert wrapper._auto_confirm_armed is True
        wrapper.pty_fd = 1
        monkeypatch.setattr(wrapper, "_stdin_fd", 0)
        monkeypatch.setattr("os.read", lambda fd, n: b"h")
        monkeypatch.setattr("os.write", lambda fd, data: len(data))

        wrapper._handle_stdin_readable()

        assert wrapper._auto_confirm_armed is False

    def test_respawn_rearms_window(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        """NEW respawn re-opens the window for the next child's boot dialogs.
        NEW respawn 시 다음 자식의 부팅 다이얼로그를 위해 윈도우가 다시 열린다.
        """
        wrapper._auto_confirm_armed = False
        wrapper._handled_confirmations = {"Use this MCP server"}

        wrapper._reset_child_detection_state()

        assert wrapper._auto_confirm_armed is True
        assert wrapper._handled_confirmations == set()

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

        # 같은 자식, 두 번째 화면: 다른 등록 prompt 변형
        wrapper.virtual_screen.feed(b"  Use this MCP server only\r\n")
        wrapper._auto_accept_confirmations()
        assert writes.count(b"\r") == 2

    def test_known_patterns_set(self) -> None:
        """Only the MCP registration prompts are auto-accepted.

        MCP 등록 prompt 만 자동 승인 대상이다 — channels 개발 플래그를 더
        이상 붙이지 않으므로 그 경고 화면은 아예 뜨지 않는다.
        """
        assert AUTO_CONFIRM_PATTERNS == (
            "Use this and all future MCP servers",
            "Use this MCP server",
        )

    def test_spawn_resets_handled_set(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A new spawn re-arms confirmations: the set is cleared.
        새 spawn 시 _handled_confirmations 초기화 — 자동 승인 재무장.
        """
        wrapper._handled_confirmations.add("Use this and all future MCP servers")
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
        # Observed, not held: the \r is forwarded untouched.
        # 붙잡지 않고 관찰만 — \r 은 그대로 forward 된다.
        assert writes == [b"\r"]

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


class TestSpawnEnvIsolation:
    """The spawned claude must not inherit CLAUDE_CODE_CHILD_SESSION (F19).

    spawn 된 claude 는 CLAUDE_CODE_CHILD_SESSION 을 상속하면 안 된다 (F19).

    Inherited, it makes an interactive claude write no transcript JSONL —
    silently disabling every transcript-based feature. Isolated to this
    single variable by experiment (docs/review/2026-07-30-verification.md §5차).
    상속되면 대화형 claude 가 transcript JSONL 을 전혀 쓰지 않아 transcript
    기반 기능 전체가 조용히 꺼진다. 실험으로 이 변수 하나로 특정됨.
    """

    def _spawn_env(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> dict:
        captured: dict = {}

        def fake_spawn(*args: object, **kwargs: object) -> MagicMock:
            captured.update(kwargs)
            child = MagicMock()
            child.fileno.return_value = 1
            return child

        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.pexpect.spawn", fake_spawn
        )
        wrapper._spawn_child()
        return captured["env"]  # type: ignore[return-value]

    def test_child_session_marker_stripped(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDE_CODE_CHILD_SESSION", "true")
        env = self._spawn_env(wrapper, monkeypatch)
        assert "CLAUDE_CODE_CHILD_SESSION" not in env

    def test_everything_else_preserved(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the proven-harmful variable goes — nothing else (rule 8).

        입증된 변수 하나만 제거한다 — 나머지는 그대로 (규칙 8).
        """
        monkeypatch.setenv("CLAUDE_CODE_CHILD_SESSION", "true")
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent-sid")
        monkeypatch.setenv("SESSION_MANAGER_SOCKET", "/tmp/x.sock")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        env = self._spawn_env(wrapper, monkeypatch)
        assert env["CLAUDECODE"] == "1"
        assert env["CLAUDE_CODE_SESSION_ID"] == "parent-sid"
        assert env["SESSION_MANAGER_SOCKET"] == "/tmp/x.sock"
        assert env["ANTHROPIC_API_KEY"] == "sk-test"


class TestJudgeWiring:
    """Wrapper-side wiring of the routing judge (R2-C3).

    라우팅 판정기의 래퍼 측 연결 (R2-C3) 테스트.
    """

    def test_judge_request_routed_to_judge_host(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[dict, object]] = []
        monkeypatch.setattr(
            wrapper.judge_host,
            "handle_request",
            lambda message, sock: bool(calls.append((message, sock))) or True,
        )
        fake_sock = object()
        message = {"client": "hook", "action": "judge_request", "prompt": "p"}
        assert wrapper._handle_hook_message(message, fake_sock) is True
        assert calls == [(message, fake_sock)]

    def test_other_hook_messages_fall_back_to_ack(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        assert (
            wrapper._handle_hook_message(
                {"client": "hook", "action": "route_switch"}, object()
            )
            is False
        )

    def _seed_two_sessions(self, wrapper: SessionManagerWrapper) -> None:
        store = SessionStore(Path(wrapper.project_path))
        store.init_project()
        store.save_session(SessionMetadata.new(name="a", title="A"))
        store.save_session(SessionMetadata.new(name="b", title="B"))

    def test_judge_starts_when_routable(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._seed_two_sessions(wrapper)
        started: list[bool] = []
        monkeypatch.setattr(
            wrapper.judge_host, "ensure_started", lambda: started.append(True)
        )
        wrapper._maybe_start_judge()
        assert started == [True]

    def test_judge_not_started_below_two_sessions(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started: list[bool] = []
        monkeypatch.setattr(
            wrapper.judge_host, "ensure_started", lambda: started.append(True)
        )
        wrapper._maybe_start_judge()
        assert started == []

    def test_judge_not_started_when_routing_off(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._seed_two_sessions(wrapper)
        config_path = (
            Path(wrapper.project_path) / ".session-manager" / "config.json"
        )
        config_path.write_text(
            json.dumps({"routing_mode": "off"}), encoding="utf-8"
        )
        started: list[bool] = []
        monkeypatch.setattr(
            wrapper.judge_host, "ensure_started", lambda: started.append(True)
        )
        wrapper._maybe_start_judge()
        assert started == []

    def test_mcp_signal_rechecks_judge_start(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._seed_two_sessions(wrapper)
        started: list[bool] = []
        monkeypatch.setattr(
            wrapper.judge_host, "ensure_started", lambda: started.append(True)
        )
        wrapper._handle_mcp_signal({"action": "current_session", "name": "a"})
        assert started == [True]


class TestRouteSwitchExecution:
    """route_switch (hook auto path) execution on the wrapper side.

    hook 자동 경로 (route_switch) 의 래퍼 측 실행 테스트.
    """

    def test_route_switch_registers_pending_with_mirror_from(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._current_session_name = "frontend"
        wrapper._handle_mcp_signal(
            {
                "client": "hook",
                "action": "route_switch",
                "target": "backend",
                "user_prompt": "로그인 API가 500을 뱉는다",
                "verdict": {"action": "SWITCH", "reason": "인증 소관"},
            }
        )
        pending = wrapper._pending_action
        assert pending is not None
        assert pending.action_type == "switch"
        assert pending.target == "backend"
        assert pending.user_prompt == "로그인 API가 500을 뱉는다"
        assert pending.handoff == {"from": "frontend", "router_reason": "인증 소관"}
        assert wrapper._current_session_name == "backend"

    def test_route_switch_without_target_ignored(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._handle_mcp_signal(
            {"client": "hook", "action": "route_switch", "user_prompt": "x"}
        )
        assert wrapper._pending_action is None

    def test_route_switch_without_verdict_reason(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._current_session_name = None
        wrapper._handle_mcp_signal(
            {"client": "hook", "action": "route_switch", "target": "backend"}
        )
        pending = wrapper._pending_action
        assert pending is not None
        assert pending.handoff == {"from": None}
        assert pending.user_prompt == ""


class TestResolveResumeArg:
    """Conversation-id resolution for the /resume injection (P2-h).

    /resume 주입 인자의 conversation id 해석 (P2-h 실측 근거) 테스트.
    """

    def _seed(self, wrapper: SessionManagerWrapper, conv_ids: list[str]) -> None:
        store = SessionStore(Path(wrapper.project_path))
        store.init_project()
        session = SessionMetadata.new(name="backend", title="API")
        session.claude_conversation_ids = conv_ids
        store.save_session(session)

    def test_latest_conversation_id_preferred(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        self._seed(wrapper, ["conv-old", "conv-new"])
        assert wrapper._resolve_resume_arg("backend") == "conv-new"

    def test_no_ids_falls_back_to_name(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        self._seed(wrapper, [])
        assert wrapper._resolve_resume_arg("backend") == "backend"

    def test_unknown_session_falls_back_to_name(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        assert wrapper._resolve_resume_arg("ghost") == "ghost"

    def test_advance_switch_injects_conversation_id(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._seed(wrapper, ["conv-a", "conv-b"])
        injected = _capture_injects(wrapper, monkeypatch)
        pending = _PendingAction(
            action_type="switch",
            target="backend",
            handoff={},
            user_prompt="hi",
            stage="await_resume_prompt",
        )
        wrapper._pending_action = pending
        wrapper._advance_switch(pending)
        assert injected == [b"/resume conv-b"]
