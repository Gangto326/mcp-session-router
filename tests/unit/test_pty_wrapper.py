"""
Unit tests for SessionManagerWrapper internals.

PTY 래퍼의 내부 로직 단위 테스트. PTY 의존 메서드는 monkeypatch 로 mock,
소켓·SIGWINCH·실런타임 동작은 통합 테스트로 이관한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from session_manager.wrapper.pty_wrapper import (
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

    def test_not_detected_inverse_too_far_from_pointer(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        # 64바이트 윈도우 밖의 inverse 시퀀스는 매칭에서 제외
        buffer = PROMPT_POINTER + b"x" * 100 + INVERSE_VIDEO_START
        assert wrapper._detect_prompt(buffer) is False

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
        """Lone \\r with matching prompt text → enter filtering, queue, signal.
        \\r 단독 + 매칭 가능한 prompt → filtering 진입, 큐잉, MCP 신호.
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

        assert wrapper.mode == "filtering"
        assert wrapper.input_queue == b"\r"
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

    def test_filtering_mode_queues_additional_input(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Once in filtering, extra stdin chunks accumulate in input_queue.
        filtering 진입 후 추가 stdin은 input_queue에 적재.
        """
        wrapper.mode = "filtering"
        wrapper.input_queue = b"\r"
        monkeypatch.setattr("os.read", lambda fd, n: b"hello")
        wrapper._stdin_fd = 0

        wrapper._handle_stdin_readable()

        assert wrapper.input_queue == b"\rhello"

    def test_intercept_done_signal_finishes_intercept(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`intercept_done` from MCP drains the queue raw and exits filtering.
        MCP의 intercept_done → 큐 원본 그대로 forward + filtering 종료.
        """
        wrapper._intercept_state = _InterceptState(command="resume", args="foo")
        wrapper.mode = "filtering"
        wrapper.input_queue = b"\rhello"
        writes: list[bytes] = []
        monkeypatch.setattr(
            "os.write", lambda fd, data: writes.append(data) or len(data)
        )
        wrapper.pty_fd = 1

        wrapper._handle_mcp_signal({"action": "intercept_done"})

        assert wrapper.mode == "passthrough"
        assert wrapper._intercept_state is None
        assert wrapper.input_queue == b""
        assert writes == [b"\rhello"]

    def test_intercept_done_ignored_when_not_active(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        """intercept_done with no active state is a no-op (no mode/queue change).
        intercept_state가 없을 때 intercept_done은 no-op.
        """
        wrapper._intercept_state = None
        wrapper.mode = "passthrough"
        wrapper.input_queue = b"untouched"

        wrapper._handle_mcp_signal({"action": "intercept_done"})

        assert wrapper.mode == "passthrough"
        assert wrapper.input_queue == b"untouched"


