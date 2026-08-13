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

from session_manager import handoff_store, summarizer
from session_manager.models import SessionMetadata
from session_manager.routing import decision_log
from session_manager.storage.file_store import SessionStore
from session_manager.transcript_excerpt import EXCERPT_MAX_CHARS
from session_manager.wrapper import wrapper_state
from session_manager.wrapper.pty_wrapper import (
    AUTO_CONFIRM_PATTERNS,
    BUSY_MARKER,
    CLEAR_COMMAND_RE,
    SessionManagerWrapper,
    _PendingRespawn,
)


@pytest.fixture
def wrapper(tmp_path: Path) -> SessionManagerWrapper:
    return SessionManagerWrapper(
        socket_path=str(tmp_path / "test.sock"),
        claude_args=[],
        project_path=str(tmp_path),
    )


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




# ---- Respawn-based transitions (redesign, docs/poc/R3-respawn.md) --------


def _seed_resumable_session(
    wrapper: SessionManagerWrapper,
    monkeypatch: pytest.MonkeyPatch,
    fake_home: Path,
    name: str = "frontend",
    conv: str = "conv-1",
) -> None:
    """Register *name* with a conversation whose transcript exists.

    transcript 가 실존하는 conversation 을 가진 세션 *name* 을 등록.
    """
    from session_manager.claude_conversation import encode_cwd

    monkeypatch.setattr(Path, "home", lambda: fake_home)
    session = SessionMetadata.new(name=name, title="t", summary="s")
    session.claude_conversation_ids = [conv]
    SessionStore(Path(wrapper.project_path)).save_session(session)
    transcript_dir = (
        fake_home / ".claude" / "projects" / encode_cwd(Path(wrapper.project_path))
    )
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (transcript_dir / f"{conv}.jsonl").write_text("{}\n", encoding="utf-8")


class TestResolveResumeConv:
    def test_unknown_session_returns_none(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        assert wrapper._resolve_resume_conv("ghost") is None

    def test_session_without_conversations_returns_none(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        SessionStore(Path(wrapper.project_path)).save_session(
            SessionMetadata.new(name="empty", title="t")
        )
        assert wrapper._resolve_resume_conv("empty") is None

    def test_stale_transcript_returns_none(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Conversation id recorded but Claude Code's own cleanup removed
        # the transcript — resuming it would kill the child at boot.
        # id 는 기록됐지만 Claude Code 자체 정리로 transcript 소멸 —
        # resume 하면 자식이 부팅에서 죽는다.
        fake_home = tmp_path / "home"
        monkeypatch.setattr(Path, "home", lambda: fake_home)
        session = SessionMetadata.new(name="stale", title="t")
        session.claude_conversation_ids = ["gone-conv"]
        SessionStore(Path(wrapper.project_path)).save_session(session)
        assert wrapper._resolve_resume_conv("stale") is None

    def test_live_transcript_returns_latest_id(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _seed_resumable_session(wrapper, monkeypatch, tmp_path / "home")
        assert wrapper._resolve_resume_conv("frontend") == "conv-1"


class TestExecuteTransition:
    def test_registers_pending_and_writes_handoff_file(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._execute_transition(
            target="backend",
            resume_conv="conv-9",
            handoff={"from": "frontend", "message": "m", "user_prompt": "dup"},
            user_prompt="원래 프롬프트",
        )
        pending = wrapper._pending_respawn
        assert pending is not None
        assert pending.target == "backend"
        assert pending.resume_conv == "conv-9"
        assert pending.from_name == "frontend"
        assert pending.user_prompt == "원래 프롬프트"
        assert pending.terminated is False
        # Mirror moves immediately; the handoff content (minus the
        # duplicated user_prompt key) lands in the pending file.
        # 미러는 즉시 이동, handoff 내용 (중복 user_prompt 키 제외) 은
        # pending 파일로.
        assert wrapper._current_session_name == "backend"
        stored = handoff_store.take_pending(Path(wrapper.project_path))
        assert stored is not None
        assert stored["target"] == "backend"
        assert stored["user_prompt"] == "원래 프롬프트"
        assert stored["handoff"] == {"from": "frontend", "message": "m"}

    def test_second_transition_dropped_while_pending(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._execute_transition(
            target="a", resume_conv=None, handoff={}, user_prompt=""
        )
        wrapper._execute_transition(
            target="b", resume_conv=None, handoff={}, user_prompt=""
        )
        assert wrapper._pending_respawn is not None
        assert wrapper._pending_respawn.target == "a"


class TestMaybeTerminateForRespawn:
    def _arm(self, wrapper: SessionManagerWrapper) -> list:
        kills: list[tuple[int, int]] = []
        wrapper.child = MagicMock(pid=4242)
        wrapper._pending_respawn = _PendingRespawn(
            target="backend", resume_conv="conv-1"
        )
        return kills

    def test_no_pending_is_noop(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kills: list = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append(pid))
        wrapper._maybe_terminate_for_respawn()
        assert kills == []

    def test_busy_holds_the_swap(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # /back mid-response or an MCP switch signalled mid-turn: the
        # in-flight reply must land in the transcript first.
        # 응답 중 /back·턴 중 MCP 전환 — 진행 중 응답이 먼저 기록돼야 한다.
        kills = self._arm(wrapper)
        monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append(pid))
        monkeypatch.setattr(
            wrapper.virtual_screen,
            "contains_near_prompt",
            lambda needle, radius: needle == BUSY_MARKER,
        )
        wrapper._maybe_terminate_for_respawn()
        assert kills == []
        assert wrapper._pending_respawn.terminated is False

    def test_idle_terminates_child_once(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kills = self._arm(wrapper)
        monkeypatch.setattr(os, "kill", lambda pid, sig: kills.append((pid, sig)))
        monkeypatch.setattr(
            wrapper.virtual_screen,
            "contains_near_prompt",
            lambda needle, radius: False,
        )
        wrapper._maybe_terminate_for_respawn()
        wrapper._maybe_terminate_for_respawn()  # 멱등 — 재전송 없음
        import signal as _signal

        assert kills == [(4242, _signal.SIGTERM)]
        assert wrapper._pending_respawn.terminated is True


class TestShouldRespawn:
    def test_false_without_pending(self, wrapper: SessionManagerWrapper) -> None:
        wrapper.child = MagicMock(exitstatus=0, signalstatus=None)
        assert wrapper._should_respawn() is False

    def test_true_with_pending(self, wrapper: SessionManagerWrapper) -> None:
        wrapper.child = MagicMock(exitstatus=None, signalstatus=15)
        wrapper._pending_respawn = _PendingRespawn(target="t", resume_conv=None)
        assert wrapper._should_respawn() is True


class TestBuildChildArgs:
    def test_plain_boot_keeps_user_args_and_guide(
        self, tmp_path: Path
    ) -> None:
        w = SessionManagerWrapper(
            socket_path=str(tmp_path / "s.sock"),
            claude_args=["--model", "opus"],
            project_path=str(tmp_path),
        )
        args = w._build_child_args()
        assert args[:2] == ["--model", "opus"]
        assert any(a.startswith("--append-system-prompt=") for a in args)
        # No trigger prompt without a pending transition.
        # pending 전환이 없으면 트리거 프롬프트도 없다.
        assert handoff_store.TRIGGER_PROMPT not in args

    def test_respawn_appends_resume_and_trigger(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._pending_respawn = _PendingRespawn(
            target="backend", resume_conv="conv-9"
        )
        args = wrapper._build_child_args()
        # `=` form is mandatory: the space form greedily swallows the
        # trailing positional trigger (measured, docs/poc/R3-respawn.md).
        # `=` 형식 필수 — 공백 형식은 뒤의 트리거를 삼킨다 (실측).
        assert "--resume=conv-9" in args
        assert args[-1] == handoff_store.TRIGGER_PROMPT
        assert "--resume" not in args

    def test_new_respawn_has_trigger_but_no_resume(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._pending_respawn = _PendingRespawn(target="fresh", resume_conv=None)
        args = wrapper._build_child_args()
        assert not any(a.startswith("--resume") for a in args)
        assert args[-1] == handoff_store.TRIGGER_PROMPT

    def test_respawn_strips_user_resume_args(self, tmp_path: Path) -> None:
        w = SessionManagerWrapper(
            socket_path=str(tmp_path / "s.sock"),
            claude_args=["--resume", "old-conv", "--model", "opus"],
            project_path=str(tmp_path),
        )
        w._pending_respawn = _PendingRespawn(target="t", resume_conv="new-conv")
        args = w._build_child_args()
        assert "old-conv" not in args
        assert "--resume=new-conv" in args
        assert "--model" in args


class TestCheckContextUsage:
    """R4-C1: rollover-pending marking at turn end.

    R4-C1: 턴 종료 시 롤오버 pending 마킹.
    """

    def _arm(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
        conv_id: str | None,
        exceeded: bool,
    ) -> list[object]:
        from session_manager.wrapper import context_monitor as cm
        from session_manager.wrapper import pty_wrapper as pw

        monkeypatch.setattr(
            pw, "get_active_conversation_id", lambda _p: conv_id
        )
        calls: list[object] = []
        usage = cm.ContextUsage(
            used_tokens=130_000,
            window_tokens=200_000,
            trigger_tokens=120_000,
            exceeded=exceeded,
            numerator_source="transcript",
            denominator_source="mapping",
        )
        monkeypatch.setattr(
            pw.context_monitor,
            "check_context_usage",
            lambda *a: calls.append(a) or usage,
        )
        return calls

    def test_marks_pending_once(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._arm(wrapper, monkeypatch, "conv-1", exceeded=True)
        wrapper._check_context_usage()
        assert wrapper._rollover_pending_conv_id == "conv-1"
        # A second exceeded check on the same conversation stays marked
        # (no re-log path — the mark is idempotent).
        # 같은 conversation 의 재검사는 마킹 유지 (마킹은 멱등).
        wrapper._check_context_usage()
        assert wrapper._rollover_pending_conv_id == "conv-1"

    def test_below_trigger_does_not_mark(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._arm(wrapper, monkeypatch, "conv-1", exceeded=False)
        wrapper._check_context_usage()
        assert wrapper._rollover_pending_conv_id is None

    def test_conversation_change_clears_mark(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._arm(wrapper, monkeypatch, "conv-1", exceeded=True)
        wrapper._check_context_usage()
        assert wrapper._rollover_pending_conv_id == "conv-1"
        # The next conversation is not full — its check clears the mark.
        # 다음 conversation 은 안 찼다 — 그 검사가 마킹을 해제한다.
        self._arm(wrapper, monkeypatch, "conv-2", exceeded=False)
        wrapper._check_context_usage()
        assert wrapper._rollover_pending_conv_id is None

    def test_no_active_conversation_is_noop(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = self._arm(wrapper, monkeypatch, None, exceeded=True)
        wrapper._check_context_usage()
        assert wrapper._rollover_pending_conv_id is None
        assert calls == []


class TestAdvanceRollover:
    """R4-C3: the handoff-turn state machine.

    R4-C3: handoff 전용 턴 상태 머신.
    """

    def _mark(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
        conv_id: str = "conv-1",
    ) -> None:
        from session_manager.wrapper import pty_wrapper as pw

        wrapper._rollover_pending_conv_id = conv_id
        wrapper._current_session_name = "backend"
        monkeypatch.setattr(
            pw, "get_active_conversation_id", lambda _p: conv_id
        )

    def test_starts_dedicated_turn_when_pending(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._mark(wrapper, monkeypatch)
        notes: list[str] = []
        monkeypatch.setattr(wrapper, "_notify_user", notes.append)
        wrapper._advance_rollover()
        state = wrapper._rollover_request_state
        assert state["phase"] == "requested"
        assert state["session"] == "backend"
        assert state["n"] == 1
        assert wrapper._pending_respawn.is_rollover_request is True
        assert wrapper._pending_respawn.resume_conv == "conv-1"
        assert any("세션을 이어갈 준비" in n for n in notes)

    def test_noop_without_pending_mark(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from session_manager.wrapper import pty_wrapper as pw

        monkeypatch.setattr(
            pw, "get_active_conversation_id", lambda _p: "conv-1"
        )
        wrapper._advance_rollover()
        assert wrapper._rollover_request_state is None
        assert wrapper._pending_respawn is None

    def test_noop_when_ready_or_conv_changed(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._mark(wrapper, monkeypatch, conv_id="conv-1")
        wrapper._rollover_ready = {"session": "backend"}
        wrapper._advance_rollover()
        assert wrapper._rollover_request_state is None
        # Conversation moved on (switch/clear) — the stale mark must not
        # fire a handoff turn for the new conversation.
        # conversation 이 바뀌면 낡은 마킹이 새 대화에 발동하면 안 된다.
        wrapper._rollover_ready = None
        wrapper._rollover_pending_conv_id = "conv-0"
        wrapper._advance_rollover()
        assert wrapper._rollover_request_state is None

    def test_finish_valid_writes_file_and_marks_ready(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from session_manager import rollover as ro

        body = (
            "# Handoff: backend #1\n"
            "## 1. 지금 바로 할 일 (재개 지점)\n다음 액션\n"
            "## 2. 사용자 요구사항\n목록\n"
        )
        monkeypatch.setattr(
            ro, "check_trigger_turn", lambda *_a, **_k: ("answered", body)
        )
        notes: list[str] = []
        monkeypatch.setattr(wrapper, "_notify_user", notes.append)
        wrapper._rollover_request_state = {
            "session": "backend",
            "n": 1,
            "conv_id": "conv-1",
            "attempts": 1,
            "phase": "writing",
        }
        wrapper._advance_rollover()
        assert wrapper._rollover_request_state is None
        ready = wrapper._rollover_ready
        assert ready["session"] == "backend"
        written = Path(ready["path"])
        assert written.read_text(encoding="utf-8") == body
        assert any("Handoff 준비 완료" in n for n in notes)

    def test_finish_invalid_retries_once(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from session_manager import rollover as ro

        monkeypatch.setattr(
            ro,
            "check_trigger_turn",
            lambda *_a, **_k: ("answered", "엉뚱한 답"),
        )
        started: list[int] = []
        monkeypatch.setattr(
            wrapper,
            "_start_handoff_turn",
            lambda _s, _c, attempts: started.append(attempts),
        )
        wrapper._rollover_request_state = {
            "session": "backend",
            "n": 1,
            "conv_id": "conv-1",
            "attempts": 1,
            "phase": "writing",
        }
        wrapper._advance_rollover()
        assert started == [2]
        assert wrapper._rollover_ready is None

    def test_finish_second_failure_writes_fallback(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from session_manager import rollover as ro
        from session_manager.wrapper import pty_wrapper as pw

        monkeypatch.setattr(
            ro, "check_trigger_turn", lambda *_a, **_k: ("answered", "")
        )
        monkeypatch.setattr(pw, "extract_full_text", lambda _p: "대화 발췌")
        monkeypatch.setattr(wrapper, "_notify_user", lambda _t: None)
        wrapper._rollover_request_state = {
            "session": "backend",
            "n": 1,
            "conv_id": "conv-1",
            "attempts": 2,
            "phase": "writing",
        }
        wrapper._advance_rollover()
        ready = wrapper._rollover_ready
        body = Path(ready["path"]).read_text(encoding="utf-8")
        assert ro.validate_handoff_text(body) is True
        assert "대화 발췌" in body

    def test_waiting_status_preserves_attempt(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Boot-edge race (measured): a turn-end signal before the trigger
        # is delivered must leave the attempt intact.
        # 부팅 에지 레이스 (실측) — 트리거 전달 전의 턴 종료 신호가 시도를
        # 소진하면 안 된다.
        from session_manager import rollover as ro

        monkeypatch.setattr(
            ro, "check_trigger_turn", lambda *_a, **_k: ("waiting", "")
        )
        state = {
            "session": "backend",
            "n": 1,
            "conv_id": "conv-1",
            "attempts": 1,
            "phase": "writing",
        }
        wrapper._rollover_request_state = state
        wrapper._advance_rollover()
        assert wrapper._rollover_request_state is state
        assert wrapper._rollover_ready is None

    def test_missing_status_consumes_attempt(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from session_manager import rollover as ro

        monkeypatch.setattr(
            ro, "check_trigger_turn", lambda *_a, **_k: ("missing", "")
        )
        started: list[int] = []
        monkeypatch.setattr(
            wrapper,
            "_start_handoff_turn",
            lambda _s, _c, attempts: started.append(attempts),
        )
        wrapper._rollover_request_state = {
            "session": "backend",
            "n": 1,
            "conv_id": "conv-1",
            "attempts": 1,
            "phase": "writing",
        }
        wrapper._advance_rollover()
        assert started == [2]

    def test_transcript_poll_revalidates_on_change(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # The turn-end signal can outrun the transcript flush (measured)
        # — a later file change alone must re-run validation.
        # 턴 종료 신호가 transcript flush 를 앞지를 수 있다 (실측) —
        # 이후의 파일 변화만으로 재검증이 돌아야 한다.
        from session_manager.wrapper import pty_wrapper as pw

        jsonl_dir = tmp_path / "proj"
        jsonl_dir.mkdir()
        monkeypatch.setattr(pw, "encode_cwd", lambda _p: "proj")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "h"))
        jsonl = tmp_path / "h" / ".claude" / "projects" / "proj" / "c1.jsonl"
        jsonl.parent.mkdir(parents=True)
        jsonl.write_text("{}\n", encoding="utf-8")

        finished: list[dict] = []
        monkeypatch.setattr(wrapper, "_finish_handoff_turn", finished.append)
        state = {
            "session": "backend",
            "n": 1,
            "conv_id": "c1",
            "attempts": 1,
            "phase": "writing",
        }
        wrapper._rollover_request_state = state
        wrapper._poll_handoff_transcript()
        assert len(finished) == 1  # first sighting establishes + checks
        wrapper._poll_handoff_transcript()
        assert len(finished) == 1  # unchanged file — no re-run
        jsonl.write_text('{}\n{"type": "assistant"}\n', encoding="utf-8")
        wrapper._poll_handoff_transcript()
        assert len(finished) == 2  # changed file — re-validated

    def test_transcript_poll_noop_outside_writing_phase(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        finished: list[dict] = []
        monkeypatch.setattr(wrapper, "_finish_handoff_turn", finished.append)
        wrapper._poll_handoff_transcript()
        wrapper._rollover_request_state = {"phase": "requested", "conv_id": "c"}
        wrapper._poll_handoff_transcript()
        assert finished == []

    def test_requested_phase_waits_for_spawn(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A turn-end edge in the OLD child (user typed during the swap
        # wait) must not validate prematurely.
        # 교체 대기 중 옛 자식의 턴 종료 에지가 조기 검증하면 안 된다.
        wrapper._rollover_request_state = {
            "session": "backend",
            "n": 1,
            "conv_id": "conv-1",
            "attempts": 1,
            "phase": "requested",
        }
        wrapper._advance_rollover()
        assert wrapper._rollover_request_state["phase"] == "requested"
        assert wrapper._rollover_ready is None


class TestObserveContextUpdate:
    """R4-C3: the context.json-based second turn-end signal.

    R4-C3: context.json 기반 제2 턴 종료 신호.
    """

    def _write_record(
        self, project: Path, conv: str = "conv-1", used: int = 10_000
    ) -> None:
        from session_manager import statusline

        statusline.write_context(
            project,
            {
                "conversation_id": conv,
                "used_tokens": used,
                "context_window_size": 200_000,
                "at": f"t-{used}",
            },
        )

    def _arm(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> list[str]:
        fired: list[str] = []
        monkeypatch.setattr(wrapper, "_on_turn_end", fired.append)
        return fired

    def test_first_observation_is_baseline_only(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # A stale file from a previous run must not fire checks at boot.
        # 이전 실행의 잔존 파일이 부팅 시 검사를 발동하면 안 된다.
        fired = self._arm(wrapper, monkeypatch)
        self._write_record(tmp_path, used=10_000)
        wrapper._observe_context_update()
        assert fired == []

    def test_usage_change_fires_turn_end(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        fired = self._arm(wrapper, monkeypatch)
        self._write_record(tmp_path, used=10_000)
        wrapper._observe_context_update()
        self._write_record(tmp_path, used=12_345)
        wrapper._observe_context_update()
        assert fired == ["context_update"]

    def test_rewrite_without_usage_change_is_silent(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # The collector rewrites the file several times per turn with
        # the same usage — only a real change may fire.
        # 수집기는 한 턴에 같은 usage 로도 여러 번 파일을 다시 쓴다 —
        # 실제 변화만 발동해야 한다.
        fired = self._arm(wrapper, monkeypatch)
        self._write_record(tmp_path, used=10_000)
        wrapper._observe_context_update()
        self._write_record(tmp_path, used=10_000)
        wrapper._observe_context_update()
        assert fired == []

    def test_missing_file_is_noop(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fired = self._arm(wrapper, monkeypatch)
        wrapper._observe_context_update()
        assert fired == []


class TestScreenBusy:
    """R4-C3: spinner-ellipsis busy marker (short turns render no hint).

    R4-C3: 스피너 말줄임표 바쁨 마커 (짧은 턴은 힌트 미렌더).
    """

    def _screen(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
        present: set[str],
    ) -> None:
        monkeypatch.setattr(
            wrapper.virtual_screen,
            "contains_near_prompt",
            lambda needle, radius: needle in present,
        )

    def test_hint_marker_counts(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._screen(wrapper, monkeypatch, {"esc to interrupt"})
        assert wrapper._screen_busy() is True

    def test_spinner_ellipsis_counts(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._screen(wrapper, monkeypatch, {"…"})
        assert wrapper._screen_busy() is True

    def test_idle_screen(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._screen(wrapper, monkeypatch, set())
        assert wrapper._screen_busy() is False


class TestRolloverSignal:
    """R4-C2: PreCompact hook's rollover signal dispatch.

    R4-C2: PreCompact hook 롤오버 신호 dispatch.
    """

    def test_marks_pending_from_signal(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._handle_mcp_signal(
            {"action": "rollover_signal", "conversation_id": "conv-9"}
        )
        assert wrapper._rollover_pending_conv_id == "conv-9"

    def test_missing_conversation_falls_back_to_active(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from session_manager.wrapper import pty_wrapper as pw

        monkeypatch.setattr(
            pw, "get_active_conversation_id", lambda _p: "conv-active"
        )
        wrapper._handle_mcp_signal({"action": "rollover_signal"})
        assert wrapper._rollover_pending_conv_id == "conv-active"

    def test_no_resolvable_conversation_is_noop(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from session_manager.wrapper import pty_wrapper as pw

        monkeypatch.setattr(pw, "get_active_conversation_id", lambda _p: None)
        wrapper._handle_mcp_signal({"action": "rollover_signal"})
        assert wrapper._rollover_pending_conv_id is None


class TestMcpConfigFlag:
    """--mcp-config= injection (F4 fix — docs/poc/R3-mcp-config.md).

    --mcp-config= 주입 (F4 수정 — docs/poc/R3-mcp-config.md).
    """

    def test_flag_is_single_equals_form_token(self) -> None:
        # `=` form is mandatory: the space form greedily swallows the
        # trailing positional trigger (measured, docs/poc/R3-respawn.md).
        # `=` 형식 필수 — 공백 형식은 뒤의 트리거를 삼킨다 (실측).
        flag = SessionManagerWrapper._mcp_config_flag()
        assert len(flag) == 1
        assert flag[0].startswith("--mcp-config=")

    def test_config_targets_server_module_in_same_venv(self) -> None:
        import json as _json
        import sys as _sys

        flag = SessionManagerWrapper._mcp_config_flag()
        config = _json.loads(flag[0].removeprefix("--mcp-config="))
        server = config["mcpServers"]["session-manager"]
        assert server["type"] == "stdio"
        assert server["command"] == _sys.executable
        assert server["args"] == ["-m", "session_manager.server"]

    def test_plain_boot_includes_mcp_config(self, tmp_path: Path) -> None:
        w = SessionManagerWrapper(
            socket_path=str(tmp_path / "s.sock"),
            claude_args=["--model", "opus"],
            project_path=str(tmp_path),
        )
        args = w._build_child_args()
        assert any(a.startswith("--mcp-config=") for a in args)
        # Additive injection only — never strict, so the user's other
        # MCP servers keep loading.
        # 추가 주입만 — strict 금지, 사용자의 타 MCP 서버 로드 유지.
        assert "--strict-mcp-config" not in args

    def test_respawn_includes_mcp_config_before_trigger(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._pending_respawn = _PendingRespawn(
            target="backend", resume_conv="conv-9"
        )
        args = wrapper._build_child_args()
        assert any(a.startswith("--mcp-config=") for a in args)
        assert args[-1] == handoff_store.TRIGGER_PROMPT


class TestStripResumeArgs:
    def test_strips_all_resume_variants(self) -> None:
        args = [
            "--resume", "aaa", "-r", "bbb", "--resume=ccc", "--continue",
            "-c", "--model", "opus",
        ]
        assert SessionManagerWrapper._strip_resume_args(args) == [
            "--model", "opus",
        ]

    def test_keeps_everything_else(self) -> None:
        args = ["--model", "opus", "--verbose"]
        assert SessionManagerWrapper._strip_resume_args(args) == args


class TestSpawnCompletion:
    """Transition bookkeeping happens at (re)spawn — the swap's success.

    전환 부기는 교체가 성공한 (re)spawn 시점에 일어난다.
    """

    def _spawn(self, wrapper: SessionManagerWrapper, monkeypatch) -> list:
        spawned: list[list[str]] = []

        class FakeChild:
            pid = 7
            def fileno(self) -> int:
                return 0

        def fake_spawn(cmd, args, **kwargs):
            spawned.append(args)
            return FakeChild()

        import pexpect

        monkeypatch.setattr(pexpect, "spawn", fake_spawn)
        wrapper._spawn_child()
        return spawned

    def test_normal_transition_records_last_transition(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wrapper._pending_respawn = _PendingRespawn(
            target="backend",
            resume_conv="conv-9",
            from_name="frontend",
            user_prompt="옮겨간 프롬프트",
        )
        self._spawn(wrapper, monkeypatch)
        assert wrapper._pending_respawn is None
        record = wrapper._last_transition
        assert record is not None
        assert record["from"] == "frontend"
        assert record["to"] == "backend"
        assert record["user_prompt"] == "옮겨간 프롬프트"

    def test_back_respawn_consumes_undo_record(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = {
            "from": "frontend", "to": "backend",
            "user_prompt": "p", "at": "2026-08-10T00:00:00+00:00",
        }
        wrapper._last_transition = dict(record)
        wrapper_state.save_last_transition(Path(wrapper.project_path), record)
        wrapper._pending_respawn = _PendingRespawn(
            target="frontend", resume_conv="conv-1", is_back=True
        )
        self._spawn(wrapper, monkeypatch)
        assert wrapper._last_transition is None
        assert (
            wrapper_state.load_last_transition(Path(wrapper.project_path)) is None
        )

    def test_spawn_uses_built_args(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        wrapper._pending_respawn = _PendingRespawn(
            target="backend", resume_conv="conv-9"
        )
        spawned = self._spawn(wrapper, monkeypatch)
        assert len(spawned) == 1
        assert "--resume=conv-9" in spawned[0]
        assert spawned[0][-1] == handoff_store.TRIGGER_PROMPT


class TestBackCommand:
    """/back over the respawn transition path.

    respawn 전환 경로 위의 /back.
    """

    RECORD = {
        "from": "frontend",
        "to": "backend",
        "user_prompt": "미스라우팅된 프롬프트\n둘째 줄",
        "at": "2026-08-10T00:00:00+00:00",
    }

    def _prepare(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> MagicMock:
        _seed_resumable_session(
            wrapper, monkeypatch, tmp_path / "home", name="frontend", conv="conv-1"
        )
        wrapper._last_transition = dict(self.RECORD)
        send = MagicMock()
        monkeypatch.setattr(wrapper.socket_server, "send", send)
        return send

    def test_back_registers_reverse_respawn(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        send = self._prepare(wrapper, monkeypatch, tmp_path)
        wrapper._handle_back_command()

        pending = wrapper._pending_respawn
        assert pending is not None
        assert pending.target == "frontend"
        assert pending.resume_conv == "conv-1"
        assert pending.is_back is True
        # The misrouted prompt travels via the pending-handoff file.
        # 잘못 이동했던 프롬프트는 pending handoff 파일로 이동한다.
        stored = handoff_store.take_pending(Path(wrapper.project_path))
        assert stored is not None
        assert stored["user_prompt"] == self.RECORD["user_prompt"]
        assert stored["handoff"]["back"] is True
        assert stored["handoff"]["from"] == "backend"

        # NOT consumed until the respawn completes.
        # respawn 완료 전에는 소비되지 않는다.
        assert wrapper._last_transition == self.RECORD

        # Precedent on the origin session, gist = first line.
        # 복귀 세션에 판례 — gist 는 첫 줄.
        origin = SessionStore(Path(wrapper.project_path)).load_session_by_name(
            "frontend"
        )
        assert len(origin.precedents) == 1
        assert origin.precedents[0].rejected == "backend"
        assert origin.precedents[0].prompt_gist == "미스라우팅된 프롬프트"

        # Calibration label + MCP pointer invalidation.
        # 보정 라벨 + MCP 포인터 무효화.
        labels = [
            e
            for e in decision_log.load_events(Path(wrapper.project_path))
            if e.get("type") == "label"
        ]
        assert [(la["label"], la["target"], la["source"]) for la in labels] == [
            ("reject", "backend", "back")
        ]
        signals = [c.args[0] for c in send.call_args_list]
        assert {"action": "session_command", "command": "back", "args": ""} in signals

    def test_back_unresolvable_origin_keeps_record(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Origin session's transcript is gone — abort BEFORE any side
        # effect so the user can retry after fixing things.
        # 원 세션 transcript 소멸 — 부수효과 전에 중단해 재시도 가능하게.
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
        wrapper._last_transition = dict(self.RECORD)
        send = MagicMock()
        monkeypatch.setattr(wrapper.socket_server, "send", send)

        wrapper._handle_back_command()

        assert wrapper._pending_respawn is None
        assert wrapper._last_transition == self.RECORD
        assert decision_log.load_events(Path(wrapper.project_path)) == []
        send.assert_not_called()

    def test_back_without_record_is_noop(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        send = MagicMock()
        monkeypatch.setattr(wrapper.socket_server, "send", send)
        wrapper._handle_back_command()
        assert wrapper._pending_respawn is None
        send.assert_not_called()

    def test_back_during_pending_transition_ignored(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        self._prepare(wrapper, monkeypatch, tmp_path)
        wrapper._pending_respawn = _PendingRespawn(target="x", resume_conv=None)
        wrapper._handle_back_command()
        assert wrapper._pending_respawn.target == "x"
        assert wrapper._last_transition == self.RECORD


class TestHandshake:
    def test_replies_with_initial_session_name(self, tmp_path: Path) -> None:
        w = SessionManagerWrapper(
            socket_path=str(tmp_path / "s.sock"),
            claude_args=["--resume", "foo"],
            project_path=str(tmp_path),
        )
        sent: list[dict] = []
        w.socket_server.send = lambda msg: bool(sent.append(msg) or True)  # type: ignore[assignment]
        w._handle_handshake_request()
        assert sent == [{"current_session_name": "foo"}]

    def test_replies_with_none_on_plain_start(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        sent: list[dict] = []
        wrapper.socket_server.send = lambda msg: bool(sent.append(msg) or True)  # type: ignore[assignment]
        wrapper._handle_handshake_request()
        assert sent == [{"current_session_name": None}]

    def test_replies_with_mirror_after_transition(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        # After a transition the respawned child's MCP must learn the
        # target session from the handshake.
        # 전환 후 재시작된 자식의 MCP 는 핸드셰이크로 대상 세션을 알아야
        # 한다.
        wrapper._execute_transition(
            target="backend", resume_conv=None, handoff={}, user_prompt=""
        )
        sent: list[dict] = []
        wrapper.socket_server.send = lambda msg: bool(sent.append(msg) or True)  # type: ignore[assignment]
        wrapper._handle_handshake_request()
        assert sent == [{"current_session_name": "backend"}]


class TestMcpSignalRouting:
    def test_switch_signal_registers_respawn(
        self,
        wrapper: SessionManagerWrapper,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _seed_resumable_session(
            wrapper, monkeypatch, tmp_path / "home", name="bar", conv="conv-bar"
        )
        wrapper._handle_mcp_signal(
            {"action": "switch", "target": "bar", "handoff": {"user_prompt": "x"}}
        )
        pending = wrapper._pending_respawn
        assert pending is not None
        assert pending.target == "bar"
        assert pending.resume_conv == "conv-bar"

    def test_switch_without_conversation_boots_fresh(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._handle_mcp_signal(
            {"action": "switch", "target": "ghost", "handoff": {}}
        )
        pending = wrapper._pending_respawn
        assert pending is not None
        assert pending.resume_conv is None

    def test_new_signal_registers_fresh_respawn(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._handle_mcp_signal(
            {
                "action": "new",
                "rename_current": "cur",
                "new_session_name": "fresh",
                "handoff": {"from": "cur"},
            }
        )
        pending = wrapper._pending_respawn
        assert pending is not None
        assert pending.target == "fresh"
        assert pending.resume_conv is None

    def test_route_switch_notifies_with_back_hint(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        notices: list[str] = []
        monkeypatch.setattr(wrapper, "_notify_user", notices.append)
        wrapper._current_session_name = "frontend"
        wrapper._handle_mcp_signal(
            {
                "action": "route_switch",
                "target": "backend",
                "user_prompt": "로그인 API 500",
                "verdict": {"reason": "인증 소관"},
            }
        )
        pending = wrapper._pending_respawn
        assert pending is not None
        assert pending.target == "backend"
        stored = handoff_store.take_pending(Path(wrapper.project_path))
        assert stored["handoff"]["from"] == "frontend"
        assert stored["handoff"]["router_reason"] == "인증 소관"
        assert notices == [
            "⇄ backend 세션으로 전환됨 (이전: frontend) — 되돌리려면 /back"
        ]

    def test_invalid_message_ignored(self, wrapper: SessionManagerWrapper) -> None:
        wrapper._handle_mcp_signal("not a dict")  # type: ignore[arg-type]
        wrapper._handle_mcp_signal({})
        assert wrapper._pending_respawn is None

    def test_switch_missing_target_ignored(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._handle_mcp_signal({"action": "switch", "handoff": {}})
        assert wrapper._pending_respawn is None

    def test_new_missing_session_name_ignored(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        wrapper._handle_mcp_signal({"action": "new", "handoff": {}})
        assert wrapper._pending_respawn is None


class TestManualMovePrecedentDrop:
    """R3-FIX2: manually resuming a session overturns precedents against it.

    R3-FIX2 — 세션으로의 수동 이동은 그 세션에 대한 판례를 뒤집는다.
    결정적 억제 게이트의 유일한 소멸 경로 (수락·롤오버 외) 이므로, 이게
    없으면 억제된 대상이 영원히 막힌다.
    """

    def _seed(self, wrapper: SessionManagerWrapper) -> SessionStore:
        from session_manager.models import PrecedentRecord

        store = SessionStore(Path(wrapper.project_path))
        frontend = SessionMetadata.new(name="frontend", title="차트")
        frontend.precedents = [
            PrecedentRecord.new(
                prompt_gist="로그인 오류", kept_in="frontend", rejected="backend"
            ),
            PrecedentRecord.new(
                prompt_gist="배포", kept_in="frontend", rejected="infra"
            ),
        ]
        store.save_session(frontend)
        backend = SessionMetadata.new(name="backend", title="API")
        backend.claude_conversation_ids = ["conv-backend"]
        store.save_session(backend)
        wrapper._current_session_name = "frontend"
        return store

    def test_resume_by_session_name_drops_matching(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        store = self._seed(wrapper)
        wrapper._drop_precedents_on_manual_move("backend")
        frontend = store.load_session_by_name("frontend")
        assert [p.rejected for p in frontend.precedents] == ["infra"]

    def test_resume_by_conversation_id_drops_matching(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        store = self._seed(wrapper)
        wrapper._drop_precedents_on_manual_move("conv-backend")
        frontend = store.load_session_by_name("frontend")
        assert [p.rejected for p in frontend.precedents] == ["infra"]

    def test_unknown_argument_is_noop(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        store = self._seed(wrapper)
        wrapper._drop_precedents_on_manual_move("서치어-없는-대상")
        frontend = store.load_session_by_name("frontend")
        assert len(frontend.precedents) == 2

    def test_without_current_session_is_noop(
        self, wrapper: SessionManagerWrapper
    ) -> None:
        store = self._seed(wrapper)
        wrapper._current_session_name = None
        wrapper._drop_precedents_on_manual_move("backend")
        frontend = store.load_session_by_name("frontend")
        assert len(frontend.precedents) == 2

    def test_observed_resume_command_triggers_drop(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # End-to-end within the wrapper: the observation path calls the
        # drop with the /resume argument.
        # 래퍼 내부 관통 — 관찰 경로가 /resume 인자로 소멸을 호출한다.
        from session_manager.wrapper.command_matcher import InterceptedCommand

        store = self._seed(wrapper)
        monkeypatch.setattr(
            wrapper.socket_server, "send", lambda msg: True
        )
        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.get_active_conversation_id",
            lambda _cwd: None,
        )
        wrapper._observe_session_command(
            InterceptedCommand(command="resume", args="backend")
        )
        frontend = store.load_session_by_name("frontend")
        assert [p.rejected for p in frontend.precedents] == ["infra"]

    def test_bare_resume_does_not_drop(
        self, wrapper: SessionManagerWrapper, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Picker destination is unknown — nothing may be invalidated.
        # picker 는 목적지 불명 — 아무것도 무효화하면 안 된다.
        from session_manager.wrapper.command_matcher import InterceptedCommand

        store = self._seed(wrapper)
        monkeypatch.setattr(wrapper.socket_server, "send", lambda msg: True)
        monkeypatch.setattr(
            "session_manager.wrapper.pty_wrapper.get_active_conversation_id",
            lambda _cwd: None,
        )
        wrapper._observe_session_command(
            InterceptedCommand(command="resume", args="")
        )
        frontend = store.load_session_by_name("frontend")
        assert len(frontend.precedents) == 2
