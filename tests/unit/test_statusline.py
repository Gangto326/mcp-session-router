"""
Unit tests for the statusline context collector.

Focus: facts are recorded only in a wrapper context, extraction is
defensive against missing fields, and no failure ever escapes.

statusline 컨텍스트 수집기 단위 테스트.

초점: 래퍼 문맥에서만 기록하고, 필드 부재에 방어적으로 추출하며,
어떤 실패도 밖으로 새지 않는지.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from session_manager import statusline
from session_manager.wrapper import wrapper_state


def _payload(**overrides) -> dict:
    """A measured-shape statusline stdin payload (docs/poc/R4-rollover.md).

    실측 형태 (docs/poc/R4-rollover.md) 의 statusline stdin payload.
    """
    base = {
        "session_id": "conv-1",
        "transcript_path": "/tmp/conv-1.jsonl",
        "cwd": "/tmp/proj",
        "model": {"id": "claude-sonnet-5", "display_name": "Sonnet 5"},
        "context_window": {
            "total_input_tokens": 41000,
            "total_output_tokens": 3,
            "context_window_size": 1_000_000,
            "used_percentage": 4,
            "current_usage": {
                "input_tokens": 2,
                "output_tokens": 3,
                "cache_creation_input_tokens": 10629,
                "cache_read_input_tokens": 30369,
            },
        },
    }
    base.update(overrides)
    return base


class TestExtractRecord:
    def test_full_payload(self) -> None:
        record = statusline._extract_record(_payload())
        assert record["context_window_size"] == 1_000_000
        assert record["used_tokens"] == 41000
        assert record["used_percentage"] == 4
        assert record["conversation_id"] == "conv-1"
        assert record["model_id"] == "claude-sonnet-5"
        assert record["at"]

    def test_no_context_window_returns_none(self) -> None:
        payload = _payload()
        del payload["context_window"]
        assert statusline._extract_record(payload) is None

    def test_invalid_window_size_returns_none(self) -> None:
        payload = _payload()
        payload["context_window"]["context_window_size"] = 0
        assert statusline._extract_record(payload) is None

    def test_missing_total_falls_back_to_usage_sum(self) -> None:
        # Numerator = input + cache_read + cache_creation, output
        # excluded (measured equality with /context).
        # 분자 = input + cache_read + cache_creation, output 제외
        # (/context 와의 일치 실측).
        payload = _payload()
        del payload["context_window"]["total_input_tokens"]
        record = statusline._extract_record(payload)
        assert record["used_tokens"] == 2 + 10629 + 30369

    def test_missing_model_is_none(self) -> None:
        payload = _payload()
        del payload["model"]
        record = statusline._extract_record(payload)
        assert record["model_id"] is None


class TestReadWrite:
    def test_roundtrip(self, tmp_path: Path) -> None:
        statusline.write_context(tmp_path, {"context_window_size": 5})
        assert statusline.read_context(tmp_path) == {"context_window_size": 5}

    def test_read_missing_is_none(self, tmp_path: Path) -> None:
        assert statusline.read_context(tmp_path) is None

    def test_read_corrupt_is_none(self, tmp_path: Path) -> None:
        path = tmp_path / ".session-manager" / statusline.CONTEXT_FILENAME
        path.parent.mkdir(parents=True)
        path.write_text("{broken", encoding="utf-8")
        assert statusline.read_context(tmp_path) is None


class TestMain:
    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdin_text: str,
        socket_env: str | None,
    ) -> None:
        if socket_env is None:
            monkeypatch.delenv("SESSION_MANAGER_SOCKET", raising=False)
        else:
            monkeypatch.setenv("SESSION_MANAGER_SOCKET", socket_env)
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin_text))
        statusline.main()

    def test_outside_wrapper_writes_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        payload = _payload(cwd=str(tmp_path))
        self._run(monkeypatch, json.dumps(payload), socket_env=None)
        assert not (tmp_path / ".session-manager").exists()
        assert capsys.readouterr().out == ""

    def test_wrapper_context_writes_and_prints(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        payload = _payload(cwd=str(tmp_path))
        self._run(monkeypatch, json.dumps(payload), socket_env="/tmp/s.sock")
        record = statusline.read_context(tmp_path)
        assert record["context_window_size"] == 1_000_000
        assert record["conversation_id"] == "conv-1"
        assert "ctx 4%" in capsys.readouterr().out

    def test_broken_stdin_is_silent(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(monkeypatch, "{broken", socket_env="/tmp/s.sock")
        assert capsys.readouterr().out == ""

    def test_payload_without_window_writes_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        payload = _payload(cwd=str(tmp_path))
        del payload["context_window"]
        self._run(monkeypatch, json.dumps(payload), socket_env="/tmp/s.sock")
        assert not (tmp_path / ".session-manager").exists()


def _write_session(project: Path, name: str, status: str | None) -> None:
    sessions = project / ".session-manager" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    data = {"name": name}
    if status is not None:
        data["status"] = status
    (sessions / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


def _write_config(project: Path, **keys) -> None:
    root = project / ".session-manager"
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(keys), encoding="utf-8")


class TestBuildStatusLine:
    def test_full(self) -> None:
        line = statusline.build_status_line("backend", "auto", 42, 3)
        assert line == "⎇ backend · auto · ctx 42% · 3 sessions"

    def test_singular_session_noun(self) -> None:
        assert statusline.build_status_line("a", "confirm", 0, 1).endswith(
            "1 session"
        )

    def test_missing_pieces_are_dropped(self) -> None:
        assert statusline.build_status_line(None, "confirm", None, 2) == (
            "confirm · 2 sessions"
        )
        assert statusline.build_status_line("a", None, 5.6, None) == (
            "⎇ a · ctx 6%"
        )

    def test_nothing_is_none(self) -> None:
        assert statusline.build_status_line(None, None, None, None) is None

    def test_bool_is_not_a_number(self) -> None:
        # json payloads cannot carry bools here, but guard the isinstance
        # trap (bool is an int subclass) anyway.
        # payload 에 bool 이 올 일은 없지만 isinstance 함정 (bool 은 int
        # 하위 타입) 을 막아 둔다.
        assert statusline.build_status_line(None, None, True, False) is None


class TestSegmentSources:
    def test_routing_mode_default_without_config(self, tmp_path: Path) -> None:
        assert statusline._load_routing_mode(tmp_path) == "confirm"

    def test_routing_mode_from_config(self, tmp_path: Path) -> None:
        _write_config(tmp_path, routing_mode="auto")
        assert statusline._load_routing_mode(tmp_path) == "auto"

    def test_routing_mode_corrupt_is_default(self, tmp_path: Path) -> None:
        root = tmp_path / ".session-manager"
        root.mkdir()
        (root / "config.json").write_text("{nope", encoding="utf-8")
        assert statusline._load_routing_mode(tmp_path) == "confirm"

    def test_count_without_dir_is_none(self, tmp_path: Path) -> None:
        assert statusline._count_active_sessions(tmp_path) is None

    def test_count_rules(self, tmp_path: Path) -> None:
        _write_session(tmp_path, "a", None)  # legacy: no status → active
        _write_session(tmp_path, "b", "active")
        _write_session(tmp_path, "c", "retired")
        _write_session(tmp_path, "d", "expired")
        bad = tmp_path / ".session-manager" / "sessions" / "e.json"
        bad.write_text("{corrupt", encoding="utf-8")
        assert statusline._count_active_sessions(tmp_path) == 2


class TestRender:
    def test_all_sources(self, tmp_path: Path) -> None:
        wrapper_state.save_current_session(tmp_path, "backend")
        _write_config(tmp_path, routing_mode="auto")
        _write_session(tmp_path, "backend", "active")
        _write_session(tmp_path, "frontend", "active")
        record = statusline._extract_record(_payload())
        assert statusline.render(tmp_path, record) == (
            "⎇ backend · auto · ctx 4% · 2 sessions"
        )

    def test_no_record_falls_back_to_context_file(self, tmp_path: Path) -> None:
        statusline.write_context(tmp_path, {"used_percentage": 37})
        assert statusline.render(tmp_path, None) == "confirm · ctx 37%"

    def test_bare_project(self, tmp_path: Path) -> None:
        # Nothing but the default mode — still a line, never a crash.
        # 기본 모드밖에 없어도 한 줄은 나온다 — 절대 죽지 않는다.
        assert statusline.render(tmp_path, None) == "confirm"


class TestMainDisplay:
    def test_prints_status_line(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        wrapper_state.save_current_session(tmp_path, "backend")
        _write_session(tmp_path, "backend", "active")
        monkeypatch.setenv("SESSION_MANAGER_SOCKET", "/tmp/s.sock")
        monkeypatch.setattr(
            "sys.stdin", io.StringIO(json.dumps(_payload(cwd=str(tmp_path))))
        )
        statusline.main()
        assert capsys.readouterr().out == (
            "⎇ backend · confirm · ctx 4% · 1 session\n"
        )

    def test_old_payload_still_displays(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # No context_window block (older Claude Code): nothing is written
        # but the session/mode segments still render.
        # context_window 블록 없음 (구버전): 기록은 없지만 세션·모드
        # 세그먼트는 그려진다.
        wrapper_state.save_current_session(tmp_path, "backend")
        payload = _payload(cwd=str(tmp_path))
        del payload["context_window"]
        monkeypatch.setenv("SESSION_MANAGER_SOCKET", "/tmp/s.sock")
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
        statusline.main()
        assert not (tmp_path / ".session-manager" / "context.json").exists()
        assert capsys.readouterr().out == "⎇ backend · confirm\n"
