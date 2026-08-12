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
