"""
Unit tests for context-usage detection (R4-C1).

Focus: source priority (override → statusline → mapping → cap alone),
the absolute cap dominating large windows, and O(tail) transcript reads.

컨텍스트 사용률 감지 (R4-C1) 단위 테스트.

초점: 소스 우선순위 (override → statusline → 매핑 → 상한 단독), 큰 창을
지배하는 절대 상한, O(꼬리) transcript 읽기.
"""

from __future__ import annotations

import json
from pathlib import Path

from session_manager import statusline
from session_manager.wrapper import context_monitor

CONV = "conv-1"


def _write_jsonl(path: Path, events: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )


def _assistant(usage: dict, model: str = "claude-haiku-4-5") -> dict:
    return {"type": "assistant", "message": {"usage": usage, "model": model}}


def _statusline_record(project: Path, **overrides) -> None:
    record = {
        "context_window_size": 1_000_000,
        "used_tokens": 40_000,
        "used_percentage": 4,
        "conversation_id": CONV,
        "model_id": "claude-sonnet-5",
        "at": "2026-08-13T00:00:00+00:00",
    }
    record.update(overrides)
    statusline.write_context(project, record)


class TestWindowForModel:
    def test_nominal_claude_is_200k(self) -> None:
        assert context_monitor.window_for_model("claude-haiku-4-5") == 200_000

    def test_one_million_marker(self) -> None:
        assert (
            context_monitor.window_for_model("claude-sonnet-4-5[1m]")
            == 1_000_000
        )

    def test_unknown_is_none(self) -> None:
        assert context_monitor.window_for_model("gpt-x") is None
        assert context_monitor.window_for_model(None) is None


class TestReadTailUsageAndModel:
    def test_reads_last_assistant(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "c.jsonl"
        _write_jsonl(
            jsonl,
            [
                _assistant({"input_tokens": 1}, model="claude-old"),
                {"type": "user", "message": {}},
                _assistant({"input_tokens": 9}, model="claude-new"),
            ],
        )
        usage, model = context_monitor.read_tail_usage_and_model(jsonl)
        assert usage == {"input_tokens": 9}
        assert model == "claude-new"

    def test_tail_window_skips_partial_first_line(self, tmp_path: Path) -> None:
        # A tiny tail_bytes cuts into the middle of a line — the partial
        # line must be skipped, the complete last event still parsed.
        # 아주 작은 tail_bytes 는 줄 중간을 자른다 — 잘린 줄은 건너뛰고
        # 온전한 마지막 이벤트는 파싱되어야 한다.
        jsonl = tmp_path / "c.jsonl"
        filler = _assistant({"input_tokens": 1}, model="claude-a")
        last = _assistant({"input_tokens": 7}, model="claude-b")
        _write_jsonl(jsonl, [filler] * 50 + [last])
        usage, model = context_monitor.read_tail_usage_and_model(
            jsonl, tail_bytes=len(json.dumps(last)) + 10
        )
        assert usage == {"input_tokens": 7}
        assert model == "claude-b"

    def test_missing_file(self, tmp_path: Path) -> None:
        assert context_monitor.read_tail_usage_and_model(
            tmp_path / "nope.jsonl"
        ) == (None, None)

    def test_no_assistant_events(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "c.jsonl"
        _write_jsonl(jsonl, [{"type": "user", "message": {}}])
        assert context_monitor.read_tail_usage_and_model(jsonl) == (None, None)


class TestReadFirstUsage:
    def test_first_assistant_usage(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "c.jsonl"
        _write_jsonl(
            jsonl,
            [
                {"type": "user", "message": {}},
                _assistant(
                    {"input_tokens": 5, "cache_creation_input_tokens": 35_000}
                ),
                _assistant({"input_tokens": 90_000}),
            ],
        )
        assert context_monitor.read_first_usage(jsonl) == 35_005

    def test_missing_file_or_no_assistant(self, tmp_path: Path) -> None:
        assert context_monitor.read_first_usage(tmp_path / "no.jsonl") is None
        jsonl = tmp_path / "c.jsonl"
        _write_jsonl(jsonl, [{"type": "user", "message": {}}])
        assert context_monitor.read_first_usage(jsonl) is None


class TestLoadRolloverConfig:
    def test_defaults_without_config(self, tmp_path: Path) -> None:
        assert context_monitor._load_rollover_config(tmp_path) == (
            60,
            120_000,
            None,
        )

    def test_reads_overrides(self, tmp_path: Path) -> None:
        config_dir = tmp_path / ".session-manager"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(
            json.dumps(
                {
                    "rollover_threshold_pct": 50,
                    "rollover_absolute_cap_tokens": 90_000,
                    "context_budget_tokens": 500_000,
                }
            ),
            encoding="utf-8",
        )
        assert context_monitor._load_rollover_config(tmp_path) == (
            50,
            90_000,
            500_000,
        )

    def test_invalid_values_fall_back(self, tmp_path: Path) -> None:
        config_dir = tmp_path / ".session-manager"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(
            json.dumps(
                {"rollover_threshold_pct": -5, "context_budget_tokens": "big"}
            ),
            encoding="utf-8",
        )
        assert context_monitor._load_rollover_config(tmp_path) == (
            60,
            120_000,
            None,
        )


class TestCheckContextUsage:
    def test_fresh_statusline_supplies_both_sides(self, tmp_path: Path) -> None:
        _statusline_record(tmp_path)
        usage = context_monitor.check_context_usage(
            tmp_path, CONV, tmp_path / "absent.jsonl"
        )
        assert usage.used_tokens == 40_000
        assert usage.window_tokens == 1_000_000
        assert usage.numerator_source == "statusline"
        assert usage.denominator_source == "statusline"
        # Cap dominates a 1M window: min(1M×60%, 120K) = 120K.
        # 1M 창은 상한이 지배: min(1M×60%, 120K) = 120K.
        assert usage.trigger_tokens == 120_000
        assert usage.exceeded is False

    def test_stale_record_falls_back_to_transcript(
        self, tmp_path: Path
    ) -> None:
        # The record describes ANOTHER conversation — it must not leak in.
        # 레코드가 다른 conversation 을 서술 — 새어들면 안 된다.
        _statusline_record(tmp_path, conversation_id="other-conv")
        jsonl = tmp_path / "c.jsonl"
        _write_jsonl(
            jsonl,
            [
                _assistant(
                    {
                        "input_tokens": 10,
                        "cache_read_input_tokens": 100_000,
                        "cache_creation_input_tokens": 30_000,
                    }
                )
            ],
        )
        usage = context_monitor.check_context_usage(tmp_path, CONV, jsonl)
        assert usage.used_tokens == 130_010
        assert usage.numerator_source == "transcript"
        assert usage.denominator_source == "mapping"
        assert usage.window_tokens == 200_000
        assert usage.exceeded is True  # 130,010 ≥ min(120K, 120K)

    def test_override_wins_denominator(self, tmp_path: Path) -> None:
        _statusline_record(tmp_path)
        config_dir = tmp_path / ".session-manager"
        (config_dir / "config.json").write_text(
            json.dumps({"context_budget_tokens": 100_000}), encoding="utf-8"
        )
        usage = context_monitor.check_context_usage(
            tmp_path, CONV, tmp_path / "absent.jsonl"
        )
        assert usage.denominator_source == "override"
        # Small window: pct side wins — 100K×60% = 60K < 120K cap.
        # 작은 창은 pct 쪽이 이긴다 — 100K×60% = 60K < 상한 120K.
        assert usage.trigger_tokens == 60_000
        assert usage.exceeded is False  # 40,000 < 60,000

    def test_unknown_model_uses_cap_alone(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "c.jsonl"
        _write_jsonl(
            jsonl, [_assistant({"input_tokens": 130_000}, model="mystery")]
        )
        usage = context_monitor.check_context_usage(tmp_path, CONV, jsonl)
        assert usage.window_tokens is None
        assert usage.denominator_source == "cap_only"
        assert usage.trigger_tokens == 120_000
        assert usage.exceeded is True

    def test_no_numerator_returns_none(self, tmp_path: Path) -> None:
        assert (
            context_monitor.check_context_usage(
                tmp_path, CONV, tmp_path / "absent.jsonl"
            )
            is None
        )
