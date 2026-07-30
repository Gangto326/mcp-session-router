"""
Unit tests for the transcript excerpt module.

Verifies defensive JSONL parsing (corrupt lines, empty/missing files),
dialogue filtering (noise prefixes, meta events, tool blocks), exchange
grouping / truncation, and last-usage lookup.

transcript 발췌 모듈 단위 테스트.

방어적 JSONL 파싱 (손상 줄, 빈 파일/부재 파일), 대화 필터링 (노이즈
프리픽스, 메타 이벤트, 도구 블록), 교환 묶기 / 절단, 마지막 usage 조회를
검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from session_manager import transcript_excerpt as te

# ---- fixture helpers -----------------------------------------------------
# 픽스처 헬퍼.


def _user(text: str, **extra: Any) -> dict:
    return {"type": "user", "message": {"content": text}, **extra}


def _tool_result_user() -> dict:
    return {
        "type": "user",
        "message": {"content": [{"type": "tool_result", "content": "big output"}]},
    }


def _assistant(*blocks: dict, usage: dict | None = None) -> dict:
    message: dict[str, Any] = {"content": list(blocks)}
    if usage is not None:
        message["usage"] = usage
    return {"type": "assistant", "message": message}


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _write_jsonl(path: Path, events: list[Any], corrupt_after: int | None = None) -> Path:
    """Write events as JSONL, optionally inserting a corrupt line.

    이벤트를 JSONL 로 기록. corrupt_after 지정 시 그 인덱스 뒤에 손상 줄 삽입.
    """
    lines = [json.dumps(e, ensure_ascii=False) for e in events]
    if corrupt_after is not None:
        lines.insert(corrupt_after + 1, '{"type": "user", "message": {truncated')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


USAGE = {
    "input_tokens": 2,
    "cache_creation_input_tokens": 100,
    "cache_read_input_tokens": 5000,
    "output_tokens": 42,
}


@pytest.fixture
def normal_jsonl(tmp_path: Path) -> Path:
    """A realistic conversation: dialogue + bookkeeping + tool events.

    현실적인 대화 — 대화 메시지 + 부기 이벤트 + 도구 이벤트 혼합.
    """
    events = [
        {"type": "file-history-snapshot", "messageId": "x"},
        _user("<command-name>/login</command-name>"),
        _user("첫 질문입니다"),
        _assistant(_text_block("첫 답변"), usage={**USAGE, "cache_read_input_tokens": 100}),
        _assistant({"type": "tool_use", "name": "Read", "input": {}}),
        _tool_result_user(),
        {"type": "system", "subtype": "turn_duration"},
        _user("둘째 질문"),
        _assistant({"type": "thinking", "thinking": "생각 중"}),
        _assistant(_text_block("둘째 답변"), usage=USAGE),
        _user("meta 메시지", isMeta=True),
        _user("셋째 질문"),
        _assistant(_text_block("셋째 답변 " + "가" * 600)),
    ]
    return _write_jsonl(tmp_path / "conv.jsonl", events)


# ---- read_tail_events ----------------------------------------------------


class TestReadTailEvents:
    def test_reads_all_when_short(self, normal_jsonl: Path) -> None:
        events = te.read_tail_events(normal_jsonl)
        assert len(events) == 13

    def test_max_lines_keeps_tail(self, normal_jsonl: Path) -> None:
        events = te.read_tail_events(normal_jsonl, max_lines=2)
        assert len(events) == 2
        assert events[-1]["type"] == "assistant"

    def test_corrupt_lines_skipped(self, tmp_path: Path) -> None:
        path = _write_jsonl(
            tmp_path / "c.jsonl", [_user("살아남은 줄")], corrupt_after=0
        )
        events = te.read_tail_events(path)
        assert len(events) == 1
        assert events[0]["message"]["content"] == "살아남은 줄"

    def test_non_dict_line_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "n.jsonl"
        path.write_text('["not", "a", "dict"]\n42\n', encoding="utf-8")
        assert te.read_tail_events(path) == []

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert te.read_tail_events(tmp_path / "nope.jsonl") == []

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert te.read_tail_events(path) == []


# ---- extract_dialogue ----------------------------------------------------


class TestExtractDialogue:
    def test_filters_and_formats(self, normal_jsonl: Path) -> None:
        events = te.read_tail_events(normal_jsonl)
        out = te.extract_dialogue(events)
        # Slash-command records, tool blocks, meta events are gone;
        # the three real exchanges survive in order.
        # 슬래시 명령 기록·도구 블록·메타 이벤트는 제거되고 실제 교환
        # 3개가 순서대로 남는다.
        assert out.startswith("user: 첫 질문입니다\nassistant: 첫 답변")
        assert "tool_result" not in out
        assert "생각 중" not in out
        assert "meta 메시지" not in out
        assert "<command-name>" not in out

    def test_max_exchanges_limits_from_tail(self, normal_jsonl: Path) -> None:
        events = te.read_tail_events(normal_jsonl)
        out = te.extract_dialogue(events, max_exchanges=1)
        assert out.startswith("user: 셋째 질문")
        assert "첫 질문" not in out
        assert "둘째 질문" not in out

    def test_max_chars_truncates_each_message(self, normal_jsonl: Path) -> None:
        events = te.read_tail_events(normal_jsonl)
        out = te.extract_dialogue(events, max_chars=20)
        for line in out.splitlines():
            role, _, text = line.partition(": ")
            assert len(text) <= 20, line

    def test_tool_result_only_region(self, tmp_path: Path) -> None:
        path = _write_jsonl(
            tmp_path / "t.jsonl",
            [_tool_result_user(), _assistant({"type": "tool_use", "name": "Bash", "input": {}})],
        )
        assert te.extract_dialogue(te.read_tail_events(path)) == ""

    def test_empty_events(self) -> None:
        assert te.extract_dialogue([]) == ""

    def test_user_text_blocks_kept(self, tmp_path: Path) -> None:
        """A user turn stored as a block list (e.g. with an image) still counts.

        블록 리스트로 저장된 user 발화 (예: 이미지 첨부) 도 발췌에 포함된다.
        """
        events = [
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "image", "source": {}},
                        {"type": "text", "text": "이 스크린샷을 봐줘"},
                    ]
                },
            },
            _assistant(_text_block("확인했습니다")),
        ]
        path = _write_jsonl(tmp_path / "img.jsonl", events)
        out = te.extract_dialogue(te.read_tail_events(path))
        assert out == "user: 이 스크린샷을 봐줘\nassistant: 확인했습니다"

    def test_interruption_marker_dropped(self, tmp_path: Path) -> None:
        """CLI-written interruption markers are not user dialogue.

        CLI 가 쓴 중단 표식은 사용자 발화가 아니다 (실측된 형태: 블록 리스트).
        """
        events = [
            _user("진짜 질문"),
            {
                "type": "user",
                "message": {
                    "content": [
                        {"type": "text", "text": "[Request interrupted by user]"}
                    ]
                },
            },
        ]
        path = _write_jsonl(tmp_path / "int.jsonl", events)
        out = te.extract_dialogue(te.read_tail_events(path))
        assert out == "user: 진짜 질문"


# ---- extract_full_text ---------------------------------------------------


class TestExtractFullText:
    def test_full_dialogue(self, normal_jsonl: Path) -> None:
        out = te.extract_full_text(normal_jsonl)
        assert "user: 첫 질문입니다" in out
        assert "assistant: 셋째 답변" in out

    def test_max_chars_keeps_tail(self, normal_jsonl: Path) -> None:
        out = te.extract_full_text(normal_jsonl, max_chars=50)
        # Budget respected and the newest dialogue survives. Here the final
        # message alone exceeds the budget, so there is no line boundary to
        # cut at — its tail is kept (documented fallback).
        # 예산을 지키고 가장 최근 대화가 남는다. 이 픽스처는 마지막 메시지
        # 하나가 예산을 넘으므로 자를 줄 경계가 없다 — 꼬리를 남긴다 (명시된 fallback).
        assert len(out) <= 50
        assert out.endswith("가" * 10)

    def test_line_boundary_cut_drops_partial_first_line(self, tmp_path: Path) -> None:
        events = [_user("첫 번째 질문입니다"), _user("두 번째 질문입니다")]
        path = _write_jsonl(tmp_path / "b.jsonl", events)
        full = te.extract_full_text(path)
        # Slice mid-way through the first line — that partial line must go.
        # 첫 줄 중간을 자르는 예산 — 잘린 줄은 통째로 버려져야 한다.
        out = te.extract_full_text(path, max_chars=len(full) - 3)
        assert out == "user: 두 번째 질문입니다"

    def test_corrupt_lines_skipped(self, tmp_path: Path) -> None:
        path = _write_jsonl(
            tmp_path / "c.jsonl",
            [_user("질문"), _assistant(_text_block("답변"))],
            corrupt_after=0,
        )
        assert te.extract_full_text(path) == "user: 질문\nassistant: 답변"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert te.extract_full_text(tmp_path / "nope.jsonl") == ""

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert te.extract_full_text(path) == ""


# ---- read_last_usage -----------------------------------------------------


class TestReadLastUsage:
    def test_returns_last_assistant_usage(self, normal_jsonl: Path) -> None:
        usage = te.read_last_usage(normal_jsonl)
        # The last assistant event has no usage; the latest one that
        # does is the "둘째 답변" event carrying USAGE.
        # 마지막 assistant 이벤트에는 usage 가 없으므로, usage 를 가진
        # 가장 최근 이벤트 ("둘째 답변") 의 값이 나와야 한다.
        assert usage == USAGE

    def test_no_usage_returns_none(self, tmp_path: Path) -> None:
        path = _write_jsonl(
            tmp_path / "u.jsonl", [_user("질문"), _assistant(_text_block("답변"))]
        )
        assert te.read_last_usage(path) is None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert te.read_last_usage(tmp_path / "nope.jsonl") is None

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        assert te.read_last_usage(path) is None
