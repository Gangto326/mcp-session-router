"""
Unit tests for the rollover handoff module.

Focus: generation numbering, the request template's fixed skeleton,
mechanical validation, transcript extraction, and a fallback that
always passes its own validation.

롤오버 handoff 모듈 단위 테스트.

초점: 세대 번호, 요청 템플릿의 고정 골격, 기계 검증, transcript 추출,
그리고 스스로의 검증을 항상 통과하는 폴백.
"""

from __future__ import annotations

import json
from pathlib import Path

from session_manager import rollover

VALID_BODY = (
    "# Handoff: backend #1\n"
    "## 1. 지금 바로 할 일 (재개 지점)\n내용\n"
    "## 2. 사용자 요구사항\n내용\n"
)


class TestNumbering:
    def test_first_is_one(self, tmp_path: Path) -> None:
        assert rollover.next_handoff_number(tmp_path, "backend") == 1

    def test_continues_after_highest(self, tmp_path: Path) -> None:
        d = rollover.handoffs_dir(tmp_path)
        d.mkdir(parents=True)
        (d / "backend-1.md").write_text("x", encoding="utf-8")
        (d / "backend-3.md").write_text("x", encoding="utf-8")
        assert rollover.next_handoff_number(tmp_path, "backend") == 4

    def test_other_sessions_and_junk_ignored(self, tmp_path: Path) -> None:
        d = rollover.handoffs_dir(tmp_path)
        d.mkdir(parents=True)
        (d / "frontend-9.md").write_text("x", encoding="utf-8")
        (d / "backend-abc.md").write_text("x", encoding="utf-8")
        assert rollover.next_handoff_number(tmp_path, "backend") == 1


class TestBuildRequest:
    def test_contains_skeleton_and_params(self, tmp_path: Path) -> None:
        request = rollover.build_request(
            tmp_path, "backend", 2, "conv-7", ["ruff 클린 유지"]
        )
        assert "# Handoff: backend #2" in request
        assert "## 1. 지금 바로 할 일" in request
        assert "## 6. 이전 대화: conv-7" in request
        assert ".session-manager/handoffs/backend-2.md" in request
        assert "- ruff 클린 유지" in request
        # The response rule: print, don't use tools (see rollover.py
        # module docstring — permission-dialog hazard).
        # 응답 규칙 — 도구 대신 본문 출력 (rollover.py docstring 참조).
        assert "도구를 사용하지 말고" in request

    def test_empty_requirements_marker(self, tmp_path: Path) -> None:
        request = rollover.build_request(tmp_path, "s", 1, "c", [])
        assert "[축적된 requirements] (없음)" in request


class TestValidate:
    def test_valid_body(self) -> None:
        assert rollover.validate_handoff_text(VALID_BODY) is True

    def test_missing_section_two(self) -> None:
        body = "# Handoff\n## 1. 지금 바로 할 일\n내용\n"
        assert rollover.validate_handoff_text(body) is False

    def test_empty_or_none(self) -> None:
        assert rollover.validate_handoff_text("") is False
        assert rollover.validate_handoff_text("   \n") is False


class TestWriteHandoff:
    def test_writes_atomically_and_creates_dir(self, tmp_path: Path) -> None:
        path = rollover.write_handoff(tmp_path, "backend", 1, VALID_BODY)
        assert path == rollover.handoff_path(tmp_path, "backend", 1)
        assert path.read_text(encoding="utf-8") == VALID_BODY


TRIGGER = "[session-manager] 세션 전환 재개"
REQUEST_AT = "2026-08-13T12:00:00+00:00"
BEFORE = "2026-08-13T11:59:00.000Z"
AFTER = "2026-08-13T12:00:05.000Z"


class TestCheckTriggerTurn:
    def _write(self, path: Path, events: list[dict]) -> None:
        path.write_text(
            "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
        )

    def _user(self, text: str, ts: str) -> dict:
        return {"type": "user", "timestamp": ts, "message": {"content": text}}

    def _assistant(self, text: str) -> dict:
        return {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": text}]},
        }

    def test_answered_after_trigger(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "c.jsonl"
        self._write(
            jsonl,
            [
                self._user("옛 프롬프트", BEFORE),
                self._assistant("옛 응답"),
                self._user(TRIGGER, AFTER),
                self._assistant("# Handoff 본문"),
            ],
        )
        status, text = rollover.check_trigger_turn(jsonl, TRIGGER, REQUEST_AT)
        assert status == "answered"
        assert text == "# Handoff 본문"

    def test_boot_edge_race_is_waiting(self, tmp_path: Path) -> None:
        # The measured race: only pre-request events exist when a boot
        # edge fires — the attempt must NOT be consumed.
        # 실측 레이스 — 부팅 에지 시점에는 요청 이전 이벤트뿐이다. 시도가
        # 소진되면 안 된다.
        jsonl = tmp_path / "c.jsonl"
        self._write(
            jsonl,
            [self._user("옛 프롬프트", BEFORE), self._assistant("2")],
        )
        assert rollover.check_trigger_turn(jsonl, TRIGGER, REQUEST_AT) == (
            "waiting",
            "",
        )

    def test_trigger_without_reply_is_waiting(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "c.jsonl"
        self._write(jsonl, [self._user(TRIGGER, AFTER)])
        assert rollover.check_trigger_turn(jsonl, TRIGGER, REQUEST_AT) == (
            "waiting",
            "",
        )

    def test_newer_foreign_user_without_trigger_is_missing(
        self, tmp_path: Path
    ) -> None:
        jsonl = tmp_path / "c.jsonl"
        self._write(jsonl, [self._user("사용자가 다른 걸 물음", AFTER)])
        assert rollover.check_trigger_turn(jsonl, TRIGGER, REQUEST_AT) == (
            "missing",
            "",
        )

    def test_missing_file_is_waiting(self, tmp_path: Path) -> None:
        assert rollover.check_trigger_turn(
            tmp_path / "no.jsonl", TRIGGER, REQUEST_AT
        ) == ("waiting", "")


class TestSuccessorInjection:
    def test_dict_and_prompt(self, tmp_path: Path) -> None:
        handoff, prompt = rollover.successor_injection(tmp_path, "backend", 2)
        assert handoff["kind"] == "rollover"
        assert handoff["from"] == "backend"
        assert handoff["handoff_file"] == ".session-manager/handoffs/backend-2.md"
        assert handoff["read"][0] == handoff["handoff_file"]
        assert ".session-manager/handoffs/backend-2.md" in prompt
        assert "재개 지점" in prompt


class TestFallback:
    def test_fallback_passes_validation(self) -> None:
        body = rollover.build_fallback_handoff("s", 3, "conv-1", "발췌 텍스트")
        assert rollover.validate_handoff_text(body) is True
        assert "conv-1" in body
        assert "발췌 텍스트" in body
