"""
Unit tests for the handoff/init/register injection formatters.

handoff/init/register 주입 텍스트 포맷터 단위 테스트.
"""

from __future__ import annotations

import json

from session_manager.wrapper.handoff_formatter import (
    format_handoff_injection,
    format_init_injection,
    format_register_injection,
)


class TestFormatHandoffInjection:
    def test_basic_shape(self) -> None:
        handoff = {
            "from": "old-session",
            "message": "context bridge",
            "instructions": ["read static-field.json"],
        }
        result = format_handoff_injection(handoff, "user request")
        assert result.startswith("[handoff]\n")
        assert "[/handoff]\n\n" in result
        assert result.endswith("user request")

    def test_json_block_is_valid_json(self) -> None:
        handoff = {"from": "a", "message": "b", "instructions": []}
        result = format_handoff_injection(handoff, "prompt")
        body = result.split("[handoff]\n", 1)[1].split("\n[/handoff]", 1)[0]
        parsed = json.loads(body)
        assert parsed == handoff

    def test_korean_preserved_not_ascii_escaped(self) -> None:
        # ensure_ascii=False 동작 검증 — 한국어가 \uXXXX 로 escape 되지 않아야 함.
        handoff = {"message": "한국어 메시지"}
        result = format_handoff_injection(handoff, "사용자 요청")
        assert "한국어 메시지" in result
        assert "사용자 요청" in result
        assert "\\u" not in result

    def test_empty_user_prompt(self) -> None:
        handoff = {"from": "a"}
        result = format_handoff_injection(handoff, "")
        assert result.endswith("[/handoff]\n\n")

    def test_indent_two(self) -> None:
        handoff = {"a": 1}
        result = format_handoff_injection(handoff, "x")
        assert '  "a": 1' in result


class TestFormatInitInjection:
    def test_contains_required_keywords(self) -> None:
        result = format_init_injection()
        assert "[자동 초기화]" in result
        assert "init_project" in result
        assert "project-context.md" in result


class TestFormatRegisterInjection:
    def test_contains_required_keywords(self) -> None:
        result = format_register_injection()
        assert "[자동 초기화]" in result
        assert "session_register" in result
