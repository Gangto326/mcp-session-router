"""
Unit tests for the PreToolUse transcript guard hook.

Main-agent transcript reads are denied; subagents and everything else
pass. Deny rides on stdout (measured format); exit code is always 0.

PreToolUse transcript 가드 hook 단위 테스트.

메인 에이전트의 transcript 읽기는 거부, 서브 에이전트와 그 외 전부는
통과. deny 는 stdout 에 실리고 (실측 형식) exit code 는 항상 0 이다.
"""

from __future__ import annotations

import json

import pytest

from session_manager.hooks import pre_tool_use as guard

TRANSCRIPT = "/Users/me/.claude/projects/-Users-me-proj/abc-123.jsonl"


def _payload(**overrides: object) -> str:
    data: dict[str, object] = {
        "session_id": "conv-1",
        "hook_event_name": "PreToolUse",
        "tool_name": "Read",
        "tool_input": {"file_path": TRANSCRIPT},
        "tool_use_id": "toolu_x",
        **overrides,
    }
    return json.dumps(data)


def _denied(capsys: pytest.CaptureFixture) -> bool:
    out = capsys.readouterr().out
    if not out:
        return False
    decision = json.loads(out)["hookSpecificOutput"]
    assert decision["hookEventName"] == "PreToolUse"
    assert "서브 에이전트" in decision["permissionDecisionReason"]
    return decision["permissionDecision"] == "deny"


class TestDeny:
    def test_main_agent_read_of_transcript_denied(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        assert guard.run(_payload()) == 0
        assert _denied(capsys)

    def test_main_agent_bash_with_transcript_path_denied(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        payload = _payload(
            tool_name="Bash",
            tool_input={"command": f"tail -50 {TRANSCRIPT} | grep usage"},
        )
        assert guard.run(payload) == 0
        assert _denied(capsys)


class TestAllow:
    def test_subagent_read_allowed(self, capsys: pytest.CaptureFixture) -> None:
        # 실측 (docs/poc/R2-hook.md §4): 서브 에이전트 호출에만 agent_id 존재
        assert guard.run(_payload(agent_id="agent-1")) == 0
        assert capsys.readouterr().out == ""

    def test_subagent_agent_type_marker_allowed(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        assert guard.run(_payload(agent_type="Explore")) == 0
        assert capsys.readouterr().out == ""

    def test_read_of_normal_file_allowed(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        payload = _payload(tool_input={"file_path": "/Users/me/proj/main.py"})
        assert guard.run(payload) == 0
        assert capsys.readouterr().out == ""

    def test_bash_without_transcript_path_allowed(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        payload = _payload(
            tool_name="Bash", tool_input={"command": "ls ~/.claude/projects"}
        )
        assert guard.run(payload) == 0
        assert capsys.readouterr().out == ""

    def test_other_tools_allowed(self, capsys: pytest.CaptureFixture) -> None:
        payload = _payload(
            tool_name="Grep",
            tool_input={"pattern": "x", "path": TRANSCRIPT},
        )
        assert guard.run(payload) == 0
        assert capsys.readouterr().out == ""


class TestGracefulDegradation:
    def test_malformed_stdin_exits_zero(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        assert guard.run("not json") == 0
        assert capsys.readouterr().out == ""

    def test_non_object_payload_exits_zero(self) -> None:
        assert guard.run("[1]") == 0

    def test_tool_input_wrong_type_allowed(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        assert guard.run(_payload(tool_input="oops")) == 0
        assert capsys.readouterr().out == ""
