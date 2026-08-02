"""
PreToolUse transcript guard hook.

Blocks the MAIN agent from reading past-conversation transcripts
(``~/.claude/projects/**/*.jsonl``) with Read or Bash, steering it to a
subagent instead. Rationale: the rollover flow (a later phase) buys a
clean context by leaving the old conversation behind — letting the main
agent re-read the raw transcript (often tens of MB) would spend that
asset right back. A subagent reads in its own disposable context and
returns only a summary.

Subagent tool calls carry ``agent_id``/``agent_type`` in the hook input
(measured: docs/poc/R2-hook.md §4) and pass through. The deny output
format is measured too (§10). Same graceful-degradation contract as the
routing hook: any internal failure exits 0 (allow).

PreToolUse transcript 가드 hook.

**메인** 에이전트가 과거 대화 transcript (``~/.claude/projects/**/*.jsonl``)
를 Read·Bash 로 직접 읽는 것을 차단하고 서브 에이전트 경로로 안내한다.
근거: 롤오버 흐름 (후속 Phase) 은 이전 대화를 두고 떠나는 대가로 깨끗한
컨텍스트를 얻는데, 메인 에이전트가 원문 transcript (수십 MB 인 경우도
있다) 를 다시 읽으면 그 자산을 도로 소모한다. 서브 에이전트는 자기
일회용 컨텍스트에서 읽고 요약만 돌려준다.

서브 에이전트의 도구 호출은 hook 입력에 ``agent_id``/``agent_type`` 이
붙으므로 (실측: docs/poc/R2-hook.md §4) 통과시킨다. deny 출력 형식도
실측 완료 (§10). 라우팅 hook 과 동일한 graceful degradation 계약 —
내부 실패는 전부 exit 0 (허용) 이다.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

from session_manager import debug_log

# Input field names (measured: docs/poc/R2-hook.md §4, §10).
# 입력 필드명 상수 (실측: docs/poc/R2-hook.md §4, §10).
FIELD_TOOL_NAME = "tool_name"
FIELD_TOOL_INPUT = "tool_input"
FIELD_SESSION_ID = "session_id"

# Presence of either field marks a subagent tool call (measured §4).
# 둘 중 하나라도 있으면 서브 에이전트의 도구 호출이다 (실측 §4).
AGENT_MARKER_FIELDS = ("agent_id", "agent_type")

# Transcript path pattern. Matches any path that goes through
# ``.claude/projects/`` and ends in ``.jsonl`` — home-anchored absolute
# paths as delivered by Read's file_path, and the same substring inside
# a Bash command line.
# transcript 경로 패턴. ``.claude/projects/`` 를 거쳐 ``.jsonl`` 로 끝나는
# 경로에 매칭 — Read 의 file_path 로 오는 절대 경로와, Bash 명령줄 안의
# 동일 부분 문자열 모두를 잡는다.
_TRANSCRIPT_RE = re.compile(r"\.claude/projects/[^\s'\"]*\.jsonl")

_DENY_REASON = (
    "과거 대화 transcript(~/.claude/projects/**/*.jsonl)는 메인 컨텍스트에서 "
    "직접 읽지 마라. 필요하면 서브 에이전트(Task 도구)를 통해 조회해 "
    "요약만 받아라."
)


def _is_transcript_access(payload: dict[str, Any]) -> bool:
    """Return True iff this tool call targets a transcript file.

    이 도구 호출이 transcript 파일을 대상으로 하면 True.
    """
    tool_input = payload.get(FIELD_TOOL_INPUT)
    if not isinstance(tool_input, dict):
        return False
    tool_name = payload.get(FIELD_TOOL_NAME)
    if tool_name == "Read":
        file_path = tool_input.get("file_path")
        return isinstance(file_path, str) and bool(
            _TRANSCRIPT_RE.search(file_path)
        )
    if tool_name == "Bash":
        command = tool_input.get("command")
        return isinstance(command, str) and bool(_TRANSCRIPT_RE.search(command))
    return False


def _deny() -> None:
    """Emit the measured deny payload (docs/poc/R2-hook.md §10).

    실측된 deny 페이로드를 출력한다 (docs/poc/R2-hook.md §10).
    """
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": _DENY_REASON,
                }
            },
            ensure_ascii=False,
        )
    )


def run(stdin_text: str) -> int:
    """
    Process one hook invocation. Always returns 0 — the decision rides
    on stdout, and a guard failure must never block a tool call.

    hook 호출 1건을 처리한다. 항상 0 을 반환한다 — 결정은 stdout 에
    실리며, 가드의 실패가 도구 호출을 막아서는 안 된다.
    """
    try:
        payload = json.loads(stdin_text)
    except ValueError:
        return 0
    if not isinstance(payload, dict):
        return 0
    try:
        if any(field in payload for field in AGENT_MARKER_FIELDS):
            # Subagent: reads in its own disposable context — allowed.
            # 서브 에이전트 — 일회용 자기 컨텍스트에서 읽으므로 허용.
            return 0
        if _is_transcript_access(payload):
            _deny()
            debug_log.log(
                "HOOK_TRANSCRIPT_GUARD",
                "SYSTEM",
                {
                    "denied": True,
                    "tool": payload.get(FIELD_TOOL_NAME),
                },
                conv_id=payload.get(FIELD_SESSION_ID),
            )
    except Exception:
        # Never let a guard bug block tool use.
        # 가드 버그가 도구 사용을 막는 일은 절대 없어야 한다.
        pass
    return 0


def main() -> None:
    try:
        debug_log.set_proc_label("hook")
        code = run(sys.stdin.read())
    except Exception:
        code = 0
    sys.exit(code)
