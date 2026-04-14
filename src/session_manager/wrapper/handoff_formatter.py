"""
Format the text payloads the wrapper injects into Claude Code's prompt.

Three text shapes are produced here, each consumed by the LLM running
inside Claude Code:

1. A handoff block — emitted right after `/resume` (or after spawning a
   fresh session) so the destination session knows which session it
   replaced, why the user switched, what to re-read from disk, and what
   the user actually asked for.
2. An init instruction — emitted on the first prompt of a project that
   has no `project-context.md` yet, telling the LLM to call
   `init_project`.
3. A register instruction — emitted when the current Claude Code session
   isn't registered with the MCP server yet, telling the LLM to call
   `session_register`.

Keeping formatting isolated here lets the I/O loop choose *when* to inject
without also owning the *shape* of the injection, which keeps both halves
testable in isolation.

Claude Code의 프롬프트에 래퍼가 주입할 텍스트를 만드는 모듈이다.

여기서 만들어지는 세 가지 텍스트는 모두 Claude Code 내부의 LLM이 곧바로
읽는 입력이 된다.

1. 핸드오프 블록 — `/resume` 직후(또는 새 세션 spawn 직후)에 주입되어,
   복귀한 세션이 어떤 세션을 대체했는지, 왜 전환됐는지, 디스크에서 무엇을
   다시 읽어야 하는지, 사용자가 실제로 요청한 내용이 무엇인지 알린다.
2. init 지시 — `project-context.md`가 아직 없는 프로젝트의 첫 프롬프트에서
   주입되어, LLM에게 `init_project` 도구를 호출하라고 안내한다.
3. register 지시 — 현재 Claude Code 세션이 MCP에 등록되지 않은 경우에
   주입되어, `session_register` 도구를 호출하라고 안내한다.

형식 책임을 이 모듈에 모아 두면 I/O 루프는 "언제" 주입할지에만 집중할 수
있고, 두 책임을 따로 테스트할 수 있다.
"""

from __future__ import annotations

import json
from typing import Any


def format_handoff_injection(handoff: dict[str, Any], user_prompt: str) -> str:
    """
    Build the `[handoff]…[/handoff]` block followed by the user's prompt.

    `[handoff]…[/handoff]` 블록과 그 뒤에 사용자 프롬프트가 따라오는
    텍스트를 만든다.
    """
    body = json.dumps(handoff, ensure_ascii=False, indent=2)
    return f"[handoff]\n{body}\n[/handoff]\n\n{user_prompt}"


def format_init_injection() -> str:
    """
    Instruction asking the LLM to bootstrap project context.

    LLM에게 프로젝트 컨텍스트를 부트스트랩하라고 요청하는 지시 텍스트.
    """
    return (
        "[자동 초기화] project-context.md가 없습니다. "
        "init_project 도구를 호출하여 프로젝트 컨텍스트를 생성하세요."
    )


def format_register_injection() -> str:
    """
    Instruction asking the LLM to register the current session.

    LLM에게 현재 세션을 MCP에 등록하라고 요청하는 지시 텍스트.
    """
    return (
        "[자동 초기화] 현재 세션이 MCP에 등록되지 않았습니다. "
        "session_register 도구를 호출하여 이름과 제목으로 등록하세요."
    )
