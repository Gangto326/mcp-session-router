"""
Format the handoff block the wrapper injects into Claude Code's prompt.

Emitted right after ``/resume`` (or after spawning a fresh session) so
the destination session knows which session it replaced, why the user
switched, what to re-read from disk, and what the user actually asked for.

래퍼가 Claude Code의 프롬프트에 주입하는 핸드오프 블록을 만드는 모듈이다.

``/resume`` 직후(또는 새 세션 spawn 직후)에 주입되어, 복귀한 세션이 어떤
세션을 대체했는지, 왜 전환됐는지, 디스크에서 무엇을 다시 읽어야 하는지,
사용자가 실제로 요청한 내용이 무엇인지 알린다.
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


