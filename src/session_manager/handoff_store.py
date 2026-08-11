"""Pending-handoff file — how a transition's context crosses the respawn.

pending handoff 파일 — 전환 컨텍스트가 respawn 을 건너는 통로.

The redesigned transition (docs/poc/R3-respawn.md) never types into the
TUI. The wrapper writes the handoff (+ the user's prompt) to this file,
respawns ``claude --resume=<conv> "<trigger>"``, and the UserPromptSubmit
hook — firing for that CLI trigger prompt — consumes the file and hands
its content to the LLM as additionalContext. Every leg of that path is
an official interface (CLI args, hooks), so no renderer coupling.

재설계된 전환 (docs/poc/R3-respawn.md) 은 TUI 에 타이핑하지 않는다.
래퍼가 handoff (+ 사용자 프롬프트) 를 이 파일에 쓰고 ``claude
--resume=<conv> "<트리거>"`` 로 재시작하면, 그 CLI 트리거 프롬프트에
발동하는 UserPromptSubmit hook 이 파일을 소비해 additionalContext 로
LLM 에 전달한다. 전 구간이 공식 인터페이스 (CLI 인자·hook) 라 렌더러
결합이 없다.

Privacy: the prompt/handoff content stays in this file (0-permission
inherited from .session-manager) — argv carries only the fixed trigger
text, since argv is world-readable via ``ps`` (same principle as the
summarizer's stdin rule).

프라이버시 — 프롬프트·handoff 내용은 이 파일에만 있다. argv 는 ``ps``
로 누구나 읽을 수 있으므로 고정 트리거 문구만 싣는다 (요약기의 stdin
규칙과 같은 원칙).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from session_manager import debug_log

_SESSION_MANAGER_DIRNAME = ".session-manager"
_HANDOFFS_DIRNAME = "handoffs"
_PENDING_FILENAME = "pending.json"

# Fixed trigger prompt the wrapper passes as the respawn's CLI initial
# prompt. The hook matches it verbatim to decide "this submission is the
# transition trigger": consume the pending file, inject its content,
# skip routing. Content-free on purpose — the real request travels in
# the file.
# 래퍼가 respawn 의 CLI 초기 프롬프트로 넘기는 고정 트리거. hook 이 이
# 문구를 그대로 매칭해 "이 제출은 전환 트리거"로 판단한다 — pending
# 파일 소비·내용 주입·라우팅 skip. 실제 요청은 파일로 이동하므로
# 트리거 자체는 의도적으로 무내용이다.
TRIGGER_PROMPT = "[session-manager] 세션 전환 재개"


def _pending_path(project_path: Path) -> Path:
    return (
        Path(project_path)
        / _SESSION_MANAGER_DIRNAME
        / _HANDOFFS_DIRNAME
        / _PENDING_FILENAME
    )


def write_pending(
    project_path: Path,
    target: str,
    handoff: dict[str, Any],
    user_prompt: str,
) -> None:
    """Persist the transition context for the post-respawn hook.

    respawn 후 hook 이 읽을 전환 컨텍스트를 영속화한다.
    """
    path = _pending_path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "target": target,
                "handoff": handoff,
                "user_prompt": user_prompt,
                "at": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)
    debug_log.log(
        "HANDOFF_PENDING",
        "WRAPPER",
        {"op": "write", "target": target},
        session=target,
    )


def take_pending(project_path: Path) -> dict[str, Any] | None:
    """Consume the pending handoff: read it and remove the file.

    pending handoff 를 소비한다 — 읽고 파일을 제거. 없거나 손상이면 None
    (손상 파일도 제거해 다음 전환을 막지 않는다).
    """
    path = _pending_path(project_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    path.unlink(missing_ok=True)
    try:
        data = json.loads(raw)
    except ValueError:
        debug_log.log(
            "HANDOFF_PENDING", "SYSTEM", {"op": "take", "result": "corrupt"}
        )
        return None
    if not isinstance(data, dict):
        return None
    debug_log.log(
        "HANDOFF_PENDING",
        "SYSTEM",
        {"op": "take", "target": data.get("target")},
        session=data.get("target"),
    )
    return data


def clear_stale_pending(project_path: Path) -> bool:
    """Drop a leftover pending file at wrapper boot.

    래퍼 부팅 시 잔류 pending 파일을 정리한다.

    A pending file at boot means a transition never reached its trigger
    (crash between write and respawn). The trigger will never come, and
    consuming it on an unrelated future prompt would inject a stale
    handoff — deletion is the safe disposal. Returns True if one existed.

    부팅 시점의 pending 파일은 전환이 트리거에 도달하지 못했다는 뜻이다
    (쓰기~respawn 사이 크래시). 트리거는 더 오지 않고, 무관한 미래
    프롬프트가 소비하면 낡은 handoff 가 주입된다 — 삭제가 안전한 처분이다.
    존재했으면 True.
    """
    path = _pending_path(project_path)
    if not path.is_file():
        return False
    path.unlink(missing_ok=True)
    debug_log.log("HANDOFF_PENDING", "WRAPPER", {"op": "clear_stale"})
    return True
