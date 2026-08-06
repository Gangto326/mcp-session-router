"""Routing decision log — the calibration data source for auto mode.

라우팅 결정 로그 — auto 모드 보정의 데이터 원천.

Why not debug_log: that log only exists when ``SESSION_MANAGER_DEBUG=1``,
so it cannot carry calibration data — auto mode would then depend on a
debugging flag. This module writes an always-on, append-only JSONL at
``.session-manager/routing-decisions.jsonl`` instead.

debug_log 를 쓰지 않는 이유: 그 로그는 ``SESSION_MANAGER_DEBUG=1`` 일
때만 존재하므로 보정 데이터를 실을 수 없다 — auto 모드가 디버깅 플래그에
종속되어 버린다. 이 모듈은 항상 켜져 있는 append 전용 JSONL 을
``.session-manager/routing-decisions.jsonl`` 에 쓴다.

Event shapes / 이벤트 형태:

- ``{"type": "proposal", "target", "confidence", "mode", "at"}`` —
  the hook proposed a SWITCH (confirm context emitted, or auto executed).
- ``{"type": "label", "label": "accept"|"reject", "target", "source",
  "at"}`` — the user's decision arrived: ``session_switch`` = accept,
  ``reject_switch`` / ``/back`` = reject.

Pairing (rule: label definition, Plan §4 R3-C4): a label consumes the
most recent unlabeled proposal with the same target. Proposals the user
ignored stay unlabeled and are EXCLUDED from calibration pairs — ignored
is not rejected. Auto switches log a proposal but produce no accept
label (the user was never asked); only a later ``/back`` labels them.

쌍 구성 (라벨 정의 — Plan §4 R3-C4): 라벨은 같은 target 의 가장 최근
미라벨 proposal 을 소비한다. 사용자가 무시한 proposal 은 미라벨로 남아
보정 쌍에서 제외된다 — 무시는 거부가 아니다. auto 전환은 proposal 만
기록하고 수용 라벨을 만들지 않는다 (사용자에게 묻지 않았으므로) — 이후
``/back`` 만이 거부 라벨을 남긴다.

Writes are single-line ``O_APPEND`` appends from multiple processes
(hook, MCP server, wrapper) — atomic for lines far below PIPE_BUF.

쓰기는 여러 프로세스 (hook·MCP 서버·래퍼) 의 한 줄 ``O_APPEND`` append
다 — PIPE_BUF 보다 훨씬 짧은 줄이라 원자적이다.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from session_manager import debug_log

_SESSION_MANAGER_DIRNAME = ".session-manager"
DECISIONS_FILENAME = "routing-decisions.jsonl"

LABEL_ACCEPT = "accept"
LABEL_REJECT = "reject"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _log_path(project_path: Path) -> Path:
    return Path(project_path) / _SESSION_MANAGER_DIRNAME / DECISIONS_FILENAME


def _append(project_path: Path, event: dict[str, Any]) -> None:
    path = _log_path(project_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Losing one calibration event must never break routing.
        # 보정 이벤트 하나의 유실이 라우팅을 깨서는 안 된다.
        debug_log.log(
            "DECISION_LOG",
            "SYSTEM",
            {"op": "append", "result": "error", "error": str(exc)},
        )


def append_proposal(
    project_path: Path, target: str, confidence: float, mode: str
) -> None:
    """Record one SWITCH proposal the router made.

    라우터가 낸 SWITCH 제안 1건을 기록한다.
    """
    _append(
        project_path,
        {
            "type": "proposal",
            "target": target,
            "confidence": confidence,
            "mode": mode,
            "at": _utc_now_iso(),
        },
    )


def append_label(
    project_path: Path, target: str, label: str, source: str
) -> None:
    """Record the user's accept/reject decision for *target*.

    *target* 에 대한 사용자의 수용/거부 결정을 기록한다.
    """
    _append(
        project_path,
        {
            "type": "label",
            "label": label,
            "target": target,
            "source": source,
            "at": _utc_now_iso(),
        },
    )


def load_events(project_path: Path) -> list[dict[str, Any]]:
    """Read every well-formed event, in file order. Corrupt lines skipped.

    정상 이벤트 전부를 파일 순서로 읽는다. 손상 줄은 건너뛴다.
    """
    path = _log_path(project_path)
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def labeled_pairs(events: list[dict[str, Any]]) -> list[tuple[float, bool]]:
    """Build (confidence, accepted) calibration pairs from raw events.

    원 이벤트에서 (confidence, 수용 여부) 보정 쌍을 만든다. 쌍 구성
    규칙은 모듈 docstring 참조 — 미라벨 proposal 과 무연고 label 은
    제외된다.
    """
    # Per-target stacks of unlabeled proposals, most recent last.
    # target 별 미라벨 proposal 스택 — 최근 것이 뒤.
    open_proposals: dict[str, list[float]] = {}
    pairs: list[tuple[float, bool]] = []
    for event in events:
        target = event.get("target")
        if not isinstance(target, str) or not target:
            continue
        if event.get("type") == "proposal":
            confidence = event.get("confidence")
            if isinstance(confidence, int | float):
                open_proposals.setdefault(target, []).append(float(confidence))
        elif event.get("type") == "label":
            label = event.get("label")
            if label not in (LABEL_ACCEPT, LABEL_REJECT):
                continue
            stack = open_proposals.get(target)
            if not stack:
                # Orphan label (e.g. a voluntary session_switch with no
                # preceding proposal) — not calibration evidence.
                # 무연고 label (제안 없이 자발 호출된 session_switch 등)
                # — 보정 증거가 아니다.
                continue
            pairs.append((stack.pop(), label == LABEL_ACCEPT))
    return pairs


def acceptance_stats(project_path: Path) -> dict[str, int]:
    """Overall accept/reject/unlabeled tallies (displayed by /router, R3-C5).

    전체 수용/거부/미라벨 집계 (/router 표시는 R3-C5).
    """
    events = load_events(project_path)
    pairs = labeled_pairs(events)
    proposals = sum(1 for e in events if e.get("type") == "proposal")
    accepted = sum(1 for _, ok in pairs if ok)
    return {
        "accepted": accepted,
        "rejected": len(pairs) - accepted,
        "unlabeled": proposals - len(pairs),
    }
