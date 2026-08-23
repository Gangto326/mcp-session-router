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

- ``{"type": "proposal", "target", "confidence", "mode", "at",
  "target_status"?, "conv_id"?}`` — the hook proposed a SWITCH (confirm
  context emitted, or auto executed). ``target_status`` (R5-C3) is the
  proposed session's status at proposal time (``active`` / ``archived``
  / ``missing``) so ``ccode --stats`` can count proposals that pointed
  at an ended session — the unmeasured premise behind dropping
  ``/retire``. ``conv_id`` is the Claude conversation the prompt was
  submitted in — the scope of the expiry rule below.
- ``{"type": "expire", "conv_id", "at"}`` — a later prompt arrived in
  ``conv_id`` while a proposal made there was still unlabeled: the user
  neither accepted nor rejected, they moved on. Every open proposal of
  that conversation is closed as ignored. (Written by the hook only when
  such an open proposal exists, so the file does not grow per prompt.)
- ``{"type": "label", "label": "accept"|"reject", "target", "source",
  "at", "kept_in"?}`` — the user's decision arrived: ``session_switch``
  = accept, ``reject_switch`` / ``/back`` = reject. ``kept_in`` (R5-C3)
  is the session the user was in when deciding, so the precedent gate's
  (current, target) pair effect can be counted.

Both extra keys are optional: files written before R5-C3 lack them and
every reader treats a missing key as "not recorded".

두 추가 키는 선택이다 — R5-C3 이전에 쓰인 파일에는 없으며 모든 읽기
쪽은 키 부재를 "기록 안 됨" 으로 다룬다. ``target_status`` 는 제안 시점
대상 세션의 상태 (끝난 세션 오제안 집계 — ``/retire`` 제거의 미측정
전제), ``kept_in`` 은 결정 당시 사용자가 있던 세션 (판례 게이트의
(현재, 대상) 쌍 효과 집계) 이다.

Pairing (rule: label definition, Plan §4 R3-C4): a label consumes the
most recent OPEN proposal with the same target. Proposals the user
ignored are EXCLUDED from calibration pairs — ignored is not rejected.
"Ignored" is made deterministic by the ``expire`` event: without it an
unlabeled proposal would wait forever and be swallowed as "accepted" by
an unrelated voluntary ``session_switch`` to the same target days later
(found in the R5-C3 review). Proposals written before ``conv_id``
existed carry no conversation and are never expired — legacy lines keep
the old behaviour rather than being silently dropped. Auto switches log
a proposal but produce no accept label (the user was never asked); only
a later ``/back`` labels them — ``/back`` runs from the new conversation,
so the origin conversation's proposal is still open when it arrives.

쌍 구성 (라벨 정의 — Plan §4 R3-C4): 라벨은 같은 target 의 가장 최근
**열린** proposal 을 소비한다. 사용자가 무시한 proposal 은 보정 쌍에서
제외된다 — 무시는 거부가 아니다. "무시" 는 ``expire`` 이벤트로 결정적이
된다: 그것이 없으면 미라벨 proposal 이 영원히 대기하다가 며칠 뒤 같은
target 으로의 무관한 자발 ``session_switch`` 에 "수용" 으로 삼켜진다
(R5-C3 재검토에서 발견). ``conv_id`` 도입 전에 쓰인 proposal 은 대화
정보가 없어 만료되지 않는다 — 옛 줄은 조용히 버려지는 대신 옛 동작을
유지한다. auto 전환은 proposal 만 기록하고 수용 라벨을 만들지 않는다
(사용자에게 묻지 않았으므로) — 이후 ``/back`` 만이 거부 라벨을 남기며,
``/back`` 은 새 대화에서 실행되므로 원 대화의 proposal 은 그때까지 열려
있다.

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
    project_path: Path,
    target: str,
    confidence: float,
    mode: str,
    target_status: str | None = None,
    conv_id: str | None = None,
) -> None:
    """Record one SWITCH proposal the router made.

    라우터가 낸 SWITCH 제안 1건을 기록한다. *target_status*·*conv_id* 는
    알 때만 기록한다 (모듈 docstring 참조).
    """
    event: dict[str, Any] = {
        "type": "proposal",
        "target": target,
        "confidence": confidence,
        "mode": mode,
        "at": _utc_now_iso(),
    }
    if target_status is not None:
        event["target_status"] = target_status
    if conv_id is not None:
        event["conv_id"] = conv_id
    _append(project_path, event)


def expire_ignored_proposals(project_path: Path, conv_id: str) -> bool:
    """Close every open proposal of *conv_id* — a new prompt arrived there.

    *conv_id* 의 열린 proposal 을 전부 닫는다 — 그 대화에 새 프롬프트가
    왔다는 뜻이다. 열린 것이 있을 때만 ``expire`` 줄을 쓰고 True 를
    돌려준다 (프롬프트마다 파일이 자라지 않도록).
    """
    if not _has_open_proposal(load_events(project_path), conv_id):
        return False
    _append(
        project_path,
        {"type": "expire", "conv_id": conv_id, "at": _utc_now_iso()},
    )
    return True


def append_label(
    project_path: Path,
    target: str,
    label: str,
    source: str,
    kept_in: str | None = None,
) -> None:
    """Record the user's accept/reject decision for *target*.

    *target* 에 대한 사용자의 수용/거부 결정을 기록한다. *kept_in* 은
    알 때만 기록한다 (모듈 docstring 참조).
    """
    event: dict[str, Any] = {
        "type": "label",
        "label": label,
        "target": target,
        "source": source,
        "at": _utc_now_iso(),
    }
    if kept_in is not None:
        event["kept_in"] = kept_in
    _append(project_path, event)


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
    return _replay(events)[0]


def _replay(
    events: list[dict[str, Any]],
) -> tuple[list[tuple[float, bool]], _OpenProposals]:
    """Walk the events once; return (calibration pairs, proposals still open).

    이벤트를 한 번 훑어 (보정 쌍, 아직 열린 proposal) 을 돌려준다 —
    "열림" 의 뜻이 라벨 소비 규칙과 정확히 같도록 한 곳에서 재생한다.
    """
    pairs: list[tuple[float, bool]] = []
    open_proposals = _OpenProposals()
    for event in events:
        kind = event.get("type")
        if kind == "expire":
            conv_id = event.get("conv_id")
            if isinstance(conv_id, str):
                open_proposals.expire(conv_id)
            continue
        target = event.get("target")
        if not isinstance(target, str) or not target:
            continue
        if kind == "proposal":
            confidence = event.get("confidence")
            if isinstance(confidence, int | float):
                conv_id = event.get("conv_id")
                open_proposals.push(
                    target,
                    float(confidence),
                    conv_id if isinstance(conv_id, str) else None,
                )
        elif kind == "label":
            label = event.get("label")
            if label not in (LABEL_ACCEPT, LABEL_REJECT):
                continue
            confidence = open_proposals.pop(target)
            if confidence is None:
                # Orphan label (e.g. a voluntary session_switch with no
                # preceding open proposal) — not calibration evidence.
                # 무연고 label (열린 제안 없이 자발 호출된 session_switch
                # 등) — 보정 증거가 아니다.
                continue
            pairs.append((confidence, label == LABEL_ACCEPT))
    return pairs, open_proposals


class _OpenProposals:
    """Per-target stacks of open proposals, most recent last.

    target 별 열린 proposal 스택 — 최근 것이 뒤. 항목은 (confidence,
    conv_id); conv_id 가 None 인 옛 줄은 만료 대상이 아니다.
    """

    def __init__(self) -> None:
        self._stacks: dict[str, list[tuple[float, str | None]]] = {}

    def push(self, target: str, confidence: float, conv_id: str | None) -> None:
        self._stacks.setdefault(target, []).append((confidence, conv_id))

    def pop(self, target: str) -> float | None:
        stack = self._stacks.get(target)
        if not stack:
            return None
        return stack.pop()[0]

    def expire(self, conv_id: str) -> None:
        for target, stack in self._stacks.items():
            self._stacks[target] = [(c, cid) for c, cid in stack if cid != conv_id]

    def has_conv(self, conv_id: str) -> bool:
        return any(
            cid == conv_id for stack in self._stacks.values() for _, cid in stack
        )


def _has_open_proposal(events: list[dict[str, Any]], conv_id: str) -> bool:
    return _replay(events)[1].has_conv(conv_id)


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
