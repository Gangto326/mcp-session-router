"""Wrapper runtime state persisted to ``.session-manager/state.json``.

``.session-manager/state.json`` 에 영속화되는 래퍼 런타임 상태.

Introduced by R3-C3 with a single key (``last_transition``) so ``/back``
survives a wrapper restart; R5 extends the same file with statusline
state. Reads are fully defensive — a missing or corrupt file is simply
empty state, never an error, because losing an undo record must not
break the wrapper.

R3-C3 이 단일 키(``last_transition``)로 도입 — ``/back`` 이 래퍼 재시작을
견디게 한다. R5 가 같은 파일을 statusline 상태로 확장한다. 읽기는 전부
방어적이다 — 파일 없음·손상은 빈 상태일 뿐 오류가 아니다. undo 기록
유실이 래퍼를 깨뜨려서는 안 되기 때문이다.

Writes are atomic (tmp + replace) but not inter-process locked: the
record is per-wrapper UX state, and the worst concurrent outcome is one
instance's undo record replacing another's — an acceptable loss, unlike
session metadata (F15).

쓰기는 원자적 (tmp + replace) 이지만 프로세스 간 잠금은 없다: 이 기록은
래퍼별 UX 상태이고, 동시 실행의 최악 결과는 한 인스턴스의 undo 기록이
다른 것을 덮는 것뿐이다 — 세션 메타데이터 (F15) 와 달리 수용 가능한
손실이다.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from session_manager import debug_log

_SESSION_MANAGER_DIRNAME = ".session-manager"
STATE_FILENAME = "state.json"

LAST_TRANSITION_KEY = "last_transition"
# Fields of one last-transition record. ``from``/``to`` are session
# names, ``user_prompt`` is the prompt that travelled with the switch
# (re-injected by /back), ``at`` is ISO8601.
# last_transition 레코드의 필드. ``from``/``to`` 는 세션 이름,
# ``user_prompt`` 는 전환과 함께 이동한 프롬프트 (/back 이 재주입),
# ``at`` 은 ISO8601.
_REQUIRED_FIELDS = ("from", "to")


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _state_path(project_path: Path) -> Path:
    return Path(project_path) / _SESSION_MANAGER_DIRNAME / STATE_FILENAME


def _load_state(project_path: Path) -> dict[str, Any]:
    path = _state_path(project_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(project_path: Path, state: dict[str, Any]) -> None:
    path = _state_path(project_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(path)
    except OSError as exc:
        # State persistence is best-effort — /back then only works
        # within the current wrapper lifetime.
        # 상태 영속화는 best-effort — 실패 시 /back 은 현재 래퍼 수명
        # 안에서만 동작한다.
        debug_log.log(
            "WRAPPER_STATE",
            "WRAPPER",
            {"op": "save", "result": "error", "error": str(exc)},
        )


def save_last_transition(project_path: Path, record: dict[str, Any]) -> None:
    """Persist the most recent wrapper-executed transition.

    가장 최근의 래퍼 실행 전환을 영속화한다.
    """
    state = _load_state(project_path)
    state[LAST_TRANSITION_KEY] = record
    _save_state(project_path, state)
    debug_log.log(
        "WRAPPER_STATE",
        "WRAPPER",
        {"op": "save_last_transition", "record": record},
    )


def load_last_transition(project_path: Path) -> dict[str, Any] | None:
    """Return the persisted last transition, or None if absent/unusable.

    영속화된 직전 전환을 반환. 없거나 사용 불가면 None.
    """
    record = _load_state(project_path).get(LAST_TRANSITION_KEY)
    if not isinstance(record, dict):
        return None
    if any(not isinstance(record.get(f), str) or not record[f] for f in _REQUIRED_FIELDS):
        return None
    return record


def clear_last_transition(project_path: Path) -> None:
    """Consume the persisted last transition (one-shot undo).

    영속화된 직전 전환을 소비한다 (1회용 undo).
    """
    state = _load_state(project_path)
    if LAST_TRANSITION_KEY not in state:
        return
    del state[LAST_TRANSITION_KEY]
    _save_state(project_path, state)
    debug_log.log(
        "WRAPPER_STATE", "WRAPPER", {"op": "clear_last_transition"}
    )
