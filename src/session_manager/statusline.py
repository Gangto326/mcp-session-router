"""Statusline collector: records context-window facts Claude Code feeds it.

statusline 수집기 — Claude Code 가 statusline 명령에 떠먹여 주는
컨텍스트 창 사실을 받아 적는다.

Claude Code invokes the configured statusline command on every status
change and passes a JSON object on stdin that includes — measured,
docs/poc/R4-rollover.md §P4-a' — ``context_window.context_window_size``
(the real window size, e.g. 1M for this environment's Sonnet 5, which no
model-name mapping could know), ``total_input_tokens``,
``used_percentage`` and the conversation id. This script persists those
facts to ``.session-manager/context.json`` so the wrapper's rollover
detection (R4-C1) reads the *actual* denominator instead of guessing
from a model table.

Claude Code 는 상태가 바뀔 때마다 등록된 statusline 명령을 실행하며
stdin 으로 JSON 을 준다. 실측 (docs/poc/R4-rollover.md §P4-a') 으로 이
JSON 에는 ``context_window.context_window_size`` (실창 크기 — 이 환경의
Sonnet 5 는 1M 로, 모델명 매핑으로는 알 수 없는 값),
``total_input_tokens``, ``used_percentage``, 대화 id 가 들어 있음이
확인됐다. 이 스크립트는 그 사실들을 ``.session-manager/context.json``
에 영속화해, 래퍼의 롤오버 감지 (R4-C1) 가 분모를 모델 표 추측이 아닌
*실제 값* 으로 읽게 한다.

Design constraints / 설계 제약:

- Outside a ccode wrapper (no ``SESSION_MANAGER_SOCKET`` in env) this
  script writes NOTHING and prints nothing — same philosophy as the F4
  fix: bare ``claude`` sessions must never grow ``.session-manager/``
  side effects.
  ccode 래퍼 밖 (env 에 ``SESSION_MANAGER_SOCKET`` 없음) 에서는 아무
  것도 쓰지 않고 아무것도 출력하지 않는다 — F4 수정과 같은 철학: 맨몸
  ``claude`` 세션이 ``.session-manager/`` 부작용을 만들면 안 된다.
- Writes go to a dedicated file (not ``state.json``): the statusline
  process fires several times per turn in its own process, and
  ``state.json`` writes are atomic but not inter-process locked — a
  shared file could drop the wrapper's ``last_transition`` (/back) in a
  read-modify-write race. A separate file removes the race entirely.
  기록은 전용 파일 (``state.json`` 아님) 에 한다: statusline 은 턴마다
  수차례 별도 프로세스로 뜨고 ``state.json`` 쓰기는 원자적이지만
  프로세스 간 잠금이 없다 — 파일을 공유하면 read-modify-write race 로
  래퍼의 ``last_transition`` (/back) 이 유실될 수 있다. 파일 분리로
  race 자체를 없앤다.
- Any failure exits 0 silently — a broken statusline must never disturb
  the user's terminal.
  어떤 실패도 조용히 exit 0 — statusline 고장이 사용자 터미널을
  어지럽혀선 안 된다.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SOCKET_ENV_VAR = "SESSION_MANAGER_SOCKET"
_SESSION_MANAGER_DIRNAME = ".session-manager"
CONTEXT_FILENAME = "context.json"

# Numerator fields inside context_window.current_usage that make up the
# context footprint (measured: their sum equals /context's display and
# total_input_tokens; output_tokens is NOT part of the footprint).
# context_window.current_usage 중 컨텍스트 점유를 구성하는 분자 필드
# (실측: 이 합이 /context 표시·total_input_tokens 와 일치, output_tokens
# 는 점유에 불포함).
_USAGE_FIELDS = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _context_path(project_path: Path) -> Path:
    return project_path / _SESSION_MANAGER_DIRNAME / CONTEXT_FILENAME


def _extract_record(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Build the context.json record from a statusline stdin payload.

    statusline stdin payload 에서 context.json 레코드를 만든다.
    Returns None when the payload lacks the context_window block (older
    Claude Code versions) — nothing useful to persist.
    context_window 블록이 없으면 (구버전 Claude Code) None — 기록할
    것이 없다.
    """
    window = payload.get("context_window")
    if not isinstance(window, dict):
        return None
    size = window.get("context_window_size")
    if not isinstance(size, int) or size <= 0:
        return None

    used = window.get("total_input_tokens")
    if not isinstance(used, int):
        usage = window.get("current_usage")
        if isinstance(usage, dict):
            used = sum(
                v
                for f in _USAGE_FIELDS
                if isinstance((v := usage.get(f)), int)
            )
        else:
            used = None

    model = payload.get("model")
    model_id = model.get("id") if isinstance(model, dict) else None
    return {
        "context_window_size": size,
        "used_tokens": used,
        "used_percentage": window.get("used_percentage"),
        "conversation_id": payload.get("session_id"),
        "transcript_path": payload.get("transcript_path"),
        "model_id": model_id,
        "at": datetime.now(UTC).isoformat(),
    }


def write_context(project_path: Path, record: dict[str, Any]) -> None:
    """Atomically persist the record (tmp + replace, own file — no race).

    레코드를 원자적으로 영속화한다 (tmp + replace, 전용 파일 — race 없음).
    """
    path = _context_path(project_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(path)


def read_context(project_path: Path) -> dict[str, Any] | None:
    """Defensive read: missing/corrupt file is None, never an error.

    방어적 읽기 — 파일 없음·손상은 None 일 뿐 오류가 아니다.
    """
    try:
        data = json.loads(_context_path(project_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def main() -> None:
    try:
        # Outside ccode: no side effects, no output (F4 philosophy).
        # ccode 밖: 부작용도 출력도 없음 (F4 철학).
        if not os.environ.get(_SOCKET_ENV_VAR, "").strip():
            return
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            return
        project_path = Path(payload.get("cwd") or os.getcwd())
        record = _extract_record(payload)
        if record is not None:
            write_context(project_path, record)
            pct = record.get("used_percentage")
            model_id = record.get("model_id") or "?"
            if isinstance(pct, int | float):
                # Minimal display — the pretty format is R5-C1's job.
                # 최소 표시 — 예쁜 형식은 R5-C1 몫.
                print(f"{model_id} · ctx {pct}%")
    except Exception:
        # A broken statusline must never disturb the terminal.
        # statusline 고장이 터미널을 어지럽혀선 안 된다.
        return


if __name__ == "__main__":
    main()
