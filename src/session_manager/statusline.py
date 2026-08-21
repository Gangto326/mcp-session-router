"""Statusline: records context-window facts and renders the router status.

statusline — 컨텍스트 창 사실을 받아 적고 (R4-C1) 라우터 상태줄을
그린다 (R5-C1).

Display (R5-C1): the line printed to stdout becomes Claude Code's bottom
status bar — ``⎇ {session} · {mode} · ctx {pct}% · {n} sessions``. Each
segment is sourced where it is freshest at render time and dropped when
unavailable: the session name from ``state.json`` (the wrapper's
mirror — the only value the wrapper alone knows), the routing mode from
``config.json`` (MCP ``set_routing_mode`` changes it without the wrapper
knowing), the active-session count from ``sessions/`` (same rule as the
UserPromptSubmit hook: missing ``status`` counts as active, corrupt
files are skipped), and the context percentage from this invocation's
stdin with ``context.json`` as a fallback for older Claude Code
payloads. This module only *reads* ``state.json`` — see below for why it
must not write there.

표시 (R5-C1): stdout 에 찍는 한 줄이 Claude Code 하단 상태줄이 된다 —
``⎇ {session} · {mode} · ctx {pct}% · {n} sessions``. 세그먼트마다
표시 시점에 가장 신선한 곳에서 읽고, 없으면 그 조각만 뺀다: 세션
이름은 ``state.json`` (래퍼 미러 — 래퍼만 아는 유일한 값), 라우팅
모드는 ``config.json`` (MCP ``set_routing_mode`` 가 래퍼 모르게 바꾼다),
활성 세션 수는 ``sessions/`` (UserPromptSubmit hook 과 같은 규칙 —
``status`` 부재는 active, 손상 파일은 skip), 컨텍스트 퍼센트는 이번
호출의 stdin, 구버전 payload 면 ``context.json`` 폴백. 이 모듈은
``state.json`` 을 *읽기만* 한다 — 쓰면 안 되는 이유는 아래.

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

from session_manager.models.config import DEFAULT_ROUTING_MODE
from session_manager.storage.file_store import _CONFIG_FILENAME, _SESSIONS_DIRNAME
from session_manager.wrapper import wrapper_state

_SOCKET_ENV_VAR = "SESSION_MANAGER_SOCKET"
_SESSION_MANAGER_DIRNAME = ".session-manager"
CONTEXT_FILENAME = "context.json"

# Status-line glyphs. ``⎇`` marks the session segment (branch-like —
# "which line of work am I on"); ``·`` separates segments.
# 상태줄 기호. ``⎇`` 는 세션 세그먼트 표시 (브랜치 느낌 — "지금 어느
# 작업 줄기에 있나"), ``·`` 는 세그먼트 구분자.
SESSION_GLYPH = "⎇"
SEGMENT_SEPARATOR = " · "

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


def _load_routing_mode(project_path: Path) -> str:
    """Read ``routing_mode`` from config.json; absent/corrupt → default.

    config.json 에서 ``routing_mode`` 를 읽는다. 부재·손상이면 기본값.
    Same rule as the UserPromptSubmit hook, re-implemented here so the
    statusline (spawned several times per turn) does not import the
    hook's judge/routing modules.
    UserPromptSubmit hook 과 같은 규칙 — statusline 은 턴마다 수차례
    뜨므로 hook 의 judge/routing 모듈을 끌어오지 않으려고 따로 둔다.
    """
    path = project_path / _SESSION_MANAGER_DIRNAME / _CONFIG_FILENAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULT_ROUTING_MODE
    if not isinstance(data, dict):
        return DEFAULT_ROUTING_MODE
    mode = data.get("routing_mode")
    return mode if isinstance(mode, str) and mode else DEFAULT_ROUTING_MODE


def _count_active_sessions(project_path: Path) -> int | None:
    """Count active sessions; None when the project has no sessions dir.

    활성 세션 수를 센다. sessions 디렉토리가 없으면 None (세그먼트 생략).
    Missing ``status`` counts as active (pre-status files); any other
    value (archived/expired/retired) is inactive; corrupt files skip.
    ``status`` 부재는 active (status 도입 전 파일), 그 외 값 (archived
    /expired/retired) 은 비활성, 손상 파일은 skip.
    """
    sessions_dir = project_path / _SESSION_MANAGER_DIRNAME / _SESSIONS_DIRNAME
    if not sessions_dir.is_dir():
        return None
    count = 0
    for path in sessions_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("status", "active") == "active":
            count += 1
    return count


def build_status_line(
    session: str | None,
    mode: str | None,
    ctx_pct: int | float | None,
    session_count: int | None,
) -> str | None:
    """Compose ``⎇ {session} · {mode} · ctx {pct}% · {n} sessions``.

    상태줄 한 줄을 조합한다. 없는 조각은 빼고 나머지로 만들며, 조각이
    하나도 없으면 None (무출력).
    """
    segments: list[str] = []
    if session:
        segments.append(f"{SESSION_GLYPH} {session}")
    if mode:
        segments.append(mode)
    if isinstance(ctx_pct, int | float) and not isinstance(ctx_pct, bool):
        segments.append(f"ctx {round(ctx_pct)}%")
    if isinstance(session_count, int) and not isinstance(session_count, bool):
        noun = "session" if session_count == 1 else "sessions"
        segments.append(f"{session_count} {noun}")
    return SEGMENT_SEPARATOR.join(segments) if segments else None


def render(project_path: Path, record: dict[str, Any] | None) -> str | None:
    """Gather every segment's source and build the line.

    세그먼트별 원천을 모아 상태줄을 만든다. ``record`` 는 이번 stdin 의
    컨텍스트 레코드 (구버전 payload 면 None → context.json 폴백).
    """
    if record is None:
        record = read_context(project_path)
    pct = record.get("used_percentage") if record else None
    return build_status_line(
        session=wrapper_state.load_current_session(project_path),
        mode=_load_routing_mode(project_path),
        ctx_pct=pct,
        session_count=_count_active_sessions(project_path),
    )


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
        line = render(project_path, record)
        if line:
            print(line)
    except Exception:
        # A broken statusline must never disturb the terminal.
        # statusline 고장이 터미널을 어지럽혀선 안 된다.
        return


if __name__ == "__main__":
    main()
