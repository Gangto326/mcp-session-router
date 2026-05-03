"""
Debug logging infrastructure shared by the wrapper and the MCP server.

Both processes (the ccode wrapper and the MCP server spawned by Claude Code)
append events to a single per-run NDJSON file so a full session can be
replayed offline.  Activated via ``SESSION_MANAGER_DEBUG=1``; otherwise
every ``log()`` call short-circuits with negligible overhead.

Wrapper와 MCP 서버가 공용으로 쓰는 디버그 로깅 인프라.

ccode wrapper와 Claude Code가 spawn하는 MCP 서버 두 프로세스가 같은
NDJSON 파일에 이벤트를 append 해, 한 ccode 세션 전체를 사후 재현·분석할
수 있게 한다. ``SESSION_MANAGER_DEBUG=1`` 일 때만 실제 기록되며, 비활성
상태에서는 ``log()`` 호출이 거의 비용 없는 no-op.

Environment variables
---------------------
- ``SESSION_MANAGER_DEBUG``       — set to a truthy value to enable.
- ``SESSION_MANAGER_RUN_ID``      — auto-populated; child processes inherit
  the wrapper's run id so all events land in one file.
- ``SESSION_MANAGER_LOG_DIR``     — override output dir
  (default: ``~/.session-manager/logs``).
- ``SESSION_MANAGER_LOG_RAW_STDIN`` — opt-in to raw stdin chunk logging
  (default: redacted with length + sha256 prefix only).

환경 변수
---------
- ``SESSION_MANAGER_DEBUG``       — truthy 값이면 활성화.
- ``SESSION_MANAGER_RUN_ID``      — 자동 채워짐. 자식 프로세스가 wrapper의
  run id를 상속하므로 모든 이벤트가 한 파일에 모인다.
- ``SESSION_MANAGER_LOG_DIR``     — 출력 디렉토리 override
  (기본 ``~/.session-manager/logs``).
- ``SESSION_MANAGER_LOG_RAW_STDIN`` — stdin chunk 평문 로깅 opt-in
  (기본은 길이 + sha256 prefix 만 redacted 기록).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---- Environment variable names ----
# 환경 변수 이름.

_DEBUG_ENV = "SESSION_MANAGER_DEBUG"
_RUN_ID_ENV = "SESSION_MANAGER_RUN_ID"
_LOG_DIR_ENV = "SESSION_MANAGER_LOG_DIR"
_RAW_STDIN_ENV = "SESSION_MANAGER_LOG_RAW_STDIN"

# Default log dir under the user's HOME so multiple projects share one
# location and the user can find logs without remembering project paths.
#
# 사용자 HOME 아래의 기본 로그 디렉토리. 프로젝트마다 흩어지지 않고 한
# 곳에 모이도록 하여, 사용자가 프로젝트 경로를 기억하지 않아도 로그를
# 찾을 수 있게 한다.
_DEFAULT_LOG_DIR = Path.home() / ".session-manager" / "logs"

# Mask any env var whose name matches these substrings, regardless of
# whitelist membership. Intentionally broad to catch typos like APIKEY,
# AUTHTOKEN, etc.
#
# 이름에 이 패턴이 들어가는 모든 환경 변수는 화이트리스트 여부와 관계없이
# 마스킹. APIKEY, AUTHTOKEN 같은 변형까지 잡도록 의도적으로 넓게 잡는다.
_SENSITIVE_ENV_PATTERN = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)

# Whitelist of env vars that are safe to dump verbatim. Anything outside
# this set is dropped from env captures (or masked if it matches the
# sensitive pattern above) — minimises accidental leakage.
#
# 평문 dump해도 안전한 환경 변수 화이트리스트. 이 집합에 없는 변수는
# env capture 시 제외 (위 sensitive 패턴에 걸리면 마스킹). 우발 유출 최소화.
_ENV_WHITELIST = frozenset(
    [
        "PATH",
        "HOME",
        "PWD",
        "TERM",
        "LANG",
        "USER",
        "SHELL",
        "SESSION_MANAGER_SOCKET",
        "SESSION_MANAGER_DEBUG",
        "SESSION_MANAGER_RUN_ID",
        "SESSION_MANAGER_LOG_DIR",
        "SESSION_MANAGER_LOG_RAW_STDIN",
    ]
)

# Threshold above which a payload field gets spilled to a side file
# instead of being inlined in the NDJSON record. Keeps each log line
# small enough for line-oriented tools (``jq``, ``grep``).
#
# payload 필드가 이 크기를 넘으면 NDJSON 레코드에 inline 하지 않고 별도
# spill 파일로 분리. 라인 단위 도구 (``jq``, ``grep``) 가 다루기 쉬운
# 크기를 유지한다.
SPILL_THRESHOLD_CHARS = 4_000


# ---- Process metadata ----
# 프로세스 메타데이터.

_proc_label: str = "wrapper"


def set_proc_label(label: str) -> None:
    """Tag this process so log records show ``wrapper`` vs ``mcp``.

    이 프로세스를 태깅한다. 로그 레코드가 ``wrapper`` 인지 ``mcp`` 인지
    구분되도록 한다.
    """
    global _proc_label
    _proc_label = label


# ---- Activation helpers ----
# 활성화 헬퍼.


def is_enabled() -> bool:
    """Return True iff debug logging is enabled via the env var.

    환경 변수로 디버그 로깅이 활성화되어 있는 경우에만 True 반환.
    """
    raw = os.environ.get(_DEBUG_ENV, "").strip().lower()
    return raw not in ("", "0", "false", "no", "off")


def get_run_id() -> str:
    """Return the wrapper's run id, exporting one to env if absent.

    Generated once per ccode session by the first caller (the wrapper),
    then inherited by Claude Code and the MCP server through the env.

    wrapper의 run id 반환. 없으면 새로 생성해 환경 변수에 export.

    ccode 세션마다 한 번 (가장 먼저 호출한 wrapper 가) 생성되고, 이후
    Claude Code 와 MCP 서버가 환경 변수를 상속해 같은 id 를 사용한다.
    """
    existing = os.environ.get(_RUN_ID_ENV, "").strip()
    if existing:
        return existing
    new_id = uuid.uuid4().hex[:16]
    os.environ[_RUN_ID_ENV] = new_id
    return new_id


def get_log_dir() -> Path:
    """Resolve and create (if missing) the log output directory.

    로그 출력 디렉토리 결정. 없으면 생성.
    """
    override = os.environ.get(_LOG_DIR_ENV, "").strip()
    log_dir = Path(override) if override else _DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_log_path() -> Path:
    """Path of the per-run NDJSON log file.

    이번 run 의 NDJSON 로그 파일 경로.
    """
    return get_log_dir() / f"{get_run_id()}.ndjson"


# ---- Spill counter for large payload bodies ----
# 큰 payload 본문 spill 카운터.

_spill_seq = 0


def _next_spill_path() -> Path:
    global _spill_seq
    _spill_seq += 1
    return get_log_dir() / f"{get_run_id()}.{_spill_seq:04d}.dump"


def spill(text: str) -> str:
    """Write a large payload body to a side file, return its filename.

    큰 payload 본문을 별도 파일에 쓰고 파일명 (basename) 만 반환.
    """
    path = _next_spill_path()
    try:
        path.write_text(text, encoding="utf-8")
    except Exception:
        # Spill must never break the main flow — return a marker filename
        # if disk write fails (e.g. quota exhausted).
        #
        # spill 실패가 본 흐름을 깨뜨리지 않도록 — 디스크 쓰기 실패 시
        # (예: quota 초과) marker 파일명만 반환.
        return f"<spill-failed seq={_spill_seq}>"
    return path.name


# ---- Masking / summarisation helpers ----
# 마스킹·요약 헬퍼.


def mask_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Filter env vars: keep whitelist as-is, mask sensitive matches, drop rest.

    환경 변수 필터: 화이트리스트는 그대로, 민감 패턴은 마스킹, 나머지는
    드랍한다.
    """
    src = env if env is not None else os.environ
    out: dict[str, str] = {}
    for k, v in src.items():
        if k in _ENV_WHITELIST:
            out[k] = v
        elif _SENSITIVE_ENV_PATTERN.search(k):
            out[k] = f"<masked len={len(v)}>"
    return out


def mask_text(text: str | None, prefix_len: int = 200) -> dict[str, Any]:
    """Summarise free-form text as length + first ``prefix_len`` chars.

    Bodies above ``SPILL_THRESHOLD_CHARS`` get spilled to a side file and
    the record carries only a reference filename — keeps NDJSON lines
    small enough for ``jq`` / ``grep`` to handle.

    자유 형식 텍스트를 길이 + 처음 ``prefix_len`` 글자로 요약한다.

    ``SPILL_THRESHOLD_CHARS`` 보다 큰 본문은 별도 파일로 spill 하고
    레코드에는 파일명 reference 만 남긴다 — NDJSON 라인이 ``jq`` /
    ``grep`` 으로 다룰 만한 크기를 유지하기 위함.
    """
    if text is None:
        return {"len": 0, "preview": None}
    info: dict[str, Any] = {"len": len(text), "preview": text[:prefix_len]}
    if len(text) > SPILL_THRESHOLD_CHARS:
        info["text_ref"] = spill(text)
    return info


def mask_stdin_chunk(chunk: bytes) -> dict[str, Any]:
    """Summarise a stdin chunk: redacted by default, raw with explicit opt-in.

    Default fields are length + 8-byte head/tail hex + sha256 prefix —
    enough to fingerprint a chunk without leaking content. When the user
    sets ``SESSION_MANAGER_LOG_RAW_STDIN=1`` the raw decoded text is
    included as well.

    stdin chunk 요약: 기본은 redacted, 명시 opt-in 시 raw 포함.

    기본 필드는 길이 + 앞·뒤 8바이트 hex + sha256 prefix — 내용 누출 없이
    chunk 를 식별 가능한 정도. ``SESSION_MANAGER_LOG_RAW_STDIN=1`` 이면
    decoded 평문도 함께 기록.
    """
    info: dict[str, Any] = {
        "len": len(chunk),
        "head_hex": chunk[:8].hex(),
        "tail_hex": chunk[-8:].hex() if len(chunk) > 8 else "",
        "sha256_prefix": hashlib.sha256(chunk).hexdigest()[:16],
    }
    if os.environ.get(_RAW_STDIN_ENV, "").strip().lower() in ("1", "true", "yes"):
        try:
            info["raw"] = chunk.decode("utf-8", errors="replace")
        except Exception:
            info["raw"] = repr(chunk)
    return info


def mask_dict_keys_only(d: dict[str, Any] | None) -> dict[str, Any]:
    """Return only key names + value lengths (use for ``variables`` style fields).

    key 이름과 값 길이만 반환 (``variables`` 같은 임의 사용자 입력 dict 용).
    """
    if not d:
        return {}
    return {k: {"len": len(str(v))} for k, v in d.items()}


# ---- Correlation id for multi-step events (tool call ↔ return) ----
# 여러 단계 이벤트 (tool call ↔ return) 묶기용 correlation id.


def new_event_id() -> str:
    """Short id used to tie a tool call event to its return event.

    도구 호출 이벤트와 그 반환 이벤트를 묶는 데 쓰는 짧은 id.
    """
    return uuid.uuid4().hex[:8]


# ---- Core log function ----
# 핵심 로그 함수.


def log(
    category: str,
    origin: str = "SYSTEM",
    payload: dict[str, Any] | None = None,
    *,
    conv_id: str | None = None,
    session: str | None = None,
) -> None:
    """Append one event to the NDJSON log if debug is enabled.

    ``category`` enumerates the source point (e.g. ``USER_KEY``,
    ``WRAPPER_INJECT``, ``MCP_TOOL_CALL``). ``origin`` is one of
    ``USER`` / ``WRAPPER`` / ``LLM`` / ``MCP_TOOL`` / ``SYSTEM`` and
    answers "who created this data?" — the most important field for
    debugging "who put that text in my context".

    디버그가 활성화되어 있으면 NDJSON 로그에 한 이벤트를 append.

    ``category`` 는 발생 지점 (``USER_KEY``, ``WRAPPER_INJECT``,
    ``MCP_TOOL_CALL`` 등) 을 enum 한다. ``origin`` 은 ``USER`` /
    ``WRAPPER`` / ``LLM`` / ``MCP_TOOL`` / ``SYSTEM`` 중 하나로 "이
    데이터를 만든 주체가 누구인가" 를 표시한다 — "왜 내 컨텍스트에 저
    텍스트가 들어왔지?" 같은 디버깅에서 가장 중요한 필드.
    """
    if not is_enabled():
        return
    record: dict[str, Any] = {
        "ts": datetime.now(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "mono_ns": time.monotonic_ns(),
        "run_id": get_run_id(),
        "proc": _proc_label,
        "pid": os.getpid(),
        "category": category,
        "origin": origin,
        "conv_id": conv_id,
        "session": session,
        "payload": payload or {},
    }
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
    except Exception as exc:
        # Fallback record so a non-serialisable payload doesn't
        # silently drop the event.
        #
        # 직렬화 실패 시 fallback 레코드 — 이벤트가 조용히 사라지지
        # 않도록 한다.
        line = json.dumps(
            {
                "ts": record["ts"],
                "mono_ns": record["mono_ns"],
                "run_id": record["run_id"],
                "proc": record["proc"],
                "pid": record["pid"],
                "category": "LOG_ERROR",
                "origin": "SYSTEM",
                "payload": {
                    "original_category": category,
                    "error": str(exc),
                },
            }
        )
    try:
        with get_log_path().open("a", encoding="utf-8") as fp:
            fp.write(line + "\n")
    except Exception:
        # Logging must never break the main flow. If disk is full or
        # the path is unwritable we silently drop — the rest of the
        # pipeline continues unaffected.
        #
        # 로깅 실패가 본 흐름을 깨뜨리지 않도록 — 디스크가 가득 차거나
        # 경로에 쓰기 불가능하면 조용히 드랍하고 나머지 파이프라인은
        # 영향을 받지 않는다.
        pass
