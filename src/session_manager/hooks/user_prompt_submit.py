"""
UserPromptSubmit routing hook entrypoint.

Claude Code runs this script every time the user submits a prompt,
before the prompt reaches the main LLM — this is what guarantees a 100%
routing trigger rate. This module implements the deterministic prefilter
stage: cases where no LLM judgment is needed exit immediately with code
0 so the prompt passes through untouched. The judgment stage (resident
judge reached over the wrapper socket) plugs into ``_route()`` in a
later commit.

Design rule (graceful degradation): any failure inside the hook must
never block the user's conversation — every exception path exits 0.

UserPromptSubmit 라우팅 hook 진입점.

Claude Code는 사용자가 프롬프트를 제출할 때마다, 프롬프트가 메인 LLM에
도달하기 전에 이 스크립트를 실행한다 — 라우팅 발동률 100%를 보장하는
장치다. 이 모듈은 결정적 프리필터 단계를 구현한다: LLM 판정이 필요
없는 경우는 즉시 exit 0으로 프롬프트를 그대로 통과시킨다. 판정 단계
(래퍼 소켓 너머의 상주 판정기)는 후속 커밋에서 ``_route()``에 연결된다.

설계 규칙 (graceful degradation): hook 내부의 어떤 실패도 사용자의
대화를 막아서는 안 된다 — 모든 예외 경로는 exit 0이다.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

from session_manager import debug_log
from session_manager.routing.judge import HOOK_REPLY_TIMEOUT_SECS
from session_manager.storage.file_store import (
    _CONFIG_FILENAME,
    _SESSION_MANAGER_DIRNAME,
    _SESSIONS_DIRNAME,
)

# The wrapper exports its socket path to Claude Code's env; hook
# processes are Claude Code's children and inherit it.
# 래퍼가 소켓 경로를 Claude Code env 로 export 하고, hook 프로세스는
# Claude Code 의 자식이므로 이를 상속한다.
_SOCKET_ENV_VAR = "SESSION_MANAGER_SOCKET"

# Stdin field names, measured from the real UserPromptSubmit payload
# (docs/poc/R2-hook.md §1 — Claude Code 2.1.220, headless and
# interactive verified identical).
# stdin 필드명 상수 — 실제 UserPromptSubmit 페이로드 실측값
# (docs/poc/R2-hook.md §1 — headless·대화형 동일 확인).
FIELD_PROMPT = "prompt"
FIELD_SESSION_ID = "session_id"
FIELD_TRANSCRIPT_PATH = "transcript_path"
FIELD_CWD = "cwd"
FIELD_PERMISSION_MODE = "permission_mode"

# Default routing mode when config.json is absent or has no routing_mode
# key. "confirm" is the Plan §1.4 default: propose, never auto-switch.
# config.json 부재·routing_mode 키 부재 시의 기본 모드. "confirm"은
# Plan §1.4의 기본값 — 제안만 하고 자동 전환하지 않는다.
DEFAULT_ROUTING_MODE = "confirm"


def _load_routing_mode(root: Path) -> str:
    """
    Read ``routing_mode`` from config.json defensively.

    The routing keys enter the Config model in a later commit; until
    then the hook reads the raw JSON so a missing file or key simply
    falls back to the default.

    config.json에서 ``routing_mode``를 방어적으로 읽는다.

    라우팅 키의 Config 모델 정식 편입은 후속 커밋이다. 그 전까지 hook은
    raw JSON을 읽어, 파일·키 부재 시 기본값으로 폴백한다.
    """
    try:
        data = json.loads((root / _CONFIG_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULT_ROUTING_MODE
    if not isinstance(data, dict):
        return DEFAULT_ROUTING_MODE
    mode = data.get("routing_mode")
    return mode if isinstance(mode, str) else DEFAULT_ROUTING_MODE


def _count_active_sessions(root: Path) -> int:
    """
    Count sessions whose status is active.

    The ``status`` field does not exist yet (a later phase adds it) —
    absence counts as active, which keeps this forward- and
    backward-compatible. Corrupt session files are skipped so one bad
    file cannot distort the count.

    status가 active인 세션 수를 센다.

    ``status`` 필드는 아직 없다 (후속 Phase에서 추가) — 부재는 active로
    간주해 상·하위 호환을 유지한다. 손상된 세션 파일은 건너뛰어 파일
    하나가 집계를 왜곡하지 못하게 한다.
    """
    sessions_dir = root / _SESSIONS_DIRNAME
    if not sessions_dir.is_dir():
        return 0
    count = 0
    for path in sessions_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("status", "active") == "active":
            count += 1
    return count


def _prefilter(payload: dict[str, Any]) -> str | None:
    """
    Apply the deterministic prefilter rules.

    Returns the name of the rule that lets the prompt pass through
    without judgment, or None when the judgment stage is needed.
    Deterministic rules only — no heuristics (Plan §3 R2-C2: the
    80-char rule was rejected for lacking evidence).

    결정적 프리필터 규칙을 적용한다.

    판정 없이 프롬프트를 통과시키는 규칙명을 반환하고, 판정 단계가
    필요하면 None을 반환한다. 결정적 규칙만 사용한다 — 휴리스틱 금지
    (Plan §3 R2-C2: 80자 규칙은 근거 부재로 기각됨).
    """
    cwd = payload.get(FIELD_CWD)
    if not isinstance(cwd, str) or not cwd:
        return "no_cwd"
    root = Path(cwd) / _SESSION_MANAGER_DIRNAME
    if not root.is_dir():
        return "no_session_manager_dir"
    if _count_active_sessions(root) <= 1:
        return "single_active_session"
    if payload.get(FIELD_PERMISSION_MODE) == "plan":
        return "plan_mode"
    if _load_routing_mode(root) == "off":
        return "routing_off"
    return None


def _request_judgment(payload: dict[str, Any]) -> dict[str, Any] | None:
    """
    One short-lived socket round-trip to the wrapper's judge host.

    Sends a thin judge_request (prompt assembly happens wrapper-side)
    and waits for the deferred reply. Returns the reply dict, or None
    on any failure — no socket, refused connection, timeout, bad frame.

    래퍼 판정 호스트로의 단발 소켓 왕복 1회.

    얇은 judge_request 를 보내고 (프롬프트 조립은 래퍼 측 담당) 지연
    회신을 기다린다. 실패 시 None — 소켓 부재·연결 거부·타임아웃·깨진
    프레임 전부.
    """
    socket_path = os.environ.get(_SOCKET_ENV_VAR, "").strip()
    if not socket_path:
        return None
    request = {
        "client": "hook",
        "action": "judge_request",
        "prompt": payload.get(FIELD_PROMPT),
        "session_id": payload.get(FIELD_SESSION_ID),
        "transcript_path": payload.get(FIELD_TRANSCRIPT_PATH),
        "cwd": payload.get(FIELD_CWD),
    }
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(HOOK_REPLY_TIMEOUT_SECS)
            sock.connect(socket_path)
            sock.sendall(
                (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
            )
            buffer = b""
            while b"\n" not in buffer:
                chunk = sock.recv(4096)
                if not chunk:
                    return None
                buffer += chunk
    except OSError:
        return None
    line = buffer.split(b"\n", 1)[0]
    try:
        reply = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return reply if isinstance(reply, dict) else None


def _route(payload: dict[str, Any]) -> None:
    """
    Judgment stage: ask the wrapper's resident judge over the socket.

    The verdict is only recorded for now — acting on it (block and
    switch) is the next commit. Every outcome, including failure, exits
    through the caller with code 0.

    판정 단계: 소켓 너머 래퍼의 상주 판정기에 묻는다.

    현재는 판정 결과를 기록만 한다 — 결과에 따른 실행(block·전환)은
    다음 커밋이다. 실패를 포함한 모든 결과가 호출자에서 exit 0 으로
    끝난다.
    """
    reply = _request_judgment(payload)
    debug_log.log(
        "HOOK_ROUTE",
        "SYSTEM",
        {"reply": reply},
        conv_id=payload.get(FIELD_SESSION_ID),
    )


def run(stdin_text: str) -> int:
    """
    Process one hook invocation. Always returns 0 (see module docstring).

    hook 호출 1건을 처리한다. 항상 0을 반환한다 (모듈 docstring 참조).
    """
    try:
        payload = json.loads(stdin_text)
    except ValueError:
        debug_log.log(
            "HOOK_PREFILTER",
            "SYSTEM",
            {"rule": "malformed_stdin", "len": len(stdin_text)},
        )
        return 0
    if not isinstance(payload, dict):
        debug_log.log(
            "HOOK_PREFILTER",
            "SYSTEM",
            {"rule": "non_object_payload"},
        )
        return 0

    try:
        rule = _prefilter(payload)
        if rule is not None:
            debug_log.log(
                "HOOK_PREFILTER",
                "SYSTEM",
                {"rule": rule},
                conv_id=payload.get(FIELD_SESSION_ID),
            )
            return 0
        debug_log.log(
            "HOOK_PREFILTER",
            "SYSTEM",
            {"rule": None, "to_judge": True},
            conv_id=payload.get(FIELD_SESSION_ID),
        )
        _route(payload)
    except Exception:
        # Never let a routing bug block the conversation.
        # 라우팅 버그가 대화를 막는 일은 절대 없어야 한다.
        pass
    return 0


def main() -> None:
    try:
        debug_log.set_proc_label("hook")
        code = run(sys.stdin.read())
    except Exception:
        code = 0
    sys.exit(code)
