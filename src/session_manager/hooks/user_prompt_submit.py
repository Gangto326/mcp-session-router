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


def _socket_round_trip(request: dict[str, Any]) -> dict[str, Any] | None:
    """
    One short-lived socket exchange with the wrapper: send one message,
    read one reply line. Returns the reply dict, or None on any failure
    — no socket, refused connection, timeout, bad frame.

    래퍼와의 단발 소켓 왕복 1회: 메시지 1건 송신, 회신 1줄 수신. 실패 시
    None — 소켓 부재·연결 거부·타임아웃·깨진 프레임 전부.
    """
    socket_path = os.environ.get(_SOCKET_ENV_VAR, "").strip()
    if not socket_path:
        return None
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


def _request_judgment(payload: dict[str, Any]) -> dict[str, Any] | None:
    """
    Ask the wrapper's judge host for a routing verdict.

    The request is thin — prompt assembly happens wrapper-side.

    래퍼 판정 호스트에 라우팅 판정을 요청한다. 요청은 얇다 — 프롬프트
    조립은 래퍼 측 담당.
    """
    return _socket_round_trip(
        {
            "client": "hook",
            "action": "judge_request",
            "prompt": payload.get(FIELD_PROMPT),
            "session_id": payload.get(FIELD_SESSION_ID),
            "transcript_path": payload.get(FIELD_TRANSCRIPT_PATH),
            "cwd": payload.get(FIELD_CWD),
        }
    )


def _calibrated_auto_threshold() -> float | None:
    """
    Confidence threshold for auto-switching, derived from the judgment
    log's (confidence, accept/reject) history.

    LLM confidence is uncalibrated, so no fixed threshold is used (rule
    8). A later phase computes this from accumulated logs (smallest
    confidence whose historical acceptance rate clears the target with
    Wilson-bound sample sufficiency). Until that exists there is no
    defensible threshold — returning None makes auto mode degrade to the
    confirm path, i.e. auto only truly activates once data has
    accumulated.

    자동 전환용 confidence 임계 — 판정 로그의 (confidence, 수용/거부)
    이력에서 산출한다.

    LLM confidence 는 보정되지 않은 값이므로 고정 임계를 쓰지 않는다
    (규칙 8). 후속 Phase 가 누적 로그에서 임계를 산출한다 (과거 수용률이
    목표치를 넘는 최소 confidence, Wilson 하한으로 표본 충분성 판정).
    그 전까지는 옹호 가능한 임계가 없다 — None 반환으로 auto 모드는
    confirm 경로로 완화된다. 즉 auto 는 데이터가 쌓여야 실제로 켜진다.
    """
    return None


# Confirm-path instruction templates. Wording follows Plan.md R2-C4
# verbatim, except the reject_switch clause (that tool arrives in a
# later phase) — until then "keep" asks for no tool call.
# confirm 경로 지시 템플릿. 문구는 Plan.md R2-C4 원문을 따르되,
# reject_switch 절만 예외 (해당 도구는 후속 Phase 에서 추가) — 그 전까지
# "유지" 선택은 도구 호출 없음으로 지시한다.
_CONFIRM_SWITCH_TEMPLATE = (
    "[session-manager 라우터] 판정: {target}으로의 전환이 적합 (근거: {evidence}). "
    "AskUserQuestion으로 사용자에게 [전환 / 현재 세션 유지]를 물은 뒤, "
    "전환 선택 시 session_switch를 호출하라. 유지 선택 시 아무 도구도 호출하지 마라."
)
_CONFIRM_NEW_TEMPLATE = (
    "[session-manager 라우터] 판정: 이 프롬프트는 기존 세션들의 소관이 아니다 "
    "(사유: {reason}). AskUserQuestion으로 사용자에게 [새 세션 생성 / 현재 세션 유지]를 "
    "물은 뒤, 생성 선택 시 session_create를 호출하라. 유지 선택 시 아무 도구도 "
    "호출하지 마라."
)


def _emit_confirm_context(verdict: dict[str, Any]) -> None:
    """
    Hand the verdict to the main LLM as additionalContext (measured
    delivery path: docs/poc/R2-hook.md §9.4). The LLM asks the user and
    calls session_switch / session_create on acceptance.

    판정을 additionalContext 로 메인 LLM 에 전달한다 (전달 경로 실측:
    docs/poc/R2-hook.md §9.4). LLM 이 사용자에게 묻고 수락 시
    session_switch / session_create 를 호출한다.
    """
    if verdict.get("action") == "SWITCH":
        context = _CONFIRM_SWITCH_TEMPLATE.format(
            target=verdict.get("target"),
            evidence=verdict.get("evidence"),
        )
    else:
        context = _CONFIRM_NEW_TEMPLATE.format(reason=verdict.get("reason"))
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": context,
                }
            },
            ensure_ascii=False,
        )
    )


def _execute_auto_switch(
    payload: dict[str, Any], verdict: dict[str, Any]
) -> bool:
    """
    Auto path: hand the switch to the wrapper, then block the prompt.

    Order matters — the block is only emitted after the wrapper ack'd
    the route_switch message. Blocking without a wrapper on the other
    side would swallow the prompt with nobody left to re-inject it.
    Returns True iff the block was emitted.

    자동 경로: 전환을 래퍼에 위임한 뒤 프롬프트를 차단한다.

    순서가 중요하다 — block 은 래퍼가 route_switch 메시지를 ack 한
    뒤에만 낸다. 반대편에 래퍼가 없는 채로 차단하면 재주입할 주체 없이
    프롬프트만 삼켜진다. block 을 냈을 때만 True 를 반환한다.
    """
    reply = _socket_round_trip(
        {
            "client": "hook",
            "action": "route_switch",
            "target": verdict.get("target"),
            "user_prompt": payload.get(FIELD_PROMPT),
            "session_id": payload.get(FIELD_SESSION_ID),
            "verdict": verdict,
        }
    )
    if reply is None or reply.get("type") != "ack":
        return False
    reason = (
        f"⇄ {verdict.get('target')} 세션으로 전환합니다 "
        f"(라우터 자동 전환 — 근거: {verdict.get('evidence')})"
    )
    print(json.dumps({"decision": "block", "reason": reason}, ensure_ascii=False))
    return True


def _route(payload: dict[str, Any]) -> None:
    """
    Judgment stage: ask the resident judge, then act on the verdict.

    STAY/ASK pass through. SWITCH/NEW go to the confirm path
    (additionalContext → the LLM asks the user) by default; the auto
    path (block + wrapper switch + re-injection) engages only in auto
    mode once a calibrated threshold exists. Every outcome exits through
    the caller with code 0.

    판정 단계: 상주 판정기에 물은 뒤 판정에 따라 행동한다.

    STAY/ASK 는 통과. SWITCH/NEW 는 기본적으로 confirm 경로
    (additionalContext → LLM 이 사용자에게 질문)로 가고, 자동 경로
    (block + 래퍼 전환 + 재주입)는 auto 모드에서 보정 임계가 존재할
    때만 발동한다. 모든 결과가 호출자에서 exit 0 으로 끝난다.
    """
    reply = _request_judgment(payload)
    debug_log.log(
        "HOOK_ROUTE",
        "SYSTEM",
        {"reply": reply},
        conv_id=payload.get(FIELD_SESSION_ID),
    )
    if reply is None or not reply.get("ok"):
        return
    verdict = reply.get("verdict")
    if not isinstance(verdict, dict):
        return
    action = verdict.get("action")
    if action not in ("SWITCH", "NEW"):
        # STAY passes silently. ASK also passes for now: the two-stage
        # re-judgment it feeds is defined over session profiles, which
        # nothing populates yet.
        # STAY 는 조용히 통과. ASK 도 당분간 통과 — ASK 가 잇는 2단
        # 재판정은 세션 profile 위에 정의되는데 아직 아무도 채우지 않는다.
        return

    cwd = payload.get(FIELD_CWD)
    root = (
        Path(cwd) / _SESSION_MANAGER_DIRNAME
        if isinstance(cwd, str) and cwd
        else None
    )
    mode = _load_routing_mode(root) if root is not None else DEFAULT_ROUTING_MODE

    if mode == "auto" and action == "SWITCH":
        threshold = _calibrated_auto_threshold()
        confidence = verdict.get("confidence")
        if (
            threshold is not None
            and isinstance(confidence, int | float)
            and confidence >= threshold
            and _execute_auto_switch(payload, verdict)
        ):
            debug_log.log(
                "HOOK_ROUTE",
                "SYSTEM",
                {"path": "auto_block", "verdict": verdict},
                conv_id=payload.get(FIELD_SESSION_ID),
            )
            return
    # NEW never auto-switches: creating a session needs a name decision,
    # which stays with the user/LLM in the confirm flow.
    # NEW 는 자동 전환하지 않는다 — 세션 생성은 이름 결정이 필요하고,
    # 그것은 confirm 흐름의 사용자·LLM 몫이다.
    _emit_confirm_context(verdict)
    debug_log.log(
        "HOOK_ROUTE",
        "SYSTEM",
        {"path": "confirm_context", "verdict": verdict},
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
