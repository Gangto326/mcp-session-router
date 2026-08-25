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

from session_manager import debug_log, handoff_store
from session_manager.models.config import (
    DEFAULT_ROUTING_MODE,
)
from session_manager.routing import decision_log
from session_manager.routing.judge import (
    HOOK_REPLY_TIMEOUT_SECS,
)
from session_manager.storage.file_store import (
    _CONFIG_FILENAME,
    _SESSION_MANAGER_DIRNAME,
    _SESSIONS_DIRNAME,
)
from session_manager.wrapper.handoff_formatter import format_handoff_injection

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

# The default routing mode ("confirm") is defined in models/config.py
# and imported above — single source of truth with the Config model.
# 기본 라우팅 모드("confirm")는 models/config.py 에 정의되어 있고 위에서
# import 한다 — Config 모델과의 단일 출처.


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
    if not isinstance(mode, str):
        return DEFAULT_ROUTING_MODE
    if mode == "auto":
        # Removed in R6-C3 — a stale config value degrades to confirm.
        # R6-C3 에서 제거 — 낡은 config 값은 confirm 으로 강등한다.
        return "confirm"
    return mode


def _count_active_sessions(root: Path) -> int:
    """
    Count sessions whose status is active.

    Absence of the ``status`` field counts as active (backward compat
    with pre-status session files). Any other value — archived, or the
    legacy retired/expired — counts as inactive, so a project whose
    extra sessions are all ended correctly skips routing. Corrupt
    session files are skipped so one bad file cannot distort the count.

    status가 active인 세션 수를 센다.

    ``status`` 필드 부재는 active 로 간주한다 (status 도입 전 세션 파일과
    의 하위 호환). 그 외 값 — archived, 옛 retired·expired — 은 전부
    비활성으로 세므로, 나머지 세션이 모두 끝난 프로젝트는 라우팅을
    올바르게 건너뛴다. 손상된 세션 파일은 건너뛰어 파일 하나가 집계를
    왜곡하지 못하게 한다.
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


TARGET_STATUS_MISSING = "missing"


def _target_status(root: Path, name: str) -> str:
    """
    Status of the session named *name*, for the proposal log (R5-C3).

    Same raw-file reading and same "absent status means active" rule as
    ``_count_active_sessions``. A name no session file carries yields
    ``"missing"`` — the judge proposed a session that does not exist,
    which is itself worth counting. Corrupt files are skipped.

    제안 로그용 (R5-C3) *name* 세션의 상태. ``_count_active_sessions``
    와 같은 원시 파일 읽기·같은 "status 부재 = active" 규칙. 어떤 세션
    파일도 갖지 않은 이름은 ``"missing"`` — 판정기가 존재하지 않는
    세션을 제안한 것이며 그 자체가 집계 대상이다. 손상 파일은 건너뛴다.
    """
    sessions_dir = root / _SESSIONS_DIRNAME
    if not sessions_dir.is_dir():
        return TARGET_STATUS_MISSING
    for path in sessions_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("name") == name:
            status = data.get("status", "active")
            return status if isinstance(status, str) else "active"
    return TARGET_STATUS_MISSING


def _conv_id_of(payload: dict[str, Any]) -> str | None:
    value = payload.get(FIELD_SESSION_ID)
    return value if isinstance(value, str) and value else None


def _expire_ignored_proposals(payload: dict[str, Any]) -> None:
    """
    A prompt arrived in this conversation: close its open proposals.

    Accepting a proposal moves the user to another conversation and
    rejecting writes a label before any further prompt, so a new prompt
    in the SAME conversation means the last proposal here was ignored.
    Closing it keeps a later unrelated session_switch from consuming it
    as an "accept" (decision_log module docstring).

    이 대화에 프롬프트가 왔다 — 열린 제안을 닫는다. 수용은 다른 대화로
    이동이고 거부는 다음 프롬프트 전에 라벨을 남기므로, **같은** 대화의
    새 프롬프트는 직전 제안이 무시됐다는 뜻이다. 닫아 두어야 나중의
    무관한 session_switch 가 그것을 "수용" 으로 소비하지 못한다
    (decision_log 모듈 docstring).
    """
    cwd = payload.get(FIELD_CWD)
    conv_id = _conv_id_of(payload)
    if not isinstance(cwd, str) or not cwd or conv_id is None:
        return
    if decision_log.expire_ignored_proposals(Path(cwd), conv_id):
        debug_log.log(
            "DECISION_LOG", "SYSTEM", {"op": "expire"}, conv_id=conv_id
        )


_STALE_CONV_TEMPLATE = (
    "[session-manager 라우터] 현재 대화는 세션 {session}의 롤오버된 이전 "
    "대화다 — 이 세션의 최신 대화가 따로 있다. 사용자의 이번 프롬프트에 "
    "답하기 전에 AskUserQuestion으로 [최신 대화로 이동 / 이 대화에서 계속]"
    "을 물어라. 이동 선택 시 session_switch(target='{session}', "
    "summary=<이 대화의 요지 한 줄>, user_prompt=<사용자의 이번 프롬프트 "
    "원문>)를 호출하라. 계속 선택 시 아무 도구도 호출하지 말고 프롬프트에 "
    "답하라."
)


def _deliver_pending_notice(payload: dict[str, Any]) -> bool:
    """
    Wrapper-notice stage (R4-C6 B): consume a pending notice file — a
    fact the wrapper discovered after the fact (stale-conversation
    entry) — and inject its instruction on this ORDINARY prompt. Returns
    True when handled; the caller then skips routing for this turn (one
    missed routing pass is harmless, same as the judge warmup window).

    래퍼 안내 단계 (R4-C6 B) — 래퍼가 사후에 발견한 사실 (만료 대화
    진입) 을 담은 notice 파일을 소비해, **일반 프롬프트**에 지시를
    주입한다. 처리했으면 True — 호출자는 이번 턴의 라우팅을 건너뛴다
    (판정 1회 미발동은 무해 — 웜업 창과 같은 원칙).
    """
    cwd = payload.get(FIELD_CWD)
    if not isinstance(cwd, str) or not cwd:
        return False
    notice = handoff_store.take_notice(Path(cwd))
    if notice is None:
        return False
    if notice.get("type") != "stale_conversation":
        # Unknown notice type (future producer newer than this hook) —
        # consume silently rather than inject a half-understood text.
        # 미지의 notice 유형 (이 hook 보다 새로운 생산자) — 어설픈 주입
        # 대신 조용히 소비한다.
        debug_log.log(
            "HOOK_NOTICE",
            "SYSTEM",
            {"result": "unknown_type", "type": notice.get("type")},
            conv_id=payload.get(FIELD_SESSION_ID),
        )
        return False
    session = notice.get("session")
    if not isinstance(session, str) or not session:
        return False
    # The notice was written at the previous turn's end for the stale
    # conversation; if the user has ALREADY moved (this prompt runs in a
    # different conversation), the warning is obsolete — drop it.
    # notice 는 직전 턴 종료 시점에 그 낡은 대화에 대해 쓰였다 — 사용자가
    # 이미 옮겼다면 (이 프롬프트가 다른 conversation 에서 돌고 있다면)
    # 경고는 낡았다 — 버린다.
    conv_id = payload.get(FIELD_SESSION_ID)
    noticed_conv = notice.get("conv_id")
    if (
        isinstance(conv_id, str)
        and isinstance(noticed_conv, str)
        and conv_id != noticed_conv
    ):
        debug_log.log(
            "HOOK_NOTICE",
            "SYSTEM",
            {"result": "conversation_moved"},
            conv_id=conv_id,
        )
        return False
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": _STALE_CONV_TEMPLATE.format(
                        session=session
                    ),
                }
            },
            ensure_ascii=False,
        )
    )
    debug_log.log(
        "HOOK_NOTICE",
        "SYSTEM",
        {"result": "delivered", "session": session},
        conv_id=conv_id,
    )
    return True


def _deliver_pending_handoff(payload: dict[str, Any]) -> bool:
    """
    Transition-trigger stage: when this submission is the respawn's
    fixed trigger prompt, consume the pending handoff file and hand its
    content to the LLM as additionalContext. Returns True when handled
    (the caller then skips prefilter/judgment — a transition trigger
    must never be re-routed).

    전환 트리거 단계 — 이 제출이 respawn 의 고정 트리거 프롬프트면
    pending handoff 파일을 소비해 additionalContext 로 LLM 에 전달한다.
    처리했으면 True — 호출자는 프리필터·판정을 건너뛴다 (전환 트리거가
    다시 라우팅되면 안 된다).
    """
    if payload.get(FIELD_PROMPT) != handoff_store.TRIGGER_PROMPT:
        return False
    cwd = payload.get(FIELD_CWD)
    if not isinstance(cwd, str) or not cwd:
        return True  # 트리거인데 cwd 불명 — 라우팅만 막고 통과
    pending = handoff_store.take_pending(Path(cwd))
    if pending is None:
        # Trigger without a file (stale trigger, double fire) — pass
        # through silently; the LLM sees only the bare trigger text.
        # 파일 없는 트리거 (낡은 트리거·이중 발동) — 조용히 통과.
        debug_log.log(
            "HOOK_HANDOFF",
            "SYSTEM",
            {"result": "no_pending_file"},
            conv_id=payload.get(FIELD_SESSION_ID),
        )
        return True
    handoff = pending.get("handoff")
    user_prompt = pending.get("user_prompt")
    context = format_handoff_injection(
        handoff if isinstance(handoff, dict) else {},
        user_prompt if isinstance(user_prompt, str) else "",
    )
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
    debug_log.log(
        "HOOK_HANDOFF",
        "SYSTEM",
        {"result": "delivered", "target": pending.get("target")},
        conv_id=payload.get(FIELD_SESSION_ID),
    )
    return True


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


def _socket_round_trip(
    request: dict[str, Any], timeout: float = HOOK_REPLY_TIMEOUT_SECS
) -> dict[str, Any] | None:
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
            sock.settimeout(timeout)
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


def _request_judgment(
    payload: dict[str, Any]
) -> dict[str, Any] | None:
    """
    Ask the wrapper's judge host for a routing verdict.

    The request is thin — prompt assembly happens wrapper-side.

    래퍼 판정 호스트에 라우팅 판정을 요청한다. 요청은 얇다 — 프롬프트
    조립은 래퍼 측 담당.
    """
    request = {
        "client": "hook",
        "action": "judge_request",
        "prompt": payload.get(FIELD_PROMPT),
        "session_id": payload.get(FIELD_SESSION_ID),
        "transcript_path": payload.get(FIELD_TRANSCRIPT_PATH),
        "cwd": payload.get(FIELD_CWD),
    }
    return _socket_round_trip(request, timeout=HOOK_REPLY_TIMEOUT_SECS)


# Confirm-path instruction templates. Wording follows Plan.md R2-C4
# verbatim, with the reject_switch clause activated in R3-C1. The NEW
# template keeps "no tool call" for the keep choice — there is no
# rejected target session to record a precedent against.
# confirm 경로 지시 템플릿. 문구는 Plan.md R2-C4 원문을 따르며,
# reject_switch 절은 R3-C1 에서 활성화되었다. NEW 템플릿의 "유지" 선택은
# 도구 호출 없음 유지 — 판례를 기록할 거부 대상 세션이 없다.
_CONFIRM_SWITCH_TEMPLATE = (
    "[session-manager 라우터] 판정: {target}으로의 전환이 적합 (근거: {evidence}). "
    "AskUserQuestion으로 사용자에게 [전환 / 현재 세션 유지]를 물은 뒤, "
    "전환 선택 시 session_switch를 호출하라. 유지 선택 시 "
    "reject_switch(rejected_target='{target}', "
    "prompt_gist=<사용자 프롬프트 요지 한 줄>)를 호출하라."
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


def _route(payload: dict[str, Any]) -> None:
    """
    Judgment stage: ask the resident judge, then act on the verdict.

    STAY/ASK pass through. SWITCH/NEW go to the confirm path
    (additionalContext → the LLM asks the user). The auto path was
    removed in R6-C3. Every outcome exits through the caller with code 0.

    판정 단계: 상주 판정기에 물은 뒤 판정에 따라 행동한다.

    STAY/ASK 는 통과. SWITCH/NEW 는 confirm 경로 (additionalContext →
    LLM 이 사용자에게 질문) 로 간다. 자동 경로는 R6-C3 에서 제거됐다.
    모든 결과가 호출자에서 exit 0 으로 끝난다.
    """
    cwd = payload.get(FIELD_CWD)
    root = (
        Path(cwd) / _SESSION_MANAGER_DIRNAME
        if isinstance(cwd, str) and cwd
        else None
    )
    mode = _load_routing_mode(root) if root is not None else DEFAULT_ROUTING_MODE

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

    target = verdict.get("target")
    confidence = verdict.get("confidence")

    # NEW never auto-switches: creating a session needs a name decision,
    # which stays with the user/LLM in the confirm flow.
    # NEW 는 자동 전환하지 않는다 — 세션 생성은 이름 결정이 필요하고,
    # 그것은 confirm 흐름의 사용자·LLM 몫이다.
    _emit_confirm_context(verdict)
    if (
        root is not None
        and action == "SWITCH"
        and isinstance(target, str)
        and target
        and isinstance(confidence, int | float)
    ):
        # Confirm proposals are the calibration source: session_switch
        # labels them accepted, reject_switch / /back rejected (R3-C4).
        # confirm 제안이 보정의 원천이다 — session_switch 가 수용 라벨,
        # reject_switch·/back 이 거부 라벨을 남긴다 (R3-C4).
        decision_log.append_proposal(
            root.parent,
            target,
            float(confidence),
            mode=mode,
            target_status=_target_status(root, target),
            conv_id=_conv_id_of(payload),
        )
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
        # Before anything else: this prompt closes any proposal still
        # open in this conversation (ignored ≠ rejected, but ignored must
        # not stay consumable either).
        # 무엇보다 먼저 — 이 프롬프트는 이 대화에 아직 열린 제안을 닫는다
        # (무시는 거부가 아니지만 소비 가능한 채로 남아서도 안 된다).
        _expire_ignored_proposals(payload)
        if _deliver_pending_handoff(payload):
            return 0
        if _deliver_pending_notice(payload):
            return 0
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
