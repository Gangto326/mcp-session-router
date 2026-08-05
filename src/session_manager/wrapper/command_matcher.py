"""Match intercepted slash commands from the input prompt text.
입력란 텍스트에서 가로채기 대상 슬래시 명령 매칭.

The matcher is intentionally strict: it only triggers on a small whitelist
of commands that affect session lifecycle (resume, exit, rename, new) and
requires the input to start with ``/``. This keeps ``/path/to/file`` and
other benign text out of the interception path.

매칭은 의도적으로 엄격하다 — 세션 lifecycle에 영향을 주는 작은 화이트리스트
(resume, exit, rename, new)에서만 trigger 되며, 입력은 반드시 ``/``로
시작해야 한다. 이로써 ``/path/to/file`` 같은 일반 텍스트는 가로채기 경로에
들어오지 않는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from session_manager import debug_log

# Commands whose execution would lose summary info if not preceded by
# session_end. Information commands (/help, /cost, /model, /clear) are NOT
# here — they don't change session identity.
# Bare /resume (no argument) is also included: while it only opens the
# picker rather than committing to a switch, the intercept's real purpose
# is to force a summary update on the leaving session for routing
# accuracy — and the user might still pick another conversation from the
# picker. Letting bare /resume slip past would leave the leaving session's
# summary stale and degrade routing precision.
# session_end 없이 실행되면 summary가 누락되는 명령들. 정보 명령
# (/help, /cost, /model, /clear 등)은 여기에 포함되지 않음 — 세션 정체성을
# 바꾸지 않으므로. 빈 /resume (인자 없음)도 포함 — picker 만 띄우는 명령이긴
# 하지만 가로채기의 진짜 목적은 떠나는 세션 summary 갱신을 통한 라우팅 정확도
# 보호이고, 사용자가 picker 에서 다른 conversation 을 고를 수도 있다.
# 빈 /resume 을 그냥 흘려보내면 떠나는 세션 summary 가 stale 로 남아 라우팅
# 정확도가 떨어진다.
KNOWN_COMMANDS: tuple[str, ...] = ("resume", "exit", "rename", "new")

# Anchored start-to-end. The argument group (\S.*?) starts with non-whitespace
# so trailing field padding doesn't get captured as args. \s*$ absorbs any
# remaining trailing whitespace.
# 문자열 처음부터 끝까지 anchored. 인자 그룹 (\S.*?)는 non-whitespace로 시작 —
# 입력란의 패딩 공백이 인자로 잡히지 않게 함. \s*$ 가 끝의 잔여 공백 흡수.
_COMMAND_RE = re.compile(
    r"^/(" + "|".join(KNOWN_COMMANDS) + r")(?:\s+(\S.*?))?\s*$"
)

# /back (R3-C3) is matched separately from KNOWN_COMMANDS on purpose:
# those are observe-and-forward (real Claude Code commands), while /back
# does not exist in Claude Code — the wrapper intercepts it and handles
# it itself, never forwarding. No arguments allowed: "/back x" is not an
# undo request and falls through to the TUI.
# /back (R3-C3) 은 의도적으로 KNOWN_COMMANDS 와 분리 매칭한다: 그쪽은
# "관찰 후 통과" (실제 Claude Code 명령) 경로이고, /back 은 Claude Code 에
# 없는 명령 — 래퍼가 가로채 자체 처리하며 절대 forward 하지 않는다.
# 인자는 허용하지 않는다: "/back x" 는 undo 요청이 아니므로 TUI 로 흘려보낸다.
_BACK_COMMAND_RE = re.compile(r"^/back\s*$")

# Heuristic: strip Ink-style placeholder hints like
# ``[conversation id or search term]`` that may follow the user's input.
# False-positive risk if a user genuinely types ``[...]`` as the argument —
# accepted as a known limitation; document in README.
# 휴리스틱 — Ink 스타일 placeholder hint 제거 (예: ``[conversation id ...]``).
# 사용자가 인자로 진짜 ``[...]``를 친 경우 false-positive 위험. 알려진 한계로
# 받아들이고 README에 명시.
_PLACEHOLDER_RE = re.compile(r"\s*\[[^\]]*\]\s*$")


@dataclass(frozen=True)
class InterceptedCommand:
    """Result of matching an intercepted slash command.
    가로채기 대상 슬래시 명령 매칭 결과.
    """

    command: str  # one of KNOWN_COMMANDS
    args: str  # empty string when no argument (e.g. /exit)


def match_intercept_command(prompt_text: str | None) -> InterceptedCommand | None:
    """Return the matched command if ``prompt_text`` is an intercept target.

    ``prompt_text``가 가로채기 대상이면 매칭 결과 반환, 아니면 None.

    Steps:
        1. Reject None / empty / whitespace-only input.
        2. Strip trailing Ink placeholder hint, if present.
        3. Match against the strict whitelist regex.

    단계:
        1. None / 빈 / 공백뿐인 입력 거부.
        2. 끝에 붙은 Ink placeholder hint 제거 (있을 경우).
        3. 엄격 화이트리스트 정규식 매칭.
    """
    if not prompt_text or not prompt_text.strip():
        debug_log.log(
            "CMD_MATCH",
            "USER",
            {"matched": False, "reason": "empty_prompt", "prompt": prompt_text},
        )
        return None
    cleaned = _PLACEHOLDER_RE.sub("", prompt_text)
    match = _COMMAND_RE.match(cleaned)
    if match is None:
        # Capture the first token so a /unknown command shows up in logs
        # (helps diagnose false-negatives when Claude Code adds a new
        # lifecycle command we haven't whitelisted).
        # 첫 토큰을 기록 — /unknown 명령이 로그에 남도록 (Claude Code 가
        # 새 lifecycle 명령을 추가했는데 화이트리스트에 없을 때 false-
        # negative 진단에 도움).
        first_token = cleaned.split(None, 1)[0] if cleaned.strip() else ""
        debug_log.log(
            "CMD_MATCH",
            "USER",
            {
                "matched": False,
                "reason": "no_regex_match",
                "prompt": prompt_text,
                "cleaned": cleaned,
                "first_token": first_token,
            },
        )
        return None
    command = match.group(1)
    args = (match.group(2) or "").strip()
    debug_log.log(
        "CMD_MATCH",
        "USER",
        {
            "matched": True,
            "command": command,
            "args": args,
            "prompt": prompt_text,
        },
    )
    return InterceptedCommand(command=command, args=args)


def match_back_command(prompt_text: str | None) -> bool:
    """Return True if ``prompt_text`` is the wrapper-native ``/back`` command.

    ``prompt_text`` 가 래퍼 자체 명령 ``/back`` 이면 True.
    """
    if not prompt_text or not prompt_text.strip():
        return False
    cleaned = _PLACEHOLDER_RE.sub("", prompt_text)
    matched = _BACK_COMMAND_RE.match(cleaned) is not None
    if matched:
        debug_log.log(
            "CMD_MATCH",
            "USER",
            {"matched": True, "command": "back", "prompt": prompt_text},
        )
    return matched
