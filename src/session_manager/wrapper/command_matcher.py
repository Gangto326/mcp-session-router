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

# Commands whose execution would lose summary info if not preceded by
# session_end. Information commands (/help, /cost, /model, /clear) are NOT
# here — they don't change session identity.
# session_end 없이 실행되면 summary가 누락되는 명령들. 정보 명령
# (/help, /cost, /model, /clear 등)은 여기에 포함되지 않음 — 세션 정체성을
# 바꾸지 않으므로.
KNOWN_COMMANDS: tuple[str, ...] = ("resume", "exit", "rename", "new")

# Anchored start-to-end. The argument group (\S.*?) starts with non-whitespace
# so trailing field padding doesn't get captured as args. \s*$ absorbs any
# remaining trailing whitespace.
# 문자열 처음부터 끝까지 anchored. 인자 그룹 (\S.*?)는 non-whitespace로 시작 —
# 입력란의 패딩 공백이 인자로 잡히지 않게 함. \s*$ 가 끝의 잔여 공백 흡수.
_COMMAND_RE = re.compile(
    r"^/(" + "|".join(KNOWN_COMMANDS) + r")(?:\s+(\S.*?))?\s*$"
)

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
        return None
    cleaned = _PLACEHOLDER_RE.sub("", prompt_text)
    match = _COMMAND_RE.match(cleaned)
    if match is None:
        return None
    command = match.group(1)
    args = (match.group(2) or "").strip()
    return InterceptedCommand(command=command, args=args)
