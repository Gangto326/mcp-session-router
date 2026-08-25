"""Router notice grammar (R5-C2).

라우터 알림 문법 (R5-C2). (/sessions 목록 렌더러는 R6-C1 에서 제거 —
직접 그리기가 Ink 렌더러와 충돌.)

Every router intervention the wrapper prints to the terminal goes
through one vocabulary so the user learns a handful of glyphs instead of
reading each sentence: ``⇄`` switch, ``✚`` new session, ``⤺`` /back
return, ``⚠`` rollover. Plain informational
lines (refusals, "nothing to undo") carry no glyph. Internal machinery
(injection, summaries, judging) stays hidden — this module only names
what is already user-visible.

래퍼가 터미널에 찍는 모든 라우터 개입은 한 어휘를 거친다 — 사용자가
문장을 읽는 대신 기호 몇 개만 익히면 되게: ``⇄`` 전환, ``✚`` 새 세션,
``⤺`` /back 복귀, ``⚠`` 롤오버. 단순 안내
(거부, "되돌릴 것 없음") 는 기호 없음. 내부 동작 (주입·요약·판정) 은
계속 은닉 — 이 모듈은 이미 사용자에게 보이는 것에 이름만 붙인다.
"""

from __future__ import annotations

import re
from enum import StrEnum


class NoticeKind(StrEnum):
    """Router intervention kinds — the value is the leading glyph.

    라우터 개입 종류 — 값이 머리 기호다.
    """

    SWITCH = "⇄"
    NEW = "✚"
    BACK = "⤺"
    ROLLOVER = "⚠"
    INFO = ""


def format_notice(kind: NoticeKind, text: str) -> str:
    """Prefix ``text`` with the kind's glyph (INFO has none).

    ``text`` 앞에 종류의 기호를 붙인다 (INFO 는 기호 없음).
    """
    return f"{kind.value} {text}" if kind.value else text


# A sentence ends at . ! ? or the CJK full stop followed by whitespace/end,
# or at a line break. Summaries are prose paragraphs ("...했습니다. ..."),
# so the first sentence is the natural one-line gist.
# 문장은 . ! ? 또는 온점(。) 뒤 공백·끝, 혹은 줄바꿈에서 끝난다. 요약은
# 산문 문단 ("...했습니다. ...") 이라 첫 문장이 자연스러운 한 줄 요지다.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?。])(?:\s|$)|\n")


def first_sentence(text: str | None) -> str:
    """Return the first sentence of ``text`` (empty string when none).

    ``text`` 의 첫 문장을 돌려준다 (없으면 빈 문자열).
    """
    if not text:
        return ""
    stripped = text.strip()
    match = _SENTENCE_END_RE.search(stripped)
    return (stripped[: match.start()] if match else stripped).strip()


def _fit(text: str, width: int | None) -> str:
    """Clip ``text`` to ``width`` columns with an ellipsis; None = no clip.

    ``text`` 를 ``width`` 칸에 말줄임표로 맞춘다. None 이면 자르지 않음.
    """
    if width is None or width <= 1 or len(text) <= width:
        return text
    return text[: width - 1] + "…"


# Row markers. ``●`` = the session the wrapper is in, ``○`` = ended
# (archived), blank = other active session.
# 행 표식. ``●`` = 래퍼가 있는 세션, ``○`` = 끝남 (archived), 공백 = 그 외
# 활성 세션.
_MARK_CURRENT = "●"
_MARK_ENDED = "○"
_MARK_OTHER = " "
