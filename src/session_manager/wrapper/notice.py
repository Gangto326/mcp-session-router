"""Router notice grammar and the ``/sessions`` listing (R5-C2).

라우터 알림 문법과 ``/sessions`` 목록 (R5-C2).

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

from session_manager.models import SessionMetadata, SessionStatus


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


# Row markers. ``●`` = the session the wrapper is in, ``⏸`` = retired,
# blank = other active session.
# 행 표식. ``●`` = 래퍼가 있는 세션, ``⏸`` = 만료, 공백 = 그 외 활성 세션.
_MARK_CURRENT = "●"
_MARK_RETIRED = "⏸"
_MARK_OTHER = " "


def format_session_list(
    sessions: list[SessionMetadata],
    current: str | None,
    width: int | None = None,
) -> list[str]:
    """Render the ``/sessions`` listing: header + one row per session.

    ``/sessions`` 목록을 그린다 — 머리줄 + 세션당 한 행.

    Active sessions first (most recently accessed on top), retired ones
    after with their retirement date; each row shows the first sentence
    of the summary. ``width`` (terminal columns) clips rows; None leaves
    them whole.

    활성 세션 먼저 (최근 접근순), 만료 세션은 만료 날짜와 함께 뒤에;
    행마다 요약 첫 문장. ``width`` (터미널 칸 수) 로 행을 자르고 None
    이면 그대로 둔다.
    """
    active = sorted(
        (s for s in sessions if s.status != SessionStatus.RETIRED),
        key=lambda s: s.last_accessed,
        reverse=True,
    )
    retired = sorted(
        (s for s in sessions if s.status == SessionStatus.RETIRED),
        key=lambda s: s.retired.at if s.retired else "",
        reverse=True,
    )
    if not active and not retired:
        return ["세션이 없습니다"]

    where = f" (현재: {current})" if current else ""
    header = f"세션 {len(active)}개{where}"
    if retired:
        header += f", 만료 {len(retired)}개"
    lines = [header]

    name_width = max(len(s.name) for s in active + retired)
    for s in active:
        mark = _MARK_CURRENT if s.name == current else _MARK_OTHER
        gist = first_sentence(s.summary)
        row = f"  {mark} {s.name.ljust(name_width)}"
        if gist:
            row += f"  — {gist}"
        lines.append(_fit(row, width))
    for s in retired:
        when = s.retired.at[:10] if s.retired else ""
        gist = first_sentence(s.summary)
        row = f"  {_MARK_RETIRED} {s.name.ljust(name_width)}  — 만료 {when}".rstrip()
        if gist:
            row += f" · {gist}"
        lines.append(_fit(row, width))
    return lines
