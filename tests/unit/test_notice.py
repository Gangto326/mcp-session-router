"""Unit tests for the router notice grammar and /sessions listing (R5-C2).

라우터 알림 문법과 /sessions 목록 단위 테스트 (R5-C2).
"""

from __future__ import annotations

from session_manager.models import SessionMetadata
from session_manager.wrapper import notice
from session_manager.wrapper.notice import NoticeKind


class TestFormatNotice:
    def test_glyph_prefix(self) -> None:
        assert notice.format_notice(NoticeKind.SWITCH, "x") == "⇄ x"
        assert notice.format_notice(NoticeKind.NEW, "x") == "✚ x"
        assert notice.format_notice(NoticeKind.BACK, "x") == "⤺ x"
        assert notice.format_notice(NoticeKind.ROLLOVER, "x") == "⚠ x"

    def test_info_has_no_glyph(self) -> None:
        assert notice.format_notice(NoticeKind.INFO, "x") == "x"

    def test_glyphs_are_distinct(self) -> None:
        glyphs = [k.value for k in NoticeKind if k.value]
        assert len(glyphs) == len(set(glyphs))


class TestFirstSentence:
    def test_korean_prose(self) -> None:
        text = "로그인 API 500 오류를 조사했습니다. 원인은 타임아웃이었습니다."
        assert notice.first_sentence(text) == "로그인 API 500 오류를 조사했습니다."

    def test_line_break_ends_sentence(self) -> None:
        assert notice.first_sentence("첫 줄\n둘째 줄.") == "첫 줄"

    def test_dot_inside_token_is_not_an_end(self) -> None:
        # "v1.2" / "file.py" — a dot not followed by whitespace continues.
        # "v1.2" / "file.py" — 뒤에 공백이 없는 점은 문장을 끝내지 않는다.
        text = "summarizer.py 를 v1.2 로 올렸습니다. 다음 단계."
        assert notice.first_sentence(text) == "summarizer.py 를 v1.2 로 올렸습니다."

    def test_no_terminator_returns_whole(self) -> None:
        assert notice.first_sentence("  끝나지 않은 문장  ") == "끝나지 않은 문장"

    def test_empty_and_none(self) -> None:
        assert notice.first_sentence(None) == ""
        assert notice.first_sentence("") == ""


def _session(name: str, summary: str | None, accessed: str) -> SessionMetadata:
    s = SessionMetadata.new(name=name, title=name)
    s.summary = summary
    s.last_accessed = accessed
    return s
