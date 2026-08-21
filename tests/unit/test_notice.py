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


class TestFormatSessionList:
    def test_empty(self) -> None:
        assert notice.format_session_list([], None) == ["세션이 없습니다"]

    def test_layout_current_order_and_retired(self) -> None:
        a = _session("backend", "로그인 API 조사. 그 다음.", "2026-08-20T00:00:00+00:00")
        b = _session("frontend", None, "2026-08-21T00:00:00+00:00")
        c = _session("old", "구 결제 모듈 작업.", "2026-08-01T00:00:00+00:00")
        c.retire("manual")
        c.retired.at = "2026-08-10T12:00:00+00:00"
        lines = notice.format_session_list([a, b, c], current="backend")
        assert lines[0] == "세션 2개 (현재: backend), 만료 1개"
        # Most recently accessed active first; current marked ●; retired last.
        # 최근 접근 활성이 먼저, 현재는 ●, 만료는 맨 뒤.
        assert lines[1] == "    frontend"
        assert lines[2] == "  ● backend   — 로그인 API 조사."
        assert lines[3] == "  ⏸ old       — 만료 2026-08-10 · 구 결제 모듈 작업."

    def test_header_without_current(self) -> None:
        a = _session("a", None, "2026-08-20T00:00:00+00:00")
        assert notice.format_session_list([a], current=None)[0] == "세션 1개"

    def test_width_clips_rows_with_ellipsis(self) -> None:
        a = _session("backend", "가" * 50, "2026-08-20T00:00:00+00:00")
        lines = notice.format_session_list([a], current=None, width=20)
        assert len(lines[1]) == 20
        assert lines[1].endswith("…")

    def test_no_width_keeps_rows_whole(self) -> None:
        a = _session("backend", "가" * 50, "2026-08-20T00:00:00+00:00")
        lines = notice.format_session_list([a], current=None, width=None)
        assert lines[1].endswith("가" * 50)
