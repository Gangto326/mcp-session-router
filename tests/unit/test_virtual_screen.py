"""Unit tests for VirtualScreen.
VirtualScreen 단위 테스트.
"""

from session_manager.wrapper.virtual_screen import PROMPT_MARKER, VirtualScreen


def test_empty_screen_returns_none():
    """No content fed → no prompt line.
    빈 화면 → None.
    """
    s = VirtualScreen(80, 24)
    assert s.get_prompt_line() is None


def test_marker_only_returns_empty_string():
    """❯ marker present but no input → empty string.
    ❯만 있고 입력 없음 → 빈 문자열.
    """
    s = VirtualScreen(80, 24)
    s.feed(PROMPT_MARKER.encode())
    assert s.get_prompt_line() == ""


def test_marker_with_inverse_cursor_only():
    """❯ + NBSP + inverse cursor (empty input) → empty string.
    ❯ + NBSP + inverse cursor (빈 입력) → 빈 문자열.
    """
    s = VirtualScreen(80, 24)
    # Mimic Ink drawing an empty prompt: ❯, NBSP, inverse space, inverse-off.
    # Ink의 빈 입력란 모방: ❯, NBSP, inverse 공백, inverse 해제.
    s.feed(PROMPT_MARKER.encode() + b"\xc2\xa0\x1b[7m \x1b[27m")
    assert s.get_prompt_line() == ""


def test_extract_simple_input():
    """❯ /help → '/help'."""
    s = VirtualScreen(80, 24)
    s.feed(PROMPT_MARKER.encode() + b" /help")
    assert s.get_prompt_line() == "/help"


def test_extract_with_color_codes():
    """ANSI SGR (color) codes are absorbed by pyte; only plain text remains.
    ANSI SGR(색상) 코드는 pyte가 흡수하고 plain text만 남음.
    """
    s = VirtualScreen(80, 24)
    s.feed(
        PROMPT_MARKER.encode()
        + b" \x1b[38;2;177;185;249m/resume foo\x1b[39m"
    )
    assert s.get_prompt_line() == "/resume foo"


def test_korean_input():
    """Hangul (UTF-8 multi-byte) input is preserved.
    한글(UTF-8 멀티바이트) 입력 보존.
    """
    s = VirtualScreen(80, 24)
    s.feed(PROMPT_MARKER.encode() + b" \xec\x95\x88\xeb\x85\x95")  # 안녕
    assert s.get_prompt_line() == "안녕"


def test_multiple_markers_returns_last():
    """When multiple ❯ rows exist, return the LAST (live) one.
    ❯가 여러 개면 마지막 (라이브) 것 반환.
    """
    s = VirtualScreen(80, 24)
    s.feed(PROMPT_MARKER.encode() + b" /old\r\n")
    s.feed(PROMPT_MARKER.encode() + b" /new")
    assert s.get_prompt_line() == "/new"


def test_resize_updates_dimensions():
    """resize() updates underlying pyte.Screen rows/cols.
    resize()로 pyte.Screen rows/cols 갱신.
    """
    s = VirtualScreen(80, 24)
    s.resize(120, 40)
    assert len(s._screen.display) == 40
    assert len(s._screen.display[0]) == 120


def test_chunked_feed_with_split_utf8():
    """UTF-8 multi-byte chars split across chunk boundaries are reassembled.
    청크 경계에 UTF-8 멀티바이트 문자가 끊겨도 재조립됨.
    """
    s = VirtualScreen(80, 24)
    # ❯ (3 bytes) + space + 안 (3 bytes) split mid-character at every byte.
    # ❯ + 공백 + 안 — 매 바이트마다 분할.
    payload = PROMPT_MARKER.encode() + b" \xec\x95\x88"
    s.feed(payload[:1])
    s.feed(payload[1:4])
    s.feed(payload[4:5])
    s.feed(payload[5:])
    assert s.get_prompt_line() == "안"


def test_partial_redraw_pattern():
    """Cursor return + line clear + redraw — pyte tracks only the final state.
    cursor return + line clear + 다시 그리기 — pyte는 최종 상태만 추적.
    """
    s = VirtualScreen(80, 24)
    s.feed(PROMPT_MARKER.encode() + b" /he")
    assert s.get_prompt_line() == "/he"
    # Partial redraw: \r returns cursor to col 0, \x1b[2K clears the line,
    # then re-write a longer text.
    # 부분 갱신: \r로 col 0 복귀, \x1b[2K로 라인 클리어, 더 긴 텍스트 재기록.
    s.feed(b"\r\x1b[2K" + PROMPT_MARKER.encode() + b" /help")
    assert s.get_prompt_line() == "/help"


def test_contains_finds_text_in_any_row():
    """contains() matches a substring across any row.
    contains()는 어느 행에서든 부분 문자열을 찾으면 True.
    """
    s = VirtualScreen(80, 24)
    s.feed(b"first row\r\nsecond row with target\r\nthird row")
    assert s.contains("target") is True
    assert s.contains("first row") is True
    assert s.contains("nonexistent") is False


def test_contains_returns_false_on_empty():
    """contains() on an empty screen returns False.
    빈 화면에서 contains()는 False.
    """
    s = VirtualScreen(80, 24)
    assert s.contains("anything") is False


def test_orphan_wide_char_stub_does_not_crash():
    """Orphan wide-char stub must not raise IndexError.
    외톨이 wide-char stub은 IndexError를 일으키지 않아야 함.

    Repro: draw a hangul char (width=2) at column 0, return cursor with
    \\r, then write a single ASCII char. pyte overwrites column 0 with
    the ASCII glyph but leaves the stub at column 1 as ``""`` — pyte
    0.8.2's Screen.display then crashes on ``char[0]`` of the empty stub
    (screens.py:241). Our _safe_display must handle this gracefully.
    재현: 0열에 한글(width=2) → ``\\r``로 cursor 복귀 → ASCII 한 글자.
    pyte는 0열을 ASCII로 덮어쓰지만 1열의 stub은 ``""``로 그대로 남고,
    pyte 0.8.2의 Screen.display는 빈 stub의 ``char[0]``에서 크래시
    (screens.py:241). _safe_display는 이를 안전하게 처리해야 함.
    """
    import pytest

    s = VirtualScreen(10, 2)
    s.feed("안".encode())
    s.feed(b"\r")
    s.feed(b"a")

    # Sanity check the bug exists in raw pyte (guards against silent
    # pyte upgrades that fix it — if this stops raising, the safe
    # wrapper may be redundant and worth revisiting).
    # raw pyte에서 버그 존재 확인 — 만약 향후 pyte가 조용히 고치면
    # 이 단언이 깨지고, safe wrapper가 불필요한지 재검토 신호가 됨.
    with pytest.raises(IndexError):
        _ = s._screen.display

    # Our wrapper must not crash and must preserve column indexing
    # (orphan stub rendered as a single space).
    # 우리 래퍼는 크래시 없이 컬럼 인덱싱 보존 (외톨이 stub은 공백 1칸).
    rows = s._safe_display()
    assert len(rows) == 2
    assert rows[0].startswith("a ")
    assert s.contains("a") is True


def test_get_prompt_line_survives_orphan_stub():
    """get_prompt_line() must survive an orphan stub elsewhere on screen.
    화면 어딘가에 외톨이 stub이 있어도 get_prompt_line()은 동작해야 함.
    """
    s = VirtualScreen(20, 3)
    # Row 0: orphan-stub trap (한글 → \r → ASCII).
    # 0행: 외톨이 stub 트랩.
    s.feed("안".encode())
    s.feed(b"\r")
    s.feed(b"a")
    # Move to row 2 and draw a normal prompt.
    # 2행으로 이동 후 정상 prompt 그리기.
    s.feed(b"\x1b[3;1H")  # CUP → row 3, col 1 (1-indexed)
    s.feed(PROMPT_MARKER.encode() + b" /help")
    assert s.get_prompt_line() == "/help"
