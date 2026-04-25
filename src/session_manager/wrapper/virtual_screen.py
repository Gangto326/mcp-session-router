"""Virtual terminal screen for capturing Claude Code's input prompt text.
가상 터미널 화면 — Claude Code 입력란 텍스트 캡처.

We feed PTY output into a pyte virtual screen; on submit (stdin \\r) the
line containing the ❯ marker holds the user's input prompt text right
before Enter is pressed.

PTY 출력을 pyte 가상 화면에 먹이고, submit (stdin \\r) 시점에 ❯ 마커가
있는 라인을 읽어 사용자가 Enter 누르기 직전의 입력 텍스트를 얻는다.
"""

from __future__ import annotations

import pyte

# figures.pointer (U+276F) — Ink draws this at the start of the input prompt.
# Ink가 입력란 시작에 그리는 마커.
PROMPT_MARKER = "❯"


class VirtualScreen:
    """pyte.Screen + ByteStream wrapper tuned for Claude Code prompt extraction.
    Claude Code 입력란 추출용 pyte 래퍼.
    """

    DEFAULT_COLS = 80
    DEFAULT_ROWS = 24

    def __init__(self, cols: int = DEFAULT_COLS, rows: int = DEFAULT_ROWS) -> None:
        # pyte.Screen constructor takes (columns, lines).
        # pyte.Screen 생성자는 (columns, lines) 순서.
        self._screen = pyte.Screen(cols, rows)
        self._stream = pyte.ByteStream(self._screen)

    def feed(self, chunk: bytes) -> None:
        """Feed a chunk of PTY output bytes into the virtual screen.
        PTY 출력 청크를 가상 화면에 먹임.
        """
        self._stream.feed(chunk)

    def resize(self, cols: int, rows: int) -> None:
        """Resize the virtual screen to match the actual PTY size.
        실제 PTY 크기에 맞춰 가상 화면 리사이즈.
        """
        # pyte.Screen.resize uses (lines, columns) keyword args — be explicit.
        # pyte.Screen.resize는 (lines, columns) 키워드 인자 — 명시 호출.
        self._screen.resize(lines=rows, columns=cols)

    def get_prompt_line(self) -> str | None:
        """Return the current input prompt text (after ❯ marker), or None.
        현재 입력란 라인 텍스트 반환 (❯ 마커 다음). 없으면 None.

        When multiple ❯ markers are present (history of submitted commands),
        the LAST one is returned — that's the live input row.
        ❯ 마커가 여러 개면 마지막 것 (현재 라이브 입력란) 반환.

        Placeholder text (e.g. ``[conversation id or search term]``) drawn by
        Ink in grey is NOT stripped here — the matcher layer handles that.
        Ink가 회색으로 그리는 placeholder text는 여기서 제거하지 않음 —
        매칭 레이어에서 처리.
        """
        for line in reversed(self._screen.display):
            idx = line.find(PROMPT_MARKER)
            if idx >= 0:
                # Strip the marker, then leading space/NBSP and trailing whitespace.
                # 마커 제거 → 선행 공백/NBSP 제거 → 후행 공백 제거.
                rest = line[idx + 1 :]
                return rest.lstrip(" \xa0").rstrip()
        return None
