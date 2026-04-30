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
from wcwidth import wcwidth

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

    def _safe_display(self) -> list[str]:
        """Render the virtual screen as text rows, robust to pyte stub bugs.
        가상 화면을 텍스트 행 리스트로 렌더 — pyte stub 버그에 견고.

        Mimics pyte.Screen.display, but guards against orphan wide-char
        stubs (empty data left over when a wide char is partially
        overwritten or clipped). pyte 0.8.2 raises IndexError in that
        case (see screens.py:241 — ``wcwidth(char[0])`` on empty string).
        Orphan stubs are rendered as a single space to preserve column
        indexing for downstream slicing/matching.
        pyte.Screen.display 모방 — 외톨이 wide-char stub(빈 문자열)을
        안전 처리. pyte 0.8.2는 이 경우 IndexError 발생
        (screens.py:241 — 빈 문자열에 ``wcwidth(char[0])``). 외톨이 stub은
        단일 공백으로 렌더해 컬럼 인덱싱을 보존, 후속 슬라이싱/매칭이 안 깨짐.
        """
        rows: list[str] = []
        for y in range(self._screen.lines):
            line = self._screen.buffer[y]
            cells: list[str] = []
            skip_next = False
            for x in range(self._screen.columns):
                if skip_next:
                    skip_next = False
                    continue
                char = line[x].data
                if not char:
                    # Orphan stub from a removed/clipped wide char.
                    # 사라진/잘린 wide char가 남긴 외톨이 stub.
                    cells.append(" ")
                    continue
                skip_next = wcwidth(char[0]) == 2
                cells.append(char)
            rows.append("".join(cells))
        return rows

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
        for line in reversed(self._safe_display()):
            idx = line.find(PROMPT_MARKER)
            if idx >= 0:
                # Strip the marker, then leading space/NBSP and trailing whitespace.
                # 마커 제거 → 선행 공백/NBSP 제거 → 후행 공백 제거.
                rest = line[idx + 1 :]
                return rest.lstrip(" \xa0").rstrip()
        return None

    def contains(self, needle: str) -> bool:
        """Return True if any row of the virtual screen contains *needle*.
        가상 화면 어느 행에라도 ``needle`` 부분 문자열이 있으면 True.

        Used to detect confirmation prompts (e.g. ``I am using this for local
        development``) so the wrapper can auto-accept them.
        confirmation prompt 감지에 사용 — 발견되면 wrapper가 자동 승인.
        """
        return any(needle in line for line in self._safe_display())
