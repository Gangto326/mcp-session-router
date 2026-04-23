#!/usr/bin/env python3
"""
Minimal mock of Claude Code's terminal UI for integration tests.

Claude Code 터미널 UI의 최소 mock. 통합 테스트에서 실제 claude 바이너리
대신 PTY에 띄워 프롬프트 감지·텍스트 주입·submit 흐름을 검증한다.

Mimics Ink's key behaviour:
- After receiving ANY data from stdin, re-render the prompt pattern
  (just like Ink re-renders the TextInput on every keystroke)
- When \\r or \\n terminates a line, process it as a command
- /resume, /rename: ack + prompt
- /exit: ack + exit(0)
- other: echo + prompt

Ink의 핵심 동작을 흉내낸다:
- stdin에서 데이터를 받을 때마다 프롬프트 패턴을 재출력 (Ink가 매
  키 입력마다 TextInput을 재렌더링하는 것과 동일)
- \\r 또는 \\n으로 줄이 끝나면 명령으로 처리
"""

from __future__ import annotations

import os
import termios
import tty

# Same bytes the real Claude Code renders for its input prompt.
# 실제 Claude Code가 입력 프롬프트에 쓰는 바이트와 동일.
PROMPT = b"\xe2\x9d\xaf \x1b[7m \x1b[27m"


def _write(data: bytes) -> None:
    os.write(1, data)


def _process_command(cmd: str) -> bool:
    """Process a complete command. Returns False if should exit.

    완성된 명령을 처리한다. 종료해야 하면 False 반환.
    """
    cmd = cmd.strip()
    if not cmd:
        return True

    if cmd.startswith("/exit"):
        _write(b"Exiting...\n")
        return False
    elif cmd.startswith("/resume"):
        target = cmd.split(maxsplit=1)[1] if " " in cmd else ""
        _write(f"Resumed {target}\n".encode())
    elif cmd.startswith("/rename"):
        name = cmd.split(maxsplit=1)[1] if " " in cmd else ""
        _write(f"Renamed to {name}\n".encode())
    else:
        _write(f"Echo: {cmd}\n".encode())
    return True


def main() -> None:
    # Set stdin to raw mode, just like the real Claude Code (Node.js
    # process.stdin.setRawMode(true)).  Without this, the PTY line
    # discipline buffers input in canonical mode and characters don't
    # reach us until a newline.
    #
    # 실제 Claude Code처럼 stdin을 raw 모드로 설정한다.  이 없이는 PTY
    # 라인 규율이 canonical 모드로 입력을 버퍼링해 개행 전까지 문자가
    # 도착하지 않는다.
    if os.isatty(0):
        old_attrs = termios.tcgetattr(0)
        tty.setraw(0)
    else:
        old_attrs = None

    _write(PROMPT)
    line_buf = b""

    while True:
        try:
            data = os.read(0, 4096)
        except OSError:
            break
        if not data:
            break

        for byte in data:
            ch = bytes([byte])
            if ch in (b"\r", b"\n"):
                # Line complete — process command, then show fresh prompt
                # 줄 완성 — 명령 처리 후 새 프롬프트 표시
                if not _process_command(line_buf.decode("utf-8", errors="replace")):
                    if old_attrs is not None:
                        termios.tcsetattr(0, termios.TCSADRAIN, old_attrs)
                    return
                line_buf = b""
                _write(PROMPT)
            else:
                line_buf += ch
                # Re-render prompt after every keystroke (like Ink)
                # Ink처럼 매 키 입력마다 프롬프트 재렌더링
                _write(PROMPT)


if __name__ == "__main__":
    main()
