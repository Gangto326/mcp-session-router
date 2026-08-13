"""
Stop hook: the contract-based turn-end signal.

Stop hook — 계약 기반 턴 종료 신호.

Claude Code fires this hook every time the main agent finishes a
response (measured: docs/poc/R4-rollover.md §Stop hook 실측 — fires per
turn in interactive TUI, including short turns that render no busy
hint). The hook forwards one short-lived socket message to the wrapper
carrying the turn's conversation id and the response body
(``last_assistant_message``), which the wrapper uses as:

- the PRIMARY turn-end signal (screen heuristics and context.json
  observation demote to fallbacks for hook-declined users),
- the handoff validation input (no transcript read — the transcript is
  measurably NOT yet flushed when Stop fires, but the payload already
  carries the response),
- the rollover-finalize entry confirmation (the successor's first turn
  delivers its conversation id directly — no directory polling).

Claude Code 는 메인 에이전트가 응답을 마칠 때마다 이 hook 을 발동한다
(실측: docs/poc/R4-rollover.md §Stop hook 실측 — 대화형 TUI 매 턴,
바쁨 힌트를 안 그리는 짧은 턴 포함). hook 은 그 턴의 conversation id
와 응답 본문 (``last_assistant_message``) 을 단발 소켓 메시지로 래퍼에
전달하고, 래퍼는 이를 다음 용도로 쓴다:

- **주** 턴 종료 신호 (화면 휴리스틱·context.json 관찰은 hook 미동의
  사용자용 폴백으로 강등),
- handoff 검증 입력 (transcript 무읽기 — Stop 발동 시점에 transcript
  는 실측상 아직 flush 전이지만 payload 에는 응답이 이미 실려 있다),
- 롤오버 finalize 진입 확인 (후계의 첫 턴이 conversation id 를 직접
  전달 — 디렉토리 폴링 불필요).

Graceful degradation: outside a wrapper (no ``SESSION_MANAGER_SOCKET``)
or on any failure, exit 0 silently — same contract as every other hook.
래퍼 밖 (``SESSION_MANAGER_SOCKET`` 부재) 이거나 어떤 실패든 조용히
exit 0 — 타 hook 과 동일 계약.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from typing import Any

from session_manager import debug_log

# Input field names (measured: docs/poc/R4-rollover.md §Stop hook 실측).
# 입력 필드명 상수 (실측: docs/poc/R4-rollover.md §Stop hook 실측).
FIELD_SESSION_ID = "session_id"
FIELD_LAST_ASSISTANT_MESSAGE = "last_assistant_message"

_SOCKET_ENV_VAR = "SESSION_MANAGER_SOCKET"

# One-shot signal timeout. Engineering parameter, same derivation as the
# PreCompact hook's: the wrapper handles the message inline in its I/O
# loop (bookkeeping only, no subprocess), so the round-trip is bounded
# by loop latency — well under a second; 2s is a wide margin.
# 단발 신호 타임아웃. PreCompact hook 과 같은 도출의 공학 파라미터 —
# 래퍼는 I/O 루프에서 즉석 처리 (부기뿐, subprocess 없음) 하므로 왕복은
# 루프 지연에 묶인다 (1초 미만). 2s 는 넉넉한 여유.
_SIGNAL_TIMEOUT_SECONDS = 2.0


def _send_turn_end(conversation_id: str, last_message: str) -> bool:
    """One short-lived socket exchange; True when the wrapper acked.

    단발 소켓 왕복 1회. 래퍼가 ack 하면 True.
    """
    socket_path = os.environ.get(_SOCKET_ENV_VAR, "").strip()
    if not socket_path:
        return False
    request: dict[str, Any] = {
        "client": "hook",
        "action": "turn_end",
        "conversation_id": conversation_id,
        "last_assistant_message": last_message,
    }
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(_SIGNAL_TIMEOUT_SECONDS)
            sock.connect(socket_path)
            sock.sendall(
                (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
            )
            buffer = b""
            while b"\n" not in buffer:
                chunk = sock.recv(4096)
                if not chunk:
                    return False
                buffer += chunk
    except OSError:
        return False
    return True


def main() -> None:
    try:
        if not os.environ.get(_SOCKET_ENV_VAR, "").strip():
            # Bare claude: no wrapper to signal (F4 philosophy).
            # 맨몸 claude — 신호 보낼 래퍼가 없다 (F4 철학).
            return
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            return
        conversation_id = payload.get(FIELD_SESSION_ID)
        if not isinstance(conversation_id, str) or not conversation_id:
            return
        last_message = payload.get(FIELD_LAST_ASSISTANT_MESSAGE)
        acked = _send_turn_end(
            conversation_id,
            last_message if isinstance(last_message, str) else "",
        )
        debug_log.log(
            "STOP_HOOK",
            "HOOK",
            {"signal_acked": acked},
            conv_id=conversation_id,
        )
    except Exception:
        # A hook bug must never disturb the turn. / hook 버그가 턴을
        # 어지럽혀선 안 된다.
        return


if __name__ == "__main__":
    main()
