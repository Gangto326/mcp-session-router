"""
PreCompact hook: block auto-compact in favour of a session rollover.

PreCompact hook — auto-compact 를 차단하고 세션 롤오버로 대신한다.

Auto-compact squashes the conversation into a lossy summary right where
context quality already degrades. The rollover flow (R4) replaces it
with a written Handoff and a fresh conversation, so when Claude Code is
about to auto-compact under a ccode wrapper, this hook blocks it and
signals the wrapper that the rollover moment has arrived — a second
trigger converging with the threshold detection (R4-C1), guaranteed to
fire even if that check missed.

auto-compact 는 컨텍스트 품질이 이미 저하되는 지점에서 대화를 손실
요약으로 뭉갠다. 롤오버 흐름 (R4) 은 그것을 Handoff 문서 + 새 대화로
대체하므로, ccode 래퍼 아래에서 auto-compact 가 임박하면 이 hook 이
차단하고 래퍼에 롤오버 시점을 신호한다 — 임계 감지 (R4-C1) 와 합류하는
제2 트리거로, 그 검사가 놓쳐도 반드시 걸린다.

Decision table / 판정표:

- ``trigger == "auto"`` AND wrapper context → block + socket signal.
- ``trigger == "manual"`` → pass (the user asked for it; Plan R4-C2).
  ``trigger == "manual"`` → 통과 (사용자 의지 존중, Plan R4-C2).
- No ``SESSION_MANAGER_SOCKET`` in env (bare claude) → pass even for
  auto: blocking without a wrapper would leave no mitigation at all and
  simply let the context overflow (F4 philosophy).
  env 에 ``SESSION_MANAGER_SOCKET`` 없음 (맨몸 claude) → auto 여도 통과.
  래퍼 없이 차단하면 대안 없이 컨텍스트만 넘친다 (F4 철학).
- Any internal failure → exit 0 pass (a hook bug must never break
  compaction — same graceful-degradation contract as the other hooks).
  내부 실패 전부 → exit 0 통과 (hook 버그가 compact 를 깨면 안 된다 —
  타 hook 과 동일한 graceful degradation 계약).

Measured facts (docs/poc/R4-rollover.md §P4-b): the stdin field is
``trigger`` (NOT the planned ``compact_reason``), value ``"manual"`` on
/compact; ``{"decision": "block", "reason": ...}`` on stdout does block
and the conversation continues. The ``"auto"`` value itself is
documented but not measured (reaching real auto-compact costs hundreds
of thousands of tokens) — if the real value differs, this hook simply
never blocks, and the R4-C1 threshold detection remains the primary
rollover path.

실측 (docs/poc/R4-rollover.md §P4-b): stdin 필드는 ``trigger`` (계획의
``compact_reason`` 아님), /compact 시 값 ``"manual"``. stdout 의
``{"decision": "block", "reason": ...}`` 이 실제로 차단하고 대화는
지속된다. ``"auto"`` 실값은 문서상 값일 뿐 미실측 (실 auto-compact
도달 비용) — 실값이 다르면 이 hook 은 block 을 안 할 뿐이고, R4-C1
임계 감지가 롤오버의 주 경로로 남는다.
"""

from __future__ import annotations

import json
import os
import socket
import sys
from typing import Any

from session_manager import debug_log

# Input field names (measured: docs/poc/R4-rollover.md §P4-b).
# 입력 필드명 상수 (실측: docs/poc/R4-rollover.md §P4-b).
FIELD_TRIGGER = "trigger"
FIELD_SESSION_ID = "session_id"

TRIGGER_AUTO = "auto"

_SOCKET_ENV_VAR = "SESSION_MANAGER_SOCKET"

# One-shot signal timeout. Engineering parameter: the wrapper handles
# the message inline in its I/O loop (mark + ack, no subprocess), so
# the round-trip is bounded by loop latency — well under a second. 2s
# keeps a wide margin without stalling compaction noticeably on failure.
# 단발 신호 타임아웃. 공학 파라미터: 래퍼는 이 메시지를 I/O 루프에서
# 즉석 처리 (마킹 + ack, subprocess 없음) 하므로 왕복은 루프 지연에
# 묶인다 — 1초 미만. 2s 는 넉넉한 여유이면서 실패 시 compact 지연도
# 체감되지 않는 값.
_SIGNAL_TIMEOUT_SECONDS = 2.0

BLOCK_REASON = (
    "[session-manager] auto-compact 를 차단했습니다 — 세션 롤오버가 "
    "Handoff 문서와 새 대화로 이어갑니다."
)


def _send_rollover_signal(conversation_id: str | None) -> bool:
    """One short-lived socket exchange; True when the wrapper acked.

    단발 소켓 왕복 1회. 래퍼가 ack 하면 True.
    """
    socket_path = os.environ.get(_SOCKET_ENV_VAR, "").strip()
    if not socket_path:
        return False
    request: dict[str, Any] = {
        "client": "hook",
        "action": "rollover_signal",
        "conversation_id": conversation_id,
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
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            return
        trigger = payload.get(FIELD_TRIGGER)
        if trigger != TRIGGER_AUTO:
            debug_log.log(
                "PRE_COMPACT", "HOOK", {"trigger": trigger, "result": "pass"}
            )
            return
        if not os.environ.get(_SOCKET_ENV_VAR, "").strip():
            # Bare claude: no rollover exists to replace the compact.
            # 맨몸 claude: compact 를 대체할 롤오버가 없다.
            debug_log.log(
                "PRE_COMPACT",
                "HOOK",
                {"trigger": trigger, "result": "pass_no_wrapper"},
            )
            return
        conversation_id = payload.get(FIELD_SESSION_ID)
        acked = _send_rollover_signal(
            conversation_id if isinstance(conversation_id, str) else None
        )
        # Block regardless of the ack: the R4-C1 threshold check will
        # re-mark the rollover on the next turn end even if the signal
        # was lost, while an auto-compact that slips through destroys
        # the conversation we meant to hand off.
        # ack 여부와 무관하게 차단한다: 신호가 유실돼도 R4-C1 임계
        # 검사가 다음 턴 종료에 재마킹하지만, 통과해버린 auto-compact
        # 는 인수인계하려던 대화를 파괴한다.
        debug_log.log(
            "PRE_COMPACT",
            "HOOK",
            {"trigger": trigger, "result": "block", "signal_acked": acked},
            conv_id=conversation_id if isinstance(conversation_id, str) else None,
        )
        print(
            json.dumps(
                {"decision": "block", "reason": BLOCK_REASON},
                ensure_ascii=False,
            )
        )
    except Exception:
        # A hook bug must never break compaction. / hook 버그가 compact
        # 를 깨면 안 된다.
        return


if __name__ == "__main__":
    main()
