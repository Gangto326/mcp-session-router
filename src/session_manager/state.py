"""In-memory state for the MCP server process."""

from __future__ import annotations

import datetime

from session_manager import debug_log
from session_manager.storage import SessionStore


class SessionManagerState:
    def __init__(self) -> None:
        self._current_session_name: str | None = None

    def get_current_session(self) -> str | None:
        return self._current_session_name

    def set_current_session(self, name: str | None) -> None:
        # Single STATE_CHANGE checkpoint — every set goes through here so
        # we can answer "when did current_session become X?" from the log.
        # 단일 STATE_CHANGE 체크포인트 — 모든 set 이 이 지점을 통과하므로
        # "current_session 이 언제 X 가 되었나?" 를 로그로 답할 수 있다.
        debug_log.log(
            "STATE_CHANGE",
            "MCP_TOOL",
            {
                "field": "current_session_name",
                "before": self._current_session_name,
                "after": name,
            },
            session=name,
        )
        self._current_session_name = name

    def resolve_from_store(
        self, store: SessionStore, active_conversation_id: str | None = None
    ) -> str | None:
        """Infer the current session when the wrapper handshake gave none.

        핸드셰이크가 현재 세션을 주지 못했을 때 세션을 추론한다.

        Prefers the session that owns *active_conversation_id* — a direct
        fact, unlike ``last_accessed``, which is only written by tool calls
        that touch a session and so can be stale for a session in active
        use. The timestamp scan stays as the last resort.

        *active_conversation_id* 를 소유한 세션을 우선한다 — ``last_accessed``
        와 달리 직접적인 사실이기 때문. ``last_accessed`` 는 세션을 건드리는
        도구 호출 시에만 기록되어 사용 중인 세션에서도 낡을 수 있다.
        타임스탬프 스캔은 최후 수단으로 유지한다.
        """
        sessions = store.list_sessions()
        if not sessions:
            debug_log.log(
                "STATE_RESOLVE",
                "SYSTEM",
                {"result": None, "reason": "store_empty"},
            )
            return None
        if active_conversation_id:
            for session in sessions:
                if active_conversation_id in session.claude_conversation_ids:
                    debug_log.log(
                        "STATE_RESOLVE",
                        "SYSTEM",
                        {
                            "result": session.name,
                            "reason": "active_conversation_match",
                            "active_conversation_id": active_conversation_id,
                        },
                        conv_id=active_conversation_id,
                        session=session.name,
                    )
                    return session.name
        latest = max(
            sessions,
            key=lambda s: datetime.datetime.fromisoformat(s.last_accessed),
        )
        debug_log.log(
            "STATE_RESOLVE",
            "SYSTEM",
            {
                "result": latest.name,
                "reason": "last_accessed_fallback",
                "last_accessed": latest.last_accessed,
                "candidates": len(sessions),
            },
        )
        return latest.name
