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

    def resolve_from_store(self, store: SessionStore) -> str | None:
        sessions = store.list_sessions()
        if not sessions:
            debug_log.log(
                "STATE_RESOLVE",
                "SYSTEM",
                {"result": None, "reason": "store_empty"},
            )
            return None
        latest = max(
            sessions,
            key=lambda s: datetime.datetime.fromisoformat(s.last_accessed),
        )
        debug_log.log(
            "STATE_RESOLVE",
            "SYSTEM",
            {
                "result": latest.name,
                "last_accessed": latest.last_accessed,
                "candidates": len(sessions),
            },
        )
        return latest.name
