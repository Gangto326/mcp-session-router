"""In-memory state for the MCP server process."""

from __future__ import annotations

import datetime

from session_manager.storage import SessionStore


class SessionManagerState:
    def __init__(self) -> None:
        self._current_session_name: str | None = None

    def get_current_session(self) -> str | None:
        return self._current_session_name

    def set_current_session(self, name: str | None) -> None:
        self._current_session_name = name

    def resolve_from_store(self, store: SessionStore) -> str | None:
        sessions = store.list_sessions()
        if not sessions:
            return None
        latest = max(
            sessions,
            key=lambda s: datetime.datetime.fromisoformat(s.last_accessed),
        )
        return latest.name
