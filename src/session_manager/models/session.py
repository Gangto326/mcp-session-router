"""Session metadata model."""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


class SessionStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    EXPIRED = "expired"


@dataclass
class TransitionRecord:
    from_session: str | None
    to_session: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_session": self.from_session,
            "to_session": self.to_session,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransitionRecord:
        return cls(
            from_session=data.get("from_session"),
            to_session=data["to_session"],
            timestamp=data["timestamp"],
        )

    @classmethod
    def new(cls, from_session: str | None, to_session: str) -> TransitionRecord:
        return cls(
            from_session=from_session,
            to_session=to_session,
            timestamp=_utc_now_iso(),
        )


@dataclass
class SessionMetadata:
    session_id: str
    name: str
    title: str
    summary: str | None
    created_at: str
    last_accessed: str
    transitions: list[TransitionRecord] = field(default_factory=list)
    status: SessionStatus = SessionStatus.ACTIVE

    @classmethod
    def new(cls, name: str, title: str, summary: str | None = None) -> SessionMetadata:
        now = _utc_now_iso()
        return cls(
            session_id=str(uuid.uuid4()),
            name=name,
            title=title,
            summary=summary,
            created_at=now,
            last_accessed=now,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "title": self.title,
            "summary": self.summary,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "transitions": [t.to_dict() for t in self.transitions],
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMetadata:
        return cls(
            session_id=data["session_id"],
            name=data["name"],
            title=data["title"],
            summary=data.get("summary"),
            created_at=data["created_at"],
            last_accessed=data["last_accessed"],
            transitions=[
                TransitionRecord.from_dict(t) for t in data.get("transitions", [])
            ],
            status=SessionStatus(data.get("status", SessionStatus.ACTIVE.value)),
        )

    def touch(self) -> None:
        self.last_accessed = _utc_now_iso()
