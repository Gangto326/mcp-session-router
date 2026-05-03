"""Session metadata model."""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from session_manager import debug_log


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
    # Claude Code conversation IDs that this metadata session has been
    # observed inside. Populated by session_register / session_switch /
    # session_create / session_end when the cwd's active_conversation_id
    # is known. Used by the routing harness to detect picker-driven
    # conversation transitions: if `active_conversation_id` matches one
    # of these, the routing is unambiguous.
    # 이 메타데이터 세션 안에서 관측된 Claude Code conversation id 목록.
    # cwd 의 active_conversation_id 가 알려진 시점에 도구들이 채운다. picker
    # 기반 conversation 전환 감지에 사용 — active_conversation_id 가 어느
    # 세션의 이 목록에 있으면 라우팅이 명확해진다.
    claude_conversation_ids: list[str] = field(default_factory=list)

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
            "claude_conversation_ids": list(self.claude_conversation_ids),
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
            # Default to empty list for legacy session files written before
            # this field existed.
            # 이 필드 도입 전에 작성된 세션 파일과의 호환을 위해 기본값 빈 리스트.
            claude_conversation_ids=list(data.get("claude_conversation_ids", [])),
        )

    def touch(self) -> None:
        self.last_accessed = _utc_now_iso()

    def link_conversation(self, conv_id: str) -> None:
        """Associate a Claude Code conversation id with this session (idempotent).

        Claude Code conversation id 를 이 세션에 연결한다. 이미 있으면 무시 (멱등).
        """
        if not conv_id:
            return
        before = list(self.claude_conversation_ids)
        if conv_id in before:
            debug_log.log(
                "CONV_LINK",
                "MCP_TOOL",
                {
                    "session": self.name,
                    "conv_id": conv_id,
                    "skipped": True,
                    "reason": "already_linked",
                    "list": before,
                },
                conv_id=conv_id,
                session=self.name,
            )
            return
        self.claude_conversation_ids.append(conv_id)
        debug_log.log(
            "CONV_LINK",
            "MCP_TOOL",
            {
                "session": self.name,
                "conv_id": conv_id,
                "skipped": False,
                "before": before,
                "after": list(self.claude_conversation_ids),
            },
            conv_id=conv_id,
            session=self.name,
        )
