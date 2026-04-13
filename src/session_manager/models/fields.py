"""Static Field model (project-wide shared context + user environment)."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


@dataclass
class StaticField:
    project_context: str = ""
    conventions: str = ""
    project_map: dict[str, str] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""

    @classmethod
    def new(cls) -> StaticField:
        return cls(updated_at=_utc_now_iso())

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_context": self.project_context,
            "conventions": self.conventions,
            "project_map": dict(self.project_map),
            "variables": dict(self.variables),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StaticField:
        return cls(
            project_context=data.get("project_context", ""),
            conventions=data.get("conventions", ""),
            project_map=dict(data.get("project_map", {})),
            variables=dict(data.get("variables", {})),
            updated_at=data.get("updated_at", ""),
        )

    def touch(self) -> None:
        self.updated_at = _utc_now_iso()
