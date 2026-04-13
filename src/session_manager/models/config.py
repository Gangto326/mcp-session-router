"""Runtime configuration model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_CLEANUP_PERIOD_DAYS = 30


@dataclass
class Config:
    socket_path: str
    cleanup_period_days: int = DEFAULT_CLEANUP_PERIOD_DAYS

    def to_dict(self) -> dict[str, Any]:
        return {
            "socket_path": self.socket_path,
            "cleanup_period_days": self.cleanup_period_days,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        return cls(
            socket_path=data["socket_path"],
            cleanup_period_days=data.get(
                "cleanup_period_days", DEFAULT_CLEANUP_PERIOD_DAYS
            ),
        )
