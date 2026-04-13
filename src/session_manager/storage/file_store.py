"""JSON file-backed storage layer for session metadata and configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from session_manager.models import Config, SessionMetadata, StaticField

_SESSION_MANAGER_DIRNAME = ".session-manager"
_SESSIONS_DIRNAME = "sessions"
_STATIC_FIELD_FILENAME = "static-field.json"
_CONFIG_FILENAME = "config.json"
_PROJECT_CONTEXT_FILENAME = "project-context.md"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _dump_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class SessionStore:
    def __init__(self, project_path: Path) -> None:
        self._root = Path(project_path) / _SESSION_MANAGER_DIRNAME
        self._sessions_dir = self._root / _SESSIONS_DIRNAME

    def init_project(self) -> None:
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    def save_session(self, session: SessionMetadata) -> None:
        path = self._sessions_dir / f"{session.session_id}.json"
        _atomic_write_text(path, _dump_json(session.to_dict()))

    def load_session(self, session_id: str) -> SessionMetadata | None:
        path = self._sessions_dir / f"{session_id}.json"
        if not path.exists():
            return None
        return SessionMetadata.from_dict(_load_json(path))

    def load_session_by_name(self, name: str) -> SessionMetadata | None:
        for session in self.list_sessions():
            if session.name == name:
                return session
        return None

    def list_sessions(self) -> list[SessionMetadata]:
        if not self._sessions_dir.exists():
            return []
        results: list[SessionMetadata] = []
        for path in sorted(self._sessions_dir.glob("*.json")):
            results.append(SessionMetadata.from_dict(_load_json(path)))
        return results

    def delete_session(self, session_id: str) -> None:
        path = self._sessions_dir / f"{session_id}.json"
        path.unlink(missing_ok=True)


class FieldStore:
    def __init__(self, project_path: Path) -> None:
        self._path = (
            Path(project_path) / _SESSION_MANAGER_DIRNAME / _STATIC_FIELD_FILENAME
        )

    def load_static(self) -> StaticField:
        if not self._path.exists():
            return StaticField.new()
        return StaticField.from_dict(_load_json(self._path))

    def save_static(self, static_field: StaticField) -> None:
        _atomic_write_text(self._path, _dump_json(static_field.to_dict()))


class ConfigStore:
    def __init__(self, project_path: Path) -> None:
        self._path = Path(project_path) / _SESSION_MANAGER_DIRNAME / _CONFIG_FILENAME

    def load_config(self) -> Config | None:
        if not self._path.exists():
            return None
        return Config.from_dict(_load_json(self._path))

    def save_config(self, config: Config) -> None:
        _atomic_write_text(self._path, _dump_json(config.to_dict()))


class ProjectContextStore:
    def __init__(self, project_path: Path) -> None:
        self._path = (
            Path(project_path) / _SESSION_MANAGER_DIRNAME / _PROJECT_CONTEXT_FILENAME
        )

    def exists(self) -> bool:
        return self._path.exists()

    def read(self) -> str:
        return self._path.read_text(encoding="utf-8")

    def write(self, content: str) -> None:
        _atomic_write_text(self._path, content)
