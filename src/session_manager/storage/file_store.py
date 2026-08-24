"""JSON file-backed storage layer for session metadata and configuration."""

from __future__ import annotations

import fcntl
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from session_manager import debug_log
from session_manager.models import Config, SessionMetadata, StaticField

_SESSION_MANAGER_DIRNAME = ".session-manager"
_SESSIONS_DIRNAME = "sessions"
_STATIC_FIELD_FILENAME = "static-field.json"
_CONFIG_FILENAME = "config.json"
_PROJECT_CONTEXT_FILENAME = "project-context.md"


def _atomic_write_text(path: Path, text: str) -> None:
    # Single STORAGE_SAVE checkpoint — every disk write under this layer
    # passes through here so the log captures the "what file changed when".
    # 단일 STORAGE_SAVE 체크포인트 — 이 레이어의 모든 디스크 쓰기가 이
    # 지점을 통과하므로 "어떤 파일이 언제 바뀌었는가" 가 로그에 잡힌다.
    debug_log.log(
        "STORAGE_SAVE",
        "MCP_TOOL",
        {
            "path": str(path),
            "len": len(text),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _dump_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    debug_log.log(
        "STORAGE_LOAD",
        "MCP_TOOL",
        {"path": str(path), "len": len(text)},
    )
    return json.loads(text)


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

    def mutate_session(
        self,
        session_id: str,
        mutator: Callable[[SessionMetadata], None],
    ) -> SessionMetadata | None:
        """
        Load-modify-save under an exclusive cross-process lock (F15).

        The MCP server process and the wrapper's worker threads both
        load-modify-save session files; without a lock, concurrent saves
        silently drop one side's field changes (atomic replace only
        prevents torn files, not lost updates). ``flock`` on a per-session
        sidecar lock file makes the whole read-modify-write one critical
        section across processes. Hold time is milliseconds, so blocking
        acquisition is fine. Returns the saved session, or None if the
        session does not exist.

        배타적 프로세스 간 잠금 아래에서 load-modify-save 를 수행한다 (F15).

        MCP 서버 프로세스와 래퍼 워커 스레드가 같은 세션 파일을
        load-modify-save 하는데, 잠금이 없으면 동시 저장에서 한쪽의 필드
        변경이 조용히 유실된다 (atomic replace 는 파일 깨짐만 막는다).
        세션별 사이드카 잠금 파일에 대한 ``flock`` 이 read-modify-write
        전체를 프로세스 간 하나의 임계 구역으로 만든다. 보유 시간이 ms
        단위라 블로킹 획득으로 충분하다. 저장된 세션을 반환하고, 세션이
        없으면 None.
        """
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self._sessions_dir / f"{session_id}.json.lock"
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                session = self.load_session(session_id)
                if session is None:
                    return None
                mutator(session)
                self.save_session(session)
                return session
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def mutate_session_by_name(
        self,
        name: str,
        mutator: Callable[[SessionMetadata], None],
    ) -> SessionMetadata | None:
        """
        Name-keyed variant of :meth:`mutate_session`. The name→id lookup
        happens outside the lock — the binding is stable (rename rewrites
        the same file), so the id-keyed critical section still protects
        the read-modify-write.

        :meth:`mutate_session` 의 이름 기반 변형. 이름→id 조회는 잠금
        밖에서 일어나지만 그 결합은 안정적이므로 (rename 도 같은 파일을
        다시 쓴다) id 기반 임계 구역이 read-modify-write 를 그대로
        보호한다.
        """
        session = self.load_session_by_name(name)
        if session is None:
            return None
        return self.mutate_session(session.session_id, mutator)

    def delete_session(self, session_id: str) -> None:
        path = self._sessions_dir / f"{session_id}.json"
        existed = path.exists()
        path.unlink(missing_ok=True)
        # Remove the F15 sidecar lock as well.
        # F15 사이드카 잠금 파일도 함께 제거.
        (self._sessions_dir / f"{session_id}.json.lock").unlink(missing_ok=True)
        debug_log.log(
            "STORAGE_DELETE",
            "MCP_TOOL",
            {
                "path": str(path),
                "session_id": session_id,
                "existed": existed,
            },
        )


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

    def mutate_static(
        self, mutator: Callable[[StaticField], bool]
    ) -> StaticField:
        """
        Load-modify-save under an exclusive cross-process lock (F15 twin).

        update_static advertises multi-session writers ("어떤 세션에서든
        갱신"), which is exactly the lost-update scenario the F15 lock on
        session files exists for — atomic replace only prevents torn
        files, not one side's changes silently vanishing. Same sidecar
        flock pattern as ``SessionStore.mutate_session``. The mutator
        returns True to save; False skips the write entirely (no
        timestamp churn on no-op calls).

        배타적 프로세스 간 잠금 아래 load-modify-save (F15 쌍둥이).

        update_static 은 다중 세션 기록자를 광고하는데 ("어떤 세션에서든
        갱신"), 그것이 정확히 세션 파일 F15 잠금이 막는 갱신 유실
        시나리오다 — atomic replace 는 파일 깨짐만 막고 한쪽 변경의
        조용한 소실은 못 막는다. ``SessionStore.mutate_session`` 과 같은
        사이드카 flock 패턴. mutator 가 True 를 반환하면 저장, False 면
        쓰기를 통째로 생략한다 (no-op 호출의 타임스탬프 공회전 방지).
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self._path.with_suffix(".json.lock")
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                static = self.load_static()
                if mutator(static):
                    self.save_static(static)
                return static
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


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
