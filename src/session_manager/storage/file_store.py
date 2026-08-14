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

    def resolve_active_successor(self, name: str) -> str | None:
        """Follow a retired session's successor chain to its living end.

        retired 세션의 successor 사슬을 살아 있는 끝까지 따라간다 (R4-C5).

        A → B → C is followed to the first ACTIVE session; visited names
        guard against cycles. On success the traversed retired nodes are
        compressed — their successor is rewritten to the final name
        (F15-locked mutate) so the next lookup is one hop. Returns None
        when the chain dead-ends (no successor, missing session, cycle)
        — the caller aborts the switch with a notice.

        A → B → C 를 첫 ACTIVE 세션까지 따라간다. 방문 집합이 순환을
        막는다. 성공 시 경유한 retired 노드들의 successor 를 최종 이름
        으로 압축해 (F15 잠금 mutate) 다음 조회를 1홉으로 만든다. 사슬이
        막히면 (successor 없음·세션 소실·순환) None — 호출자는 안내 후
        전환을 중단한다.
        """
        from session_manager.models.session import SessionStatus

        visited: list[str] = []
        current = name
        while True:
            if current in visited:
                debug_log.log(
                    "SUCCESSOR_RESOLVE",
                    "WRAPPER",
                    {"start": name, "result": "cycle", "visited": visited},
                )
                return None
            visited.append(current)
            session = self.load_session_by_name(current)
            if session is None:
                debug_log.log(
                    "SUCCESSOR_RESOLVE",
                    "WRAPPER",
                    {"start": name, "result": "missing", "at": current},
                )
                return None
            if session.status != SessionStatus.RETIRED:
                final = current
                break
            successor = session.retired.successor if session.retired else None
            if not successor:
                debug_log.log(
                    "SUCCESSOR_RESOLVE",
                    "WRAPPER",
                    {"start": name, "result": "no_heir", "at": current},
                )
                return None
            current = successor
        # Path compression — every traversed retired node points to the
        # living end afterwards.
        # 경로 압축 — 경유한 retired 노드가 이후 살아 있는 끝을 직접
        # 가리킨다.
        for node in visited[:-1]:
            self.mutate_session_by_name(
                node,
                lambda s: setattr(s.retired, "successor", final)
                if s.retired is not None
                else None,
            )
        debug_log.log(
            "SUCCESSOR_RESOLVE",
            "WRAPPER",
            {"start": name, "result": final, "hops": len(visited) - 1},
        )
        return final

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
