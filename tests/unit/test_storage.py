"""Tests for the JSON file-backed storage layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from session_manager.models import Config, SessionMetadata, SessionStatus, StaticField
from session_manager.storage import (
    ConfigStore,
    FieldStore,
    ProjectContextStore,
    SessionStore,
)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    return tmp_path


class TestSessionStore:
    def test_list_sessions_when_dir_missing_returns_empty(
        self, project_root: Path
    ) -> None:
        store = SessionStore(project_root)
        assert store.list_sessions() == []

    def test_load_session_when_missing_returns_none(self, project_root: Path) -> None:
        store = SessionStore(project_root)
        assert store.load_session("nonexistent-id") is None

    def test_load_session_by_name_when_missing_returns_none(
        self, project_root: Path
    ) -> None:
        store = SessionStore(project_root)
        assert store.load_session_by_name("missing") is None

    def test_delete_session_when_missing_is_idempotent(
        self, project_root: Path
    ) -> None:
        store = SessionStore(project_root)
        store.delete_session("nonexistent-id")  # should not raise

    def test_save_then_load_session_roundtrip(self, project_root: Path) -> None:
        store = SessionStore(project_root)
        session = SessionMetadata.new(name="auth-fix", title="인증 수정")
        store.save_session(session)

        loaded = store.load_session(session.session_id)
        assert loaded == session

    def test_save_then_load_by_name(self, project_root: Path) -> None:
        store = SessionStore(project_root)
        session = SessionMetadata.new(name="auth-fix", title="인증 수정")
        store.save_session(session)

        assert store.load_session_by_name("auth-fix") == session

    def test_save_without_init_project_creates_directory(
        self, project_root: Path
    ) -> None:
        store = SessionStore(project_root)
        assert not (project_root / ".session-manager" / "sessions").exists()

        session = SessionMetadata.new(name="auth-fix", title="Foo")
        store.save_session(session)

        assert (project_root / ".session-manager" / "sessions").exists()
        assert store.load_session(session.session_id) == session

    def test_save_overwrites_existing_session(self, project_root: Path) -> None:
        store = SessionStore(project_root)
        session = SessionMetadata.new(name="auth-fix", title="원본")
        store.save_session(session)

        session.title = "수정됨"
        session.summary = "새 요약"
        store.save_session(session)

        loaded = store.load_session(session.session_id)
        assert loaded is not None
        assert loaded.title == "수정됨"
        assert loaded.summary == "새 요약"

    def test_list_sessions_returns_multiple_entries_deterministically(
        self, project_root: Path
    ) -> None:
        store = SessionStore(project_root)
        a = SessionMetadata.new(name="a", title="A")
        b = SessionMetadata.new(name="b", title="B")
        c = SessionMetadata.new(name="c", title="C")
        for s in (b, a, c):
            store.save_session(s)

        listed = store.list_sessions()
        assert len(listed) == 3
        # 동일 입력에 대해 반복 호출해도 순서가 동일
        assert store.list_sessions() == listed

    def test_delete_session_removes_file(self, project_root: Path) -> None:
        store = SessionStore(project_root)
        session = SessionMetadata.new(name="auth-fix", title="Foo")
        store.save_session(session)
        assert store.load_session(session.session_id) is not None

        store.delete_session(session.session_id)
        assert store.load_session(session.session_id) is None

    def test_init_project_creates_sessions_dir(self, project_root: Path) -> None:
        store = SessionStore(project_root)
        store.init_project()
        assert (project_root / ".session-manager" / "sessions").is_dir()


class TestResolveActiveSuccessor:
    """R4-C5 successor chain resolution + path compression.

    R4-C5 후계 사슬 해석 + 경로 압축.
    """

    def _save(
        self,
        store: SessionStore,
        name: str,
        *,
        retired_to: str | None = None,
        retired: bool = False,
    ) -> SessionMetadata:
        session = SessionMetadata.new(name=name, title=name)
        if retired or retired_to is not None:
            session.retire("manual", successor=retired_to)
        store.save_session(session)
        return session

    def test_active_session_resolves_to_itself(self, project_root: Path) -> None:
        store = SessionStore(project_root)
        self._save(store, "a")
        assert store.resolve_active_successor("a") == "a"

    def test_single_hop(self, project_root: Path) -> None:
        store = SessionStore(project_root)
        self._save(store, "a", retired_to="b")
        self._save(store, "b")
        assert store.resolve_active_successor("a") == "b"

    def test_multi_hop_chain(self, project_root: Path) -> None:
        store = SessionStore(project_root)
        self._save(store, "a", retired_to="b")
        self._save(store, "b", retired_to="c")
        self._save(store, "c")
        assert store.resolve_active_successor("a") == "c"

    def test_multi_hop_compresses_traversed_nodes(self, project_root: Path) -> None:
        store = SessionStore(project_root)
        self._save(store, "a", retired_to="b")
        self._save(store, "b", retired_to="c")
        self._save(store, "c")
        store.resolve_active_successor("a")
        # Both traversed retired nodes now point straight at the end.
        # 경유한 retired 노드 둘 다 이제 끝을 직접 가리킨다.
        for name in ("a", "b"):
            reloaded = store.load_session_by_name(name)
            assert reloaded is not None
            assert reloaded.status is SessionStatus.RETIRED  # still retired
            assert reloaded.retired is not None
            assert reloaded.retired.successor == "c"

    def test_cycle_returns_none(self, project_root: Path) -> None:
        store = SessionStore(project_root)
        self._save(store, "a", retired_to="b")
        self._save(store, "b", retired_to="a")
        assert store.resolve_active_successor("a") is None

    def test_no_heir_returns_none(self, project_root: Path) -> None:
        store = SessionStore(project_root)
        self._save(store, "a", retired=True)
        assert store.resolve_active_successor("a") is None

    def test_missing_session_returns_none(self, project_root: Path) -> None:
        store = SessionStore(project_root)
        assert store.resolve_active_successor("ghost") is None

    def test_chain_to_missing_session_returns_none(self, project_root: Path) -> None:
        store = SessionStore(project_root)
        self._save(store, "a", retired_to="gone")
        assert store.resolve_active_successor("a") is None

    def test_dead_end_does_not_compress(self, project_root: Path) -> None:
        # A failed resolution must leave the chain untouched — /revive of
        # the missing link should heal the original topology.
        # 해석 실패는 사슬을 건드리지 않아야 한다 — 끊긴 고리를 /revive
        # 하면 원래 위상이 그대로 살아나야 한다.
        store = SessionStore(project_root)
        self._save(store, "a", retired_to="b")
        self._save(store, "b", retired_to="gone")
        assert store.resolve_active_successor("a") is None
        reloaded = store.load_session_by_name("a")
        assert reloaded is not None
        assert reloaded.retired is not None
        assert reloaded.retired.successor == "b"


class TestFieldStore:
    def test_load_when_missing_returns_fresh_default(
        self, project_root: Path
    ) -> None:
        store = FieldStore(project_root)
        loaded = store.load_static()
        assert loaded.project_context == ""
        assert loaded.conventions == ""
        assert loaded.project_map == {}
        assert loaded.variables == {}
        assert loaded.updated_at != ""  # StaticField.new() sets this

    def test_save_then_load_roundtrip(self, project_root: Path) -> None:
        store = FieldStore(project_root)
        field = StaticField.new()
        field.project_context = "React + TypeScript 모노레포"
        field.project_map = {"src/auth/": "인증 모듈"}
        field.variables = {
            "환경변수": ["DATABASE_URL", "OPENAI_API_KEY"],
            "API 키": {"OpenAI": "sk-..."},
        }
        store.save_static(field)

        assert store.load_static() == field

    def test_save_creates_directory_if_missing(self, project_root: Path) -> None:
        store = FieldStore(project_root)
        assert not (project_root / ".session-manager").exists()

        store.save_static(StaticField.new())
        assert (project_root / ".session-manager" / "static-field.json").is_file()


class TestConfigStore:
    def test_load_when_missing_returns_none(self, project_root: Path) -> None:
        store = ConfigStore(project_root)
        assert store.load_config() is None

    def test_save_then_load_roundtrip(self, project_root: Path) -> None:
        store = ConfigStore(project_root)
        config = Config(socket_path="/tmp/session-manager-xyz.sock", cleanup_period_days=14)
        store.save_config(config)

        assert store.load_config() == config

    def test_save_with_default_cleanup_period(self, project_root: Path) -> None:
        store = ConfigStore(project_root)
        config = Config(socket_path="/tmp/x.sock")
        store.save_config(config)

        loaded = store.load_config()
        assert loaded is not None
        assert loaded.cleanup_period_days == 30


class TestProjectContextStore:
    def test_exists_false_when_missing(self, project_root: Path) -> None:
        store = ProjectContextStore(project_root)
        assert store.exists() is False

    def test_read_when_missing_raises_file_not_found(
        self, project_root: Path
    ) -> None:
        store = ProjectContextStore(project_root)
        with pytest.raises(FileNotFoundError):
            store.read()

    def test_write_then_exists_and_read(self, project_root: Path) -> None:
        store = ProjectContextStore(project_root)
        content = "# 프로젝트 맥락\n\n- auth: 인증 모듈\n"
        store.write(content)

        assert store.exists() is True
        assert store.read() == content

    def test_write_creates_directory_if_missing(self, project_root: Path) -> None:
        store = ProjectContextStore(project_root)
        assert not (project_root / ".session-manager").exists()

        store.write("hello")
        assert (project_root / ".session-manager" / "project-context.md").is_file()


class TestAtomicWrite:
    def test_no_tmp_residue_after_session_save(self, project_root: Path) -> None:
        store = SessionStore(project_root)
        store.save_session(SessionMetadata.new(name="a", title="A"))
        assert list(project_root.rglob("*.tmp")) == []

    def test_no_tmp_residue_after_field_save(self, project_root: Path) -> None:
        store = FieldStore(project_root)
        store.save_static(StaticField.new())
        assert list(project_root.rglob("*.tmp")) == []

    def test_no_tmp_residue_after_config_save(self, project_root: Path) -> None:
        store = ConfigStore(project_root)
        store.save_config(Config(socket_path="/tmp/x.sock"))
        assert list(project_root.rglob("*.tmp")) == []

    def test_no_tmp_residue_after_project_context_write(
        self, project_root: Path
    ) -> None:
        store = ProjectContextStore(project_root)
        store.write("hello")
        assert list(project_root.rglob("*.tmp")) == []

    def test_leftover_tmp_does_not_break_load(self, project_root: Path) -> None:
        """크래시로 .tmp가 남아도 list_sessions는 정상 파일만 반환해야 한다."""
        store = SessionStore(project_root)
        session = SessionMetadata.new(name="a", title="A")
        store.save_session(session)

        # 이전 저장 도중 크래시 시뮬레이션: 현재 세션의 .tmp 잔재 생성
        sessions_dir = project_root / ".session-manager" / "sessions"
        leftover = sessions_dir / f"{session.session_id}.json.tmp"
        leftover.write_text("partial-bytes", encoding="utf-8")

        listed = store.list_sessions()
        assert len(listed) == 1
        assert listed[0] == session


class TestMutateSession:
    """Locked read-modify-write (F15) — lost-update prevention.

    잠금 하의 read-modify-write (F15) — 동시 저장 유실 방어 테스트.
    """

    def _make(self, tmp_path: Path) -> tuple[SessionStore, SessionMetadata]:
        store = SessionStore(tmp_path)
        store.init_project()
        session = SessionMetadata.new(name="s1", title="T")
        store.save_session(session)
        return store, session

    def test_mutator_applied_and_saved(self, tmp_path: Path) -> None:
        store, session = self._make(tmp_path)

        result = store.mutate_session(
            session.session_id, lambda s: s.requirements.append("테스트 필수")
        )

        assert result is not None
        reloaded = store.load_session(session.session_id)
        assert reloaded is not None
        assert reloaded.requirements == ["테스트 필수"]

    def test_missing_session_returns_none(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        store.init_project()
        assert store.mutate_session("ghost", lambda s: None) is None
        assert store.mutate_session_by_name("ghost", lambda s: None) is None

    def test_mutate_by_name(self, tmp_path: Path) -> None:
        store, session = self._make(tmp_path)
        result = store.mutate_session_by_name("s1", lambda s: s.requirements.append("r"))
        assert result is not None
        assert result.session_id == session.session_id

    def test_concurrent_mutations_not_lost(self, tmp_path: Path) -> None:
        # 두 스레드가 같은 세션에 각각 N회 append — 유실이 있으면 2N 미만이 된다.
        # (flock 은 open file description 단위라 동일 프로세스의 별도 open
        # 끼리도 상호 배제된다 — 프로세스 간 배제와 같은 메커니즘.)
        import threading

        store, session = self._make(tmp_path)
        n = 50

        def worker(tag: str) -> None:
            for i in range(n):
                item = f"{tag}-{i}"
                store.mutate_session(
                    session.session_id,
                    lambda s, item=item: s.requirements.append(item),
                )

        t1 = threading.Thread(target=worker, args=("a",))
        t2 = threading.Thread(target=worker, args=("b",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        reloaded = store.load_session(session.session_id)
        assert reloaded is not None
        assert len(reloaded.requirements) == 2 * n

    def test_delete_removes_lock_sidecar(self, tmp_path: Path) -> None:
        store, session = self._make(tmp_path)
        store.mutate_session(session.session_id, lambda s: None)
        lock_path = (
            tmp_path / ".session-manager" / "sessions" / f"{session.session_id}.json.lock"
        )
        assert lock_path.exists()

        store.delete_session(session.session_id)
        assert not lock_path.exists()
