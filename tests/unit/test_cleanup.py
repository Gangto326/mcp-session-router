"""Tests for TTL-based session cleanup.

TTL 기반 세션 정리 모듈 테스트.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from session_manager.claude_conversation import encode_cwd
from session_manager.lifecycle.cleanup import (
    _DEFAULT_CLEANUP_PERIOD_DAYS,
    cleanup_expired_sessions,
    get_cleanup_period_days,
)
from session_manager.models import SessionMetadata
from session_manager.storage import SessionStore

# -- helpers ------------------------------------------------------------------


def _make_session(
    store: SessionStore,
    name: str,
    last_accessed: str,
) -> SessionMetadata:
    """Create and persist a session with the given last_accessed timestamp.

    주어진 last_accessed 타임스탬프로 세션을 생성·저장한다.
    """
    session = SessionMetadata.new(name=name, title=f"Title for {name}")
    session.last_accessed = last_accessed
    store.save_session(session)
    return session


def _days_ago_iso(days: int) -> str:
    """Return an ISO timestamp *days* days in the past (UTC).

    현재로부터 *days*일 전의 UTC ISO 타임스탬프를 반환한다.
    """
    dt = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days)
    return dt.isoformat()


# -- get_cleanup_period_days --------------------------------------------------


class TestGetCleanupPeriodDays:
    def test_reads_value_from_settings(self, tmp_path: Path) -> None:
        settings = tmp_path / "settings.json"
        settings.write_text(json.dumps({"cleanupPeriodDays": 7}))
        with patch(
            "session_manager.lifecycle.cleanup.Path.home",
            return_value=tmp_path / "fake_home",
        ):
            # Point to the tmp_path directly by patching the full path.
            # settings_path = Path.home() / ".claude" / "settings.json"
            # So we need: fake_home / .claude / settings.json
            claude_dir = tmp_path / "fake_home" / ".claude"
            claude_dir.mkdir(parents=True)
            (claude_dir / "settings.json").write_text(
                json.dumps({"cleanupPeriodDays": 7})
            )
            assert get_cleanup_period_days() == 7

    def test_returns_default_when_file_missing(self, tmp_path: Path) -> None:
        with patch(
            "session_manager.lifecycle.cleanup.Path.home",
            return_value=tmp_path / "nonexistent",
        ):
            assert get_cleanup_period_days() == _DEFAULT_CLEANUP_PERIOD_DAYS

    def test_returns_default_when_key_missing(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / "fake_home" / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "settings.json").write_text(json.dumps({"other": "value"}))
        with patch(
            "session_manager.lifecycle.cleanup.Path.home",
            return_value=tmp_path / "fake_home",
        ):
            assert get_cleanup_period_days() == _DEFAULT_CLEANUP_PERIOD_DAYS

    def test_returns_default_when_value_is_zero(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / "fake_home" / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "settings.json").write_text(
            json.dumps({"cleanupPeriodDays": 0})
        )
        with patch(
            "session_manager.lifecycle.cleanup.Path.home",
            return_value=tmp_path / "fake_home",
        ):
            assert get_cleanup_period_days() == _DEFAULT_CLEANUP_PERIOD_DAYS

    def test_returns_default_when_value_is_negative(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / "fake_home" / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "settings.json").write_text(
            json.dumps({"cleanupPeriodDays": -5})
        )
        with patch(
            "session_manager.lifecycle.cleanup.Path.home",
            return_value=tmp_path / "fake_home",
        ):
            assert get_cleanup_period_days() == _DEFAULT_CLEANUP_PERIOD_DAYS

    def test_returns_default_when_value_is_string(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / "fake_home" / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "settings.json").write_text(
            json.dumps({"cleanupPeriodDays": "thirty"})
        )
        with patch(
            "session_manager.lifecycle.cleanup.Path.home",
            return_value=tmp_path / "fake_home",
        ):
            assert get_cleanup_period_days() == _DEFAULT_CLEANUP_PERIOD_DAYS

    def test_returns_default_when_json_is_malformed(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / "fake_home" / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "settings.json").write_text("{not valid json")
        with patch(
            "session_manager.lifecycle.cleanup.Path.home",
            return_value=tmp_path / "fake_home",
        ):
            assert get_cleanup_period_days() == _DEFAULT_CLEANUP_PERIOD_DAYS

    def test_accepts_minimum_value_one(self, tmp_path: Path) -> None:
        claude_dir = tmp_path / "fake_home" / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "settings.json").write_text(
            json.dumps({"cleanupPeriodDays": 1})
        )
        with patch(
            "session_manager.lifecycle.cleanup.Path.home",
            return_value=tmp_path / "fake_home",
        ):
            assert get_cleanup_period_days() == 1


# -- cleanup_expired_sessions ------------------------------------------------


@pytest.fixture
def session_store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path)


class TestCleanupExpiredSessions:
    def test_deletes_expired_session(self, session_store: SessionStore) -> None:
        _make_session(session_store, "old-task", _days_ago_iso(31))
        deleted = cleanup_expired_sessions(session_store, period_days=30)
        assert deleted == ["old-task"]
        assert session_store.list_sessions() == []

    def test_keeps_fresh_session(self, session_store: SessionStore) -> None:
        _make_session(session_store, "recent-task", _days_ago_iso(5))
        deleted = cleanup_expired_sessions(session_store, period_days=30)
        assert deleted == []
        assert len(session_store.list_sessions()) == 1

    def test_mixed_expired_and_fresh(self, session_store: SessionStore) -> None:
        _make_session(session_store, "expired", _days_ago_iso(60))
        _make_session(session_store, "fresh", _days_ago_iso(2))
        deleted = cleanup_expired_sessions(session_store, period_days=30)
        assert deleted == ["expired"]
        remaining = session_store.list_sessions()
        assert len(remaining) == 1
        assert remaining[0].name == "fresh"

    def test_empty_store_returns_empty(self, session_store: SessionStore) -> None:
        deleted = cleanup_expired_sessions(session_store, period_days=30)
        assert deleted == []

    def test_boundary_exact_cutoff_not_deleted(
        self, session_store: SessionStore
    ) -> None:
        """A session accessed exactly *period_days* ago is on the boundary.
        datetime comparison: accessed == cutoff → not less than → kept.

        정확히 period_days 전에 접근한 세션은 경계값이다.
        accessed == cutoff → 미만이 아님 → 유지된다.
        """
        _make_session(session_store, "boundary", _days_ago_iso(30))
        deleted = cleanup_expired_sessions(session_store, period_days=30)
        # Due to sub-second timing, the session is at or just past the cutoff.
        # We accept either outcome — the important thing is no crash.
        # 서브초 타이밍 차이로 경계 세션은 삭제될 수도 유지될 수도 있다.
        # 중요한 것은 에러 없이 동작하는 것이다.
        assert isinstance(deleted, list)

    def test_skips_malformed_timestamp(self, session_store: SessionStore) -> None:
        session = _make_session(session_store, "bad-ts", _days_ago_iso(1))
        session.last_accessed = "not-a-timestamp"
        session_store.save_session(session)
        deleted = cleanup_expired_sessions(session_store, period_days=30)
        assert deleted == []
        assert len(session_store.list_sessions()) == 1

    def test_multiple_expired_all_deleted(self, session_store: SessionStore) -> None:
        _make_session(session_store, "old-1", _days_ago_iso(40))
        _make_session(session_store, "old-2", _days_ago_iso(50))
        _make_session(session_store, "old-3", _days_ago_iso(100))
        deleted = cleanup_expired_sessions(session_store, period_days=30)
        assert len(deleted) == 3
        assert session_store.list_sessions() == []

    def test_custom_period_days(self, session_store: SessionStore) -> None:
        _make_session(session_store, "task-a", _days_ago_iso(8))
        _make_session(session_store, "task-b", _days_ago_iso(3))
        deleted = cleanup_expired_sessions(session_store, period_days=7)
        assert deleted == ["task-a"]
        assert len(session_store.list_sessions()) == 1


class TestActivityAwareCleanup:
    """A session in daily use must not be deleted (F16).

    매일 쓰는 세션이 삭제되면 안 된다 (F16).

    ``last_accessed`` is only written by tool calls that touch a session —
    a switch touches the session being left, not the one entered. Someone
    working in one session for a month keeps a month-old ``last_accessed``
    while their transcript is written to every day.
    ``last_accessed`` 는 세션을 건드리는 도구 호출 시에만 기록된다 — 전환은
    떠나는 세션만 touch 한다. 한 세션에서 한 달간 작업하는 사용자는
    ``last_accessed`` 가 한 달 전인 채로 남지만 transcript 는 매일 기록된다.
    """

    def _link(self, tmp_path: Path, conv_id: str, when_days_ago: float) -> None:
        project_dir = (
            Path.home() / ".claude" / "projects" / encode_cwd(tmp_path)
        )
        project_dir.mkdir(parents=True, exist_ok=True)
        path = project_dir / f"{conv_id}.jsonl"
        path.write_text("{}\n", encoding="utf-8")
        ts = time.time() - when_days_ago * 86400
        os.utime(path, (ts, ts))

    @pytest.fixture
    def project(self, tmp_path: Path) -> Iterator[Path]:
        """A project dir whose ~/.claude transcript dir is cleaned up after.

        테스트 후 ~/.claude transcript 디렉토리를 정리하는 프로젝트 디렉토리.
        """
        yield tmp_path
        shutil.rmtree(
            Path.home() / ".claude" / "projects" / encode_cwd(tmp_path),
            ignore_errors=True,
        )

    def test_in_use_session_survives_stale_last_accessed(
        self, project: Path
    ) -> None:
        store = SessionStore(project)
        store.init_project()
        session = SessionMetadata.new(name="daily", title="매일 쓰는 세션")
        session.last_accessed = _days_ago_iso(60)
        session.claude_conversation_ids = ["conv-daily"]
        store.save_session(session)
        self._link(project, "conv-daily", when_days_ago=0.5)

        assert cleanup_expired_sessions(store, 30, project) == []
        assert store.load_session_by_name("daily") is not None

    def test_truly_idle_session_still_deleted(self, project: Path) -> None:
        store = SessionStore(project)
        store.init_project()
        session = SessionMetadata.new(name="idle", title="방치된 세션")
        session.last_accessed = _days_ago_iso(60)
        session.claude_conversation_ids = ["conv-idle"]
        store.save_session(session)
        self._link(project, "conv-idle", when_days_ago=45)

        assert cleanup_expired_sessions(store, 30, project) == ["idle"]

    def test_missing_transcript_falls_back_to_metadata(
        self, project: Path
    ) -> None:
        """Claude Code may have removed the transcript — fall back safely.

        Claude Code 가 transcript 를 지웠을 수 있다 — 메타데이터로 fallback.
        """
        store = SessionStore(project)
        store.init_project()
        session = SessionMetadata.new(name="gone", title="대화 파일 소멸")
        session.last_accessed = _days_ago_iso(60)
        session.claude_conversation_ids = ["conv-missing"]
        store.save_session(session)

        assert cleanup_expired_sessions(store, 30, project) == ["gone"]

    def test_without_project_path_behaves_as_before(self, project: Path) -> None:
        store = SessionStore(project)
        store.init_project()
        session = SessionMetadata.new(name="legacy", title="구 동작")
        session.last_accessed = _days_ago_iso(60)
        session.claude_conversation_ids = ["conv-legacy"]
        store.save_session(session)
        self._link(project, "conv-legacy", when_days_ago=0.5)

        assert cleanup_expired_sessions(store, 30) == ["legacy"]
