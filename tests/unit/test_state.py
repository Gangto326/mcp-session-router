"""Tests for SessionManagerState."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from session_manager.models import SessionMetadata
from session_manager.state import SessionManagerState
from session_manager.storage import SessionStore


@pytest.fixture
def state() -> SessionManagerState:
    return SessionManagerState()


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path)


class TestCurrentSessionAccessors:
    def test_initial_current_session_is_none(
        self, state: SessionManagerState
    ) -> None:
        assert state.get_current_session() is None

    def test_set_current_session_stores_value(
        self, state: SessionManagerState
    ) -> None:
        state.set_current_session("foo")
        assert state.get_current_session() == "foo"

    def test_set_current_session_to_none_clears_value(
        self, state: SessionManagerState
    ) -> None:
        state.set_current_session("foo")
        state.set_current_session(None)
        assert state.get_current_session() is None


class TestResolveFromStore:
    def test_empty_store_returns_none(
        self, state: SessionManagerState, store: SessionStore
    ) -> None:
        assert state.resolve_from_store(store) is None

    def test_single_session_returns_its_name(
        self, state: SessionManagerState, store: SessionStore
    ) -> None:
        store.save_session(SessionMetadata.new(name="only", title="Only"))
        assert state.resolve_from_store(store) == "only"

    def test_returns_most_recently_accessed_session_name(
        self, state: SessionManagerState, store: SessionStore
    ) -> None:
        older = SessionMetadata.new(name="older", title="O")
        time.sleep(0.002)
        newer = SessionMetadata.new(name="newer", title="N")
        store.save_session(older)
        store.save_session(newer)

        assert state.resolve_from_store(store) == "newer"

    def test_touch_changes_resolution_order(
        self, state: SessionManagerState, store: SessionStore
    ) -> None:
        a = SessionMetadata.new(name="a", title="A")
        time.sleep(0.002)
        b = SessionMetadata.new(name="b", title="B")
        store.save_session(a)
        store.save_session(b)
        assert state.resolve_from_store(store) == "b"

        time.sleep(0.002)
        a.touch()
        store.save_session(a)
        assert state.resolve_from_store(store) == "a"

    def test_resolve_does_not_mutate_state(
        self, state: SessionManagerState, store: SessionStore
    ) -> None:
        store.save_session(SessionMetadata.new(name="foo", title="Foo"))
        _ = state.resolve_from_store(store)
        assert state.get_current_session() is None


class TestHandshakeScenarios:
    """계획서 §10.6 — ccode 시작 모드별 초기화 흐름."""

    def test_resume_flag_sets_current_from_handshake(
        self, state: SessionManagerState, store: SessionStore
    ) -> None:
        # 래퍼 핸드셰이크 응답: "foo"
        handshake_value: str | None = "foo"
        if handshake_value is not None:
            state.set_current_session(handshake_value)
        else:
            state.set_current_session(state.resolve_from_store(store))

        assert state.get_current_session() == "foo"

    def test_continue_flag_with_existing_sessions_resolves_to_latest(
        self, state: SessionManagerState, store: SessionStore
    ) -> None:
        older = SessionMetadata.new(name="older", title="O")
        time.sleep(0.002)
        newer = SessionMetadata.new(name="newer", title="N")
        store.save_session(older)
        store.save_session(newer)

        # 래퍼 핸드셰이크 응답: null
        handshake_value: str | None = None
        if handshake_value is not None:
            state.set_current_session(handshake_value)
        else:
            state.set_current_session(state.resolve_from_store(store))

        assert state.get_current_session() == "newer"

    def test_first_run_with_empty_store_remains_none(
        self, state: SessionManagerState, store: SessionStore
    ) -> None:
        # 래퍼 핸드셰이크 응답: null, store 비어있음
        handshake_value: str | None = None
        if handshake_value is not None:
            state.set_current_session(handshake_value)
        else:
            state.set_current_session(state.resolve_from_store(store))

        assert state.get_current_session() is None
