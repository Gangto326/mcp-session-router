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


class TestResolveWithActiveConversation:
    """Active-conversation match beats the last_accessed scan (F14).

    활성 conversation 매칭이 last_accessed 스캔보다 우선한다 (F14).

    ``last_accessed`` is written only by tool calls that touch a session,
    so the newest timestamp is not evidence of where the user actually is.
    Conversation ownership is a direct fact.
    ``last_accessed`` 는 세션을 건드리는 도구 호출 시에만 기록되므로 최신
    타임스탬프가 사용자의 현 위치를 증명하지 못한다. conversation 소유는
    직접적인 사실이다.
    """

    def test_conversation_owner_wins_over_newer_timestamp(
        self, tmp_path: Path
    ) -> None:
        store = SessionStore(tmp_path)
        store.init_project()
        owner = SessionMetadata.new(name="owner", title="실제 위치")
        owner.last_accessed = "2026-07-01T00:00:00+00:00"
        owner.claude_conversation_ids = ["conv-here"]
        store.save_session(owner)
        newer = SessionMetadata.new(name="newer", title="타임스탬프만 최신")
        newer.last_accessed = "2026-07-30T00:00:00+00:00"
        store.save_session(newer)

        state = SessionManagerState()
        assert state.resolve_from_store(store, "conv-here") == "owner"

    def test_falls_back_to_last_accessed_when_unmatched(
        self, tmp_path: Path
    ) -> None:
        store = SessionStore(tmp_path)
        store.init_project()
        old = SessionMetadata.new(name="old", title="예전")
        old.last_accessed = "2026-07-01T00:00:00+00:00"
        store.save_session(old)
        recent = SessionMetadata.new(name="recent", title="최근")
        recent.last_accessed = "2026-07-30T00:00:00+00:00"
        store.save_session(recent)

        state = SessionManagerState()
        assert state.resolve_from_store(store, "conv-unknown") == "recent"
        assert state.resolve_from_store(store, None) == "recent"


class TestResolveExcludesRetired:
    """Retired sessions never resolve as current (R4-C6 prep).

    만료 세션은 current 로 추론되지 않는다 (R4-C6 선행 수정).

    check_session hides retired sessions, so surfacing one as current
    would contradict the candidate list the LLM sees (observed in the
    C5 e2e boot).

    check_session 이 만료 세션을 숨기므로 current 로 노출되면 LLM 이 보는
    후보 목록과 모순된다 (C5 e2e 부팅에서 실관측).
    """

    def test_conversation_match_on_retired_redirects_to_successor(
        self, tmp_path: Path
    ) -> None:
        store = SessionStore(tmp_path)
        store.init_project()
        store.save_session(SessionMetadata.new(name="heir", title="후계"))
        gone = SessionMetadata.new(name="gone", title="만료")
        gone.claude_conversation_ids = ["conv-old"]
        gone.retire("manual", successor="heir")
        store.save_session(gone)

        state = SessionManagerState()
        assert state.resolve_from_store(store, "conv-old") == "heir"

    def test_living_owner_wins_over_retired_owner(self, tmp_path: Path) -> None:
        # A conversation linked to both (the C5 stale-link shape): the
        # living owner is a direct fact and must win regardless of scan
        # order.
        # 양쪽에 링크된 conversation (C5 낡은 링크 형태) — 살아 있는
        # 소유자가 직접 사실이므로 스캔 순서와 무관하게 이겨야 한다.
        store = SessionStore(tmp_path)
        store.init_project()
        gone = SessionMetadata.new(name="a-gone", title="만료")
        gone.claude_conversation_ids = ["conv-x"]
        gone.retire("manual", successor="z-alive")
        store.save_session(gone)
        alive = SessionMetadata.new(name="z-alive", title="생존")
        alive.claude_conversation_ids = ["conv-x"]
        store.save_session(alive)

        state = SessionManagerState()
        assert state.resolve_from_store(store, "conv-x") == "z-alive"

    def test_retired_match_without_successor_falls_back_to_scan(
        self, tmp_path: Path
    ) -> None:
        store = SessionStore(tmp_path)
        store.init_project()
        gone = SessionMetadata.new(name="gone", title="만료")
        gone.claude_conversation_ids = ["conv-old"]
        gone.retire("manual")
        gone.last_accessed = "2026-07-30T00:00:00+00:00"
        store.save_session(gone)
        alive = SessionMetadata.new(name="alive", title="생존")
        alive.last_accessed = "2026-07-01T00:00:00+00:00"
        store.save_session(alive)

        state = SessionManagerState()
        assert state.resolve_from_store(store, "conv-old") == "alive"

    def test_timestamp_scan_excludes_retired(self, tmp_path: Path) -> None:
        store = SessionStore(tmp_path)
        store.init_project()
        gone = SessionMetadata.new(name="gone", title="만료")
        gone.retire("manual")
        gone.last_accessed = "2026-07-30T00:00:00+00:00"
        store.save_session(gone)
        alive = SessionMetadata.new(name="alive", title="생존")
        alive.last_accessed = "2026-07-01T00:00:00+00:00"
        store.save_session(alive)

        state = SessionManagerState()
        assert state.resolve_from_store(store, None) == "alive"

    def test_successor_resolve_failure_falls_back_to_scan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The chain walk writes on success (path compression); a storage
        # failure there must degrade to the timestamp fallback instead of
        # breaking the MCP boot path (same guard as the other call sites).
        # 사슬 추적은 성공 시 쓰기 (경로 압축) 를 겸한다 — 그 스토리지
        # 실패는 MCP 부팅 경로를 깨지 말고 타임스탬프 폴백으로 열화해야
        # 한다 (다른 호출처들과 같은 보호).
        # The successor ("heir") and the scan winner ("recent") differ on
        # purpose: the assertion below proves the FALLBACK ran, not a
        # redirect that happened to give the same name.
        # 후계 ("heir") 와 스캔 승자 ("recent") 를 일부러 다르게 둔다 —
        # 아래 단정이 "우연히 같은 이름의 재지향" 이 아니라 폴백 경로
        # 실행을 증명하도록.
        store = SessionStore(tmp_path)
        store.init_project()
        gone = SessionMetadata.new(name="gone", title="만료")
        gone.claude_conversation_ids = ["conv-old"]
        gone.retire("manual", successor="heir")
        store.save_session(gone)
        heir = SessionMetadata.new(name="heir", title="후계")
        heir.last_accessed = "2026-07-01T00:00:00+00:00"
        store.save_session(heir)
        recent = SessionMetadata.new(name="recent", title="최근")
        recent.last_accessed = "2026-07-30T00:00:00+00:00"
        store.save_session(recent)

        def boom(name: str) -> str | None:
            raise OSError("disk error")

        monkeypatch.setattr(store, "resolve_active_successor", boom)

        state = SessionManagerState()
        assert state.resolve_from_store(store, "conv-old") == "recent"

    def test_all_retired_still_resolves_latest(self, tmp_path: Path) -> None:
        # A stale answer beats no answer: with nothing living, keep the
        # old behaviour instead of returning None.
        # 낡은 답이 무답보다 낫다 — 살아 있는 세션이 없으면 기존 동작
        # (최신 타임스탬프) 을 유지한다.
        store = SessionStore(tmp_path)
        store.init_project()
        a = SessionMetadata.new(name="a", title="A")
        a.retire("manual")
        a.last_accessed = "2026-07-01T00:00:00+00:00"
        store.save_session(a)
        b = SessionMetadata.new(name="b", title="B")
        b.retire("manual")
        b.last_accessed = "2026-07-30T00:00:00+00:00"
        store.save_session(b)

        state = SessionManagerState()
        assert state.resolve_from_store(store, None) == "b"
