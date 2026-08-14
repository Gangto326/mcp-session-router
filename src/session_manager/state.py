"""In-memory state for the MCP server process."""

from __future__ import annotations

import datetime

from session_manager import debug_log
from session_manager.models.session import SessionStatus
from session_manager.storage import SessionStore


class SessionManagerState:
    def __init__(self) -> None:
        self._current_session_name: str | None = None

    def get_current_session(self) -> str | None:
        return self._current_session_name

    def set_current_session(self, name: str | None) -> None:
        # Single STATE_CHANGE checkpoint — every set goes through here so
        # we can answer "when did current_session become X?" from the log.
        # 단일 STATE_CHANGE 체크포인트 — 모든 set 이 이 지점을 통과하므로
        # "current_session 이 언제 X 가 되었나?" 를 로그로 답할 수 있다.
        debug_log.log(
            "STATE_CHANGE",
            "MCP_TOOL",
            {
                "field": "current_session_name",
                "before": self._current_session_name,
                "after": name,
            },
            session=name,
        )
        self._current_session_name = name

    def resolve_from_store(
        self, store: SessionStore, active_conversation_id: str | None = None
    ) -> str | None:
        """Infer the current session when the wrapper handshake gave none.

        핸드셰이크가 현재 세션을 주지 못했을 때 세션을 추론한다.

        Prefers the session that owns *active_conversation_id* — a direct
        fact, unlike ``last_accessed``, which is only written by tool calls
        that touch a session and so can be stale for a session in active
        use. The timestamp scan stays as the last resort.

        *active_conversation_id* 를 소유한 세션을 우선한다 — ``last_accessed``
        와 달리 직접적인 사실이기 때문. ``last_accessed`` 는 세션을 건드리는
        도구 호출 시에만 기록되어 사용 중인 세션에서도 낡을 수 있다.
        타임스탬프 스캔은 최후 수단으로 유지한다.

        Retired sessions are never resolved to directly (R4-C6 prep):
        check_session hides them, so surfacing one as *current* would
        contradict the list the LLM sees (observed in the C5 e2e boot).
        A conversation match on a retired session is redirected to its
        living successor; the timestamp scan skips retired sessions
        unless nothing else exists.

        만료 세션은 직접 결과가 되지 않는다 (R4-C6 선행 수정):
        check_session 이 만료 세션을 숨기므로, current 로 노출되면 LLM 이
        보는 목록과 모순된다 (C5 e2e 부팅에서 실관측). 만료 세션에 대한
        conversation 매칭은 살아 있는 후계로 재지향하고, 타임스탬프
        스캔은 만료 세션을 제외한다 (남는 게 없을 때만 포함).
        """
        sessions = store.list_sessions()
        if not sessions:
            debug_log.log(
                "STATE_RESOLVE",
                "SYSTEM",
                {"result": None, "reason": "store_empty"},
            )
            return None
        if active_conversation_id:
            # Two passes: a living owner is a direct fact and wins; a
            # retired owner is only a lead — follow its successor chain.
            # 2단 스캔 — 살아 있는 소유자는 직접 사실이라 우선, 만료
            # 소유자는 단서일 뿐이라 후계 사슬을 따라간다.
            retired_match = None
            for session in sessions:
                if active_conversation_id not in session.claude_conversation_ids:
                    continue
                if session.status != SessionStatus.RETIRED:
                    debug_log.log(
                        "STATE_RESOLVE",
                        "SYSTEM",
                        {
                            "result": session.name,
                            "reason": "active_conversation_match",
                            "active_conversation_id": active_conversation_id,
                        },
                        conv_id=active_conversation_id,
                        session=session.name,
                    )
                    return session.name
                if retired_match is None:
                    retired_match = session
            if retired_match is not None:
                # Guarded like the other two call sites (wrapper redirect,
                # session_switch pre-resolution): the chain walk WRITES on
                # success (path compression), and a storage failure there
                # must degrade to the timestamp fallback, not break the
                # MCP boot path this resolution runs on.
                # 다른 두 호출처 (래퍼 재지향·session_switch 선해석) 와
                # 같은 보호 — 사슬 추적은 성공 시 쓰기 (경로 압축) 를
                # 겸하므로, 스토리지 실패는 이 추론이 도는 MCP 부팅
                # 경로를 깨지 말고 타임스탬프 폴백으로 열화해야 한다.
                try:
                    successor = store.resolve_active_successor(
                        retired_match.name
                    )
                except Exception:
                    successor = None
                if successor is not None:
                    debug_log.log(
                        "STATE_RESOLVE",
                        "SYSTEM",
                        {
                            "result": successor,
                            "reason": "retired_match_redirected",
                            "retired": retired_match.name,
                            "active_conversation_id": active_conversation_id,
                        },
                        conv_id=active_conversation_id,
                        session=successor,
                    )
                    return successor
                # Dead-ended chain: fall through to the timestamp scan —
                # a stale current is still better than pointing at a
                # session the candidate list hides.
                # 사슬이 막히면 타임스탬프 스캔으로 폴백 — 후보 목록이
                # 숨기는 세션을 가리키는 것보다는 낡은 추론이 낫다.
        living = [s for s in sessions if s.status != SessionStatus.RETIRED]
        pool = living or sessions
        latest = max(
            pool,
            key=lambda s: datetime.datetime.fromisoformat(s.last_accessed),
        )
        debug_log.log(
            "STATE_RESOLVE",
            "SYSTEM",
            {
                "result": latest.name,
                "reason": "last_accessed_fallback",
                "last_accessed": latest.last_accessed,
                "candidates": len(pool),
                "retired_excluded": len(sessions) - len(living),
            },
        )
        return latest.name
