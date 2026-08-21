"""Session metadata model."""

from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from session_manager import debug_log


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


class SessionStatus(StrEnum):
    """Session lifecycle: the single "ended" state is ARCHIVED.

    세션 수명 주기 — "끝남" 상태는 ARCHIVED 하나다.

    ACTIVE sessions are routing candidates everywhere (check_session,
    judge input, boot-time current inference, statusline count).
    ARCHIVED is entered when the LLM calls ``session_end`` on the user's
    explicit wish; it is hidden from every candidate list and comes back
    to ACTIVE automatically the moment a transition targets it
    (``reactivate``). Never entered by time or inactivity — a dormant
    session must stay routable. Legacy values ``retired``/``expired``
    (R4-C5 model, consolidated 2026-08-21) load as ARCHIVED.

    ACTIVE 세션은 모든 곳 (check_session·판정 입력·부팅 시 현재 추론·
    statusline 집계) 에서 라우팅 후보다. ARCHIVED 는 사용자가 명시적으로
    끝내겠다고 해 LLM 이 ``session_end`` 를 부를 때 진입하며, 모든 후보
    목록에서 숨겨지고 전환 대상이 되는 순간 자동으로 ACTIVE 로 돌아온다
    (``reactivate``). 시간·미접근으로는 절대 진입하지 않는다 — 휴면
    세션은 라우팅 가능해야 한다. 옛 값 ``retired``/``expired`` (R4-C5
    모델, 2026-08-21 통합) 는 ARCHIVED 로 읽는다.
    """

    ACTIVE = "active"
    ARCHIVED = "archived"


# Status strings written by the retired R4-C5 model; read as ARCHIVED so
# old session files keep loading. Nothing writes them any more.
# 폐기된 R4-C5 모델이 기록하던 상태 문자열 — 옛 세션 파일이 계속
# 로드되도록 ARCHIVED 로 읽는다. 더 이상 기록하는 곳은 없다.
_LEGACY_ENDED_STATUSES = frozenset({"retired", "expired"})


def parse_status(value: object) -> SessionStatus:
    """Parse a stored status string, mapping legacy ended values.

    저장된 상태 문자열을 해석한다 — 옛 끝남 값은 ARCHIVED 로 사상.
    """
    if value in _LEGACY_ENDED_STATUSES:
        return SessionStatus.ARCHIVED
    return SessionStatus(value if isinstance(value, str) else SessionStatus.ACTIVE.value)


@dataclass
class TransitionRecord:
    from_session: str | None
    to_session: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_session": self.from_session,
            "to_session": self.to_session,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransitionRecord:
        return cls(
            from_session=data.get("from_session"),
            to_session=data["to_session"],
            timestamp=data["timestamp"],
        )

    @classmethod
    def new(cls, from_session: str | None, to_session: str) -> TransitionRecord:
        return cls(
            from_session=from_session,
            to_session=to_session,
            timestamp=_utc_now_iso(),
        )


@dataclass
class PrecedentRecord:
    """One rejected switch proposal ("precedent", R3-C1).

    Recorded when the user declines a router SWITCH proposal and stays in
    the current session. Fed back into the judge prompt so the same
    proposal is not repeated. Invalidation is event-based, not time-based
    (rule 8 — no TTL): a precedent dies when its kept_in session rolls
    over (``clear_precedents``) or when a later switch to the same
    rejected target is accepted (``drop_precedents_for``).

    거부된 전환 제안 한 건 ("판례", R3-C1).

    사용자가 라우터의 SWITCH 제안을 거부하고 현재 세션에 머물 때
    기록된다. 판정 프롬프트에 다시 입력되어 같은 제안의 반복을 막는다.
    무효화는 시간이 아니라 이벤트 기반이다 (규칙 8 — TTL 없음): kept_in
    세션이 롤오버되거나 (``clear_precedents``), 이후 같은 rejected 대상으로의
    전환이 수용되면 (``drop_precedents_for``) 소멸한다.
    """

    # One-line gist of the prompt that triggered the rejected proposal.
    # 거부된 제안을 유발한 프롬프트의 한 줄 요지.
    prompt_gist: str
    # Session the user chose to stay in (also where this record is stored).
    # 사용자가 머물기로 한 세션 (이 기록의 저장 위치이기도 하다).
    kept_in: str
    # Proposed switch target the user rejected.
    # 사용자가 거부한 전환 제안 대상.
    rejected: str
    # ISO8601 timestamp of the rejection. Judge input sorts by this,
    # most recent first.
    # 거부 시각 (ISO8601). 판정 입력은 이 값으로 최근 우선 정렬한다.
    at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_gist": self.prompt_gist,
            "kept_in": self.kept_in,
            "rejected": self.rejected,
            "at": self.at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrecedentRecord:
        return cls(
            prompt_gist=data["prompt_gist"],
            kept_in=data["kept_in"],
            rejected=data["rejected"],
            at=data["at"],
        )

    @classmethod
    def new(cls, prompt_gist: str, kept_in: str, rejected: str) -> PrecedentRecord:
        return cls(
            prompt_gist=prompt_gist,
            kept_in=kept_in,
            rejected=rejected,
            at=_utc_now_iso(),
        )


@dataclass
class SessionMetadata:
    session_id: str
    name: str
    title: str
    summary: str | None
    created_at: str
    last_accessed: str
    transitions: list[TransitionRecord] = field(default_factory=list)
    status: SessionStatus = SessionStatus.ACTIVE
    # Claude Code conversation IDs that this metadata session has been
    # observed inside. Populated by session_register / session_switch /
    # session_create / session_end when the cwd's active_conversation_id
    # is known. Used by the routing harness to detect picker-driven
    # conversation transitions: if `active_conversation_id` matches one
    # of these, the routing is unambiguous.
    # 이 메타데이터 세션 안에서 관측된 Claude Code conversation id 목록.
    # cwd 의 active_conversation_id 가 알려진 시점에 도구들이 채운다. picker
    # 기반 conversation 전환 감지에 사용 — active_conversation_id 가 어느
    # 세션의 이 목록에 있으면 라우팅이 명확해진다.
    claude_conversation_ids: list[str] = field(default_factory=list)
    # Session-scoped user instructions/constraints (e.g. "tests required
    # for this work") extracted by the background summarizer. Distinct
    # from global conventions — these apply to this session only.
    # 백그라운드 요약기가 추출하는 세션 한정 사용자 지시·제약 (예: "이
    # 작업은 테스트 필수"). 전역 컨벤션과 달리 이 세션에만 적용된다.
    requirements: list[str] = field(default_factory=list)
    # When the summary was last refreshed. Compared against
    # ``last_accessed`` at boot to find sessions whose summary refresh
    # was lost to a forced exit (R1-C3).
    # summary 가 마지막으로 갱신된 시각. 부팅 시 ``last_accessed`` 와
    # 비교해 강제 종료로 요약이 누락된 세션을 찾는다 (R1-C3).
    summary_updated_at: str | None = None
    # Dialogue length the last summary was built from, and the conversation
    # it was measured in. The pair is what makes periodic refresh work:
    # growth is measured against this baseline, and the conversation id
    # scopes it — after a rollover or switch the new conversation starts
    # from zero, so comparing against the old one would make the growth
    # negative and silence the trigger for a long time.
    # 마지막 요약이 대상으로 한 대화 길이와, 그것을 측정한 conversation.
    # 이 쌍이 주기 갱신을 성립시킨다: 증가량을 이 기준값과 비교하며,
    # conversation id 가 그 범위를 한정한다 — 롤오버·전환 후 새 conversation
    # 은 0 에서 시작하므로 옛 기준과 비교하면 증가량이 음수가 되어 트리거가
    # 오랫동안 침묵한다.
    summary_dialogue_chars: int = 0
    summary_dialogue_conversation_id: str | None = None
    # Rich context for second-pass routing (key files, decisions,
    # detailed state). Populated from R2 onwards.
    # 2차 라우팅 판정용 리치 컨텍스트 (핵심 파일, 결정 사항, 상세 상태).
    # R2 부터 채워진다.
    profile: str | None = None
    # Rejected switch proposals recorded while staying in this session
    # (R3-C1). This session is always the kept_in side.
    # 이 세션에 머물면서 기록된 거부 판례 목록 (R3-C1). 이 세션이 항상
    # kept_in 쪽이다.
    precedents: list[PrecedentRecord] = field(default_factory=list)
    # Topic-mixing tally (R3-C2): incremented when a rooting check finds
    # that a rejected topic took root here (multi-turn continuation).
    # No threshold constant exists on purpose (rule 8) — the raw value
    # (with its evidence quotes below) goes straight into the judge
    # input, and the judge weighs it against other signals.
    # 주제 혼합도 집계 (R3-C2) — 정착 확인이 "거부된 주제가 이 세션에
    # 뿌리내렸다(복수 턴 진행)"고 판정할 때마다 1 증가. 임계 상수는
    # 의도적으로 없다 (규칙 8) — 원값이 (아래 근거 인용과 함께) 판정자
    # 입력에 그대로 들어가고, 가중은 판정기가 다른 신호와 함께 결정한다.
    mixing_score: int = 0
    # Evidence quotes from rooted=true rooting checks, accumulated in
    # order. Shown to the judge next to the raw score.
    # rooted=true 정착 확인의 근거 인용 누적 목록. 원값 곁에 판정자에게
    # 표기된다.
    mixing_evidence: list[str] = field(default_factory=list)

    @classmethod
    def new(cls, name: str, title: str, summary: str | None = None) -> SessionMetadata:
        now = _utc_now_iso()
        return cls(
            session_id=str(uuid.uuid4()),
            name=name,
            title=title,
            summary=summary,
            created_at=now,
            last_accessed=now,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "title": self.title,
            "summary": self.summary,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "transitions": [t.to_dict() for t in self.transitions],
            "status": self.status.value,
            "claude_conversation_ids": list(self.claude_conversation_ids),
            "requirements": list(self.requirements),
            "summary_updated_at": self.summary_updated_at,
            "profile": self.profile,
            "summary_dialogue_chars": self.summary_dialogue_chars,
            "summary_dialogue_conversation_id": self.summary_dialogue_conversation_id,
            "precedents": [p.to_dict() for p in self.precedents],
            "mixing_score": self.mixing_score,
            "mixing_evidence": list(self.mixing_evidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMetadata:
        return cls(
            session_id=data["session_id"],
            name=data["name"],
            title=data["title"],
            summary=data.get("summary"),
            created_at=data["created_at"],
            last_accessed=data["last_accessed"],
            transitions=[
                TransitionRecord.from_dict(t) for t in data.get("transitions", [])
            ],
            status=parse_status(data.get("status", SessionStatus.ACTIVE.value)),
            # Default to empty list for legacy session files written before
            # this field existed.
            # 이 필드 도입 전에 작성된 세션 파일과의 호환을 위해 기본값 빈 리스트.
            claude_conversation_ids=list(data.get("claude_conversation_ids", [])),
            # Same backward-compat defaults for the R1-C6 fields.
            # R1-C6 필드들도 같은 방식의 하위 호환 기본값.
            requirements=list(data.get("requirements", [])),
            summary_updated_at=data.get("summary_updated_at"),
            profile=data.get("profile"),
            summary_dialogue_chars=data.get("summary_dialogue_chars", 0),
            summary_dialogue_conversation_id=data.get(
                "summary_dialogue_conversation_id"
            ),
            # Backward-compat default for session files written before
            # R3-C1 introduced this field.
            # R3-C1 이 이 필드를 도입하기 전에 작성된 세션 파일과의 하위
            # 호환 기본값.
            precedents=[
                PrecedentRecord.from_dict(p) for p in data.get("precedents", [])
            ],
            # Backward-compat defaults for the R3-C2 mixing fields.
            # R3-C2 혼합도 필드의 하위 호환 기본값.
            mixing_score=data.get("mixing_score", 0),
            mixing_evidence=list(data.get("mixing_evidence", [])),
        )

    def touch(self) -> None:
        self.last_accessed = _utc_now_iso()

    def reactivate(self) -> bool:
        """Bring an ARCHIVED session back to ACTIVE; True if it changed.

        ARCHIVED 세션을 ACTIVE 로 되돌린다. 바뀌었으면 True.
        Called when a transition targets the session — using an ended
        session again is what "not ended any more" means, so no separate
        command is needed.
        전환이 이 세션을 대상으로 할 때 호출된다 — 끝낸 세션을 다시
        쓰는 것이 곧 "더는 끝난 게 아님" 이므로 별도 명령이 필요 없다.
        """
        if self.status == SessionStatus.ACTIVE:
            return False
        self.status = SessionStatus.ACTIVE
        debug_log.log("SESSION_REACTIVATE", "WRAPPER", {}, session=self.name)
        return True

    def link_conversation(self, conv_id: str) -> None:
        """Associate a Claude Code conversation id with this session (idempotent).

        Claude Code conversation id 를 이 세션에 연결한다. 이미 있으면 무시 (멱등).
        """
        if not conv_id:
            return
        before = list(self.claude_conversation_ids)
        if conv_id in before:
            debug_log.log(
                "CONV_LINK",
                "MCP_TOOL",
                {
                    "session": self.name,
                    "conv_id": conv_id,
                    "skipped": True,
                    "reason": "already_linked",
                    "list": before,
                },
                conv_id=conv_id,
                session=self.name,
            )
            return
        self.claude_conversation_ids.append(conv_id)
        debug_log.log(
            "CONV_LINK",
            "MCP_TOOL",
            {
                "session": self.name,
                "conv_id": conv_id,
                "skipped": False,
                "before": before,
                "after": list(self.claude_conversation_ids),
            },
            conv_id=conv_id,
            session=self.name,
        )

    def clear_precedents(self) -> None:
        """Invalidation (a) — drop every precedent of this session.

        Called when this (kept_in) session rolls over: the rollover
        changes the session's topical make-up, so old rejections no
        longer describe it. The trigger hookup lands with the rollover
        phase; the contract lives here with the data.

        이벤트 무효화 (a) — 이 세션의 판례를 전부 소멸시킨다.

        이 (kept_in) 세션이 롤오버될 때 호출된다: 롤오버는 세션의 주제
        구성을 바꾸므로 과거 거부 기록이 더는 세션을 설명하지 못한다.
        발동 지점 연결은 롤오버 Phase 에서 하고, 계약은 데이터 곁인
        여기에 둔다.
        """
        if not self.precedents:
            return
        dropped = len(self.precedents)
        self.precedents = []
        debug_log.log(
            "PRECEDENT",
            "MCP_TOOL",
            {"op": "clear", "session": self.name, "dropped": dropped},
            session=self.name,
        )

    def drop_precedents_for(self, rejected_target: str) -> None:
        """Invalidation (b) — the precedent was overturned.

        Called when a switch to *rejected_target* is later accepted:
        the user's acceptance contradicts the recorded rejection, so
        only the precedents against that target die.

        이벤트 무효화 (b) — 선례 뒤집힘.

        이후 *rejected_target* 으로의 전환이 수용될 때 호출된다: 수용은
        기록된 거부와 모순되므로, 그 대상에 대한 판례만 소멸한다.
        """
        kept = [p for p in self.precedents if p.rejected != rejected_target]
        dropped = len(self.precedents) - len(kept)
        if dropped == 0:
            return
        self.precedents = kept
        debug_log.log(
            "PRECEDENT",
            "MCP_TOOL",
            {
                "op": "drop_for",
                "session": self.name,
                "rejected_target": rejected_target,
                "dropped": dropped,
            },
            session=self.name,
        )
