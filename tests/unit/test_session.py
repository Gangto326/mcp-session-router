"""Tests for session metadata model."""

from __future__ import annotations

import json
import time
import uuid

from session_manager.models import (
    RETIRE_REASONS,
    PrecedentRecord,
    RetiredRecord,
    SessionMetadata,
    SessionStatus,
    TransitionRecord,
)


def _roundtrip(obj: SessionMetadata) -> SessionMetadata:
    encoded = json.dumps(obj.to_dict(), ensure_ascii=False)
    return SessionMetadata.from_dict(json.loads(encoded))


class TestSessionMetadataFactory:
    def test_new_generates_valid_uuid(self) -> None:
        session = SessionMetadata.new(name="auth-fix", title="인증 수정")
        uuid.UUID(session.session_id)

    def test_new_sets_created_at_and_last_accessed_identical(self) -> None:
        session = SessionMetadata.new(name="auth-fix", title="인증 수정")
        assert session.created_at == session.last_accessed
        assert session.created_at.endswith("+00:00")

    def test_new_summary_defaults_to_none(self) -> None:
        session = SessionMetadata.new(name="auth-fix", title="인증 수정")
        assert session.summary is None

    def test_new_status_defaults_to_active(self) -> None:
        session = SessionMetadata.new(name="auth-fix", title="인증 수정")
        assert session.status is SessionStatus.ACTIVE

    def test_new_transitions_default_empty(self) -> None:
        session = SessionMetadata.new(name="auth-fix", title="인증 수정")
        assert session.transitions == []


class TestSessionMetadataRoundtrip:
    def test_roundtrip_preserves_all_fields(self) -> None:
        original = SessionMetadata.new(
            name="auth-fix",
            title="인증 모듈 디버깅",
            summary="JWT 검증 로직 수정 완료. refresh API 구현 남음.",
        )
        original.transitions.append(
            TransitionRecord.new(from_session="prev", to_session="auth-fix")
        )
        restored = _roundtrip(original)
        assert restored == original

    def test_roundtrip_with_null_summary(self) -> None:
        original = SessionMetadata.new(name="auth-fix", title="인증 수정")
        assert original.summary is None
        restored = _roundtrip(original)
        assert restored.summary is None

    def test_roundtrip_with_archived_status(self) -> None:
        original = SessionMetadata.new(name="auth-fix", title="인증 수정")
        original.status = SessionStatus.ARCHIVED
        restored = _roundtrip(original)
        assert restored.status is SessionStatus.ARCHIVED

    def test_roundtrip_preserves_korean_and_emoji_in_summary(self) -> None:
        original = SessionMetadata.new(
            name="auth-fix",
            title="인증 모듈 🔐 디버깅",
            summary="한글 요약과 🚀 이모지 섞인 문장.",
        )
        restored = _roundtrip(original)
        assert restored.title == original.title
        assert restored.summary == original.summary

    def test_roundtrip_with_empty_title_does_not_raise(self) -> None:
        original = SessionMetadata.new(name="noop", title="")
        restored = _roundtrip(original)
        assert restored.title == ""


class TestSessionMetadataR1C6Fields:
    """New fields: requirements / summary_updated_at / profile.

    신규 필드 — requirements / summary_updated_at / profile.
    """

    def test_new_defaults(self) -> None:
        session = SessionMetadata.new(name="auth-fix", title="인증 수정")
        assert session.requirements == []
        assert session.summary_updated_at is None
        assert session.profile is None

    def test_roundtrip_preserves_new_fields(self) -> None:
        original = SessionMetadata.new(name="auth-fix", title="인증 수정")
        original.requirements = ["이 작업은 테스트 필수", "커밋은 사용자가 직접"]
        original.summary_updated_at = "2026-07-30T12:00:00+00:00"
        original.profile = "핵심 파일: auth.py — JWT 검증 재작성 중"
        restored = _roundtrip(original)
        assert restored == original

    def test_legacy_file_without_new_fields_loads_with_defaults(self) -> None:
        # A session JSON written before R1-C6 has none of the new keys.
        # R1-C6 이전에 작성된 세션 JSON 에는 신규 키가 하나도 없다.
        legacy = SessionMetadata.new(name="old", title="옛 세션").to_dict()
        for key in ("requirements", "summary_updated_at", "profile"):
            del legacy[key]
        restored = SessionMetadata.from_dict(legacy)
        assert restored.requirements == []
        assert restored.summary_updated_at is None
        assert restored.profile is None


class TestSessionMetadataPrecedents:
    """R3-C1 precedents: roundtrip, backward compat, event invalidation.

    R3-C1 판례 — 라운드트립, 하위 호환, 이벤트 무효화.
    """

    def _record(self, rejected: str = "backend") -> PrecedentRecord:
        return PrecedentRecord.new(
            prompt_gist="로그인 API 500 조사",
            kept_in="frontend",
            rejected=rejected,
        )

    def test_new_defaults_to_empty(self) -> None:
        session = SessionMetadata.new(name="frontend", title="차트")
        assert session.precedents == []

    def test_precedent_record_new_sets_iso_timestamp(self) -> None:
        record = self._record()
        assert record.at.endswith("+00:00")

    def test_roundtrip_preserves_precedents(self) -> None:
        original = SessionMetadata.new(name="frontend", title="차트")
        original.precedents.append(self._record())
        restored = _roundtrip(original)
        assert restored == original
        assert restored.precedents[0].rejected == "backend"

    def test_legacy_file_without_precedents_loads_with_default(self) -> None:
        # A session JSON written before R3-C1 has no precedents key.
        # R3-C1 이전에 작성된 세션 JSON 에는 precedents 키가 없다.
        legacy = SessionMetadata.new(name="old", title="옛 세션").to_dict()
        del legacy["precedents"]
        restored = SessionMetadata.from_dict(legacy)
        assert restored.precedents == []

    def test_clear_precedents_drops_all(self) -> None:
        session = SessionMetadata.new(name="frontend", title="차트")
        session.precedents = [self._record("backend"), self._record("infra")]
        session.clear_precedents()
        assert session.precedents == []

    def test_drop_precedents_for_removes_only_matching_target(self) -> None:
        session = SessionMetadata.new(name="frontend", title="차트")
        session.precedents = [
            self._record("backend"),
            self._record("infra"),
            self._record("backend"),
        ]
        session.drop_precedents_for("backend")
        assert [p.rejected for p in session.precedents] == ["infra"]

    def test_drop_precedents_for_non_matching_is_noop(self) -> None:
        session = SessionMetadata.new(name="frontend", title="차트")
        session.precedents = [self._record("backend")]
        session.drop_precedents_for("없는-세션")
        assert [p.rejected for p in session.precedents] == ["backend"]


class TestSessionMetadataMixing:
    """R3-C2 mixing fields: defaults, roundtrip, backward compat.

    R3-C2 혼합도 필드 — 기본값, 라운드트립, 하위 호환.
    """

    def test_new_defaults(self) -> None:
        session = SessionMetadata.new(name="frontend", title="차트")
        assert session.mixing_score == 0
        assert session.mixing_evidence == []

    def test_roundtrip_preserves_mixing_fields(self) -> None:
        original = SessionMetadata.new(name="frontend", title="차트")
        original.mixing_score = 2
        original.mixing_evidence = ["차트 얘기 3턴", "백엔드 파일 수정 동반"]
        restored = _roundtrip(original)
        assert restored == original

    def test_legacy_file_without_mixing_fields_loads_with_defaults(self) -> None:
        # A session JSON written before R3-C2 has neither mixing key.
        # R3-C2 이전에 작성된 세션 JSON 에는 혼합도 키가 없다.
        legacy = SessionMetadata.new(name="old", title="옛 세션").to_dict()
        for key in ("mixing_score", "mixing_evidence"):
            del legacy[key]
        restored = SessionMetadata.from_dict(legacy)
        assert restored.mixing_score == 0
        assert restored.mixing_evidence == []


class TestSessionMetadataRetirement:
    """R4-C5 retirement: status, record roundtrip, retire/revive.

    R4-C5 세션 만료 — 상태·기록 라운드트립·retire/revive.
    """

    def test_new_defaults(self) -> None:
        session = SessionMetadata.new(name="frontend", title="차트")
        assert session.status is SessionStatus.ACTIVE
        assert session.retired is None

    def test_retire_sets_status_and_record(self) -> None:
        session = SessionMetadata.new(name="frontend", title="차트")
        session.retire("manual")
        assert session.status is SessionStatus.RETIRED
        assert session.retired is not None
        assert session.retired.reason == "manual"
        assert session.retired.successor is None
        assert session.retired.at.endswith("+00:00")

    def test_retire_with_successor(self) -> None:
        session = SessionMetadata.new(name="frontend", title="차트")
        session.retire("polluted", successor="frontend-2")
        assert session.retired is not None
        assert session.retired.reason == "polluted"
        assert session.retired.successor == "frontend-2"

    def test_retire_with_unknown_reason_falls_back_to_manual(self) -> None:
        session = SessionMetadata.new(name="frontend", title="차트")
        session.retire("cosmic-rays")
        assert session.retired is not None
        assert session.retired.reason == "manual"

    def test_retired_reason_vocabulary(self) -> None:
        # "rolled_over" is gone — a rollover keeps the same session, so
        # nothing could ever record it.
        # "rolled_over" 는 제거됐다 — 롤오버는 같은 세션을 유지하므로
        # 기록할 주체가 없었다.
        assert RETIRE_REASONS == ("polluted", "abandoned", "manual")

    def test_legacy_rolled_over_file_loads_as_written(self) -> None:
        # Backward compat: the vocabulary gates writes (retire), not
        # reads — a file written before the removal keeps its value.
        # 하위 호환 — 어휘는 쓰기 (retire) 만 통제하고 읽기는 통제하지
        # 않는다. 제거 이전에 쓰인 파일은 값을 그대로 유지한다.
        session = SessionMetadata.new(name="old", title="옛 세션")
        session.retire("manual", successor="old-2")
        data = session.to_dict()
        data["retired"]["reason"] = "rolled_over"
        restored = SessionMetadata.from_dict(data)
        assert restored.retired is not None
        assert restored.retired.reason == "rolled_over"
        assert restored.status is SessionStatus.RETIRED

    def test_revive_restores_active_and_clears_record(self) -> None:
        session = SessionMetadata.new(name="frontend", title="차트")
        session.retire("manual")
        session.revive()
        assert session.status is SessionStatus.ACTIVE
        assert session.retired is None

    def test_roundtrip_preserves_retirement(self) -> None:
        original = SessionMetadata.new(name="frontend", title="차트")
        original.retire("polluted", successor="frontend-clean")
        restored = _roundtrip(original)
        assert restored == original
        assert restored.status is SessionStatus.RETIRED

    def test_active_session_serialises_null_retired(self) -> None:
        data = SessionMetadata.new(name="frontend", title="차트").to_dict()
        assert data["retired"] is None

    def test_legacy_file_without_retired_loads_with_default(self) -> None:
        # A session JSON written before R4-C5 has no retired key.
        # R4-C5 이전에 작성된 세션 JSON 에는 retired 키가 없다.
        legacy = SessionMetadata.new(name="old", title="옛 세션").to_dict()
        del legacy["retired"]
        restored = SessionMetadata.from_dict(legacy)
        assert restored.retired is None

    def test_retired_record_from_dict_defends_bad_types(self) -> None:
        record = RetiredRecord.from_dict({"successor": 42})
        assert record.reason == "manual"
        assert record.successor is None
        assert record.at == ""


class TestSessionMetadataTouch:
    def test_touch_updates_last_accessed(self) -> None:
        session = SessionMetadata.new(name="auth-fix", title="인증 수정")
        initial = session.last_accessed
        time.sleep(0.001)
        session.touch()
        assert session.last_accessed > initial
        assert session.created_at == initial


class TestSessionStatus:
    def test_status_values(self) -> None:
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.ARCHIVED.value == "archived"
        assert SessionStatus.EXPIRED.value == "expired"
        assert SessionStatus.RETIRED.value == "retired"

    def test_status_str_comparison(self) -> None:
        assert SessionStatus.ACTIVE == "active"

    def test_status_from_string_restores_enum(self) -> None:
        assert SessionStatus("expired") is SessionStatus.EXPIRED


class TestTransitionRecord:
    def test_new_sets_timestamp(self) -> None:
        record = TransitionRecord.new(from_session="A", to_session="B")
        assert record.from_session == "A"
        assert record.to_session == "B"
        assert record.timestamp.endswith("+00:00")

    def test_new_supports_null_from_session(self) -> None:
        record = TransitionRecord.new(from_session=None, to_session="first")
        assert record.from_session is None
        assert record.to_session == "first"

    def test_roundtrip(self) -> None:
        record = TransitionRecord.new(from_session="A", to_session="B")
        restored = TransitionRecord.from_dict(record.to_dict())
        assert restored == record

    def test_roundtrip_with_null_from_session(self) -> None:
        record = TransitionRecord.new(from_session=None, to_session="first")
        restored = TransitionRecord.from_dict(record.to_dict())
        assert restored == record
        assert restored.from_session is None

    def test_dict_uses_snake_case_keys(self) -> None:
        record = TransitionRecord.new(from_session="A", to_session="B")
        data = record.to_dict()
        assert set(data.keys()) == {"from_session", "to_session", "timestamp"}
