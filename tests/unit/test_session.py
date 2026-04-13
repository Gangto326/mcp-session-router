"""Tests for session metadata model."""

from __future__ import annotations

import json
import time
import uuid

from session_manager.models import (
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
