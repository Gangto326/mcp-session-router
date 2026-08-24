"""Tests for the StaticField model (provenance-tracked since R5-C4).

StaticField 모델 테스트 (R5-C4 부터 출처 추적).
"""

from __future__ import annotations

import json
import time

import pytest

from session_manager.models import StaticField
from session_manager.models.fields import (
    SOURCE_AUTO,
    SOURCE_USER,
    StaticEntry,
    UnsupportedStaticSchemaError,
)


def _roundtrip(obj: StaticField) -> StaticField:
    encoded = json.dumps(obj.to_dict(), ensure_ascii=False)
    return StaticField.from_dict(json.loads(encoded))


class TestStaticFieldFactory:
    def test_new_sets_updated_at(self) -> None:
        field = StaticField.new()
        assert field.updated_at.endswith("+00:00")

    def test_new_defaults_are_empty(self) -> None:
        field = StaticField.new()
        assert field.project_context.value == ""
        assert field.conventions.value == ""
        assert field.project_map == {}
        assert field.variables == {}


class TestStaticEntry:
    def test_set_keeps_previous_value(self) -> None:
        entry = StaticEntry()
        assert entry.set("localhost", SOURCE_AUTO) is True
        assert entry.set("db.example.com", SOURCE_USER) is True
        assert entry.value == "db.example.com"
        assert entry.prev_value == "localhost"
        assert entry.source == SOURCE_USER
        assert entry.updated_at != ""

    def test_set_same_value_is_noop(self) -> None:
        # Rewriting the same value must not erase history by shifting
        # prev_value onto itself.
        # 같은 값을 다시 써도 prev_value 가 자기 자신으로 밀려 이력이
        # 지워지면 안 된다.
        entry = StaticEntry()
        entry.set("a", SOURCE_AUTO)
        entry.set("b", SOURCE_AUTO)
        stamp = entry.updated_at
        assert entry.set("b", SOURCE_USER) is False
        assert entry.prev_value == "a"
        assert entry.source == SOURCE_AUTO
        assert entry.updated_at == stamp


class TestStaticFieldRoundtrip:
    def test_roundtrip_empty(self) -> None:
        field = StaticField.new()
        restored = _roundtrip(field)
        assert restored == field

    def test_roundtrip_with_nested_heterogeneous_variables(self) -> None:
        field = StaticField.new()
        field.set_project_context(
            "React + TypeScript 모노레포, turborepo 사용", SOURCE_AUTO
        )
        field.set_conventions("ESLint + Prettier, Jest 테스트", SOURCE_USER)
        field.merge_project_map(
            {
                "src/auth/": "인증 모듈 (JWT, OAuth)",
                "src/api/": "REST API 엔드포인트",
            },
            SOURCE_AUTO,
        )
        field.merge_variables(
            {
                "서버 접속": {"staging": "ssh deploy@staging.example.com"},
                "환경변수": ["DATABASE_URL", "OPENAI_API_KEY"],
                "API 키": {"OpenAI": "sk-abc123..."},
            },
            SOURCE_USER,
        )
        restored = _roundtrip(field)
        assert restored == field
        assert restored.variables["환경변수"].value == [
            "DATABASE_URL",
            "OPENAI_API_KEY",
        ]
        assert restored.variables["환경변수"].source == SOURCE_USER

    def test_roundtrip_missing_optional_fields_uses_defaults(self) -> None:
        restored = StaticField.from_dict(
            {"schema": 2, "updated_at": "2026-04-13T00:00:00+00:00"}
        )
        assert restored.project_context.value == ""
        assert restored.conventions.value == ""
        assert restored.project_map == {}
        assert restored.variables == {}
        assert restored.updated_at == "2026-04-13T00:00:00+00:00"


class TestLegacyMigration:
    """Pre-R5-C4 flat files migrate on load (module docstring).

    R5-C4 이전 평면 파일은 로드 시 마이그레이션된다 (모듈 docstring).
    """

    LEGACY = {
        "project_context": "React 모노레포",
        "conventions": "ESLint",
        "project_map": {"src/auth/": "인증 모듈"},
        # A dict that LOOKS like an entry — the schema marker, not value
        # shape, must decide the format.
        # entry 처럼 생긴 dict — 형식 판별은 값 모양이 아니라 schema
        # 마커가 해야 한다.
        "variables": {"조회 응답": {"value": 42, "source": "api"}},
        "updated_at": "2026-04-13T00:00:00+00:00",
    }

    def test_flat_values_become_auto_entries(self) -> None:
        field = StaticField.from_dict(self.LEGACY)
        assert field.project_context.value == "React 모노레포"
        assert field.project_context.source == SOURCE_AUTO
        assert field.project_context.updated_at == "2026-04-13T00:00:00+00:00"
        assert field.project_context.prev_value is None
        assert field.project_map["src/auth/"].value == "인증 모듈"

    def test_entry_shaped_legacy_value_survives_verbatim(self) -> None:
        field = StaticField.from_dict(self.LEGACY)
        assert field.variables["조회 응답"].value == {
            "value": 42,
            "source": "api",
        }
        assert field.variables["조회 응답"].source == SOURCE_AUTO

    def test_unknown_schema_is_refused_not_destroyed(self) -> None:
        # A future schema 3 (or corrupted marker) must not be re-wrapped
        # as legacy — that would silently destroy its provenance.
        # 미래의 schema 3 (또는 오염된 마커) 을 구식으로 재포장하면
        # 출처가 조용히 파괴된다 — 거부해야 한다.
        for schema in (3, "2", 0):
            with pytest.raises(UnsupportedStaticSchemaError):
                StaticField.from_dict({"schema": schema, "updated_at": "t"})

    def test_schema2_hand_edited_flat_value_is_preserved(self) -> None:
        # Inside a schema-2 file a non-entry value (hand edit) wraps as
        # a migrated entry instead of vanishing into an empty one.
        # schema 2 파일 안의 entry 아닌 값 (손편집) 은 빈 항목으로
        # 사라지는 대신 이행 항목으로 감싸 보존된다.
        field = StaticField.from_dict(
            {
                "schema": 2,
                "project_context": "손으로 적은 평면값",
                "variables": {"HOST": "localhost"},
                "updated_at": "t",
            }
        )
        assert field.project_context.value == "손으로 적은 평면값"
        assert field.variables["HOST"].value == "localhost"

    def test_migrated_file_saves_as_schema_2(self) -> None:
        migrated = StaticField.from_dict(self.LEGACY)
        data = migrated.to_dict()
        assert data["schema"] == 2
        assert StaticField.from_dict(data) == migrated


class TestMerge:
    def test_merge_updates_only_given_keys(self) -> None:
        field = StaticField.new()
        field.merge_variables({"A": "1", "B": "2"}, SOURCE_AUTO)
        changed = field.merge_variables({"B": "20"}, SOURCE_USER)
        assert changed == ["variables.B"]
        assert field.variables["A"].value == "1"  # untouched / 유지
        assert field.variables["B"].value == "20"

    def test_variables_history_drops_value_keeps_timestamp(self) -> None:
        # variables may hold secrets: the old value is scrubbed, only
        # the fact of overwriting (prev_updated_at) is kept.
        # variables 는 비밀 가능 — 옛 값은 파기, 덮어썼다는 사실
        # (prev_updated_at) 만 남는다.
        field = StaticField.new()
        field.merge_variables({"TOKEN": "old-secret"}, SOURCE_AUTO)
        first_at = field.variables["TOKEN"].updated_at
        field.merge_variables({"TOKEN": "new-secret"}, SOURCE_AUTO)
        entry = field.variables["TOKEN"]
        assert entry.prev_value is None
        assert entry.prev_updated_at == first_at
        assert "old-secret" not in json.dumps(field.to_dict())

    def test_project_map_history_keeps_value(self) -> None:
        # Non-secret fields keep full prev_value for revert.
        # 비밀 아닌 필드는 되돌리기용 prev_value 를 온전히 보존.
        field = StaticField.new()
        field.merge_project_map({"src/": "옛 설명"}, SOURCE_AUTO)
        field.merge_project_map({"src/": "새 설명"}, SOURCE_USER)
        assert field.project_map["src/"].prev_value == "옛 설명"

    def test_merge_reports_no_change_for_same_value(self) -> None:
        field = StaticField.new()
        field.merge_project_map({"src/": "루트"}, SOURCE_AUTO)
        assert field.merge_project_map({"src/": "루트"}, SOURCE_AUTO) == []

    def test_set_text_fields_report_change(self) -> None:
        field = StaticField.new()
        assert field.set_project_context("ctx", SOURCE_AUTO) == ["project_context"]
        assert field.set_project_context("ctx", SOURCE_AUTO) == []
        assert field.set_conventions("conv", SOURCE_USER) == ["conventions"]


class TestStaticFieldTouch:
    def test_touch_updates_timestamp(self) -> None:
        field = StaticField.new()
        initial = field.updated_at
        time.sleep(0.001)
        field.touch()
        assert field.updated_at > initial
