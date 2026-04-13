"""Tests for StaticField model."""

from __future__ import annotations

import json
import time

from session_manager.models import StaticField


def _roundtrip(obj: StaticField) -> StaticField:
    encoded = json.dumps(obj.to_dict(), ensure_ascii=False)
    return StaticField.from_dict(json.loads(encoded))


class TestStaticFieldFactory:
    def test_new_sets_updated_at(self) -> None:
        field = StaticField.new()
        assert field.updated_at.endswith("+00:00")

    def test_new_defaults_are_empty(self) -> None:
        field = StaticField.new()
        assert field.project_context == ""
        assert field.conventions == ""
        assert field.project_map == {}
        assert field.variables == {}


class TestStaticFieldRoundtrip:
    def test_roundtrip_empty(self) -> None:
        field = StaticField.new()
        restored = _roundtrip(field)
        assert restored == field

    def test_roundtrip_with_nested_heterogeneous_variables(self) -> None:
        field = StaticField.new()
        field.project_context = "React + TypeScript 모노레포, turborepo 사용"
        field.conventions = "ESLint + Prettier, Jest 테스트"
        field.project_map = {
            "src/auth/": "인증 모듈 (JWT, OAuth)",
            "src/api/": "REST API 엔드포인트",
        }
        field.variables = {
            "서버 접속": {"staging": "ssh deploy@staging.example.com"},
            "환경변수": ["DATABASE_URL", "OPENAI_API_KEY"],
            "API 키": {"OpenAI": "sk-abc123..."},
        }
        restored = _roundtrip(field)
        assert restored == field
        assert restored.variables["환경변수"] == [
            "DATABASE_URL",
            "OPENAI_API_KEY",
        ]

    def test_roundtrip_missing_optional_fields_uses_defaults(self) -> None:
        restored = StaticField.from_dict({"updated_at": "2026-04-13T00:00:00+00:00"})
        assert restored.project_context == ""
        assert restored.conventions == ""
        assert restored.project_map == {}
        assert restored.variables == {}
        assert restored.updated_at == "2026-04-13T00:00:00+00:00"


class TestStaticFieldTouch:
    def test_touch_updates_timestamp(self) -> None:
        field = StaticField.new()
        initial = field.updated_at
        time.sleep(0.001)
        field.touch()
        assert field.updated_at > initial
