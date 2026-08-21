"""Unit tests for the wrapper state file (state.json).

래퍼 상태 파일 (state.json) 단위 테스트.
"""

from __future__ import annotations

import json
from pathlib import Path

from session_manager.wrapper import wrapper_state

RECORD = {
    "from": "frontend",
    "to": "backend",
    "user_prompt": "로그인 API 500 조사",
    "at": "2026-08-05T00:00:00+00:00",
}


def _state_file(project: Path) -> Path:
    return project / ".session-manager" / "state.json"


class TestLastTransitionRoundtrip:
    def test_save_then_load(self, tmp_path: Path) -> None:
        wrapper_state.save_last_transition(tmp_path, RECORD)
        assert wrapper_state.load_last_transition(tmp_path) == RECORD

    def test_load_without_file(self, tmp_path: Path) -> None:
        assert wrapper_state.load_last_transition(tmp_path) is None

    def test_clear_consumes(self, tmp_path: Path) -> None:
        wrapper_state.save_last_transition(tmp_path, RECORD)
        wrapper_state.clear_last_transition(tmp_path)
        assert wrapper_state.load_last_transition(tmp_path) is None

    def test_clear_without_file_is_noop(self, tmp_path: Path) -> None:
        wrapper_state.clear_last_transition(tmp_path)
        assert not _state_file(tmp_path).exists()

    def test_save_preserves_other_keys(self, tmp_path: Path) -> None:
        # R5 will store statusline state in the same file — writes must
        # not clobber foreign keys.
        # R5 가 같은 파일에 statusline 상태를 둔다 — 쓰기가 다른 키를
        # 덮어쓰면 안 된다.
        path = _state_file(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"statusline": {"x": 1}}), encoding="utf-8"
        )
        wrapper_state.save_last_transition(tmp_path, RECORD)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["statusline"] == {"x": 1}
        assert data["last_transition"] == RECORD
        wrapper_state.clear_last_transition(tmp_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {"statusline": {"x": 1}}


class TestDefensiveLoad:
    def test_corrupt_file_returns_none(self, tmp_path: Path) -> None:
        path = _state_file(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("{깨진 JSON", encoding="utf-8")
        assert wrapper_state.load_last_transition(tmp_path) is None

    def test_non_dict_state_returns_none(self, tmp_path: Path) -> None:
        path = _state_file(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("[1, 2]", encoding="utf-8")
        assert wrapper_state.load_last_transition(tmp_path) is None

    def test_record_missing_required_field_returns_none(
        self, tmp_path: Path
    ) -> None:
        broken = {k: v for k, v in RECORD.items() if k != "to"}
        wrapper_state.save_last_transition(tmp_path, broken)
        assert wrapper_state.load_last_transition(tmp_path) is None

    def test_record_empty_field_returns_none(self, tmp_path: Path) -> None:
        wrapper_state.save_last_transition(tmp_path, {**RECORD, "from": ""})
        assert wrapper_state.load_last_transition(tmp_path) is None


class TestCurrentSession:
    def test_save_then_load(self, tmp_path: Path) -> None:
        wrapper_state.save_current_session(tmp_path, "backend")
        assert wrapper_state.load_current_session(tmp_path) == "backend"

    def test_load_without_file(self, tmp_path: Path) -> None:
        assert wrapper_state.load_current_session(tmp_path) is None

    def test_none_removes_key(self, tmp_path: Path) -> None:
        # An unregistered start must not display the previous run's name.
        # 미등록 시작이 이전 실행의 이름을 표시하면 안 된다.
        wrapper_state.save_current_session(tmp_path, "backend")
        wrapper_state.save_current_session(tmp_path, None)
        assert wrapper_state.load_current_session(tmp_path) is None
        data = json.loads(_state_file(tmp_path).read_text(encoding="utf-8"))
        assert "current_session" not in data

    def test_none_without_file_is_noop(self, tmp_path: Path) -> None:
        wrapper_state.save_current_session(tmp_path, None)
        assert not _state_file(tmp_path).exists()

    def test_preserves_last_transition(self, tmp_path: Path) -> None:
        wrapper_state.save_last_transition(tmp_path, RECORD)
        wrapper_state.save_current_session(tmp_path, "backend")
        assert wrapper_state.load_last_transition(tmp_path) == RECORD
        wrapper_state.clear_last_transition(tmp_path)
        assert wrapper_state.load_current_session(tmp_path) == "backend"

    def test_non_string_value_is_none(self, tmp_path: Path) -> None:
        path = _state_file(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"current_session": 7}), encoding="utf-8")
        assert wrapper_state.load_current_session(tmp_path) is None
