"""Unit tests for the pending-handoff file store.

pending handoff 파일 스토어 단위 테스트.
"""

from __future__ import annotations

import json
from pathlib import Path

from session_manager import handoff_store


def _pending_path(project: Path) -> Path:
    return project / ".session-manager" / "handoffs" / "pending.json"


class TestWriteTake:
    def test_roundtrip(self, tmp_path: Path) -> None:
        handoff_store.write_pending(
            tmp_path,
            target="backend",
            handoff={"from": "frontend", "message": "m"},
            user_prompt="원래 프롬프트",
        )
        data = handoff_store.take_pending(tmp_path)
        assert data is not None
        assert data["target"] == "backend"
        assert data["handoff"] == {"from": "frontend", "message": "m"}
        assert data["user_prompt"] == "원래 프롬프트"
        assert data["at"].endswith("+00:00")
        # Consumed: the file is gone and a second take yields nothing.
        # 소비됨 — 파일이 사라지고 재소비는 None.
        assert not _pending_path(tmp_path).exists()
        assert handoff_store.take_pending(tmp_path) is None

    def test_take_without_file(self, tmp_path: Path) -> None:
        assert handoff_store.take_pending(tmp_path) is None

    def test_corrupt_file_consumed_and_none(self, tmp_path: Path) -> None:
        path = _pending_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("{깨진", encoding="utf-8")
        assert handoff_store.take_pending(tmp_path) is None
        # Corrupt files must not wedge future transitions.
        # 손상 파일이 다음 전환을 막으면 안 된다.
        assert not path.exists()

    def test_overwrite_keeps_latest(self, tmp_path: Path) -> None:
        handoff_store.write_pending(tmp_path, "a", {}, "첫")
        handoff_store.write_pending(tmp_path, "b", {}, "둘")
        assert handoff_store.take_pending(tmp_path)["target"] == "b"


class TestStaleClear:
    def test_clears_leftover_file(self, tmp_path: Path) -> None:
        handoff_store.write_pending(tmp_path, "a", {}, "p")
        assert handoff_store.clear_stale_pending(tmp_path) is True
        assert handoff_store.take_pending(tmp_path) is None

    def test_noop_without_file(self, tmp_path: Path) -> None:
        assert handoff_store.clear_stale_pending(tmp_path) is False


class TestTrigger:
    def test_trigger_is_fixed_and_content_free(self) -> None:
        # argv is ps-visible: the trigger must never carry user content.
        # argv 는 ps 로 노출된다 — 트리거에 사용자 내용이 실리면 안 된다.
        assert handoff_store.TRIGGER_PROMPT == "[session-manager] 세션 전환 재개"

    def test_pending_file_content_is_valid_json(self, tmp_path: Path) -> None:
        handoff_store.write_pending(tmp_path, "a", {"k": "한글"}, "프롬프트")
        raw = json.loads(_pending_path(tmp_path).read_text(encoding="utf-8"))
        assert raw["handoff"]["k"] == "한글"
