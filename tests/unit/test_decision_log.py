"""Unit tests for the routing decision log (R3-C4 calibration source).

라우팅 결정 로그 (R3-C4 보정 원천) 단위 테스트.
"""

from __future__ import annotations

from pathlib import Path

from session_manager.routing import decision_log


def _propose(project: Path, target: str = "backend", confidence: float = 0.9) -> None:
    decision_log.append_proposal(project, target, confidence, mode="confirm")


def _label(project: Path, target: str = "backend", label: str = "accept") -> None:
    decision_log.append_label(project, target, label, source="test")


class TestAppendAndLoad:
    def test_events_in_file_order(self, tmp_path: Path) -> None:
        _propose(tmp_path, confidence=0.8)
        _label(tmp_path)
        events = decision_log.load_events(tmp_path)
        assert [e["type"] for e in events] == ["proposal", "label"]
        assert events[0]["confidence"] == 0.8
        assert events[1]["label"] == "accept"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert decision_log.load_events(tmp_path) == []

    def test_corrupt_lines_skipped(self, tmp_path: Path) -> None:
        _propose(tmp_path)
        log_path = tmp_path / ".session-manager" / "routing-decisions.jsonl"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write("깨진 줄 {{{\n")
        _label(tmp_path)
        assert len(decision_log.load_events(tmp_path)) == 2


class TestLabeledPairs:
    def test_label_consumes_most_recent_proposal_of_same_target(
        self, tmp_path: Path
    ) -> None:
        _propose(tmp_path, confidence=0.7)
        _propose(tmp_path, confidence=0.9)
        _label(tmp_path, label="accept")
        pairs = decision_log.labeled_pairs(decision_log.load_events(tmp_path))
        # The newer proposal (0.9) is consumed; the older stays unlabeled.
        # 최근 제안 (0.9) 이 소비되고 이전 것은 미라벨로 남는다.
        assert pairs == [(0.9, True)]

    def test_ignored_proposals_excluded(self, tmp_path: Path) -> None:
        # Ignored is not rejected — no pair is produced.
        # 무시는 거부가 아니다 — 쌍이 생기지 않는다.
        _propose(tmp_path)
        assert (
            decision_log.labeled_pairs(decision_log.load_events(tmp_path)) == []
        )

    def test_orphan_label_excluded(self, tmp_path: Path) -> None:
        # A voluntary session_switch with no preceding proposal.
        # 선행 제안 없는 자발 session_switch.
        _label(tmp_path)
        assert (
            decision_log.labeled_pairs(decision_log.load_events(tmp_path)) == []
        )

    def test_target_scoped_pairing(self, tmp_path: Path) -> None:
        _propose(tmp_path, target="backend", confidence=0.9)
        _propose(tmp_path, target="infra", confidence=0.6)
        _label(tmp_path, target="infra", label="reject")
        pairs = decision_log.labeled_pairs(decision_log.load_events(tmp_path))
        assert pairs == [(0.6, False)]

    def test_unknown_label_value_ignored(self, tmp_path: Path) -> None:
        _propose(tmp_path)
        _label(tmp_path, label="maybe")
        assert (
            decision_log.labeled_pairs(decision_log.load_events(tmp_path)) == []
        )


class TestAcceptanceStats:
    def test_tallies(self, tmp_path: Path) -> None:
        _propose(tmp_path)
        _label(tmp_path, label="accept")
        _propose(tmp_path)
        _label(tmp_path, label="reject")
        _propose(tmp_path)  # 무시됨 — unlabeled
        stats = decision_log.acceptance_stats(tmp_path)
        assert stats == {"accepted": 1, "rejected": 1, "unlabeled": 1}

    def test_empty(self, tmp_path: Path) -> None:
        assert decision_log.acceptance_stats(tmp_path) == {
            "accepted": 0,
            "rejected": 0,
            "unlabeled": 0,
        }
