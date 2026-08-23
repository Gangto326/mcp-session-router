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


class TestOptionalKeys:
    # R5-C3 schema extension: kept_in / target_status are written only
    # when given, so old files and new files share one reader.
    # R5-C3 스키마 확장 — kept_in / target_status 는 줄 때만 기록되어
    # 옛 파일과 새 파일이 같은 읽기 코드를 공유한다.
    def test_proposal_records_target_status_when_given(self, tmp_path: Path) -> None:
        decision_log.append_proposal(
            tmp_path, "backend", 0.9, mode="confirm", target_status="archived"
        )
        assert decision_log.load_events(tmp_path)[0]["target_status"] == "archived"

    def test_proposal_omits_target_status_by_default(self, tmp_path: Path) -> None:
        _propose(tmp_path)
        assert "target_status" not in decision_log.load_events(tmp_path)[0]

    def test_label_records_kept_in_when_given(self, tmp_path: Path) -> None:
        decision_log.append_label(
            tmp_path, "backend", "reject", source="test", kept_in="frontend"
        )
        assert decision_log.load_events(tmp_path)[0]["kept_in"] == "frontend"

    def test_label_omits_kept_in_by_default(self, tmp_path: Path) -> None:
        _label(tmp_path)
        assert "kept_in" not in decision_log.load_events(tmp_path)[0]

    def test_pairing_ignores_extra_keys(self, tmp_path: Path) -> None:
        decision_log.append_proposal(
            tmp_path, "backend", 0.9, mode="confirm", target_status="active"
        )
        decision_log.append_label(
            tmp_path, "backend", "accept", source="test", kept_in="frontend"
        )
        pairs = decision_log.labeled_pairs(decision_log.load_events(tmp_path))
        assert pairs == [(0.9, True)]


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


class TestExpiry:
    """Ignored proposals must not be consumable later (R5-C3 review).

    무시된 제안은 나중에 소비되어선 안 된다 (R5-C3 재검토).
    """

    def test_stale_proposal_not_swallowed_by_later_voluntary_switch(
        self, tmp_path: Path
    ) -> None:
        # Monday: proposal ignored. Friday: voluntary switch to the same
        # target. Without expiry this paired as "accepted" — the bug.
        # 월요일 제안 무시, 금요일 같은 대상으로 자발 전환. 만료가 없으면
        # "수용" 으로 짝지어졌다 — 그 버그.
        decision_log.append_proposal(
            tmp_path, "backend", 0.9, mode="confirm", conv_id="conv-a"
        )
        assert decision_log.expire_ignored_proposals(tmp_path, "conv-a") is True
        _label(tmp_path, target="backend", label="accept")
        assert decision_log.labeled_pairs(decision_log.load_events(tmp_path)) == []

    def test_expire_only_written_when_something_is_open(self, tmp_path: Path) -> None:
        assert decision_log.expire_ignored_proposals(tmp_path, "conv-a") is False
        assert decision_log.load_events(tmp_path) == []
        decision_log.append_proposal(
            tmp_path, "backend", 0.9, mode="confirm", conv_id="conv-a"
        )
        _label(tmp_path, target="backend", label="reject")
        # Already labeled → nothing open → no expire line.
        # 이미 라벨됨 → 열린 것 없음 → expire 줄 없음.
        assert decision_log.expire_ignored_proposals(tmp_path, "conv-a") is False
        assert [e["type"] for e in decision_log.load_events(tmp_path)] == [
            "proposal",
            "label",
        ]

    def test_expiry_is_scoped_to_conversation(self, tmp_path: Path) -> None:
        # Auto switch: proposal in conv-a, user now prompts in conv-b.
        # conv-b prompts must not close conv-a's proposal — /back from
        # conv-b still has to label it.
        # auto 전환: conv-a 의 제안, 사용자는 이제 conv-b 에서 입력. conv-b
        # 의 프롬프트가 conv-a 제안을 닫으면 안 된다 — conv-b 에서의 /back
        # 이 아직 라벨해야 한다.
        decision_log.append_proposal(
            tmp_path, "backend", 0.95, mode="auto", conv_id="conv-a"
        )
        assert decision_log.expire_ignored_proposals(tmp_path, "conv-b") is False
        _label(tmp_path, target="backend", label="reject")
        assert decision_log.labeled_pairs(decision_log.load_events(tmp_path)) == [
            (0.95, False)
        ]

    def test_legacy_proposal_without_conv_id_never_expires(
        self, tmp_path: Path
    ) -> None:
        _propose(tmp_path, confidence=0.8)  # no conv_id — pre-R5-C3 line
        assert decision_log.expire_ignored_proposals(tmp_path, "conv-a") is False
        _label(tmp_path, label="accept")
        assert decision_log.labeled_pairs(decision_log.load_events(tmp_path)) == [
            (0.8, True)
        ]

    def test_expire_closes_all_open_proposals_of_that_conversation(
        self, tmp_path: Path
    ) -> None:
        decision_log.append_proposal(
            tmp_path, "backend", 0.9, mode="confirm", conv_id="conv-a"
        )
        decision_log.append_proposal(
            tmp_path, "infra", 0.7, mode="confirm", conv_id="conv-a"
        )
        decision_log.expire_ignored_proposals(tmp_path, "conv-a")
        _label(tmp_path, target="backend", label="accept")
        _label(tmp_path, target="infra", label="reject")
        assert decision_log.labeled_pairs(decision_log.load_events(tmp_path)) == []
        stats = decision_log.acceptance_stats(tmp_path)
        assert stats == {"accepted": 0, "rejected": 0, "unlabeled": 2}


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
