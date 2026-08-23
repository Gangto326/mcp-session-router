"""Unit tests for ``ccode --stats`` aggregation (R5-C3).

``ccode --stats`` 집계 (R5-C3) 단위 테스트. 합성 이벤트·세션·디버그
레코드로 순수 집계 함수와 출처 적재를 검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from session_manager.routing import stats


def _proposal(target: str, confidence: float, mode: str = "confirm", **extra: Any) -> dict:
    return {
        "type": "proposal",
        "target": target,
        "confidence": confidence,
        "mode": mode,
        "at": "t",
        **extra,
    }


def _label(target: str, label: str, source: str, **extra: Any) -> dict:
    return {
        "type": "label",
        "label": label,
        "target": target,
        "source": source,
        "at": "t",
        **extra,
    }


# One project's worth of synthetic history, mixing pre- and post-R5-C3
# lines (the "docs" pair lacks kept_in / target_status).
# 한 프로젝트 분량의 합성 이력 — R5-C3 전후 줄이 섞여 있다 ("docs" 쌍은
# kept_in / target_status 가 없다).
EVENTS = [
    _proposal("backend", 0.9, target_status="active"),
    _label("backend", "accept", "session_switch", kept_in="frontend"),
    _proposal("infra", 0.7, target_status="archived"),
    _label("infra", "reject", "reject_switch", kept_in="backend"),
    _proposal("infra", 0.95, mode="auto", target_status="active"),
    _label("infra", "reject", "back", kept_in="backend"),
    _proposal("docs", 0.6),
    _label("docs", "reject", "reject_switch"),
    _proposal("docs", 0.5, target_status="missing"),  # ignored / 무시됨
]


class TestProposals:
    def test_counts_by_mode_and_target_status(self) -> None:
        p = stats.compute_stats(EVENTS, [])["proposals"]
        assert p["total"] == 5
        assert p["by_mode"] == {"confirm": 4, "auto": 1}
        assert p["auto_switches"] == 1
        # Pre-R5-C3 lines have no target_status → "unrecorded", never
        # silently folded into a real status.
        # R5-C3 이전 줄은 target_status 가 없다 → "unrecorded", 실제
        # 상태로 조용히 섞이지 않는다.
        assert p["by_target_status"] == {
            "active": 2,
            "archived": 1,
            "missing": 1,
            "unrecorded": 1,
        }


class TestAcceptance:
    def test_reuses_decision_log_pairing(self) -> None:
        a = stats.compute_stats(EVENTS, [])["acceptance"]
        assert (a["accepted"], a["rejected"], a["unlabeled"]) == (1, 3, 1)
        assert a["overall_rate"] == 0.25
        assert a["back_count"] == 1
        assert a["labels_by_source"] == {
            "session_switch": 1,
            "reject_switch": 2,
            "back": 1,
        }

    def test_recent_window_is_most_recent_half(self) -> None:
        # Same proportional window as get_routing_status (rule 8).
        # get_routing_status 와 같은 비례 창 (규칙 8).
        a = stats.compute_stats(EVENTS, [])["acceptance"]
        assert a["recent_window"] == 2
        assert a["recent_rate"] == 0.0

    def test_empty_rates_are_none(self) -> None:
        a = stats.compute_stats([], [])["acceptance"]
        assert a["overall_rate"] is None
        assert a["recent_rate"] is None
        assert a["recent_window"] == 0


class TestPrecedent:
    def test_reject_pairs_keyed_by_kept_in_and_target(self) -> None:
        pr = stats.compute_stats(EVENTS, [])["precedent"]
        assert pr["reject_pairs"] == {"backend → infra": 2}
        assert pr["unrecorded"] == 1


class TestDebug:
    RECORDS = [
        {"run_id": "r1", "category": "WRAPPER_BOOT", "payload": {}},
        {"run_id": "r1", "category": "HOOK_PREFILTER", "payload": {"rule": "routing_off"}},
        {
            "run_id": "r1",
            "category": "HOOK_PREFILTER",
            "payload": {"rule": None, "to_judge": True},
        },
        {
            "run_id": "r1",
            "category": "HOOK_PREFILTER",
            "payload": {"rule": None, "to_judge": True},
        },
        {
            "run_id": "r1",
            "category": "JUDGE",
            "payload": {"op": "verdict", "elapsed_s": 2.0, "verdict": {"action": "SWITCH"}},
        },
        {
            "run_id": "r1",
            "category": "JUDGE",
            "payload": {"op": "precedent_suppress", "target": "infra"},
        },
        {
            "run_id": "r2",
            "category": "JUDGE",
            "payload": {"op": "verdict", "elapsed_s": 1.0, "verdict": {"action": "STAY"}},
        },
        # Non-verdict JUDGE ops must not count as judgments.
        # verdict 가 아닌 JUDGE op 는 판정으로 세지 않는다.
        {"run_id": "r2", "category": "JUDGE", "payload": {"op": "warmup", "result": "ok"}},
    ]

    def test_prefilter_and_judge_aggregates(self) -> None:
        d = stats.compute_stats([], [], self.RECORDS)["debug"]
        assert d["runs"] == 2
        assert d["prefilter"]["total"] == 3
        assert d["prefilter"]["to_judge"] == 2
        assert d["prefilter"]["trigger_rate"] == 2 / 3
        assert d["prefilter"]["skipped_by_rule"] == {"routing_off": 1}
        assert d["judge"]["verdicts"] == 2
        assert d["judge"]["by_action"] == {"SWITCH": 1, "STAY": 1}
        assert d["judge"]["avg_elapsed_ms"] == 1500
        assert d["judge"]["precedent_suppressed"] == 1

    def test_no_records_means_not_measured(self) -> None:
        assert stats.compute_stats([], [], None)["debug"] is None
        assert stats.compute_stats([], [], [])["debug"] is None

    def test_no_prefilter_records_rate_is_none(self) -> None:
        d = stats.compute_stats([], [], [self.RECORDS[0]])["debug"]
        assert d["prefilter"]["trigger_rate"] is None
        assert d["judge"]["avg_elapsed_ms"] is None


class TestSessions:
    def test_sorted_by_mixing_score_desc_then_name(self) -> None:
        rows = stats.compute_stats(
            [],
            [
                {"name": "b", "mixing_score": 1, "mixing_evidence": ["x"]},
                {"name": "a", "mixing_score": 1},
                {"name": "c", "status": "archived", "mixing_score": 5, "mixing_evidence": []},
            ],
        )["sessions"]
        assert [r["name"] for r in rows] == ["c", "a", "b"]
        assert rows[0]["status"] == "archived"
        assert rows[1]["status"] == "active"  # absent status = active
        assert rows[2]["mixing_evidence_count"] == 1


class TestLoadDebugRecords:
    def _write_run(self, log_dir: Path, run_id: str, project: str) -> None:
        records = [
            {"run_id": run_id, "category": "WRAPPER_BOOT", "payload": {"project_path": project}},
            {
                "run_id": run_id,
                "category": "HOOK_PREFILTER",
                "payload": {"rule": None, "to_judge": True},
            },
        ]
        (log_dir / f"{run_id}.ndjson").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )

    def test_only_runs_booted_in_this_project(self, tmp_path: Path) -> None:
        # The log dir is shared across projects — other projects' runs
        # must not leak into this project's trigger rate.
        # 로그 디렉터리는 프로젝트 공용 — 다른 프로젝트의 run 이 이
        # 프로젝트의 발동률에 섞이면 안 된다.
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        project = tmp_path / "proj"
        project.mkdir()
        self._write_run(log_dir, "mine", str(project))
        self._write_run(log_dir, "other", str(tmp_path / "elsewhere"))
        records = stats.load_debug_records(project, log_dir)
        assert {r["run_id"] for r in records} == {"mine"}
        assert len(records) == 2

    def test_corrupt_lines_skipped(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        project = tmp_path / "proj"
        project.mkdir()
        self._write_run(log_dir, "mine", str(project))
        with open(log_dir / "mine.ndjson", "a", encoding="utf-8") as fh:
            fh.write("{{{ broken\n")
        assert len(stats.load_debug_records(project, log_dir)) == 2

    def test_missing_log_dir(self, tmp_path: Path) -> None:
        assert stats.load_debug_records(tmp_path, tmp_path / "nope") == []


class TestFormat:
    def test_table_mentions_not_measured_without_debug(self) -> None:
        text = stats.format_stats(stats.compute_stats(EVENTS, []))
        assert stats.NOT_MEASURED in text
        assert "수용률 전체      25%" in text
        assert "backend → infra 2" in text

    def test_table_with_debug(self) -> None:
        text = stats.format_stats(stats.compute_stats(EVENTS, [], TestDebug.RECORDS))
        assert stats.NOT_MEASURED not in text
        assert "발동률           67%" in text
        assert "1500 ms" in text


class TestRunStats:
    def _project(self, tmp_path: Path) -> Path:
        project = tmp_path / "proj"
        (project / ".session-manager" / "sessions").mkdir(parents=True)
        with open(
            project / ".session-manager" / "routing-decisions.jsonl", "w", encoding="utf-8"
        ) as fh:
            for e in EVENTS:
                fh.write(json.dumps(e) + "\n")
        (project / ".session-manager" / "sessions" / "a.json").write_text(
            json.dumps({"name": "backend", "mixing_score": 2}), encoding="utf-8"
        )
        (project / ".session-manager" / "sessions" / "bad.json").write_text("{{{", encoding="utf-8")
        return project

    def test_json_output_round_trips(self, tmp_path: Path) -> None:
        project = self._project(tmp_path)
        out = stats.run_stats(project, tmp_path / "no-logs", as_json=True)
        data = json.loads(out)
        assert data["proposals"]["total"] == 5
        assert data["debug"] is None
        # Corrupt session file skipped, valid one kept.
        # 손상 세션 파일은 건너뛰고 정상 파일만 남는다.
        assert [s["name"] for s in data["sessions"]] == ["backend"]

    def test_table_output(self, tmp_path: Path) -> None:
        project = self._project(tmp_path)
        out = stats.run_stats(project, tmp_path / "no-logs", as_json=False)
        assert out.startswith("라우팅 통계")
