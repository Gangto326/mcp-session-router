"""
Unit tests for the routing judge's pure logic.

Prompt assembly and verdict parsing — no subprocess involved.

라우팅 판정기 순수 로직 단위 테스트 — 프롬프트 조립과 판정 파싱.
subprocess 는 개입하지 않는다.
"""

from __future__ import annotations

import pytest

from session_manager.routing import judge


class TestBuildJudgePrompt:
    def test_contains_all_sections(self) -> None:
        prompt = judge.build_judge_prompt(
            prompt="로그인 API가 500을 뱉는다",
            excerpt="user: 차트 고쳐줘\nassistant: 수정했습니다",
            sessions=[
                {
                    "name": "frontend",
                    "title": "차트 작업",
                    "summary": "React 차트 리팩토링 진행 중",
                    "last_accessed": "2026-08-01T00:00:00+00:00",
                },
                {"name": "backend", "title": "API", "summary": "JWT 교체 완료"},
            ],
            current_name="frontend",
        )
        assert "로그인 API가 500을 뱉는다" in prompt
        assert "user: 차트 고쳐줘" in prompt
        assert "- frontend (현재 세션): 차트 작업 — React 차트 리팩토링 진행 중" in prompt
        assert "- backend: API — JWT 교체 완료" in prompt

    def test_empty_excerpt_and_sessions(self) -> None:
        prompt = judge.build_judge_prompt(
            prompt="p", excerpt="", sessions=[], current_name=None
        )
        assert "[현재 세션 최근 대화] (없음)" in prompt
        assert "[세션 목록] (없음)" in prompt

    def test_missing_summary_rendered_as_placeholder(self) -> None:
        text = judge.format_sessions(
            [{"name": "s1", "title": "t", "summary": None}], current_name=None
        )
        assert "(요약 없음)" in text

    def test_mixing_score_shown_when_present(self) -> None:
        text = judge.format_sessions(
            [{"name": "s1", "title": "t", "summary": "x", "mixing_score": 2}],
            current_name=None,
        )
        assert "mixing_score: 2" in text

    def test_mixing_evidence_shown_next_to_score(self) -> None:
        # R3-C2: the raw score and its rooted-evidence quotes go to the
        # judge as-is — no threshold applied on the wrapper side.
        # R3-C2 — 원값 점수와 rooted 근거 인용이 그대로 판정자에게 간다.
        # 래퍼 측 임계 적용은 없다.
        text = judge.format_sessions(
            [
                {
                    "name": "s1",
                    "title": "t",
                    "summary": "x",
                    "mixing_score": 1,
                    "mixing_evidence": ["차트 얘기가 3턴 이어짐"],
                }
            ],
            current_name=None,
        )
        assert "mixing_score: 1" in text
        assert "차트 얘기가 3턴 이어짐" in text

    def test_empty_mixing_evidence_not_shown(self) -> None:
        text = judge.format_sessions(
            [
                {
                    "name": "s1",
                    "title": "t",
                    "summary": "x",
                    "mixing_score": 0,
                    "mixing_evidence": [],
                }
            ],
            current_name=None,
        )
        assert "mixing_evidence" not in text


class TestParseVerdict:
    def test_clean_json(self) -> None:
        verdict = judge.parse_verdict(
            '{"action":"SWITCH","target":"backend","confidence":0.9,'
            '"evidence":"JWT 교체 완료","reason":"인증 관련"}'
        )
        assert verdict is not None
        assert verdict.action == "SWITCH"
        assert verdict.target == "backend"
        assert verdict.confidence == 0.9
        assert verdict.evidence == "JWT 교체 완료"

    def test_fenced_json(self) -> None:
        # 실측 7회 중 1회는 지시에도 펜스를 붙였다 (docs/poc/R2-hook.md §9.3)
        verdict = judge.parse_verdict(
            '```json\n{"action":"STAY","target":null,"confidence":0.95,'
            '"evidence":null,"reason":"연속 작업"}\n```'
        )
        assert verdict is not None
        assert verdict.action == "STAY"
        assert verdict.target is None

    def test_garbage_returns_none(self) -> None:
        assert judge.parse_verdict("죄송합니다, 판정할 수 없습니다.") is None
        assert judge.parse_verdict("") is None
        assert judge.parse_verdict("{broken json") is None

    def test_invalid_action_returns_none(self) -> None:
        assert judge.parse_verdict('{"action":"MAYBE","confidence":1}') is None

    def test_non_dict_returns_none(self) -> None:
        assert judge.parse_verdict("[1, 2]") is None

    def test_switch_without_evidence_demoted_to_stay(self) -> None:
        verdict = judge.parse_verdict(
            '{"action":"SWITCH","target":"backend","confidence":0.9,'
            '"evidence":null,"reason":"감이 그렇다"}'
        )
        assert verdict is not None
        assert verdict.action == "STAY"
        assert verdict.reason.startswith("demoted_no_evidence")

    def test_switch_without_target_demoted_to_stay(self) -> None:
        verdict = judge.parse_verdict(
            '{"action":"SWITCH","target":null,"confidence":0.9,'
            '"evidence":"인용","reason":"r"}'
        )
        assert verdict is not None
        assert verdict.action == "STAY"

    def test_confidence_clamped_and_defaulted(self) -> None:
        high = judge.parse_verdict('{"action":"STAY","confidence":3.5}')
        assert high is not None and high.confidence == 1.0
        low = judge.parse_verdict('{"action":"STAY","confidence":-1}')
        assert low is not None and low.confidence == 0.0
        missing = judge.parse_verdict('{"action":"STAY"}')
        assert missing is not None and missing.confidence == 0.0

    def test_stay_factory(self) -> None:
        verdict = judge.Verdict.stay("judge_timeout")
        assert verdict.action == "STAY"
        assert verdict.confidence == 0.0
        assert verdict.reason == "judge_timeout"


class TestRefute:
    """Second-pass refutation prompt and parsing (R3-C4).

    2차 반박 검증 프롬프트·파싱 (R3-C4).
    """

    def test_build_prompt_contains_verdict_and_evidence(self) -> None:
        prompt = judge.build_refute_prompt(
            {"action": "SWITCH", "target": "backend", "evidence": "JWT 교체 완료"}
        )
        assert "반박하라" in prompt
        assert "JWT 교체 완료" in prompt
        assert "refuted" in prompt

    def test_parse_clean_true_and_false(self) -> None:
        assert judge.parse_refute('{"refuted": true, "reason": "근거 빈약"}') == {
            "refuted": True,
            "reason": "근거 빈약",
        }
        assert judge.parse_refute('{"refuted": false, "reason": "타당"}') == {
            "refuted": False,
            "reason": "타당",
        }

    def test_parse_fenced(self) -> None:
        parsed = judge.parse_refute('```json\n{"refuted": false, "reason": "r"}\n```')
        assert parsed is not None
        assert parsed["refuted"] is False

    def test_parse_garbage_returns_none(self) -> None:
        assert judge.parse_refute("반박할 수 없습니다") is None

    def test_parse_non_bool_refuted_returns_none(self) -> None:
        assert judge.parse_refute('{"refuted": "yes"}') is None


class TestWilsonLowerBound:
    def test_zero_samples_is_zero(self) -> None:
        assert judge.wilson_lower_bound(0, 0) == 0.0

    def test_all_successes_closed_form(self) -> None:
        # For p̂=1 the bound reduces to 1/(1+z²/n).
        # p̂=1 이면 하한은 1/(1+z²/n) 로 닫힌 형태가 된다.
        z = judge.WILSON_Z_ONE_SIDED_95
        n = 60
        assert judge.wilson_lower_bound(n, n) == pytest.approx(
            1 / (1 + z * z / n)
        )

    def test_bound_below_point_estimate(self) -> None:
        assert judge.wilson_lower_bound(9, 10) < 0.9

    def test_more_samples_tighter_bound(self) -> None:
        assert judge.wilson_lower_bound(60, 60) > judge.wilson_lower_bound(10, 10)


class TestCalibratedThreshold:
    def test_empty_pairs_none(self) -> None:
        assert judge.calibrated_threshold([], 0.05) is None

    def test_insufficient_samples_none(self) -> None:
        # 10/10 accepts: lower bound ≈ 0.787 < 0.95 — the bound itself
        # decides sample sufficiency (rule 8), no minimum-count constant.
        # 10/10 수용 — 하한 약 0.787 < 0.95. 표본 충분성은 하한 자체가
        # 판정한다 (규칙 8, 최소 개수 상수 없음).
        pairs = [(0.9, True)] * 10
        assert judge.calibrated_threshold(pairs, 0.05) is None

    def test_sufficient_accepts_yield_smallest_confidence(self) -> None:
        # 60/60 at 0.9 qualifies (bound 0.957); low-confidence rejects
        # below the candidate do not disqualify it.
        # 0.9 에서 60/60 수용은 자격 충족 (하한 0.957). 후보 아래의
        # 저확신 거부는 자격에 영향 없다.
        pairs = [(0.9, True)] * 60 + [(0.5, False)] * 10
        assert judge.calibrated_threshold(pairs, 0.05) == 0.9

    def test_smallest_qualifying_candidate_wins(self) -> None:
        pairs = [(0.9, True)] * 60 + [(0.7, True)] * 60
        # conf ≥ 0.7 subset = 120/120 → qualifies → 0.7 is smaller.
        # conf ≥ 0.7 부분집합 120/120 → 자격 충족 → 더 작은 0.7 선택.
        assert judge.calibrated_threshold(pairs, 0.05) == 0.7

    def test_rejects_at_high_confidence_block_qualification(self) -> None:
        pairs = [(0.9, True)] * 60 + [(0.9, False)] * 5
        assert judge.calibrated_threshold(pairs, 0.05) is None

    def test_looser_tolerance_lowers_the_bar(self) -> None:
        pairs = [(0.9, True)] * 20
        # 20/20: bound ≈ 0.881 — insufficient at 0.05, sufficient at 0.15.
        # 20/20 하한 약 0.881 — tolerance 0.05 엔 미달, 0.15 면 충족.
        assert judge.calibrated_threshold(pairs, 0.05) is None
        assert judge.calibrated_threshold(pairs, 0.15) == 0.9
