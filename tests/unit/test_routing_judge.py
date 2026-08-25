"""
Unit tests for the routing judge's pure logic.

Prompt assembly and verdict parsing — no subprocess involved.

라우팅 판정기 순수 로직 단위 테스트 — 프롬프트 조립과 판정 파싱.
subprocess 는 개입하지 않는다.
"""

from __future__ import annotations

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
