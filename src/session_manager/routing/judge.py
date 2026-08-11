"""
Routing judge — prompt assembly and verdict parsing (pure logic).

The judge decides, for each user prompt, whether the current session
should keep it (STAY), another session should take it (SWITCH), a new
session is warranted (NEW), or the user must be asked (ASK). The actual
model call runs in a resident process managed by
``wrapper/judge_host.py``; this module only builds the judge input and
interprets its output, so everything here is unit-testable without a
subprocess.

라우팅 판정기 — 프롬프트 조립과 판정 파싱 (순수 로직).

판정기는 사용자 프롬프트마다 현재 세션 유지(STAY) / 타 세션 이관
(SWITCH) / 새 세션(NEW) / 사용자 질의(ASK)를 결정한다. 실제 모델 호출은
``wrapper/judge_host.py``가 관리하는 상주 프로세스에서 수행되며, 이
모듈은 판정 입력 조립과 출력 해석만 담당한다 — 따라서 전부 subprocess
없이 단위 테스트 가능하다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

# ---- Model / timing constants -------------------------------------------
# 모델·시간 상수.

JUDGE_MODEL = "haiku"

# Engineering parameters, derived from measurement (rule 8 — no guessed
# constants). Measured on warmed disposable processes with thinking
# suppressed (docs/poc/R2-hook.md §9.2):
#   - judgment round-trip, n=7: 1.63–3.09s (max 3.09s)
#   - warmup round, n=5: 3.54–4.21s (max 4.21s)
# Derivations:
#   JUDGE_TIMEOUT_SECS   = measured max 3.1s × 2 (API variance margin)
#   WARMUP_TIMEOUT_SECS  = measured max 4.2s × 2
#   HOOK_REPLY_TIMEOUT_SECS = JUDGE_TIMEOUT_SECS + 1s IPC margin — the
#     wrapper times out first and replies STAY, so the hook-side timeout
#     is a backstop, not the normal path.
#
# 공학 파라미터 — 실측 도출 (규칙 8, 추정 상수 금지). 웜업된 1회용
# 프로세스 + thinking 억제에서 실측 (docs/poc/R2-hook.md §9.2):
#   - 판정 왕복 n=7: 1.63~3.09s (최대 3.09s)
#   - 웜업 라운드 n=5: 3.54~4.21s (최대 4.21s)
# 도출식:
#   JUDGE_TIMEOUT_SECS   = 실측 최대 3.1s × 여유 2배
#   WARMUP_TIMEOUT_SECS  = 실측 최대 4.2s × 2배
#   HOOK_REPLY_TIMEOUT_SECS = 판정 타임아웃 + IPC 여유 1s — 정상 경로는
#     래퍼가 먼저 타임아웃해 STAY 를 회신하는 것이고, hook 측 값은 백스톱.
JUDGE_TIMEOUT_SECS = 6.2
WARMUP_TIMEOUT_SECS = 8.4
HOOK_REPLY_TIMEOUT_SECS = JUDGE_TIMEOUT_SECS + 1.0

# Hook-side reply timeout when an auto gate is attached to the request:
# the judge host may then run one extra refute round in the same warm
# process (R3-C4), so the wall time is up to two judgment round-trips.
# Derivation: 2 × JUDGE_TIMEOUT_SECS + the same 1s IPC margin.
# auto 게이트가 동봉된 요청의 hook 측 회신 타임아웃 — 판정 호스트가 같은
# 웜 프로세스에서 반박 라운드 1회를 추가로 돌릴 수 있으므로 (R3-C4) 벽시계
# 시간은 판정 왕복 최대 2회다. 도출식: 2 × JUDGE_TIMEOUT_SECS + IPC 여유 1s.
AUTO_HOOK_REPLY_TIMEOUT_SECS = 2 * JUDGE_TIMEOUT_SECS + 1.0

# ---- Actions -------------------------------------------------------------
# 판정 액션.

ACTION_STAY = "STAY"
ACTION_SWITCH = "SWITCH"
ACTION_NEW = "NEW"
ACTION_ASK = "ASK"
VALID_ACTIONS = (ACTION_STAY, ACTION_SWITCH, ACTION_NEW, ACTION_ASK)

# ---- Judge prompt --------------------------------------------------------
# 판정 프롬프트.

# Rule text is verbatim from Plan.md R2-C3. The final output-format line
# is an addition beyond the verbatim rule (reported deviation): with
# thinking suppressed, output length is the dominant latency term
# (docs/poc/R2-hook.md §9.1), and 1 of 7 measured runs emitted a
# markdown fence despite "JSON으로만" — so the format instruction is
# made explicit (and the parser still strips fences defensively).
#
# 규칙 문구는 Plan.md R2-C3 원문 그대로다. 마지막 출력 형식 줄은 원문에
# 없는 추가분 (보고된 변형): thinking 억제 하에서는 출력 길이가 지연의
# 지배 항이고 (docs/poc/R2-hook.md §9.1), 실측 7회 중 1회는 "JSON으로만"
# 지시에도 마크다운 펜스를 출력했다 — 형식 지시를 명시하되 파서도
# 방어적으로 펜스를 제거한다.
PROMPT_TEMPLATE = """너는 코딩 세션 라우터다. 사용자의 새 프롬프트가 어느 세션 소관인지 판정하라.
[현재 세션 최근 대화] {excerpt}
[세션 목록] {sessions}
[새 프롬프트] {prompt}
규칙:
- action: STAY | SWITCH | NEW | ASK
- SWITCH를 내려면 evidence에 대상 세션 summary의 실제 구절을 인용하라.
  인용할 구절이 없으면 SWITCH를 내리지 마라.
- 여러 세션이 그럴듯하면 ASK. 확신이 없으면 STAY.
- mixing_score가 높은 세션일수록 그 세션으로의 SWITCH confidence를 낮춰라.
JSON으로만 응답:
{{"action": "...", "target": "세션명|null", "confidence": 0.0~1.0,
 "evidence": "인용 구절|null", "reason": "한 문장"}}
마크다운 코드펜스 없이 한 줄의 raw JSON만 출력하라. reason은 한 문장으로 간결하게."""

# Warmup prompt for a freshly spawned judge process. Creates the system
# prompt cache so the real judgment runs warm (measured 5.6s cold vs
# 1.6-3.1s warm — docs/poc/R2-hook.md §9.2).
# 새로 spawn 된 판정 프로세스의 웜업 프롬프트. 시스템 프롬프트 캐시를
# 생성해 실제 판정이 warm 으로 돈다 (실측 cold 5.6s vs warm 1.6~3.1s).
WARMUP_PROMPT = "OK라고만 답하라."

# Second-pass refutation before an auto switch (R3-C4). Wording is
# verbatim from Plan.md. Runs as one extra turn in the SAME warm judge
# process right before its retirement: an independent process would be
# cold (measured 5.2s+ floor) and the retire/re-warm cycle (3.5-4.2s)
# makes a separate refute request nearly always hit "unavailable".
# Trade-off accepted and documented: refuting one's own judgment in the
# same conversation risks self-consistency bias.
# 자동 전환 직전의 2차 반박 검증 (R3-C4). 문구는 Plan.md 원문. 은퇴 직전의
# **같은 웜 프로세스**에 1턴 추가로 돌린다 — 독립 프로세스는 cold (실측
# 바닥 5.2s+) 이고, 은퇴·재웜업 주기 (3.5~4.2s) 탓에 별도 반박 요청은
# 거의 항상 unavailable 이 된다. 수용한 트레이드오프: 같은 대화 안에서
# 자기 판정을 반박하면 자기일관성 편향 위험이 있다 (명시 기록).
REFUTE_PROMPT_TEMPLATE = """다음 라우팅 판정을 반박하라: {verdict}
반박에 성공하면 refuted=true. {{"refuted": true|false, "reason": "..."}}"""


@dataclass
class Verdict:
    """One routing judgment result.

    라우팅 판정 결과 한 건.
    """

    action: str
    target: str | None = None
    confidence: float = 0.0
    evidence: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target": self.target,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "reason": self.reason,
        }

    @classmethod
    def stay(cls, reason: str) -> Verdict:
        """Fallback STAY verdict (timeouts, parse failures, demotions).

        폴백 STAY 판정 (타임아웃·파싱 실패·강등).
        """
        return cls(action=ACTION_STAY, confidence=0.0, reason=reason)


def format_sessions(
    sessions: list[dict[str, Any]], current_name: str | None
) -> str:
    """Render the active-session list block for the judge prompt.

    판정 프롬프트의 활성 세션 목록 블록을 렌더링한다. 현재 세션에는
    "(현재 세션)" 표시를 붙여 STAY 판정의 기준점을 명확히 한다.
    """
    if not sessions:
        return "(없음)"
    lines = []
    for s in sessions:
        name = s.get("name", "?")
        marker = " (현재 세션)" if current_name is not None and name == current_name else ""
        summary = s.get("summary") or "(요약 없음)"
        parts = [f"- {name}{marker}: {s.get('title', '')} — {summary}"]
        extras = []
        if s.get("last_accessed"):
            extras.append(f"last_accessed: {s['last_accessed']}")
        # Raw mixing signal (R3-C2): the score and its rooted-evidence
        # quotes are shown as-is; the prompt's rule ("높을수록 confidence
        # 를 낮춰라") lets the judge do the weighing — no threshold here.
        # 혼합도 원신호 (R3-C2) — 점수와 rooted 근거 인용을 그대로 표기.
        # 가중은 프롬프트 규칙("높을수록 confidence 를 낮춰라")에 따라
        # 판정기가 수행하며, 여기에 임계는 없다.
        if s.get("mixing_score") is not None:
            extras.append(f"mixing_score: {s['mixing_score']}")
        if s.get("mixing_evidence"):
            quotes = json.dumps(s["mixing_evidence"], ensure_ascii=False)
            extras.append(f"mixing_evidence: {quotes}")
        if extras:
            parts.append(f" ({', '.join(extras)})")
        lines.append("".join(parts))
    return "\n".join(lines)


def build_judge_prompt(
    prompt: str,
    excerpt: str,
    sessions: list[dict[str, Any]],
    current_name: str | None = None,
) -> str:
    """Assemble the full judge input.

    판정기 입력 전문을 조립한다.

    Precedents are deliberately NOT part of the judge input (R3-FIX2):
    measured 3/3, the model inverted their meaning and cited a rejection
    as evidence FOR switching. Suppressing repeat proposals is a MUST,
    so it moved to the deterministic layer — the judge host demotes a
    SWITCH whose target has a live precedent (판단이 아니라 보장).

    판례는 의도적으로 판정 입력에서 제외한다 (R3-FIX2): 실측 3/3 으로
    모델이 의미를 반전 해석해 거부 기록을 전환 **찬성** 근거로 인용했다.
    거부된 제안의 반복 억제는 "반드시 일어나야 하는 일"이므로 결정적
    계층으로 이전 — 판정 호스트가 판례 대상으로의 SWITCH 를 강등한다.
    """
    return PROMPT_TEMPLATE.format(
        excerpt=excerpt.strip() or "(없음)",
        sessions=format_sessions(sessions, current_name),
        prompt=prompt,
    )


def _strip_fences(text: str) -> str:
    """Cut the response down to the outermost JSON object.

    Handles markdown fences and stray prose: measurement showed 1 of 7
    runs wrapped the JSON in ```json fences despite instructions.

    응답에서 최외곽 JSON 객체만 잘라낸다. 마크다운 펜스·잡담 방어 —
    실측 7회 중 1회는 지시에도 ```json 펜스가 붙었다.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return ""
    return text[start : end + 1]


def parse_verdict(text: str) -> Verdict | None:
    """Parse the model response into a Verdict, or None if unusable.

    모델 응답을 Verdict 로 파싱한다. 사용 불가 응답이면 None.

    A SWITCH without evidence (or without target) is demoted to STAY —
    the prompt requires quoting the target summary, and an unquoted
    SWITCH is exactly the hallucination pattern the rule exists to stop.

    evidence(또는 target) 없는 SWITCH 는 STAY 로 강등한다 — 프롬프트가
    대상 summary 인용을 요구하며, 인용 없는 SWITCH 가 바로 그 규칙이
    막으려는 환각 패턴이다.
    """
    body = _strip_fences(text)
    if not body:
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    action = data.get("action")
    if action not in VALID_ACTIONS:
        return None

    target = data.get("target")
    if not isinstance(target, str) or not target:
        target = None
    evidence = data.get("evidence")
    if not isinstance(evidence, str) or not evidence:
        evidence = None
    reason = data.get("reason")
    if not isinstance(reason, str):
        reason = ""
    confidence_raw = data.get("confidence")
    if isinstance(confidence_raw, int | float):
        confidence = min(1.0, max(0.0, float(confidence_raw)))
    else:
        confidence = 0.0

    if action == ACTION_SWITCH and (evidence is None or target is None):
        return Verdict.stay(f"demoted_no_evidence: {reason}".strip())

    return Verdict(
        action=action,
        target=target,
        confidence=confidence,
        evidence=evidence,
        reason=reason,
    )


# ---- Second-pass refutation (R3-C4) --------------------------------------
# 2차 반박 검증 (R3-C4).


def build_refute_prompt(verdict: dict[str, Any]) -> str:
    """Render the refutation prompt for one verdict (evidence included).

    판정 1건 (evidence 포함) 에 대한 반박 프롬프트를 렌더링한다.
    """
    return REFUTE_PROMPT_TEMPLATE.format(
        verdict=json.dumps(verdict, ensure_ascii=False)
    )


def parse_refute(text: str) -> dict[str, Any] | None:
    """Parse the refutation answer; None if unusable.

    반박 응답을 파싱한다. 사용 불가면 None — 호출자는 반박 실패를
    refuted 와 동일하게 (confirm 강등) 다룬다.
    """
    body = _strip_fences(text)
    if not body:
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("refuted"), bool):
        return None
    reason = data.get("reason")
    return {
        "refuted": data["refuted"],
        "reason": reason if isinstance(reason, str) else "",
    }


# ---- Calibrated auto threshold (R3-C4) -----------------------------------
# 보정 기반 자동 전환 임계 (R3-C4).

# Engineering parameter: normal quantile for a ONE-SIDED 95% Wilson lower
# bound. We only act on the lower bound (is the acceptance rate provably
# above target?), so the test is one-sided; 95% is the standard
# confidence level, giving z = Φ⁻¹(0.95) = 1.645.
# 공학 파라미터 — **단측** 95% Wilson 하한의 정규 분위수. 우리는 하한만
# 사용하므로 (수용률이 목표치 위임이 입증되는가) 검정은 단측이고, 95% 는
# 표준 신뢰수준 → z = Φ⁻¹(0.95) = 1.645.
WILSON_Z_ONE_SIDED_95 = 1.645


def wilson_lower_bound(
    successes: int, n: int, z: float = WILSON_Z_ONE_SIDED_95
) -> float:
    """Wilson score interval lower bound for a binomial proportion.

    이항 비율의 Wilson score 구간 하한. n=0 이면 0.0 (증거 없음).
    """
    if n <= 0:
        return 0.0
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def calibrated_threshold(
    pairs: list[tuple[float, bool]], tolerance: float
) -> float | None:
    """
    Smallest confidence whose historical acceptance clears the target.

    From the (confidence, accepted) history, return the smallest
    candidate threshold t such that among past proposals with
    confidence >= t, the Wilson lower bound of the acceptance rate is
    >= 1 - tolerance. Returns None when no candidate qualifies — sample
    insufficiency is decided by the bound itself, not by an arbitrary
    minimum count (rule 8): with few samples the lower bound simply
    cannot reach the target.

    과거 수용률이 목표를 넘는 **최소** confidence 를 산출한다.

    (confidence, 수용) 이력에서, confidence >= t 인 과거 제안들의 수용률
    Wilson 하한이 1 - tolerance 이상인 최소 후보 t 를 반환한다. 만족하는
    후보가 없으면 None. 표본 충분성은 임의 최소 개수가 아니라 하한
    자체가 판정한다 (규칙 8) — 표본이 적으면 하한이 목표에 닿지 못한다.
    """
    if not pairs:
        return None
    target = 1.0 - tolerance
    qualifying = []
    for candidate in sorted({confidence for confidence, _ in pairs}):
        subset = [accepted for confidence, accepted in pairs if confidence >= candidate]
        if wilson_lower_bound(sum(subset), len(subset)) >= target:
            qualifying.append(candidate)
    return min(qualifying) if qualifying else None
