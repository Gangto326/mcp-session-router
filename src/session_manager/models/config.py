"""Runtime configuration model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_CLEANUP_PERIOD_DAYS = 30

# Routing modes (Plan §1.4). "confirm" is the default: the router only
# proposes switches (the LLM asks the user); "off" disables the
# routing gate entirely (hook prefilter passes everything through).
#
# Note: the UserPromptSubmit hook reads this key from config.json raw
# (not through this model) — the hook must survive any config state,
# and this model requires socket_path. Keep the default in sync via
# DEFAULT_ROUTING_MODE, which the hook imports.
#
# 라우팅 모드 (Plan §1.4). 기본은 "confirm" — 라우터는 전환을 제안만
# 하고 (LLM 이 사용자에게 질문), 고확신 자동 전환을 허용하며, "off" 는 라우팅 게이트 전체를 끈다
# (hook 프리필터가 전부 통과시킴).
#
# 참고: UserPromptSubmit hook 은 이 키를 모델이 아니라 config.json raw
# 로 읽는다 — hook 은 어떤 config 상태에서도 살아야 하는데 이 모델은
# socket_path 를 요구한다. 기본값은 hook 이 import 하는
# DEFAULT_ROUTING_MODE 로 동기화한다.
# "auto" was removed in R6-C3 (never activated — calibration data never
# accumulated). A config file still carrying "auto" is read as "confirm"
# defensively (see the mode loader).
# "auto" 는 R6-C3 에서 제거 (발동 이력 0 — 보정 데이터가 쌓인 적 없음).
# config 에 남은 "auto" 값은 방어적으로 "confirm" 으로 읽는다 (모드 로더 참조).
ROUTING_MODES = ("confirm", "off")
DEFAULT_ROUTING_MODE = "confirm"


# Rollover trigger (R4-C1). Both are empirical parameters (rule 8):
#
# - DEFAULT_ROLLOVER_THRESHOLD_PCT: the Claude Code team's proactive
#   compact guidance is 50-60% of the window (Thariq Shihipar), and
#   effective context is 50-65% of nominal (NVIDIA RULER). 60 is the
#   upper bound of that guidance; auto-compact's hard limit (~83%)
#   stays comfortably above the trigger.
# - DEFAULT_ROLLOVER_ABSOLUTE_CAP_TOKENS: quality degradation onset is
#   an absolute token count, not a percentage (Chroma "Context Rot" —
#   task-dependent onset at 32K~100K+); the 50-60% team guidance was
#   given against a 200K window = 100~120K absolute. The cap prevents
#   a 1M window from computing 60% = 600K, far past the degradation
#   zone. Actual trigger = min(window × pct, cap).
# - DEFAULT_CONTEXT_BUDGET_TOKENS: user override for the denominator
#   (window size). None = auto-detect (statusline collector first,
#   model mapping fallback — R4-C1).
#
# 롤오버 트리거 (R4-C1). 둘 다 경험 파라미터 (규칙 8):
#
# - DEFAULT_ROLLOVER_THRESHOLD_PCT: Claude Code 팀의 proactive compact
#   권고가 창의 50~60% (Thariq Shihipar), RULER 실효 컨텍스트가 공칭의
#   50~65% (NVIDIA RULER). 60 은 그 권고의 상한이며 auto-compact 하드
#   한계 (~83%) 보다 충분히 이르다.
# - DEFAULT_ROLLOVER_ABSOLUTE_CAP_TOKENS: 품질 저하 시작점은 %가 아니라
#   절대 토큰량 (Chroma "Context Rot" — 과제별 32K~100K대 시작); 팀 권고
#   50~60% 도 200K 창 기준 = 절대 100~120K. 1M 창에서 60% = 600K 처럼
#   저하 구간을 지나치는 것을 상한이 막는다. 실제 트리거 =
#   min(창 × pct, 상한).
# - DEFAULT_CONTEXT_BUDGET_TOKENS: 분모 (창 크기) 사용자 override.
#   None = 자동 감지 (statusline 수집기 우선, 모델 매핑 폴백 — R4-C1).
DEFAULT_ROLLOVER_THRESHOLD_PCT = 60
DEFAULT_ROLLOVER_ABSOLUTE_CAP_TOKENS = 120_000
DEFAULT_CONTEXT_BUDGET_TOKENS: int | None = None


@dataclass
class Config:
    socket_path: str
    cleanup_period_days: int = DEFAULT_CLEANUP_PERIOD_DAYS
    routing_mode: str = DEFAULT_ROUTING_MODE
    rollover_threshold_pct: int = DEFAULT_ROLLOVER_THRESHOLD_PCT
    rollover_absolute_cap_tokens: int = DEFAULT_ROLLOVER_ABSOLUTE_CAP_TOKENS
    context_budget_tokens: int | None = DEFAULT_CONTEXT_BUDGET_TOKENS

    def to_dict(self) -> dict[str, Any]:
        return {
            "socket_path": self.socket_path,
            "cleanup_period_days": self.cleanup_period_days,
            "routing_mode": self.routing_mode,
            "rollover_threshold_pct": self.rollover_threshold_pct,
            "rollover_absolute_cap_tokens": self.rollover_absolute_cap_tokens,
            "context_budget_tokens": self.context_budget_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        return cls(
            socket_path=data["socket_path"],
            cleanup_period_days=data.get(
                "cleanup_period_days", DEFAULT_CLEANUP_PERIOD_DAYS
            ),
            routing_mode=data.get("routing_mode", DEFAULT_ROUTING_MODE),
            rollover_threshold_pct=data.get(
                "rollover_threshold_pct", DEFAULT_ROLLOVER_THRESHOLD_PCT
            ),
            rollover_absolute_cap_tokens=data.get(
                "rollover_absolute_cap_tokens",
                DEFAULT_ROLLOVER_ABSOLUTE_CAP_TOKENS,
            ),
            context_budget_tokens=data.get(
                "context_budget_tokens", DEFAULT_CONTEXT_BUDGET_TOKENS
            ),
        )
