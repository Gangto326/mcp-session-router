"""Runtime configuration model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_CLEANUP_PERIOD_DAYS = 30

# Routing modes (Plan §1.4). "confirm" is the default: the router only
# proposes switches (the LLM asks the user); "auto" allows high-confidence
# automatic switching once calibration data exists; "off" disables the
# routing gate entirely (hook prefilter passes everything through).
#
# Note: the UserPromptSubmit hook reads this key from config.json raw
# (not through this model) — the hook must survive any config state,
# and this model requires socket_path. Keep the default in sync via
# DEFAULT_ROUTING_MODE, which the hook imports.
#
# 라우팅 모드 (Plan §1.4). 기본은 "confirm" — 라우터는 전환을 제안만
# 하고 (LLM 이 사용자에게 질문), "auto" 는 보정 데이터가 쌓인 뒤
# 고확신 자동 전환을 허용하며, "off" 는 라우팅 게이트 전체를 끈다
# (hook 프리필터가 전부 통과시킴).
#
# 참고: UserPromptSubmit hook 은 이 키를 모델이 아니라 config.json raw
# 로 읽는다 — hook 은 어떤 config 상태에서도 살아야 하는데 이 모델은
# socket_path 를 요구한다. 기본값은 hook 이 import 하는
# DEFAULT_ROUTING_MODE 로 동기화한다.
ROUTING_MODES = ("auto", "confirm", "off")
DEFAULT_ROUTING_MODE = "confirm"

# Policy parameter (rule 8, 3-way classification): the mis-switch rate
# the user is willing to tolerate in auto mode. The auto gate targets an
# acceptance rate of 1 - tolerance. The default (0.05 = 1 mis-switch
# allowed per 20) is a CHOICE about risk preference, not a measured
# fact — users adjust it in config.json.
# 정책 파라미터 (규칙 8, 3분류): auto 모드에서 사용자가 감수할 오전환율.
# auto 게이트의 목표 수용률 = 1 - tolerance. 기본값 (0.05 = 20회 중 1회
# 오전환 허용) 은 측정된 사실이 아니라 위험 선호의 **선택**이다 —
# 사용자가 config.json 에서 조정한다.
DEFAULT_AUTO_ERROR_TOLERANCE = 0.05


@dataclass
class Config:
    socket_path: str
    cleanup_period_days: int = DEFAULT_CLEANUP_PERIOD_DAYS
    routing_mode: str = DEFAULT_ROUTING_MODE
    auto_error_tolerance: float = DEFAULT_AUTO_ERROR_TOLERANCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "socket_path": self.socket_path,
            "cleanup_period_days": self.cleanup_period_days,
            "routing_mode": self.routing_mode,
            "auto_error_tolerance": self.auto_error_tolerance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        return cls(
            socket_path=data["socket_path"],
            cleanup_period_days=data.get(
                "cleanup_period_days", DEFAULT_CLEANUP_PERIOD_DAYS
            ),
            routing_mode=data.get("routing_mode", DEFAULT_ROUTING_MODE),
            auto_error_tolerance=data.get(
                "auto_error_tolerance", DEFAULT_AUTO_ERROR_TOLERANCE
            ),
        )
