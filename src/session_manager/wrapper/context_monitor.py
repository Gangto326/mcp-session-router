"""Context-usage detection for rollover (R4-C1).

롤오버를 위한 컨텍스트 사용률 감지 (R4-C1).

At every turn end the wrapper asks this module "how full is the active
conversation?". The answer combines a numerator (tokens occupied) and a
denominator (window size), each resolved by priority:

denominator: config ``context_budget_tokens`` override
           → statusline collector record (``context.json``, only when it
             describes the *active* conversation — a stale record from a
             previous conversation must not leak in)
           → model-id mapping (nominal sizes; measured to be WRONG for
             plan-dependent variants like this environment's 1M Sonnet 5,
             which is exactly why the statusline source ranks above it —
             docs/poc/R4-rollover.md §P4-a')
           → None (absolute cap alone decides)

numerator:   same statusline record when fresh
           → transcript tail (last assistant event's usage; the sum
             ``input + cache_read + cache_creation`` equals /context's
             display within 0.2% — §P4-a)

trigger point = min(window × rollover_threshold_pct, absolute cap); the
cap alone when the window is unknown. Rationale for both constants sits
with them in models/config.py.

래퍼는 매 턴 종료마다 이 모듈에 "활성 대화가 얼마나 찼나"를 묻는다.
답은 분자 (점유 토큰) 와 분모 (창 크기) 의 조합이며 각각 우선순위로
결정된다:

분모: config ``context_budget_tokens`` override
    → statusline 수집 레코드 (``context.json``, *활성* 대화를 서술할
      때만 — 이전 대화의 낡은 레코드가 새어들면 안 된다)
    → 모델 id 매핑 (공칭 크기. 이 환경의 1M Sonnet 5 처럼 플랜 의존
      변형에서 틀림이 실측됐다 — 그래서 statusline 이 위 순위다,
      docs/poc/R4-rollover.md §P4-a')
    → None (절대 상한 단독 판단)

분자: 신선한 statusline 레코드
    → transcript 꼬리 (마지막 assistant 이벤트 usage.
      ``input + cache_read + cache_creation`` 합 = /context 표시와
      0.2% 이내 일치 — §P4-a)

트리거 = min(창 × rollover_threshold_pct, 절대 상한). 창 미상이면 상한
단독. 두 상수의 근거는 models/config.py 에 병기.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from session_manager import debug_log, statusline

# Numerator fields of an assistant event's usage dict (P4-a measured:
# their sum equals /context's display; output_tokens excluded).
# assistant 이벤트 usage dict 의 분자 필드 (P4-a 실측: 합 = /context
# 표시, output_tokens 제외).
USAGE_FIELDS = (
    "input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)

# Fallback mapping: model id → nominal window tokens. Nominal only —
# the standard window of every current Claude model is 200K and 1M
# variants carry the "[1m]" marker in their id (Claude Code model
# selector format). A plan-level 1M upgrade with an unmarked id (this
# environment's Sonnet 5, §P4-a') is undetectable here; the statusline
# source above this fallback carries the real value, and the absolute
# cap bounds the trigger regardless (min(200K×60%, 120K) = min(1M-truth
# trigger, 120K) = 120K — the cap dominates for any window ≥ 200K).
# 폴백 매핑: 모델 id → 공칭 창 토큰. 공칭만 담는다 — 현행 Claude 모델의
# 표준 창은 전부 200K 이고 1M 변형은 id 에 "[1m]" 마커를 단다 (Claude
# Code 모델 선택기 형식). 마커 없는 플랜 단위 1M 승격 (이 환경의
# Sonnet 5, §P4-a') 은 여기서 감지 불가 — 그 실값은 이 폴백보다 위
# 순위인 statusline 소스가 나르고, 어차피 절대 상한이 트리거를 지배한다
# (창 ≥ 200K 면 min(창×60%, 120K) = 120K).
NOMINAL_WINDOW_TOKENS = 200_000
ONE_MILLION_MARKER = "[1m]"
ONE_MILLION_WINDOW_TOKENS = 1_000_000


@dataclass
class ContextUsage:
    """One turn-end measurement. / 턴 종료 1회의 측정 결과."""

    used_tokens: int
    window_tokens: int | None
    trigger_tokens: int
    exceeded: bool
    numerator_source: str  # "statusline" | "transcript"
    denominator_source: str  # "override" | "statusline" | "mapping" | "cap_only"


def window_for_model(model_id: str | None) -> int | None:
    """Nominal window for a model id, None when unknown.

    모델 id 의 공칭 창 크기. 미상이면 None.
    """
    if not isinstance(model_id, str) or not model_id.startswith("claude"):
        return None
    if ONE_MILLION_MARKER in model_id:
        return ONE_MILLION_WINDOW_TOKENS
    return NOMINAL_WINDOW_TOKENS


def read_tail_usage_and_model(
    jsonl_path: Path, tail_bytes: int = 65536
) -> tuple[dict[str, Any] | None, str | None]:
    """Last assistant event's (usage, model) from the file tail only.

    파일 꼬리에서만 마지막 assistant 이벤트의 (usage, model) 를 읽는다.

    Runs on the PTY loop every turn end, so it must stay O(tail_bytes)
    no matter how large the transcript grows (the R1-C4 lesson: a full
    rescan of a tens-of-MB transcript stalls the terminal for hundreds
    of ms). An assistant event lands every turn, so a 64KB tail misses
    one only under pathological tool-result sizes — returning (None,
    None) then simply skips this turn's check; the next turn retries.

    매 턴 종료마다 PTY 루프에서 돌므로 transcript 가 아무리 커져도
    O(tail_bytes) 여야 한다 (R1-C4 교훈: 수십 MB 전체 재스캔은 터미널을
    수백 ms 멈춘다). assistant 이벤트는 매 턴 생기므로 64KB 꼬리에
    없는 경우는 병리적 도구 결과 크기뿐 — 그때 (None, None) 반환은 이번
    턴 검사를 건너뛸 뿐이고 다음 턴이 재시도한다.
    """
    try:
        with jsonl_path.open("rb") as fp:
            fp.seek(0, 2)
            size = fp.tell()
            fp.seek(max(0, size - tail_bytes))
            data = fp.read()
    except OSError:
        return None, None
    # The first line may be a partial (we sought mid-line); parse errors
    # are skipped anyway.
    # 첫 줄은 중간에서 잘렸을 수 있다 — 어차피 파싱 실패는 건너뛴다.
    usage: dict[str, Any] | None = None
    model: str | None = None
    for raw in data.decode("utf-8", errors="replace").splitlines():
        try:
            event = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(event, dict) or event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        if isinstance(message.get("usage"), dict):
            usage = message["usage"]
            model = message.get("model") if isinstance(
                message.get("model"), str
            ) else model
    return usage, model


def read_first_usage(jsonl_path: Path) -> int | None:
    """Birth footprint: the FIRST assistant event's usage sum.

    태생 점유량 — 첫 assistant 이벤트의 usage 합.

    A conversation can never shrink below what its very first turn
    already occupied (system prompt, tools, guide). If that birth
    footprint meets the rollover trigger, rolling over cannot improve
    anything — the successor would be born equally full and the wrapper
    would roll over forever (observed with an aggressively low
    ``context_budget_tokens`` — R4-C4 e2e, 4 rollovers in one run).
    Head-scan stops at the first hit; None when no assistant event.

    대화는 첫 턴이 이미 점유한 양 (시스템 프롬프트·도구·가이드) 아래로
    줄어들 수 없다. 태생 점유가 롤오버 트리거 이상이면 롤오버는 아무것도
    개선하지 못한다 — 후계도 똑같이 찬 채 태어나 래퍼가 영원히 롤오버를
    반복한다 (공격적으로 낮춘 ``context_budget_tokens`` 로 실관측 —
    R4-C4 e2e, 1회 실행에 롤오버 4번). 머리부터 첫 히트에서 중단. 없으면
    None.
    """
    try:
        with jsonl_path.open(encoding="utf-8") as fp:
            for raw in fp:
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                if (
                    not isinstance(event, dict)
                    or event.get("type") != "assistant"
                ):
                    continue
                message = event.get("message")
                if not isinstance(message, dict):
                    continue
                usage = message.get("usage")
                if not isinstance(usage, dict):
                    continue
                total = sum(
                    v
                    for f in USAGE_FIELDS
                    if isinstance((v := usage.get(f)), int)
                )
                if total > 0:
                    return total
    except OSError:
        return None
    return None


def _load_rollover_config(project_path: Path) -> tuple[int, int, int | None]:
    """(threshold_pct, cap_tokens, budget_override) from raw config.json.

    config.json raw 에서 (threshold_pct, cap_tokens, budget_override).

    Raw read (not the Config model, which requires socket_path) so the
    check survives any config state — same rationale as the hook.
    Config 모델이 아닌 raw 읽기 (모델은 socket_path 를 요구) — 어떤
    config 상태에서도 검사가 살아야 한다는 hook 과 같은 근거.
    """
    from session_manager.models.config import (
        DEFAULT_CONTEXT_BUDGET_TOKENS,
        DEFAULT_ROLLOVER_ABSOLUTE_CAP_TOKENS,
        DEFAULT_ROLLOVER_THRESHOLD_PCT,
    )

    path = project_path / ".session-manager" / "config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}

    def _int_or(key: str, default: int | None) -> int | None:
        value = data.get(key, default)
        return value if isinstance(value, int) and value > 0 else default

    return (
        _int_or("rollover_threshold_pct", DEFAULT_ROLLOVER_THRESHOLD_PCT),
        _int_or(
            "rollover_absolute_cap_tokens",
            DEFAULT_ROLLOVER_ABSOLUTE_CAP_TOKENS,
        ),
        _int_or("context_budget_tokens", DEFAULT_CONTEXT_BUDGET_TOKENS),
    )


def check_context_usage(
    project_path: Path, active_conv_id: str, jsonl_path: Path
) -> ContextUsage | None:
    """Measure the active conversation's context fullness once.

    활성 대화의 컨텍스트 사용률을 1회 측정한다.

    Returns None when no numerator is obtainable this turn (no fresh
    statusline record and no usage in the transcript tail) — the caller
    just tries again next turn end.

    이번 턴에 분자를 얻을 수 없으면 (신선한 statusline 레코드도,
    transcript 꼬리의 usage 도 없음) None — 호출자는 다음 턴 종료에
    다시 시도할 뿐이다.
    """
    threshold_pct, cap_tokens, budget_override = _load_rollover_config(
        project_path
    )

    record = statusline.read_context(project_path)
    fresh = (
        isinstance(record, dict)
        and record.get("conversation_id") == active_conv_id
    )

    used: int | None = None
    numerator_source = "statusline"
    model_id: str | None = None
    if fresh and isinstance(record.get("used_tokens"), int):
        used = record["used_tokens"]
        model_id = record.get("model_id")
    else:
        usage, model_id = read_tail_usage_and_model(jsonl_path)
        if isinstance(usage, dict):
            total = sum(
                v for f in USAGE_FIELDS if isinstance((v := usage.get(f)), int)
            )
            if total > 0:
                used = total
                numerator_source = "transcript"
    if used is None:
        return None

    window: int | None
    if budget_override is not None:
        window, denominator_source = budget_override, "override"
    elif fresh and isinstance(record.get("context_window_size"), int):
        window, denominator_source = record["context_window_size"], "statusline"
    else:
        window = window_for_model(model_id)
        denominator_source = "mapping" if window is not None else "cap_only"

    trigger = (
        min(window * threshold_pct // 100, cap_tokens)
        if window is not None
        else cap_tokens
    )
    usage_result = ContextUsage(
        used_tokens=used,
        window_tokens=window,
        trigger_tokens=trigger,
        exceeded=used >= trigger,
        numerator_source=numerator_source,
        denominator_source=denominator_source,
    )
    if usage_result.exceeded:
        debug_log.log(
            "CONTEXT_USAGE",
            "WRAPPER",
            {
                "used_tokens": used,
                "window_tokens": window,
                "trigger_tokens": trigger,
                "numerator_source": numerator_source,
                "denominator_source": denominator_source,
            },
            conv_id=active_conv_id,
        )
    return usage_result
