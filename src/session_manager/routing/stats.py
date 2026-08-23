"""Routing statistics for ``ccode --stats`` (R5-C3).

``ccode --stats`` 라우팅 통계 (R5-C3).

What this answers: "how well has the router behaved in this project?" —
how often it proposed a switch, how often the user accepted, how many
auto switches were undone with ``/back``, how often the prefilter let a
prompt reach the judge, how long a judgment takes, and how mixed each
session's topics are. These numbers are the measurement base for tuning
thresholds (rule 8: no constant without evidence).

이 모듈이 답하는 질문: "이 프로젝트에서 라우터가 얼마나 잘 동작했나" —
전환 제안 빈도, 사용자 수용 빈도, ``/back`` 으로 되돌린 auto 전환 수,
프리필터가 판정기까지 보낸 비율, 판정 소요 시간, 세션별 주제 혼합도.
임계값 조정의 측정 근거다 (규칙 8 — 근거 없는 상수 금지).

Two data sources of different nature / 성격이 다른 데이터 출처 둘:

1. ``.session-manager/routing-decisions.jsonl`` (``decision_log``) —
   always written. Proposals, accept/reject labels, ``/back``.
2. ``~/.session-manager/logs/<run_id>.ndjson`` (``debug_log``) — written
   only under ``SESSION_MANAGER_DEBUG=1``. Prefilter trigger rate, judge
   verdict distribution, judgment latency, precedent suppression. The
   log dir is shared by every project, so only runs whose
   ``WRAPPER_BOOT`` event names this project are read. Items that need
   this source are reported as "not measured" when no such run exists —
   never silently as zero.

   로그 디렉터리는 모든 프로젝트가 공유하므로 ``WRAPPER_BOOT`` 이벤트의
   project_path 가 이 프로젝트인 run 만 읽는다. 이 출처가 필요한 항목은
   run 이 없으면 "측정 안 됨" 으로 보고한다 — 조용히 0 으로 내지 않는다.

Window for "recent" / "최근" 창: the chronologically most recent HALF
of labeled pairs — the same proportional window ``get_routing_status``
uses (rule 8: reuse, no new constant).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from session_manager.routing import decision_log

STATS_FLAG = "--stats"
JSON_FLAG = "--json"

# decision_log event fields written since R5-C3 (see that module's
# docstring). Older lines lack them → counted as "unrecorded".
# R5-C3 부터 기록되는 decision_log 필드 — 옛 줄에는 없다 → "unrecorded".
_UNRECORDED = "unrecorded"

# Presentation label for items that need the debug log.
# 디버그 로그가 필요한 항목의 표시 문구.
NOT_MEASURED = "측정 안 됨 (SESSION_MANAGER_DEBUG=1 로 실행하면 수집됨)"


# ---- Data loading / 데이터 적재 ----


def load_debug_records(project_path: Path, log_dir: Path) -> list[dict[str, Any]]:
    """Every debug record from runs booted in *project_path*.

    *project_path* 에서 부팅된 run 들의 디버그 레코드 전부. 파일 1개가
    run 1개이며 ``WRAPPER_BOOT`` 의 project_path 로 프로젝트를 식별한다.
    손상 줄·읽기 실패는 건너뛴다.
    """
    if not log_dir.is_dir():
        return []
    wanted = _normalize_path(project_path)
    records: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.ndjson")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        run: list[dict[str, Any]] = []
        belongs = False
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            run.append(record)
            if record.get("category") == "WRAPPER_BOOT":
                booted = record.get("payload", {}).get("project_path")
                if isinstance(booted, str) and _normalize_path(Path(booted)) == wanted:
                    belongs = True
        if belongs:
            records.extend(run)
    return records


def _normalize_path(path: Path) -> str:
    try:
        return os.path.realpath(path)
    except OSError:
        return str(path)


# ---- Aggregation (pure) / 집계 (순수 함수) ----


def compute_stats(
    events: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    debug_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate raw inputs into the stats dict printed by ``--stats``.

    원 입력을 ``--stats`` 가 출력하는 dict 로 집계한다. *sessions* 는
    세션 파일의 dict (``SessionMetadata.to_dict()`` 형태). *debug_records*
    가 None 이면 디버그 항목은 None (측정 안 됨) 으로 채운다.
    """
    return {
        "proposals": _proposal_stats(events),
        "acceptance": _acceptance(events),
        "precedent": _precedent_stats(events),
        "debug": _debug_stats(debug_records) if debug_records else None,
        "sessions": _session_stats(sessions),
    }


def _proposal_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    proposals = [e for e in events if e.get("type") == "proposal"]
    by_mode = Counter(str(e.get("mode", _UNRECORDED)) for e in proposals)
    by_status = Counter(str(e.get("target_status", _UNRECORDED)) for e in proposals)
    return {
        "total": len(proposals),
        "by_mode": dict(by_mode),
        "auto_switches": by_mode.get("auto", 0),
        # Proposals that pointed at an ended or nonexistent session —
        # the premise behind dropping /retire, now countable.
        # 끝난·존재하지 않는 세션을 가리킨 제안 — /retire 제거의 전제,
        # 이제 셀 수 있다.
        "by_target_status": dict(by_status),
    }


def _acceptance(events: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = decision_log.labeled_pairs(events)
    proposals = sum(1 for e in events if e.get("type") == "proposal")
    accepted = sum(1 for _, ok in pairs if ok)
    labels = [e for e in events if e.get("type") == "label"]
    by_source = Counter(str(e.get("source", _UNRECORDED)) for e in labels)
    recent = pairs[len(pairs) // 2 :]
    return {
        "accepted": accepted,
        "rejected": len(pairs) - accepted,
        "unlabeled": proposals - len(pairs),
        "overall_rate": _rate(pairs),
        "recent_rate": _rate(recent),
        "recent_window": len(recent),
        "back_count": by_source.get("back", 0),
        "labels_by_source": dict(by_source),
    }


def _rate(pairs: list[tuple[float, bool]]) -> float | None:
    if not pairs:
        return None
    return sum(1 for _, ok in pairs if ok) / len(pairs)


def _precedent_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Rejections by (kept_in, target) pair — the precedent gate's unit.

    (kept_in, target) 쌍별 거부 수 — 판례 게이트의 단위. kept_in 이
    없는 옛 라벨은 ``unrecorded`` 로 따로 센다.
    """
    rejects = [
        e
        for e in events
        if e.get("type") == "label" and e.get("label") == decision_log.LABEL_REJECT
    ]
    pairs: Counter[str] = Counter()
    unrecorded = 0
    for e in rejects:
        kept_in = e.get("kept_in")
        if not isinstance(kept_in, str):
            unrecorded += 1
            continue
        pairs[f"{kept_in} → {e.get('target')}"] += 1
    return {"reject_pairs": dict(pairs), "unrecorded": unrecorded}


def _debug_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    prefilter = [r for r in records if r.get("category") == "HOOK_PREFILTER"]
    to_judge = sum(1 for r in prefilter if r.get("payload", {}).get("to_judge"))
    rules = Counter(
        str(r.get("payload", {}).get("rule"))
        for r in prefilter
        if not r.get("payload", {}).get("to_judge")
    )
    judge = [r.get("payload", {}) for r in records if r.get("category") == "JUDGE"]
    verdicts = [p for p in judge if p.get("op") == "verdict"]
    actions = Counter(str(p.get("verdict", {}).get("action")) for p in verdicts)
    elapsed = [p["elapsed_s"] for p in verdicts if isinstance(p.get("elapsed_s"), int | float)]
    suppressed = sum(1 for p in judge if p.get("op") == "precedent_suppress")
    runs = {r.get("run_id") for r in records if r.get("run_id")}
    return {
        "runs": len(runs),
        "prefilter": {
            "total": len(prefilter),
            "to_judge": to_judge,
            "trigger_rate": (to_judge / len(prefilter)) if prefilter else None,
            "skipped_by_rule": dict(rules),
        },
        "judge": {
            "verdicts": len(verdicts),
            "by_action": dict(actions),
            "avg_elapsed_ms": (round(sum(elapsed) / len(elapsed) * 1000) if elapsed else None),
            "precedent_suppressed": suppressed,
        },
    }


def _session_stats(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for s in sessions:
        score = s.get("mixing_score", 0)
        evidence = s.get("mixing_evidence", [])
        rows.append(
            {
                "name": str(s.get("name", "?")),
                "status": str(s.get("status", "active")),
                "mixing_score": score if isinstance(score, int) else 0,
                "mixing_evidence_count": (len(evidence) if isinstance(evidence, list) else 0),
            }
        )
    rows.sort(key=lambda r: (-r["mixing_score"], r["name"]))
    return rows


# ---- Presentation / 표시 ----


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.0f}%"


def _counter_line(counts: dict[str, int]) -> str:
    if not counts:
        return "—"
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))


def format_stats(stats: dict[str, Any]) -> str:
    """Render the stats dict as one human-readable table.

    통계 dict 를 사람이 읽는 표 한 장으로 만든다.
    """
    p, a, pr = stats["proposals"], stats["acceptance"], stats["precedent"]
    lines = [
        "라우팅 통계 (ccode --stats)",
        "",
        "[제안·수용]  출처: .session-manager/routing-decisions.jsonl",
        f"  전환 제안        {p['total']}  (모드별: {_counter_line(p['by_mode'])})",
        f"  수용 / 거부 / 무시  {a['accepted']} / {a['rejected']} / {a['unlabeled']}",
        f"  수용률 전체      {_pct(a['overall_rate'])}",
        f"  수용률 최근      {_pct(a['recent_rate'])}"
        f"  (최근 {a['recent_window']}건 — 라벨 표본의 최근 절반)",
        f"  자동 전환        {p['auto_switches']}",
        f"  /back 되돌림     {a['back_count']}",
        f"  제안 대상 상태   {_counter_line(p['by_target_status'])}",
        f"  거부 (현재 → 대상) {_counter_line(pr['reject_pairs'])}"
        + (f"  [기록 전 라벨 {pr['unrecorded']}]" if pr["unrecorded"] else ""),
        "",
        "[발동·판정]  출처: ~/.session-manager/logs (SESSION_MANAGER_DEBUG=1)",
    ]
    d = stats["debug"]
    if d is None:
        lines.append(f"  {NOT_MEASURED}")
    else:
        pf, j = d["prefilter"], d["judge"]
        lines += [
            f"  디버그 run       {d['runs']}",
            f"  발동률           {_pct(pf['trigger_rate'])}"
            f"  (프롬프트 {pf['total']}건 중 판정 {pf['to_judge']}건)",
            f"  프리필터 skip    {_counter_line(pf['skipped_by_rule'])}",
            f"  판정 분포        {_counter_line(j['by_action'])}",
            "  평균 판정 시간   "
            + ("—" if j["avg_elapsed_ms"] is None else f"{j['avg_elapsed_ms']} ms"),
            f"  판례 억제        {j['precedent_suppressed']}",
        ]
    lines += ["", "[세션별 혼합도]  출처: .session-manager/sessions/*.json"]
    if not stats["sessions"]:
        lines.append("  세션 없음")
    for s in stats["sessions"]:
        lines.append(
            f"  {s['name']:<20} {s['status']:<9} mixing_score {s['mixing_score']}"
            f"  (근거 {s['mixing_evidence_count']}건)"
        )
    return "\n".join(lines)


def run_stats(project_path: Path, log_dir: Path, as_json: bool) -> str:
    """Load every source for *project_path* and render the stats.

    *project_path* 의 모든 출처를 읽어 통계를 만든다. ``as_json`` 이면
    dict 를 JSON 으로, 아니면 표로 돌려준다.
    """
    events = decision_log.load_events(project_path)
    sessions = _load_session_dicts(project_path)
    debug_records = load_debug_records(project_path, log_dir)
    stats = compute_stats(events, sessions, debug_records or None)
    if as_json:
        return json.dumps(stats, ensure_ascii=False, indent=2)
    return format_stats(stats)


def _load_session_dicts(project_path: Path) -> list[dict[str, Any]]:
    # Raw files, not SessionStore: a corrupt session file must not crash
    # a read-only report.
    # SessionStore 대신 원시 파일 — 손상된 세션 파일 하나가 읽기 전용
    # 보고를 죽여서는 안 된다.
    sessions_dir = Path(project_path) / ".session-manager" / "sessions"
    if not sessions_dir.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(sessions_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            out.append(data)
    return out
