"""Extract dialogue text and usage info from Claude Code conversation JSONL.

Claude Code 대화 JSONL 에서 대화 텍스트와 usage 정보를 뽑아내는 모듈.

Claude Code persists each conversation as an append-only JSONL file
(``~/.claude/projects/<encoded-cwd>/<conversation-id>.jsonl``). Besides
the actual user/assistant messages it contains many bookkeeping events
(file-history snapshots, mode changes, tool results, ...). This module
filters those out and returns only the human-readable dialogue — the
raw material for the background summarizer (R1) and the routing judge
(R2) — plus the token-usage counters needed for context-fullness
detection (R4).

Claude Code 는 대화를 append 전용 JSONL 파일로 영속화한다. 그 안에는
실제 user/assistant 메시지 외에 부기용 이벤트 (파일 이력 스냅샷, 모드
변경, 도구 결과 등) 가 잔뜩 섞여 있다. 이 모듈은 그것들을 걸러내고
사람이 읽을 수 있는 대화만 반환한다 — 백그라운드 요약기 (R1) 와 라우팅
판정기 (R2) 의 원재료이며, 컨텍스트 사용률 감지 (R4) 용 토큰 usage
카운터도 함께 제공한다.

All field names were confirmed against real transcripts — see
``docs/poc/R1-summarizer.md`` §4. The JSONL format has no stability
guarantee, so every function parses defensively: on any failure it
returns an empty result / ``None`` and records the reason via
``debug_log`` instead of raising.

모든 필드명은 실제 transcript 실측으로 확정했다 — ``docs/poc/R1-summarizer.md``
§4 참조. JSONL 형식은 안정성 보장이 없으므로 모든 함수는 방어적으로
파싱한다: 어떤 실패에서도 예외 대신 빈 결과 / ``None`` 을 반환하고
사유를 ``debug_log`` 에 기록한다.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from session_manager import debug_log

# ---- JSONL field constants (confirmed in docs/poc/R1-summarizer.md §4) ----
# JSONL 필드 상수 (docs/poc/R1-summarizer.md §4 에서 실측 확정).

# Event types that carry dialogue. Everything else (system, ai-title,
# attachment, file-history-*, last-prompt, mode, permission-mode,
# queue-operation, ...) is bookkeeping and gets ignored.
#
# 대화를 담는 이벤트 타입. 그 외 (system, ai-title, attachment,
# file-history-*, last-prompt, mode, permission-mode, queue-operation
# 등) 는 부기용이므로 무시한다.
EVENT_TYPE_USER = "user"
EVENT_TYPE_ASSISTANT = "assistant"

# Assistant content block type that carries visible prose. ``thinking``
# and ``tool_use`` blocks are excluded from excerpts.
#
# 사람이 읽는 산문을 담는 assistant content 블록 타입. ``thinking`` 과
# ``tool_use`` 블록은 발췌에서 제외한다.
BLOCK_TYPE_TEXT = "text"

# A user event whose text starts with one of these prefixes was written by
# the CLI, not typed by the user as dialogue: slash-command records and
# interruption markers. Confirmed against real transcripts — the marker
# form appears as a *list* content block, which is why the list branch
# below must filter too.
#
# 텍스트가 이 프리픽스로 시작하는 user 이벤트는 사용자가 대화로 입력한 것이
# 아니라 CLI 가 쓴 것이다 — 슬래시 명령 기록과 중단 표식. 실제 transcript 로
# 확인했으며, 중단 표식은 *list* content 블록으로 나타나므로 아래 list 분기도
# 같은 필터를 적용해야 한다.
NOISE_PREFIXES = (
    "<command-name>",
    "<local-command-caveat>",
    "<local-command-stdout>",
    "[Request interrupted",
)

# ``message.usage`` keys of an assistant event. The context footprint of
# a turn is approximately the sum of the three input-side counters.
#
# assistant 이벤트의 ``message.usage`` 키. 한 턴의 컨텍스트 점유량은
# 입력측 카운터 3개의 합으로 근사된다.
USAGE_KEY_INPUT = "input_tokens"
USAGE_KEY_CACHE_READ = "cache_read_input_tokens"
USAGE_KEY_CACHE_CREATION = "cache_creation_input_tokens"
USAGE_KEY_OUTPUT = "output_tokens"


def read_tail_events(jsonl_path: Path, max_lines: int = 100) -> list[dict]:
    """Parse up to the last *max_lines* lines of a conversation JSONL.

    대화 JSONL 의 끝에서 최대 *max_lines* 줄을 JSON 파싱해 반환.

    Corrupt lines are skipped (the format has no stability guarantee and
    the active conversation's last line may be a partial write). Returns
    an empty list when the file is missing or unreadable.

    손상 줄은 건너뛴다 (형식 안정성 보장이 없고, 활성 대화의 마지막 줄은
    쓰다 만 상태일 수 있다). 파일이 없거나 읽을 수 없으면 빈 리스트 반환.
    """
    try:
        with jsonl_path.open(encoding="utf-8") as fp:
            tail = deque(fp, maxlen=max_lines)
    except OSError as exc:
        debug_log.log(
            "TRANSCRIPT_EXCERPT",
            "SYSTEM",
            {
                "op": "read_tail_events",
                "result": "unreadable",
                "path": str(jsonl_path),
                "error": str(exc),
            },
        )
        return []
    events: list[dict] = []
    skipped = 0
    for raw in tail:
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if isinstance(event, dict):
            events.append(event)
        else:
            skipped += 1
    if skipped:
        debug_log.log(
            "TRANSCRIPT_EXCERPT",
            "SYSTEM",
            {
                "op": "read_tail_events",
                "result": "partial",
                "path": str(jsonl_path),
                "skipped_lines": skipped,
                "parsed_events": len(events),
            },
        )
    return events


def _dialogue_text(event: dict) -> tuple[str, str] | None:
    """Return ``(role, text)`` if *event* is a dialogue message, else None.

    *event* 가 대화 메시지이면 ``(role, text)`` 를, 아니면 None 을 반환.

    Filters applied / 적용 필터:
    - only ``user`` / ``assistant`` event types
    - ``isMeta`` user events dropped (CLI-injected meta messages)
    - user content: plain string *and* ``text`` blocks of a list both kept
      (a user turn with an attached image is stored as a block list, and
      interruption markers arrive as list-of-text) — ``tool_result`` blocks
      dropped
    - CLI-written text (``NOISE_PREFIXES``) dropped from either shape
    - assistant: only ``text`` blocks kept (no thinking / tool_use)
    """
    event_type = event.get("type")
    if event_type not in (EVENT_TYPE_USER, EVENT_TYPE_ASSISTANT):
        return None
    if event.get("isMeta"):
        return None
    message = event.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if event_type == EVENT_TYPE_USER:
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = _join_text_blocks(content)
        else:
            return None
        if not text or text.startswith(NOISE_PREFIXES):
            return None
        return (EVENT_TYPE_USER, text)
    if not isinstance(content, list):
        return None
    text = _join_text_blocks(content)
    if not text:
        return None
    return (EVENT_TYPE_ASSISTANT, text)


def _join_text_blocks(content: list) -> str:
    """Concatenate the ``text`` blocks of a content list, ignoring the rest.

    content 리스트에서 ``text`` 블록만 이어 붙이고 나머지는 무시.
    """
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == BLOCK_TYPE_TEXT
    ]
    return "\n".join(p.strip() for p in parts if p and p.strip())


def extract_dialogue(
    events: list[dict], max_exchanges: int = 3, max_chars: int = 500
) -> str:
    """Format the last *max_exchanges* user↔assistant exchanges as text.

    마지막 *max_exchanges* 개의 user↔assistant 교환을 텍스트로 형식화.

    An "exchange" starts at a user message and spans the assistant
    messages that follow it. Each message is truncated to *max_chars*.
    Output format is ``user: ...\\nassistant: ...`` — the shape the
    routing judge's prompt expects. Returns "" when nothing qualifies.

    "교환" 은 user 메시지에서 시작해 그 뒤의 assistant 메시지들까지다.
    각 메시지는 *max_chars* 로 절단. 출력 형식은 라우팅 판정 프롬프트가
    기대하는 ``user: ...\\nassistant: ...``. 해당 메시지가 없으면 "" 반환.
    """
    lines: list[tuple[str, str]] = []
    for event in events:
        pair = _dialogue_text(event)
        if pair is not None:
            lines.append(pair)
    # Walk backwards counting user messages — the start of each exchange —
    # until max_exchanges of them are covered.
    #
    # 뒤에서부터 user 메시지 (각 교환의 시작점) 를 세어 max_exchanges 개가
    # 확보되는 지점까지 거슬러 올라간다.
    start = 0
    user_seen = 0
    for i in range(len(lines) - 1, -1, -1):
        if lines[i][0] == EVENT_TYPE_USER:
            user_seen += 1
            if user_seen >= max_exchanges:
                start = i
                break
    return "\n".join(
        f"{role}: {text[:max_chars]}" for role, text in lines[start:]
    )


def extract_full_text(jsonl_path: Path, max_chars: int = 30000) -> str:
    """Extract the whole dialogue (tail-truncated to *max_chars*) for summarising.

    요약 생성용 — 대화 전체를 같은 필터로 추출하고 뒤에서부터 *max_chars*
    로 절단해 반환.

    Streams the entire file (not just a tail window) so long assistant
    messages early in the conversation cannot push the parse off a
    line-count cliff; the char budget is applied to the joined result,
    keeping the most recent dialogue. Returns "" on any failure.

    파일 전체를 스트리밍한다 (줄 수 기반 tail 이 아니라). 문자 예산은
    합쳐진 결과에 적용되어 가장 최근 대화가 남는다. 실패 시 "" 반환.
    """
    lines: list[str] = []
    skipped = 0
    try:
        with jsonl_path.open(encoding="utf-8") as fp:
            for raw in fp:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if not isinstance(event, dict):
                    skipped += 1
                    continue
                pair = _dialogue_text(event)
                if pair is not None:
                    lines.append(f"{pair[0]}: {pair[1]}")
    except OSError as exc:
        debug_log.log(
            "TRANSCRIPT_EXCERPT",
            "SYSTEM",
            {
                "op": "extract_full_text",
                "result": "unreadable",
                "path": str(jsonl_path),
                "error": str(exc),
            },
        )
        return ""
    if skipped:
        debug_log.log(
            "TRANSCRIPT_EXCERPT",
            "SYSTEM",
            {
                "op": "extract_full_text",
                "result": "partial",
                "path": str(jsonl_path),
                "skipped_lines": skipped,
            },
        )
    full = "\n".join(lines)
    if len(full) <= max_chars:
        return full
    # Cut at a line boundary so the excerpt never opens mid-sentence with a
    # half message that has lost its "user:"/"assistant:" prefix. When a
    # single message is itself longer than the whole budget there is no
    # boundary to cut at — keep its tail rather than returning nothing.
    # 줄 경계에서 자른다 — 발췌가 "user:"/"assistant:" 접두사를 잃은 문장
    # 중간부터 시작하지 않도록. 단, 메시지 하나가 예산보다 길면 자를 경계가
    # 없으므로 빈 결과 대신 그 꼬리를 남긴다.
    tail = full[-max_chars:]
    newline = tail.find("\n")
    return tail[newline + 1 :] if newline != -1 else tail


def read_last_usage(jsonl_path: Path) -> dict[str, Any] | None:
    """Return the ``usage`` dict of the last assistant event, or None.

    마지막 assistant 이벤트의 ``usage`` dict 를 반환. 없으면 None.

    The caller (context-fullness detection, R4) approximates the current
    context footprint as ``input_tokens + cache_read_input_tokens +
    cache_creation_input_tokens`` of this dict.

    호출자 (컨텍스트 사용률 감지, R4) 는 이 dict 의 ``input_tokens +
    cache_read_input_tokens + cache_creation_input_tokens`` 합으로 현재
    컨텍스트 점유량을 근사한다.
    """
    last_usage: dict[str, Any] | None = None
    try:
        with jsonl_path.open(encoding="utf-8") as fp:
            for raw in fp:
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("type") != EVENT_TYPE_ASSISTANT:
                    continue
                message = event.get("message")
                if not isinstance(message, dict):
                    continue
                usage = message.get("usage")
                if isinstance(usage, dict):
                    last_usage = usage
    except OSError as exc:
        debug_log.log(
            "TRANSCRIPT_EXCERPT",
            "SYSTEM",
            {
                "op": "read_last_usage",
                "result": "unreadable",
                "path": str(jsonl_path),
                "error": str(exc),
            },
        )
        return None
    if last_usage is None:
        debug_log.log(
            "TRANSCRIPT_EXCERPT",
            "SYSTEM",
            {
                "op": "read_last_usage",
                "result": "no_usage",
                "path": str(jsonl_path),
            },
        )
    return last_usage
