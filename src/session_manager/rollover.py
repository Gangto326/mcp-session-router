"""Handoff document management for session rollover (R4-C3).

세션 롤오버의 Handoff 문서 관리 (R4-C3).

When a conversation approaches its context limit, the LLM that knows
the work best — the one inside that conversation — is asked to produce
a resumption brief ("Handoff"). This module owns the request text (the
Plan §5 template, verbatim), the handoff file naming/paths, the
mechanical validation of a produced handoff, and the excerpt-based
fallback used when the LLM fails twice.

대화가 컨텍스트 한계에 다가가면, 작업을 가장 잘 아는 LLM — 그 대화 속
LLM — 에게 재개 지시서 ("Handoff") 작성을 요청한다. 이 모듈은 요청문
(Plan §5 템플릿 원문), handoff 파일 이름·경로, 산출물의 기계 검증,
그리고 LLM 이 2회 실패했을 때의 발췌 기반 폴백을 담당한다.

Design deviation from the original Plan wording (approved 2026-08-13):
the LLM does NOT write the file itself. An unattended dedicated turn
that calls the Write tool can stall on a permission dialog with nobody
there to answer it. Instead the LLM prints the handoff content as its
response body, and the wrapper extracts it from the transcript at turn
end and writes the file — no dialog can exist, and the deterministic
layer does the execution ("판단은 LLM, 실행은 래퍼").

Plan 원안 문구와의 의도적 차이 (2026-08-13 승인): LLM 이 파일을 직접
쓰지 않는다. 무인 전용 턴에서 Write 도구를 쓰면 권한 다이얼로그에
아무도 답할 수 없어 턴이 멈출 수 있다. 대신 LLM 은 handoff 내용을 응답
본문으로 출력하고, 래퍼가 턴 종료 시 transcript 에서 추출해 파일로
쓴다 — 다이얼로그가 원천적으로 없고, 실행은 결정적 계층이 맡는다
("판단은 LLM, 실행은 래퍼").
"""

from __future__ import annotations

import re
from pathlib import Path

from session_manager import debug_log
from session_manager.transcript_excerpt import read_tail_events

_SESSION_MANAGER_DIRNAME = ".session-manager"
_HANDOFFS_DIRNAME = "handoffs"

# Mandatory section headings the validator checks. §1 (resume point) and
# §2 (user requirements) are the sections the successor conversation
# cannot reconstruct from anywhere else — the rest degrade gracefully.
# 검증기가 확인하는 필수 섹션. §1 (재개 지점) 과 §2 (사용자 요구사항) 는
# 후계 대화가 다른 어디서도 복원할 수 없는 섹션이다 — 나머지는 빠져도
# 완만히 열화한다.
_REQUIRED_SECTION_RES = (
    re.compile(r"^##\s*1\.", re.MULTILINE),
    re.compile(r"^##\s*2\.", re.MULTILINE),
)

# Handoff request template (Plan §5 R4-C3 — original text, section
# numbers/titles must not be altered). The two format slots are the
# document skeleton parameters; the request wraps it with the response
# rule (print, don't write — see module docstring).
# Handoff 요청 템플릿 (Plan §5 R4-C3 — 원문, 섹션 번호·제목 변형 금지).
# format 슬롯은 문서 골격 파라미터이고, 요청문이 응답 규칙 (파일 쓰기
# 대신 본문 출력 — 모듈 docstring) 으로 감싼다.
_REQUEST_TEMPLATE = """\
[session-manager] 컨텍스트 한계가 가깝다. 아래 구조의 Handoff 문서를 작성하라.
과거 서술은 미래 작업에 필요한 만큼만 담아라.
규칙: 도구를 사용하지 말고, 응답 본문에 Handoff 문서 내용만 출력하라
(래퍼가 응답을 {relpath} 파일로 저장한다). 문서 외 인사·설명을 덧붙이지 마라.

# Handoff: {session} #{n}
## 1. 지금 바로 할 일 (재개 지점)     ← 다음 액션, 착수 파일, 접근. 가장 상세히
## 2. 사용자 요구사항                  ← 아래 requirements 목록을 검토·병합하고,
                                        대화에서 추가로 파악한 세션 한정 지시 포함
## 3. 남은 작업 (우선순위 순)
## 4. 제약·결정 사항                   ← 미래 작업을 구속하는 것만, 이유 포함
## 5. 완료된 것                        ← 한 줄 목록 + 파일 경로만
## 6. 이전 대화: {conversation_id}
이 세션이 사실상 2개 이상의 주제로 갈라져 있으면 ## 3~5를 주제별 하위 섹션으로 나눠라.
[축적된 requirements] {requirements}"""


def handoffs_dir(project_path: Path) -> Path:
    return Path(project_path) / _SESSION_MANAGER_DIRNAME / _HANDOFFS_DIRNAME


def handoff_path(project_path: Path, session: str, n: int) -> Path:
    return handoffs_dir(project_path) / f"{session}-{n}.md"


def next_handoff_number(project_path: Path, session: str) -> int:
    """Next generation number for *session*'s handoffs (1-based).

    *session* handoff 의 다음 세대 번호 (1부터).
    """
    highest = 0
    try:
        for path in handoffs_dir(project_path).glob(f"{session}-*.md"):
            suffix = path.stem[len(session) + 1 :]
            if suffix.isdigit():
                highest = max(highest, int(suffix))
    except OSError:
        pass
    return highest + 1


def build_request(
    project_path: Path,
    session: str,
    n: int,
    conversation_id: str,
    requirements: list[str],
) -> str:
    """The dedicated-turn prompt asking the LLM for the handoff body.

    LLM 에게 handoff 본문을 요청하는 전용 턴 프롬프트.
    """
    relpath = handoff_path(project_path, session, n).relative_to(
        Path(project_path)
    )
    formatted = (
        "\n".join(f"- {r}" for r in requirements) if requirements else "(없음)"
    )
    return _REQUEST_TEMPLATE.format(
        relpath=relpath,
        session=session,
        n=n,
        conversation_id=conversation_id,
        requirements=formatted,
    )


def validate_handoff_text(text: str) -> bool:
    """Mechanical check: both mandatory sections present.

    기계 검증 — 필수 섹션 둘 다 존재하는가.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    return all(pattern.search(text) for pattern in _REQUIRED_SECTION_RES)


def write_handoff(project_path: Path, session: str, n: int, text: str) -> Path:
    """Atomically persist a handoff body produced by the LLM (or fallback).

    LLM (또는 폴백) 이 산출한 handoff 본문을 원자적으로 저장한다.
    """
    path = handoff_path(project_path, session, n)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    debug_log.log(
        "ROLLOVER_HANDOFF",
        "WRAPPER",
        {"op": "write", "path": str(path), "chars": len(text)},
        session=session,
    )
    return path


def _is_after(timestamp: str, since_iso: str) -> bool:
    """Chronological comparison tolerant of 'Z' vs '+00:00' suffixes.

    'Z' 와 '+00:00' 접미를 모두 수용하는 시간 비교.
    """
    from datetime import datetime

    try:
        ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        since = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
    except ValueError:
        return False
    return ts > since


def _event_text(message: dict) -> str:
    """Plain text of a message's content (string or text blocks).

    메시지 content 의 평문 (문자열 또는 text 블록).
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(p for p in parts if p)


def check_trigger_turn(
    jsonl_path: Path,
    trigger: str,
    since_iso: str,
    max_lines: int = 200,
) -> tuple[str, str]:
    """Where does the handoff request stand in the transcript?

    handoff 요청이 transcript 에서 어떤 상태인가?

    Returns (status, text) with status one of:
    - "waiting":  the request's trigger has not appeared yet (child
      still booting / turn streaming) — NOT a failed attempt.
    - "answered": an assistant message follows the trigger; *text* is
      its body, ready for validation.
    - "missing":  a NEWER non-trigger user event exists but the trigger
      never arrived — the delivery was lost; the attempt failed.

    Validation must anchor on transcript content, not on screen edges:
    the dedicated-turn child's BOOT renders spinner frames that produce
    a busy falling edge BEFORE the trigger prompt is even delivered
    (measured — R4-C3 e2e race, 2026-08-13), so an edge-anchored check
    reads the previous conversation's reply and burns every attempt.
    Only user events with a timestamp after *since_iso* (the request's
    registration time) count — the pre-existing conversation always
    ends with an older user prompt.

    (status, text) 반환. status:
    - "waiting":  요청의 트리거가 아직 없다 (자식 부팅 중·턴 스트리밍
      중) — 실패 시도가 아니다.
    - "answered": 트리거 뒤에 assistant 메시지가 있다. *text* 가 그
      본문 — 검증 준비 완료.
    - "missing":  더 새로운 비트리거 user 이벤트는 있는데 트리거가 끝내
      안 왔다 — 전달 유실, 시도 실패.

    검증은 화면 에지가 아니라 transcript 내용에 앵커해야 한다: 전용 턴
    자식의 **부팅** 화면이 그리는 스피너가 트리거 전달 **이전에** 바쁨
    하강 에지를 만들고 (실측 — R4-C3 e2e 레이스, 2026-08-13), 에지 앵커
    검증은 직전 대화의 응답을 읽어 시도를 전부 태운다. *since_iso* (요청
    등록 시각) 이후 타임스탬프의 user 이벤트만 센다 — 기존 대화는 항상
    그보다 오래된 user 프롬프트로 끝나 있기 때문이다.
    """
    events = read_tail_events(jsonl_path, max_lines=max_lines)
    trigger_index: int | None = None
    newer_foreign_user = False
    for i, event in enumerate(events):
        if event.get("type") != "user":
            continue
        timestamp = event.get("timestamp")
        if not isinstance(timestamp, str) or not _is_after(timestamp, since_iso):
            # Unparsable or older-than-request events belong to the
            # previous life of this conversation.
            # 파싱 불가·요청 이전 이벤트는 이 대화의 이전 생애의 것.
            continue
        message = event.get("message")
        text = _event_text(message) if isinstance(message, dict) else ""
        if trigger in text:
            trigger_index = i
        else:
            newer_foreign_user = True
    if trigger_index is None:
        return ("missing", "") if newer_foreign_user else ("waiting", "")
    for event in events[trigger_index + 1 :]:
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        if not isinstance(message, dict):
            continue
        text = _event_text(message)
        if text.strip():
            return ("answered", text)
    return ("waiting", "")


def successor_injection(
    project_path: Path, session: str, n: int
) -> tuple[dict, str]:
    """(handoff dict, user prompt) for the successor conversation's first turn.

    후계 대화 첫 턴에 실릴 (handoff dict, 사용자 프롬프트).

    The dict rides the pending file and reaches the LLM as the
    ``[handoff]`` block (existing injection path, unchanged); the prompt
    is the Plan §5 read instruction — handoff file first, the shared
    static field and project context when present.

    dict 는 pending 파일을 타고 ``[handoff]`` 블록으로 LLM 에 도달한다
    (기존 주입 경로 무변경). 프롬프트는 Plan §5 의 읽기 지시 — handoff
    파일 우선, 공유 static field·project context 는 있으면.
    """
    relpath = str(
        handoff_path(project_path, session, n).relative_to(Path(project_path))
    )
    handoff = {
        "kind": "rollover",
        "from": session,
        "handoff_file": relpath,
        "read": [
            relpath,
            ".session-manager/static-field.json",
            ".session-manager/project-context.md",
        ],
    }
    prompt = (
        f"[session-manager 롤오버] 이전 대화가 컨텍스트 한계에 도달해 이 새 "
        f"대화로 이어졌다. {relpath} 를 읽고 (있으면 "
        ".session-manager/static-field.json 과 "
        ".session-manager/project-context.md 도), §1 재개 지점부터 작업을 "
        "이어가라."
    )
    return handoff, prompt


def build_fallback_handoff(
    session: str, n: int, conversation_id: str, excerpt: str
) -> str:
    """Excerpt-based fallback body — low quality, but rollover proceeds.

    발췌 기반 폴백 본문 — 품질은 낮아도 롤오버는 진행된다 (Plan R4-C3).

    The mandatory sections are present (so validation passes) but marked
    as fallback; the raw dialogue excerpt substitutes for the missing
    judgment.

    필수 섹션은 갖춰 검증을 통과하되 폴백임을 명시하고, 판단 대신 원시
    대화 발췌를 싣는다.
    """
    return (
        f"# Handoff: {session} #{n}\n"
        "## 1. 지금 바로 할 일 (재개 지점)\n"
        "(자동 발췌 폴백 — LLM handoff 생성 실패. 아래 §3 의 대화 발췌에서\n"
        "재개 지점을 판단하라.)\n"
        "## 2. 사용자 요구사항\n"
        "(폴백 — 세션 메타데이터의 requirements 필드를 참조하라.)\n"
        "## 3. 남은 작업 (우선순위 순)\n"
        "### 최근 대화 발췌 (원시)\n"
        f"{excerpt}\n"
        f"## 6. 이전 대화: {conversation_id}\n"
    )
