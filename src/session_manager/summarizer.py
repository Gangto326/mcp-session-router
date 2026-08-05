"""Background summarizer — refresh session summaries without the main LLM.

백그라운드 요약기 — 메인 LLM 의 협조 없이 세션 summary 를 갱신한다.

Why this exists: the previous design asked the in-session LLM to write a
summary while leaving a session, which silently failed on timeouts,
``/clear``, forced exits and races, leaving stale summaries that mislead
the routing judge. This module decouples that: leaving a session merely
drops a small task file into a queue, and a daemon-thread worker inside
the wrapper process later calls ``claude -p`` (a one-shot headless
invocation) to produce the summary from the transcript itself.

존재 이유: 기존 설계는 세션을 떠나는 시점에 세션 안의 LLM 에게 summary
작성을 부탁했는데, timeout·``/clear``·강제 종료·race 에서 조용히 실패해
낡은 summary 가 라우팅 판정을 오도했다. 이 모듈은 그것을 분리한다 —
세션을 떠날 때는 작은 작업 파일 하나를 큐에 떨어뜨릴 뿐이고, 래퍼
프로세스 안의 데몬 스레드 워커가 나중에 ``claude -p`` (단발 headless
호출) 로 transcript 로부터 직접 summary 를 만든다.

Design decisions from the PoC (docs/poc/R1-summarizer.md):

- **Excerpt path only** — both departed and active sessions are
  summarised from an ``extract_full_text()`` excerpt fed to a one-shot
  haiku call. The ``--resume`` path either pollutes the original
  transcript (plain resume) or costs up to 60x more (``--fork-session``,
  and >200k-token conversations cannot use haiku at all).
- **Instruction after transcript** — haiku ignores a leading instruction
  and continues the conversation instead; the summary prompt must place
  the transcript first and the instruction last.
- **Neutral cwd** — ``claude -p`` records a junk conversation JSONL under
  the cwd's project directory, which would corrupt the wrapper's
  mtime-based active-conversation tracking if run from the project root.
  The subprocess therefore runs from a dedicated neutral directory and
  its junk transcript is deleted right after.

PoC (docs/poc/R1-summarizer.md) 에서 확정한 설계 결정:

- **발췌 경로 단일화** — departed/active 모두 ``extract_full_text()``
  발췌를 haiku 단발 호출에 넘긴다. ``--resume`` 경로는 원본 오염 (일반
  resume) 또는 최대 60배 비용 (``--fork-session``, 20만 토큰 초과 대화는
  haiku 사용 불가) 문제가 있다.
- **지시문 후치** — haiku 는 앞에 놓인 지시를 무시하고 대화를 이어가
  버린다. 요약 프롬프트는 transcript 를 앞에, 지시를 뒤에 배치해야 한다.
- **중립 cwd** — ``claude -p`` 는 cwd 프로젝트 디렉토리에 정크 대화
  JSONL 을 남겨, 프로젝트 루트에서 실행하면 래퍼의 mtime 기반 활성 대화
  추적을 교란한다. subprocess 는 전용 중립 디렉토리에서 실행하고 정크
  transcript 는 직후 삭제한다.

Failure policy: a task is retried once; a second failure marks the task
file as failed (kept on disk for diagnosis). A malformed model response
is **never** saved — a stale summary is better than a wrong one.

실패 정책: 작업은 1회 재시도하고, 두 번째 실패 시 큐 파일에 실패 마킹해
디스크에 남긴다 (진단용). 형식이 깨진 모델 응답은 **절대** 저장하지
않는다 — 낡은 summary 가 잘못된 summary 보다 낫다.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from session_manager import debug_log
from session_manager.claude_conversation import encode_cwd
from session_manager.storage.file_store import SessionStore
from session_manager.transcript_excerpt import dialogue_length, extract_full_text

# ---- Queue layout --------------------------------------------------------
# 큐 배치.

# One JSON file per task under <project>/.session-manager/summary-queue/.
# File-based so pending work survives process crashes.
#
# <project>/.session-manager/summary-queue/ 아래 작업당 JSON 파일 하나.
# 파일 기반이라 프로세스가 죽어도 대기 작업이 살아남는다.
_SESSION_MANAGER_DIRNAME = ".session-manager"
QUEUE_DIRNAME = "summary-queue"

# Suffix appended when a worker claims a task. Claimed files fall out of
# the ``*.json`` listing, which is what hides them from other workers.
# 워커가 작업을 선점할 때 붙이는 접미사. 선점된 파일은 ``*.json`` 목록에서
# 빠지며, 그것이 다른 워커에게서 감추는 메커니즘이다.
CLAIM_SUFFIX = ".processing"

# Task kinds. departed/active run standalone through the excerpt path;
# rooting_check (R3-C2) never runs standalone — it waits in the queue
# and rides the next active refresh of the same session (see
# process_queue).
#
# 작업 종류. departed/active 는 발췌 경로로 단독 처리된다. rooting_check
# (R3-C2) 는 단독 처리되지 않는다 — 큐에서 대기하다가 같은 세션의 다음
# active 갱신에 편승한다 (process_queue 참조).
KIND_DEPARTED = "departed"
KIND_ACTIVE = "active"
KIND_ROOTING_CHECK = "rooting_check"
_SUPPORTED_KINDS = (KIND_DEPARTED, KIND_ACTIVE)

# Extra-field keys (R3-C2).
#
# ``EXTRA_FROM_REJECT`` marks the immediate active refresh enqueued by
# reject_switch. A rooting check never rides that task: at rejection
# time the rejected topic has just appeared, so "continued beyond a
# single exchange" cannot be true yet — the check rides the *next*
# refresh instead (periodic growth / /clear), which is the "거부 N턴 후"
# of the plan without introducing a numeric constant (rule 8).
# ``EXTRA_REJECTED_TOPIC`` carries the topic the rooting check asks about.
#
# extra 필드 키 (R3-C2).
#
# ``EXTRA_FROM_REJECT`` 는 reject_switch 가 즉시 적재한 active 갱신 표시.
# 정착 확인은 그 작업에 편승하지 않는다 — 거부 시점엔 거부된 주제가 방금
# 나타나 "단발 문답을 넘어 이어졌는가"가 참일 수 없으므로, *다음* 갱신
# (주기 증가량 / /clear) 에 편승한다. 이것이 수치 상수 없이 (규칙 8) 계획의
# "거부 N턴 후"를 구현하는 방식이다.
# ``EXTRA_REJECTED_TOPIC`` 은 정착 확인이 묻는 주제를 담는다.
EXTRA_FROM_REJECT = "from_reject"
EXTRA_REJECTED_TOPIC = "rejected_topic"

# ---- Headless call parameters -------------------------------------------
# headless 호출 파라미터.

SUMMARY_MODEL = "haiku"
# PoC measured 14–41s per call (CLI boot included); 120s leaves headroom.
# PoC 실측 호출당 14~41초 (CLI 부팅 포함). 120초면 여유가 있다.
SUBPROCESS_TIMEOUT_SECS = 120

# Neutral cwd for headless calls — junk transcripts land in this
# directory's project namespace instead of the real project's.
#
# headless 호출용 중립 cwd — 정크 transcript 가 실제 프로젝트가 아니라 이
# 디렉토리의 프로젝트 네임스페이스에 쌓이게 한다.
_NEUTRAL_CWD = Path.home() / ".session-manager" / "headless-tmp"

# Headless isolation (see run_headless_summary docstring for the measured
# rationale). Shared by every headless call the project makes — the routing
# judge (R2) must use these too.
# headless 격리 (실측 근거는 run_headless_summary docstring). 프로젝트의 모든
# headless 호출이 공유한다 — 라우팅 판정기 (R2) 도 동일하게 적용해야 한다.
_EMPTY_MCP_CONFIG = '{"mcpServers":{}}'
_SOCKET_ENV_VAR = "SESSION_MANAGER_SOCKET"

# Summary prompt. Rule text is verbatim from Plan.md R1-C2; the layout
# (transcript first, instruction after, non-participant notice) follows
# the PoC finding that haiku otherwise continues the conversation.
#
# 요약 프롬프트. 규칙 문구는 Plan.md R1-C2 원문 그대로이고, 배치
# (transcript 선행, 지시 후행, 비참여자 고지) 는 haiku 가 대화를 이어가
# 버리는 PoC 발견을 따른다.
_PROMPT_TEMPLATE = """[대화 기록 시작]
{excerpt}
[대화 기록 끝]

위는 한 코딩 세션의 대화 기록이다. 너는 이 대화의 참여자가 아니다.
기록을 읽고 이 세션의 작업을 요약하라.
규칙:
- summary: 2~4문장. 주 작업을 먼저, 부수 작업은 "이 과정에서 ~도"로 종속 서술.
  다룬 코드 영역(where), 수행 작업(what), 상태(done/in-progress/remaining) 포함.
- requirements: 사용자가 명시한 이 세션 한정 지시·제약을 목록으로.
  (전역 컨벤션이 아니라 이 작업에만 해당하는 것만)
- transcript에 실제로 있는 작업만 서술하라. 추측으로 범위를 넓히지 마라.
JSON으로만 응답: {{"summary": "...", "requirements": ["..."], "title": "..."}}"""

# Rooting-check question (R3-C2), appended after the summary instruction
# when a rooting_check task rides the refresh. Question text is verbatim
# from Plan.md R3-C2.
#
# 정착 확인 질문 (R3-C2) — rooting_check 작업이 갱신에 편승할 때 요약
# 지시 뒤에 덧붙인다. 질문 문구는 Plan.md R3-C2 원문 그대로다.
_ROOTING_QUESTION_TEMPLATE = (
    '추가 질문: 이 대화에서 "{rejected_topic}" 관련 작업이 단발 문답을 넘어\n'
    "이어졌는가? 복수 턴 진행 또는 파일 수정 동반 시에만 yes.\n"
    '{{"rooted": true|false, "evidence": "근거 인용|null"}}'
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class SummaryTask:
    """One unit of summarisation work, persisted as a queue file.

    큐 파일로 영속화되는 요약 작업 한 건.
    """

    session_name: str
    conversation_id: str
    kind: str
    requested_at: str = field(default_factory=_utc_now_iso)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_name": self.session_name,
            "conversation_id": self.conversation_id,
            "kind": self.kind,
            "requested_at": self.requested_at,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SummaryTask:
        return cls(
            session_name=data["session_name"],
            conversation_id=data["conversation_id"],
            kind=data["kind"],
            requested_at=data.get("requested_at", _utc_now_iso()),
            extra=dict(data.get("extra", {})),
        )


def _queue_dir(project_path: Path) -> Path:
    return Path(project_path) / _SESSION_MANAGER_DIRNAME / QUEUE_DIRNAME


def enqueue(project_path: Path, task: SummaryTask) -> Path | None:
    """Persist *task* as a new queue file; return its path, or None if a duplicate.

    *task* 를 새 큐 파일로 영속화하고 경로를 반환. 중복이면 None.

    Idempotent on (session, conversation, kind): a rapid A→B→A switch would
    otherwise queue the same work twice and pay for two identical summaries.
    An already-queued task summarises the transcript as it stands when the
    worker runs, so it subsumes the later request.

    (세션, conversation, kind) 에 대해 멱등 — A→B→A 처럼 빠르게 오가면 같은
    작업이 두 번 적재되어 동일 요약 비용을 두 번 낸다. 이미 대기 중인 작업은
    워커 실행 시점의 transcript 를 요약하므로 나중 요청을 포함한다.
    """
    queue_dir = _queue_dir(project_path)
    queue_dir.mkdir(parents=True, exist_ok=True)
    for _, pending in load_pending_tasks(project_path):
        if (
            pending.session_name == task.session_name
            and pending.conversation_id == task.conversation_id
            and pending.kind == task.kind
        ):
            debug_log.log(
                "SUMMARIZER",
                "WRAPPER",
                {
                    "op": "enqueue",
                    "result": "skipped_duplicate",
                    "kind": task.kind,
                    "conversation_id": task.conversation_id,
                },
                conv_id=task.conversation_id,
                session=task.session_name,
            )
            return None
    path = queue_dir / f"{uuid.uuid4()}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(task.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(path)
    debug_log.log(
        "SUMMARIZER",
        "WRAPPER",
        {
            "op": "enqueue",
            "path": str(path),
            "kind": task.kind,
            "conversation_id": task.conversation_id,
        },
        conv_id=task.conversation_id,
        session=task.session_name,
    )
    return path


def load_pending_tasks(project_path: Path) -> list[tuple[Path, SummaryTask]]:
    """Return (path, task) pairs for queue files not yet marked failed.

    실패 마킹되지 않은 큐 파일들의 (경로, 작업) 쌍을 반환.

    Ordered by ``requested_at`` so older work runs first. Corrupt queue
    files are skipped (and logged), never raised on.

    ``requested_at`` 순 정렬로 오래된 작업부터 처리. 손상된 큐 파일은
    건너뛰고 로그만 남긴다.
    """
    queue_dir = _queue_dir(project_path)
    if not queue_dir.is_dir():
        return []
    pending: list[tuple[Path, SummaryTask]] = []
    for path in queue_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("failed_at"):
                continue
            pending.append((path, SummaryTask.from_dict(data)))
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            debug_log.log(
                "SUMMARIZER",
                "WRAPPER",
                {
                    "op": "load_pending_tasks",
                    "result": "corrupt_task_skipped",
                    "path": str(path),
                    "error": str(exc),
                },
            )
    pending.sort(key=lambda pair: pair[1].requested_at)
    return pending


def _mark_failed(path: Path, task: SummaryTask, error: str) -> None:
    """Mark a queue file as failed in place (kept on disk for diagnosis).

    큐 파일에 실패를 마킹한다 (진단용으로 디스크에 남긴다).
    """
    data = task.to_dict()
    data["failed_at"] = _utc_now_iso()
    data["error"] = error
    try:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        # Marking failure must not raise — worst case the task is retried
        # on the next pass and fails again.
        # 실패 마킹 자체가 예외를 내면 안 된다 — 최악의 경우 다음 pass 에서
        # 재시도되어 다시 실패할 뿐이다.
        pass
    debug_log.log(
        "SUMMARIZER",
        "WRAPPER",
        {
            "op": "mark_failed",
            "path": str(path),
            "kind": task.kind,
            "error": error,
        },
        conv_id=task.conversation_id,
        session=task.session_name,
    )


def _conversation_jsonl_path(project_path: Path, conversation_id: str) -> Path:
    """Locate the Claude Code transcript for *conversation_id* of this project.

    이 프로젝트의 *conversation_id* 에 해당하는 Claude Code transcript 경로.
    """
    return (
        Path.home()
        / ".claude"
        / "projects"
        / encode_cwd(Path(project_path))
        / f"{conversation_id}.jsonl"
    )


def _sweep_junk_transcripts() -> None:
    """Delete every junk transcript left under the neutral cwd's project dir.

    중립 cwd 프로젝트 디렉토리에 남은 정크 transcript 를 모두 삭제.

    Swept before each headless call rather than after, because the response
    envelope that carries the junk session id is unavailable on timeout or
    parse failure — an after-only cleanup leaks transcripts containing the
    excerpted dialogue. Unlinking a file another process still has open is
    harmless on POSIX (that process keeps writing through its fd).

    호출 "후" 가 아니라 "전" 에 쓸어낸다 — 정크 session id 를 담은 응답
    envelope 은 타임아웃·파싱 실패 시 오지 않으므로, 사후 정리만으로는 발췌
    대화가 담긴 transcript 가 남는다. 다른 프로세스가 열어 둔 파일을 unlink
    해도 POSIX 에서는 무해하다 (그 프로세스는 fd 로 계속 쓴다).
    """
    junk_dir = Path.home() / ".claude" / "projects" / encode_cwd(_NEUTRAL_CWD)
    if not junk_dir.is_dir():
        return
    removed = 0
    for entry in junk_dir.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
            removed += 1
        except OSError:
            # Cleanup must never break summarisation.
            # 정리 실패가 요약을 깨뜨리지 않도록 한다.
            continue
    if removed:
        debug_log.log(
            "SUMMARIZER",
            "WRAPPER",
            {"op": "sweep_junk", "removed": removed, "dir": str(junk_dir)},
        )


def run_headless_summary(prompt: str) -> str | None:
    """Run ``claude -p`` once and return its ``result`` text, or None.

    ``claude -p`` 를 1회 실행하고 응답의 ``result`` 텍스트를 반환. 실패 시 None.

    Isolation applied to every headless call (measured, see below):

    - **No MCP servers.** ``claude -p`` otherwise loads user-scope MCP
      servers regardless of cwd — including session-manager itself, whose
      server would then connect back to the wrapper socket and whose tools
      the summariser model could call. Measured cost of that load: 30,021
      vs 6,750 input tokens for the same one-line prompt (~23K wasted per
      call). ``--strict-mcp-config`` with an empty config disables it.
    - **Socket env stripped**, so a stray MCP server spawned by any other
      means cannot reach the wrapper's socket.
    - **Prompt on stdin**, never argv — argv is world-readable via ``ps``
      and the prompt carries the excerpted conversation.

    모든 headless 호출에 적용하는 격리 (실측 근거 포함):

    - **MCP 서버 무로드.** 그렇지 않으면 cwd 와 무관하게 user scope MCP 서버가
      로드된다 — session-manager 자신도 포함되어, 그 서버가 래퍼 소켓에 다시
      접속하고 요약 모델이 세션 도구를 호출할 수 있게 된다. 실측 비용: 같은 한 줄
      프롬프트에서 입력 토큰 30,021 vs 6,750 (호출당 약 23K 낭비).
    - **소켓 환경 변수 제거** — 다른 경로로 MCP 서버가 떠도 래퍼 소켓에 닿지 못한다.
    - **프롬프트는 stdin** (argv 금지) — argv 는 ``ps`` 로 누구나 읽을 수 있는데
      프롬프트에는 발췌한 대화가 들어 있다.

    Any subprocess/JSON failure returns None (logged).
    subprocess/JSON 실패는 전부 None 반환 (로그 기록).
    """
    _NEUTRAL_CWD.mkdir(parents=True, exist_ok=True)
    # Sweep before the call — see _sweep_junk_transcripts for why not after.
    # 호출 전에 쓸어낸다 — 사후가 아닌 이유는 _sweep_junk_transcripts 참조.
    _sweep_junk_transcripts()
    env = {k: v for k, v in os.environ.items() if k != _SOCKET_ENV_VAR}
    try:
        proc = subprocess.run(
            [
                "claude",
                "-p",
                "--model",
                SUMMARY_MODEL,
                "--output-format",
                "json",
                "--strict-mcp-config",
                "--mcp-config",
                _EMPTY_MCP_CONFIG,
            ],
            input=prompt,
            cwd=_NEUTRAL_CWD,
            env=env,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        debug_log.log(
            "SUMMARIZER",
            "WRAPPER",
            {"op": "run_headless_summary", "result": "subprocess_error", "error": str(exc)},
        )
        return None
    # The CLI can emit either a JSON envelope (even for some errors) or a
    # bare plain-text error line — defend against both (PoC §4-6).
    # CLI 는 JSON envelope (일부 오류 포함) 또는 평문 오류 한 줄을 낼 수
    # 있다 — 둘 다 방어 (PoC §4-6).
    try:
        envelope = json.loads(proc.stdout)
    except json.JSONDecodeError:
        debug_log.log(
            "SUMMARIZER",
            "WRAPPER",
            {
                "op": "run_headless_summary",
                "result": "non_json_output",
                "returncode": proc.returncode,
                "stdout": debug_log.mask_text(proc.stdout),
                "stderr": debug_log.mask_text(proc.stderr),
            },
        )
        return None
    if not isinstance(envelope, dict) or envelope.get("is_error"):
        debug_log.log(
            "SUMMARIZER",
            "WRAPPER",
            {
                "op": "run_headless_summary",
                "result": "cli_error",
                "returncode": proc.returncode,
                "stdout": debug_log.mask_text(proc.stdout),
            },
        )
        return None
    result = envelope.get("result")
    return result if isinstance(result, str) else None


def _parse_summary_response(text: str) -> dict[str, Any] | None:
    """Parse the model's JSON answer, tolerating a markdown code fence.

    모델의 JSON 응답을 파싱. markdown 코드펜스로 감싼 경우도 허용.

    Returns None unless a non-empty string ``summary`` is present —
    the caller must then keep the existing summary untouched.

    비어 있지 않은 문자열 ``summary`` 가 없으면 None — 호출자는 기존
    summary 를 건드리지 않아야 한다.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence line and the trailing fence.
        # 여는 펜스 줄과 닫는 펜스를 제거.
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rsplit("```", 1)[0]
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        debug_log.log(
            "SUMMARIZER",
            "WRAPPER",
            {
                "op": "parse_summary_response",
                "result": "unparseable",
                "raw": debug_log.mask_text(text),
            },
        )
        return None
    summary = data.get("summary") if isinstance(data, dict) else None
    if not isinstance(summary, str) or not summary.strip():
        debug_log.log(
            "SUMMARIZER",
            "WRAPPER",
            {
                "op": "parse_summary_response",
                "result": "missing_summary",
                "raw": debug_log.mask_text(text),
            },
        )
        return None
    return data


def _parse_json_objects(text: str) -> list[dict[str, Any]]:
    """Parse every top-level JSON object in *text* (fence-tolerant).

    *text* 안의 최상위 JSON 객체를 전부 파싱한다 (코드펜스 허용).

    The rooting-check question shows its own answer shape, so the model
    may emit the summary JSON and the rooted JSON as two separate
    objects instead of one merged object. A whole-text ``json.loads``
    would then fail and lose the summary too — this scanner recovers
    both objects whichever way the model chose.

    정착 확인 질문은 자체 응답 형식을 제시하므로, 모델이 summary JSON 과
    rooted JSON 을 병합하지 않고 별도 객체 둘로 낼 수 있다. 전문
    ``json.loads`` 는 그 경우 실패해 summary 까지 잃는다 — 이 스캐너는
    모델이 어느 쪽을 택했든 두 객체를 모두 회수한다.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rsplit("```", 1)[0]
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    idx = 0
    while True:
        start = stripped.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(stripped, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        idx = end
    return objects


def _rooting_question(rooting_task: SummaryTask | None) -> str | None:
    """Render the rooting-check question for a riding task, if usable.

    편승 작업의 정착 확인 질문을 렌더링한다 (사용 가능할 때만).
    """
    if rooting_task is None:
        return None
    topic = rooting_task.extra.get(EXTRA_REJECTED_TOPIC)
    if not isinstance(topic, str) or not topic.strip():
        return None
    return _ROOTING_QUESTION_TEMPLATE.format(rejected_topic=topic.strip())


def _process_task(
    project_path: Path,
    task: SummaryTask,
    run: Callable[[str], str | None],
    transcript_dir: Path | None,
    rooting_task: SummaryTask | None = None,
) -> str | None:
    """Try to summarise one task. Returns None on success, error string on failure.

    작업 한 건의 요약을 시도. 성공 시 None, 실패 시 오류 문자열 반환.

    With *rooting_task* attached, the rooting question is appended to
    the same headless call and the mixing tally is updated from the
    ``rooted`` answer. A missing/odd ``rooted`` answer never fails the
    task — the summary still saves, the check outcome is just logged.

    *rooting_task* 가 붙으면 같은 headless 호출에 정착 질문을 덧붙이고
    ``rooted`` 응답으로 혼합도를 갱신한다. ``rooted`` 응답 누락·이상은
    작업 실패로 치지 않는다 — summary 는 저장하고 확인 결과는 로그만.
    """
    if task.kind not in _SUPPORTED_KINDS:
        return f"unsupported_kind: {task.kind}"
    if transcript_dir is not None:
        jsonl_path = transcript_dir / f"{task.conversation_id}.jsonl"
    else:
        jsonl_path = _conversation_jsonl_path(project_path, task.conversation_id)
    excerpt = extract_full_text(jsonl_path)
    if not excerpt:
        return "empty_excerpt"
    question = _rooting_question(rooting_task)
    prompt = _PROMPT_TEMPLATE.format(excerpt=excerpt)
    if question is not None:
        prompt = f"{prompt}\n\n{question}"
    response = run(prompt)
    if response is None:
        return "headless_call_failed"
    rooted_answer: dict[str, Any] | None = None
    if question is not None:
        # The model may merge both answers into one object or emit two
        # separate objects — recover summary and rooted from either shape.
        # 모델은 두 응답을 한 객체로 병합할 수도, 별도 객체 둘로 낼 수도
        # 있다 — 어느 형태든 summary 와 rooted 를 회수한다.
        objects = _parse_json_objects(response)
        parsed = next(
            (
                o
                for o in objects
                if isinstance(o.get("summary"), str) and o["summary"].strip()
            ),
            None,
        )
        rooted_answer = next((o for o in objects if "rooted" in o), None)
        if parsed is None:
            debug_log.log(
                "SUMMARIZER",
                "WRAPPER",
                {
                    "op": "parse_summary_response",
                    "result": "missing_summary",
                    "raw": debug_log.mask_text(response),
                },
            )
    else:
        parsed = _parse_summary_response(response)
    if parsed is None:
        return "unparseable_response"
    rooted = rooted_answer.get("rooted") if rooted_answer is not None else None
    rooted_evidence = (
        rooted_answer.get("evidence") if rooted_answer is not None else None
    )
    store = SessionStore(project_path)

    def apply(session: Any) -> None:
        session.summary = parsed["summary"].strip()
        title = parsed.get("title")
        if isinstance(title, str) and title.strip():
            session.title = title.strip()
        requirements = parsed.get("requirements")
        if isinstance(requirements, list):
            session.requirements = [
                r.strip() for r in requirements if isinstance(r, str) and r.strip()
            ]
        # Freshness marker only — ``last_accessed`` is deliberately NOT
        # touched: a background refresh is not a user access, and bumping it
        # would make idle sessions look recently used to the router.
        # 신선도 표시만 갱신 — ``last_accessed`` 는 의도적으로 건드리지 않는다.
        # 백그라운드 갱신은 사용자 접근이 아니며, 갱신하면 놀고 있는 세션이
        # 라우터에게 방금 쓴 세션처럼 보인다.
        session.summary_updated_at = _utc_now_iso()
        # Baseline for periodic refresh: how much dialogue this summary covers.
        # Scoped to the conversation it was measured in (see the model's field
        # comment for why the pairing matters).
        # 주기 갱신 기준값 — 이 요약이 포괄하는 대화의 양. 측정한 conversation
        # 범위로 한정한다 (쌍으로 두는 이유는 모델 필드 주석 참조).
        session.summary_dialogue_chars = dialogue_length(jsonl_path)
        session.summary_dialogue_conversation_id = task.conversation_id
        # Mixing tally (R3-C2): only an explicit boolean true counts.
        # rooted=false is a judge-calibration signal (logged below), and
        # anything else is a malformed answer — neither touches the score.
        # 혼합도 집계 (R3-C2) — 명시적 boolean true 만 집계한다.
        # rooted=false 는 판정 보정 신호 (아래 로그), 그 외는 형식 이상 —
        # 어느 쪽도 점수를 건드리지 않는다.
        if rooted is True:
            session.mixing_score += 1
            if isinstance(rooted_evidence, str) and rooted_evidence.strip():
                session.mixing_evidence.append(rooted_evidence.strip())

    # Locked read-modify-write (F15) — a concurrent MCP-side save (e.g.
    # transitions append) must not be lost under this summary update.
    # 잠금 하의 read-modify-write (F15) — 동시에 일어나는 MCP 측 저장
    # (transitions append 등) 이 이 요약 갱신에 유실되면 안 된다.
    session = store.mutate_session_by_name(task.session_name, apply)
    if session is None:
        return f"session_not_found: {task.session_name}"
    if question is not None:
        debug_log.log(
            "SUMMARIZER",
            "WRAPPER",
            {
                "op": "rooting_check",
                "rooted": rooted,
                "evidence": rooted_evidence,
                "answered": rooted_answer is not None,
                "mixing_score": session.mixing_score,
            },
            conv_id=task.conversation_id,
            session=task.session_name,
        )
    debug_log.log(
        "SUMMARIZER",
        "WRAPPER",
        {
            "op": "process_task",
            "result": "saved",
            "kind": task.kind,
            "summary_len": len(session.summary),
            "requirements": parsed.get("requirements"),
        },
        conv_id=task.conversation_id,
        session=task.session_name,
    )
    return None


def _claim(path: Path) -> Path | None:
    """Atomically claim a queue file; return the claimed path or None.

    큐 파일을 원자적으로 선점한다. 성공 시 선점된 경로, 실패 시 None.

    Two ccode instances on the same project share this queue, and paying
    twice for the same summary is real money. ``rename`` is atomic on
    POSIX, so exactly one claimant wins. Claimed files no longer match the
    ``*.json`` glob, hiding them from the other worker.

    같은 프로젝트에서 ccode 두 개가 이 큐를 공유하며, 같은 요약을 두 번
    결제하는 것은 실제 비용이다. POSIX 에서 ``rename`` 은 원자적이므로 정확히
    한 쪽만 선점에 성공한다. 선점된 파일은 ``*.json`` glob 에 걸리지 않아
    다른 워커에게서 감춰진다.
    """
    claimed = path.with_name(path.name + CLAIM_SUFFIX)
    try:
        path.rename(claimed)
    except OSError:
        return None
    return claimed


def sweep_stale_queue_files(project_path: Path, period_days: int) -> int:
    """Delete failed/abandoned queue files older than *period_days*.

    *period_days* 보다 오래된 실패·유기 큐 파일을 삭제하고 삭제 건수 반환.

    Failed tasks are kept on disk for diagnosis and claimed files can be
    orphaned by a crash; neither is ever cleaned by the normal path, so
    both accumulate forever without this. Reuses the project's existing
    retention period rather than introducing another time constant.

    실패 작업은 진단용으로 디스크에 남고, 선점된 파일은 크래시로 고아가 될
    수 있다. 정상 경로에서는 어느 쪽도 정리되지 않아 이 sweep 없이는 영원히
    쌓인다. 새 시간 상수를 만들지 않고 프로젝트의 기존 보존 기간을 재사용한다.
    """
    queue_dir = _queue_dir(project_path)
    if not queue_dir.is_dir():
        return 0
    cutoff = datetime.now(UTC).timestamp() - period_days * 86400
    removed = 0
    for path in queue_dir.iterdir():
        try:
            if path.stat().st_mtime >= cutoff:
                continue
            path.unlink()
            removed += 1
        except OSError:
            continue
    if removed:
        debug_log.log(
            "SUMMARIZER",
            "WRAPPER",
            {"op": "sweep_stale_queue_files", "removed": removed},
        )
    return removed


def _claim_rooting_check(
    project_path: Path, session_name: str
) -> tuple[Path, SummaryTask] | None:
    """Claim the oldest pending rooting check for *session_name*, if any.

    *session_name* 의 가장 오래된 대기 정착 확인을 선점한다 (있을 때).

    One check per refresh: with several pending, the oldest rides now
    and the rest ride later refreshes — a single question per call keeps
    the combined response parseable.

    갱신당 확인 1건 — 여럿이 대기 중이면 가장 오래된 것이 지금 편승하고
    나머지는 이후 갱신에 편승한다. 호출당 질문 하나가 결합 응답을
    파싱 가능하게 유지한다.
    """
    for path, pending in load_pending_tasks(project_path):
        if pending.kind != KIND_ROOTING_CHECK:
            continue
        if pending.session_name != session_name:
            continue
        topic = pending.extra.get(EXTRA_REJECTED_TOPIC)
        if not isinstance(topic, str) or not topic.strip():
            # Unusable check (no topic) — leave it for the stale sweep.
            # 사용 불가 확인 (주제 없음) — 오래된 파일 sweep 에 맡긴다.
            continue
        claimed = _claim(path)
        if claimed is not None:
            return claimed, pending
    return None


def _restore_claimed(claimed: Path) -> None:
    """Return a claimed queue file to pending state (undo the claim rename).

    선점된 큐 파일을 대기 상태로 되돌린다 (선점 rename 의 역방향).
    """
    try:
        claimed.rename(claimed.with_name(claimed.name.removesuffix(CLAIM_SUFFIX)))
    except OSError:
        # Restore failure leaves an orphaned claim file; the stale sweep
        # collects it eventually. Never raise from cleanup.
        # 복원 실패는 고아 선점 파일을 남기지만 오래된 파일 sweep 이 결국
        # 수거한다. 정리 경로에서 예외를 내지 않는다.
        pass


def process_queue(
    project_path: Path,
    run: Callable[[str], str | None] = run_headless_summary,
    transcript_dir: Path | None = None,
) -> int:
    """Process every pending task once; return the number summarised.

    대기 작업을 한 차례 전부 처리하고 요약 성공 건수를 반환.

    Tasks run strictly one at a time, each claimed atomically so a second
    ccode instance on the same project cannot process it too. A failing
    task is retried once within the same pass, then marked failed. *run*
    and *transcript_dir* are injectable for tests.

    Rooting checks (R3-C2) are never processed standalone: they wait in
    the queue and ride an active refresh of their session — except the
    refresh enqueued by the rejection itself (``EXTRA_FROM_REJECT``),
    which is too early to judge rooting. If the ride fails, the check is
    restored to pending and rides a later refresh.

    작업은 한 번에 하나씩 처리되며, 원자적으로 선점되어 같은 프로젝트의 두
    번째 ccode 인스턴스가 중복 처리하지 못한다. 실패한 작업은 같은 pass 안에서
    1회 재시도 후 실패 마킹. *run* 과 *transcript_dir* 는 테스트용 주입점.

    정착 확인 (R3-C2) 은 단독 처리되지 않는다 — 큐에서 대기하다 같은 세션의
    active 갱신에 편승하되, 거부 자신이 적재한 갱신 (``EXTRA_FROM_REJECT``)
    은 정착 판정에 너무 이르므로 제외한다. 편승한 갱신이 실패하면 대기
    상태로 복원되어 이후 갱신에 편승한다.
    """
    done = 0
    for path, task in load_pending_tasks(project_path):
        if task.kind == KIND_ROOTING_CHECK:
            # Waits for an active refresh to ride — see docstring.
            # active 갱신 편승 대기 — docstring 참조.
            continue
        claimed = _claim(path)
        if claimed is None:
            # Another worker took it (or it was consumed) — skip.
            # 다른 워커가 가져갔거나 이미 소비됨 — 건너뛴다.
            continue
        rooting: tuple[Path, SummaryTask] | None = None
        if task.kind == KIND_ACTIVE and not task.extra.get(EXTRA_FROM_REJECT):
            rooting = _claim_rooting_check(project_path, task.session_name)
        rooting_task = rooting[1] if rooting is not None else None
        error = _process_task(
            project_path, task, run, transcript_dir, rooting_task=rooting_task
        )
        if error is not None:
            error = _process_task(
                project_path, task, run, transcript_dir, rooting_task=rooting_task
            )
        if error is None:
            claimed.unlink(missing_ok=True)
            if rooting is not None:
                rooting[0].unlink(missing_ok=True)
            done += 1
        else:
            _mark_failed(claimed, task, error)
            if rooting is not None:
                _restore_claimed(rooting[0])
    return done


class SummarizerWorker:
    """Daemon-thread worker draining the summary queue inside the wrapper.

    래퍼 프로세스 안에서 요약 큐를 비우는 데몬 스레드 워커.

    The subprocess call dominates each task, so the GIL is irrelevant.
    ``wake()`` nudges the loop right after an enqueue; otherwise the
    queue is re-checked every *poll_interval* seconds (crash-recovery
    tasks written by other processes are picked up too).

    작업당 비용은 subprocess 호출이 지배적이므로 GIL 은 무관하다.
    ``wake()`` 는 enqueue 직후 루프를 즉시 깨우고, 그 외에는
    *poll_interval* 초마다 큐를 재확인한다 (다른 프로세스가 남긴 크래시
    복구 작업도 함께 집어간다).
    """

    def __init__(
        self,
        project_path: Path,
        poll_interval: float = 30.0,
        run: Callable[[str], str | None] = run_headless_summary,
    ) -> None:
        self._project_path = Path(project_path)
        self._poll_interval = poll_interval
        self._run = run
        self._wakeup = threading.Event()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="summarizer-worker", daemon=True
        )
        self._thread.start()

    def wake(self) -> None:
        self._wakeup.set()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stopping.set()
        self._wakeup.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                process_queue(self._project_path, run=self._run)
            except Exception as exc:
                # The worker must survive anything — a dead worker means
                # summaries silently stop updating.
                # 워커는 무슨 일이 있어도 살아남아야 한다 — 워커가 죽으면
                # summary 갱신이 조용히 멈춘다.
                debug_log.log(
                    "SUMMARIZER",
                    "WRAPPER",
                    {"op": "worker_loop", "result": "error", "error": str(exc)},
                )
            self._wakeup.wait(timeout=self._poll_interval)
            self._wakeup.clear()
