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
from session_manager.transcript_excerpt import extract_full_text

# ---- Queue layout --------------------------------------------------------
# 큐 배치.

# One JSON file per task under <project>/.session-manager/summary-queue/.
# File-based so pending work survives process crashes.
#
# <project>/.session-manager/summary-queue/ 아래 작업당 JSON 파일 하나.
# 파일 기반이라 프로세스가 죽어도 대기 작업이 살아남는다.
_SESSION_MANAGER_DIRNAME = ".session-manager"
QUEUE_DIRNAME = "summary-queue"

# Task kinds. departed/active are handled identically today (excerpt
# path unified by the PoC); the split is kept because R3 attaches a
# rooting-check question to active-refresh tasks.
#
# 작업 종류. departed/active 는 현재 동일하게 처리되지만 (PoC 로 발췌
# 경로 단일화), R3 가 active 갱신 작업에 정착 확인 질문을 붙이므로 구분은
# 유지한다.
KIND_DEPARTED = "departed"
KIND_ACTIVE = "active"
KIND_ROOTING_CHECK = "rooting_check"
_SUPPORTED_KINDS = (KIND_DEPARTED, KIND_ACTIVE)

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


def enqueue(project_path: Path, task: SummaryTask) -> Path:
    """Persist *task* as a new queue file and return its path.

    *task* 를 새 큐 파일로 영속화하고 경로를 반환.
    """
    queue_dir = _queue_dir(project_path)
    queue_dir.mkdir(parents=True, exist_ok=True)
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


def _cleanup_junk_transcript(junk_session_id: str) -> None:
    """Delete the junk transcript a headless call recorded under the neutral cwd.

    headless 호출이 중립 cwd 아래에 남긴 정크 transcript 를 삭제.
    """
    junk_dir = Path.home() / ".claude" / "projects" / encode_cwd(_NEUTRAL_CWD)
    junk_file = junk_dir / f"{junk_session_id}.jsonl"
    junk_file.unlink(missing_ok=True)
    shutil.rmtree(junk_dir / junk_session_id, ignore_errors=True)


def run_headless_summary(prompt: str) -> str | None:
    """Run ``claude -p`` once and return its ``result`` text, or None.

    ``claude -p`` 를 1회 실행하고 응답의 ``result`` 텍스트를 반환. 실패 시 None.

    Runs from the neutral cwd and deletes the junk transcript the call
    leaves behind. Any subprocess/JSON failure returns None (logged).

    중립 cwd 에서 실행하고 호출이 남긴 정크 transcript 를 삭제한다.
    subprocess/JSON 실패는 전부 None 반환 (로그 기록).
    """
    _NEUTRAL_CWD.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [
                "claude",
                "-p",
                "--model",
                SUMMARY_MODEL,
                "--output-format",
                "json",
                prompt,
            ],
            cwd=_NEUTRAL_CWD,
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
    if isinstance(envelope, dict) and envelope.get("session_id"):
        _cleanup_junk_transcript(str(envelope["session_id"]))
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


def _process_task(
    project_path: Path,
    task: SummaryTask,
    run: Callable[[str], str | None],
    transcript_dir: Path | None,
) -> str | None:
    """Try to summarise one task. Returns None on success, error string on failure.

    작업 한 건의 요약을 시도. 성공 시 None, 실패 시 오류 문자열 반환.
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
    response = run(_PROMPT_TEMPLATE.format(excerpt=excerpt))
    if response is None:
        return "headless_call_failed"
    parsed = _parse_summary_response(response)
    if parsed is None:
        return "unparseable_response"
    store = SessionStore(project_path)
    session = store.load_session_by_name(task.session_name)
    if session is None:
        return f"session_not_found: {task.session_name}"
    session.summary = parsed["summary"].strip()
    title = parsed.get("title")
    if isinstance(title, str) and title.strip():
        session.title = title.strip()
    # ``requirements`` is logged but not persisted until the model gains
    # the field in R1-C6.
    # ``requirements`` 는 R1-C6 에서 모델에 필드가 생기기 전까지 로그만 남긴다.
    store.save_session(session)
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


def process_queue(
    project_path: Path,
    run: Callable[[str], str | None] = run_headless_summary,
    transcript_dir: Path | None = None,
) -> int:
    """Process every pending task once; return the number summarised.

    대기 작업을 한 차례 전부 처리하고 요약 성공 건수를 반환.

    Tasks run strictly one at a time. Each failing task is retried once
    within the same pass, then marked failed. *run* and *transcript_dir*
    are injectable for tests.

    작업은 엄격히 한 번에 하나씩 처리된다. 실패한 작업은 같은 pass 안에서
    1회 재시도 후 실패 마킹. *run* 과 *transcript_dir* 는 테스트용 주입점.
    """
    done = 0
    for path, task in load_pending_tasks(project_path):
        error = _process_task(project_path, task, run, transcript_dir)
        if error is not None:
            error = _process_task(project_path, task, run, transcript_dir)
        if error is None:
            path.unlink(missing_ok=True)
            done += 1
        else:
            _mark_failed(path, task, error)
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
