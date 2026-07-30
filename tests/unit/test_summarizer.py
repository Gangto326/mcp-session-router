"""
Unit tests for the background summarizer.

Verifies queue persistence (enqueue / pending listing / failure marking),
the one-retry policy, response parsing (code fence, malformed JSON), the
"never save a bad response" rule, and the worker thread lifecycle. The
headless CLI call is mocked throughout except for its own envelope tests.

백그라운드 요약기 단위 테스트.

큐 영속화 (enqueue / 대기 목록 / 실패 마킹), 1회 재시도 정책, 응답 파싱
(코드펜스, 깨진 JSON), "잘못된 응답은 저장하지 않는다" 규칙, 워커 스레드
생명주기를 검증한다. headless CLI 호출은 자체 envelope 테스트를 제외하면
전부 mock.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from session_manager import summarizer
from session_manager.models import SessionMetadata
from session_manager.storage.file_store import SessionStore

# ---- fixture helpers -----------------------------------------------------
# 픽스처 헬퍼.

CONV_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

GOOD_RESPONSE = json.dumps(
    {
        "summary": "라우터 개선 작업을 진행했다.",
        "requirements": ["테스트 필수"],
        "title": "라우터 개선",
    },
    ensure_ascii=False,
)


def _write_transcript(transcript_dir: Path, conv_id: str = CONV_ID) -> Path:
    """Write a minimal dialogue transcript the excerpt filter accepts.

    발췌 필터가 통과시키는 최소한의 대화 transcript 를 기록.
    """
    transcript_dir.mkdir(parents=True, exist_ok=True)
    events = [
        {"type": "user", "message": {"content": "라우터를 고쳐줘"}},
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "고쳤습니다"}]},
        },
    ]
    path = transcript_dir / f"{conv_id}.jsonl"
    path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A tmp project with one registered session named ``work``.

    ``work`` 세션 하나가 등록된 tmp 프로젝트.
    """
    store = SessionStore(tmp_path)
    store.init_project()
    session = SessionMetadata.new(name="work", title="이전 제목", summary="이전 요약")
    store.save_session(session)
    return tmp_path


@pytest.fixture
def transcripts(tmp_path: Path) -> Path:
    transcript_dir = tmp_path / "transcripts"
    _write_transcript(transcript_dir)
    return transcript_dir


def _task(kind: str = summarizer.KIND_DEPARTED) -> summarizer.SummaryTask:
    return summarizer.SummaryTask(
        session_name="work", conversation_id=CONV_ID, kind=kind
    )


class RunRecorder:
    """Mock for the headless call recording prompts, replaying canned answers.

    headless 호출 mock — 프롬프트를 기록하고 준비된 응답을 순서대로 반환.
    """

    def __init__(self, responses: list[str | None]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str | None:
        self.prompts.append(prompt)
        if not self.responses:
            return None
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


# ---- queue persistence ---------------------------------------------------


class TestQueue:
    def test_enqueue_writes_task_file(self, project: Path) -> None:
        path = summarizer.enqueue(project, _task())
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["session_name"] == "work"
        assert data["conversation_id"] == CONV_ID
        assert data["kind"] == "departed"
        assert data["requested_at"]

    def test_pending_sorted_by_requested_at(self, project: Path) -> None:
        newer = summarizer.SummaryTask(
            session_name="work",
            conversation_id=CONV_ID,
            kind="active",
            requested_at="2026-07-30T12:00:00+00:00",
        )
        older = summarizer.SummaryTask(
            session_name="work",
            conversation_id=CONV_ID,
            kind="departed",
            requested_at="2026-07-30T11:00:00+00:00",
        )
        summarizer.enqueue(project, newer)
        summarizer.enqueue(project, older)
        kinds = [t.kind for _, t in summarizer.load_pending_tasks(project)]
        assert kinds == ["departed", "active"]

    def test_failed_marked_tasks_excluded(self, project: Path) -> None:
        path = summarizer.enqueue(project, _task())
        summarizer._mark_failed(path, _task(), "boom")
        assert summarizer.load_pending_tasks(project) == []
        # The file itself stays on disk for diagnosis.
        # 파일 자체는 진단용으로 디스크에 남는다.
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["failed_at"]
        assert data["error"] == "boom"

    def test_corrupt_task_file_skipped(self, project: Path) -> None:
        summarizer.enqueue(project, _task())
        queue_dir = project / ".session-manager" / "summary-queue"
        (queue_dir / "broken.json").write_text("{not json", encoding="utf-8")
        assert len(summarizer.load_pending_tasks(project)) == 1

    def test_no_queue_dir(self, tmp_path: Path) -> None:
        assert summarizer.load_pending_tasks(tmp_path) == []


# ---- process_queue -------------------------------------------------------


class TestProcessQueue:
    def test_success_updates_session_and_consumes_task(
        self, project: Path, transcripts: Path
    ) -> None:
        task_path = summarizer.enqueue(project, _task())
        run = RunRecorder([GOOD_RESPONSE])
        done = summarizer.process_queue(project, run=run, transcript_dir=transcripts)
        assert done == 1
        assert not task_path.exists()
        session = SessionStore(project).load_session_by_name("work")
        assert session is not None
        assert session.summary == "라우터 개선 작업을 진행했다."
        assert session.title == "라우터 개선"
        # Prompt layout: transcript first, instruction after (PoC finding).
        # 프롬프트 배치 — transcript 선행, 지시 후행 (PoC 발견 반영).
        assert run.prompts[0].startswith("[대화 기록 시작]")
        assert "user: 라우터를 고쳐줘" in run.prompts[0]

    def test_success_persists_requirements_and_freshness(
        self, project: Path, transcripts: Path
    ) -> None:
        store = SessionStore(project)
        before = store.load_session_by_name("work")
        assert before is not None
        summarizer.enqueue(project, _task())
        run = RunRecorder([GOOD_RESPONSE])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 1
        session = store.load_session_by_name("work")
        assert session is not None
        assert session.requirements == ["테스트 필수"]
        assert session.summary_updated_at is not None
        # A background refresh is not a user access — last_accessed must
        # stay untouched so idle sessions don't look freshly used.
        # 백그라운드 갱신은 사용자 접근이 아니다 — last_accessed 는 그대로
        # 유지되어야 놀고 있는 세션이 방금 쓴 것처럼 보이지 않는다.
        assert session.last_accessed == before.last_accessed

    def test_non_string_requirements_entries_dropped(
        self, project: Path, transcripts: Path
    ) -> None:
        summarizer.enqueue(project, _task())
        response = json.dumps(
            {
                "summary": "요약.",
                "requirements": ["유효한 항목", 42, "  ", None],
                "title": "제목",
            },
            ensure_ascii=False,
        )
        run = RunRecorder([response])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 1
        session = SessionStore(project).load_session_by_name("work")
        assert session is not None
        assert session.requirements == ["유효한 항목"]

    def test_code_fenced_response_accepted(
        self, project: Path, transcripts: Path
    ) -> None:
        summarizer.enqueue(project, _task())
        run = RunRecorder([f"```json\n{GOOD_RESPONSE}\n```"])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 1

    def test_retry_once_then_succeed(self, project: Path, transcripts: Path) -> None:
        summarizer.enqueue(project, _task())
        run = RunRecorder([None, GOOD_RESPONSE])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 1
        assert len(run.prompts) == 2

    def test_persistent_failure_marks_task(
        self, project: Path, transcripts: Path
    ) -> None:
        task_path = summarizer.enqueue(project, _task())
        run = RunRecorder([None])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 0
        assert len(run.prompts) == 2  # original + one retry / 원 호출 + 재시도 1회
        data = json.loads(task_path.read_text(encoding="utf-8"))
        assert data["error"] == "headless_call_failed"

    def test_bad_response_never_saved(self, project: Path, transcripts: Path) -> None:
        summarizer.enqueue(project, _task())
        run = RunRecorder(["요약이 아니라 잡담입니다"])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 0
        session = SessionStore(project).load_session_by_name("work")
        assert session is not None
        assert session.summary == "이전 요약"
        assert session.title == "이전 제목"

    def test_missing_summary_field_never_saved(
        self, project: Path, transcripts: Path
    ) -> None:
        summarizer.enqueue(project, _task())
        run = RunRecorder([json.dumps({"title": "제목만 있음"}, ensure_ascii=False)])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 0
        session = SessionStore(project).load_session_by_name("work")
        assert session is not None
        assert session.summary == "이전 요약"

    def test_missing_transcript_fails_without_calling_model(
        self, project: Path, tmp_path: Path
    ) -> None:
        task_path = summarizer.enqueue(project, _task())
        run = RunRecorder([GOOD_RESPONSE])
        empty_dir = tmp_path / "no-transcripts"
        assert summarizer.process_queue(project, run=run, transcript_dir=empty_dir) == 0
        assert run.prompts == []
        data = json.loads(task_path.read_text(encoding="utf-8"))
        assert data["error"] == "empty_excerpt"

    def test_unsupported_kind_fails_without_calling_model(
        self, project: Path, transcripts: Path
    ) -> None:
        task_path = summarizer.enqueue(project, _task(kind=summarizer.KIND_ROOTING_CHECK))
        run = RunRecorder([GOOD_RESPONSE])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 0
        assert run.prompts == []
        data = json.loads(task_path.read_text(encoding="utf-8"))
        assert data["error"].startswith("unsupported_kind")

    def test_unknown_session_marks_task(self, project: Path, transcripts: Path) -> None:
        task = summarizer.SummaryTask(
            session_name="ghost", conversation_id=CONV_ID, kind="departed"
        )
        task_path = summarizer.enqueue(project, task)
        run = RunRecorder([GOOD_RESPONSE])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 0
        data = json.loads(task_path.read_text(encoding="utf-8"))
        assert data["error"].startswith("session_not_found")


# ---- run_headless_summary (subprocess envelope handling) -----------------


class _FakeProc:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


class TestRunHeadlessSummary:
    @pytest.fixture(autouse=True)
    def _no_junk_cleanup(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep the test away from the real ~/.claude junk directory.

        실제 ~/.claude 정크 디렉토리를 건드리지 않게 격리.
        """
        self.cleaned: list[str] = []
        monkeypatch.setattr(
            summarizer, "_cleanup_junk_transcript", self.cleaned.append
        )

    def _patch_subprocess(
        self, monkeypatch: pytest.MonkeyPatch, result: Any
    ) -> None:
        def fake_run(*args: Any, **kwargs: Any) -> Any:
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(summarizer.subprocess, "run", fake_run)

    def test_success_returns_result_and_cleans_junk(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        envelope = json.dumps(
            {"is_error": False, "session_id": "junk-id", "result": GOOD_RESPONSE}
        )
        self._patch_subprocess(monkeypatch, _FakeProc(envelope))
        assert summarizer.run_headless_summary("p") == GOOD_RESPONSE
        assert self.cleaned == ["junk-id"]

    def test_cli_error_envelope_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        envelope = json.dumps(
            {"is_error": True, "session_id": "junk-id", "result": "Prompt is too long"}
        )
        self._patch_subprocess(monkeypatch, _FakeProc(envelope, returncode=1))
        assert summarizer.run_headless_summary("p") is None
        # Even a failed call records a junk transcript — still cleaned.
        # 실패한 호출도 정크 transcript 를 남기므로 역시 정리한다.
        assert self.cleaned == ["junk-id"]

    def test_plain_text_error_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_subprocess(
            monkeypatch, _FakeProc("No conversation found with session ID: x", 1)
        )
        assert summarizer.run_headless_summary("p") is None

    def test_timeout_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_subprocess(
            monkeypatch, subprocess.TimeoutExpired(cmd="claude", timeout=1)
        )
        assert summarizer.run_headless_summary("p") is None

    def test_spawn_failure_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_subprocess(monkeypatch, OSError("claude not on PATH"))
        assert summarizer.run_headless_summary("p") is None


# ---- worker thread -------------------------------------------------------


class TestSummarizerWorker:
    def test_processes_after_wake_and_survives_errors(
        self, project: Path, transcripts: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # transcript_dir is not exposed on the worker (production always
        # uses the real location), so redirect the resolver instead.
        # 워커는 transcript_dir 를 노출하지 않으므로 (프로덕션은 항상 실제
        # 위치 사용) 경로 해석 함수를 우회시킨다.
        monkeypatch.setattr(
            summarizer,
            "_conversation_jsonl_path",
            lambda _p, conv_id: transcripts / f"{conv_id}.jsonl",
        )
        run = RunRecorder([GOOD_RESPONSE])
        worker = summarizer.SummarizerWorker(project, poll_interval=60.0, run=run)
        worker.start()
        try:
            summarizer.enqueue(project, _task())
            worker.wake()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if summarizer.load_pending_tasks(project) == []:
                    break
                time.sleep(0.05)
            session = SessionStore(project).load_session_by_name("work")
            assert session is not None
            assert session.summary == "라우터 개선 작업을 진행했다."
        finally:
            worker.stop()

    def test_start_is_idempotent_and_stop_joins(self, project: Path) -> None:
        worker = summarizer.SummarizerWorker(project, poll_interval=60.0)
        worker.start()
        worker.start()
        worker.stop()
        assert worker._thread is None
