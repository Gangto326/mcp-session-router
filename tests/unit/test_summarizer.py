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
import os
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


def _failed_records(project: Path) -> list[dict[str, Any]]:
    """Read every failure-marked queue record.

    실패 마킹된 큐 레코드를 모두 읽는다. 처리 중 선점으로 파일명이 바뀌므로
    이름이 아니라 내용으로 찾는다.
    """
    queue_dir = project / ".session-manager" / "summary-queue"
    records: list[dict[str, Any]] = []
    for path in queue_dir.iterdir():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("failed_at"):
            records.append(data)
    return records


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
        assert path is not None
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

    def test_duplicate_task_skipped(self, project: Path) -> None:
        """A→B→A must not pay for the same summary twice.

        A→B→A 왕복이 같은 요약 비용을 두 번 내지 않게 한다.
        """
        first = summarizer.enqueue(project, _task())
        second = summarizer.enqueue(project, _task())
        assert first is not None
        assert second is None
        assert len(summarizer.load_pending_tasks(project)) == 1

    def test_different_kind_is_not_duplicate(self, project: Path) -> None:
        summarizer.enqueue(project, _task(kind=summarizer.KIND_DEPARTED))
        assert summarizer.enqueue(project, _task(kind=summarizer.KIND_ACTIVE)) is not None
        assert len(summarizer.load_pending_tasks(project)) == 2

    def test_requeue_allowed_after_consumption(
        self, project: Path, transcripts: Path
    ) -> None:
        """Dedup keys on *pending* work only — a later refresh must still run.

        중복 판정은 *대기 중* 작업에만 적용 — 처리 후의 재요청은 다시 실행되어야 한다.
        """
        summarizer.enqueue(project, _task())
        summarizer.process_queue(
            project, run=RunRecorder([GOOD_RESPONSE]), transcript_dir=transcripts
        )
        assert summarizer.enqueue(project, _task()) is not None


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
        summarizer.enqueue(project, _task())
        run = RunRecorder([None])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 0
        assert len(run.prompts) == 2  # original + one retry / 원 호출 + 재시도 1회
        failed = _failed_records(project)
        assert [r["error"] for r in failed] == ["headless_call_failed"]
        # A failed task must not be retried forever on every pass.
        # 실패한 작업이 매 pass 마다 무한 재시도되면 안 된다.
        assert summarizer.load_pending_tasks(project) == []

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
        summarizer.enqueue(project, _task())
        run = RunRecorder([GOOD_RESPONSE])
        empty_dir = tmp_path / "no-transcripts"
        assert summarizer.process_queue(project, run=run, transcript_dir=empty_dir) == 0
        assert run.prompts == []
        assert [r["error"] for r in _failed_records(project)] == ["empty_excerpt"]

    def test_unsupported_kind_fails_without_calling_model(
        self, project: Path, transcripts: Path
    ) -> None:
        # An unknown kind (corrupt/foreign queue file) is marked failed.
        # rooting_check is NOT this case — it stays pending to ride an
        # active refresh (see TestRootingCheck).
        # 알 수 없는 kind (손상·외부 큐 파일) 는 실패 마킹된다.
        # rooting_check 는 이 경우가 아니다 — active 갱신 편승을 위해
        # 대기한다 (TestRootingCheck 참조).
        summarizer.enqueue(project, _task(kind="mystery"))
        run = RunRecorder([GOOD_RESPONSE])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 0
        assert run.prompts == []
        assert _failed_records(project)[0]["error"].startswith("unsupported_kind")

    def test_unknown_session_marks_task(self, project: Path, transcripts: Path) -> None:
        task = summarizer.SummaryTask(
            session_name="ghost", conversation_id=CONV_ID, kind="departed"
        )
        summarizer.enqueue(project, task)
        run = RunRecorder([GOOD_RESPONSE])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 0
        assert _failed_records(project)[0]["error"].startswith("session_not_found")


# ---- rooting check (R3-C2) -----------------------------------------------


ROOTED_TRUE_RESPONSE = json.dumps(
    {
        "summary": "차트 리팩토링과 백엔드 문의를 진행했다.",
        "requirements": [],
        "title": "혼합 작업",
        "rooted": True,
        "evidence": "백엔드 얘기가 3턴 이어짐",
    },
    ensure_ascii=False,
)

ROOTED_FALSE_RESPONSE = json.dumps(
    {
        "summary": "차트 리팩토링을 진행했다.",
        "requirements": [],
        "title": "차트",
        "rooted": False,
        "evidence": None,
    },
    ensure_ascii=False,
)


def _rooting_task(topic: str = "백엔드 로그인 오류") -> summarizer.SummaryTask:
    return summarizer.SummaryTask(
        session_name="work",
        conversation_id=CONV_ID,
        kind=summarizer.KIND_ROOTING_CHECK,
        extra={summarizer.EXTRA_REJECTED_TOPIC: topic},
    )


class TestRootingCheck:
    """R3-C2: the rooting check rides an active refresh and feeds mixing.

    R3-C2 — 정착 확인이 active 갱신에 편승해 혼합도를 갱신한다.
    """

    def _mixing(self, project: Path) -> tuple[int, list[str]]:
        session = SessionStore(project).load_session_by_name("work")
        assert session is not None
        return session.mixing_score, session.mixing_evidence

    def test_never_processed_standalone(
        self, project: Path, transcripts: Path
    ) -> None:
        summarizer.enqueue(project, _rooting_task())
        run = RunRecorder([GOOD_RESPONSE])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 0
        assert run.prompts == []
        # Still pending (not failed) — waits for an active refresh.
        # 실패가 아니라 대기 유지 — active 갱신을 기다린다.
        pending = summarizer.load_pending_tasks(project)
        assert [t.kind for _, t in pending] == [summarizer.KIND_ROOTING_CHECK]

    def test_rides_active_refresh_and_scores_on_rooted_true(
        self, project: Path, transcripts: Path
    ) -> None:
        summarizer.enqueue(project, _rooting_task())
        summarizer.enqueue(project, _task(kind=summarizer.KIND_ACTIVE))
        run = RunRecorder([ROOTED_TRUE_RESPONSE])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 1
        # The single call carried both the summary instruction and the
        # verbatim rooting question with the rejected topic.
        # 호출 한 번에 요약 지시와 원문 정착 질문(거부 주제 포함)이 실렸다.
        assert len(run.prompts) == 1
        assert "추가 질문" in run.prompts[0]
        assert "백엔드 로그인 오류" in run.prompts[0]
        score, evidence = self._mixing(project)
        assert score == 1
        assert evidence == ["백엔드 얘기가 3턴 이어짐"]
        session = SessionStore(project).load_session_by_name("work")
        assert session.summary == "차트 리팩토링과 백엔드 문의를 진행했다."
        assert summarizer.load_pending_tasks(project) == []

    def test_rooted_false_logs_only(
        self, project: Path, transcripts: Path
    ) -> None:
        summarizer.enqueue(project, _rooting_task())
        summarizer.enqueue(project, _task(kind=summarizer.KIND_ACTIVE))
        run = RunRecorder([ROOTED_FALSE_RESPONSE])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 1
        score, evidence = self._mixing(project)
        assert score == 0
        assert evidence == []
        # The check is consumed either way — rooted=false is an answer.
        # 어느 쪽이든 확인은 소비된다 — rooted=false 도 응답이다.
        assert summarizer.load_pending_tasks(project) == []

    def test_split_json_objects_response_parsed(
        self, project: Path, transcripts: Path
    ) -> None:
        # The model may answer with two separate JSON objects (summary
        # first, rooted second) instead of one merged object.
        # 모델이 병합 객체 대신 별도 JSON 객체 둘 (summary, rooted 순) 로
        # 답할 수 있다.
        summarizer.enqueue(project, _rooting_task())
        summarizer.enqueue(project, _task(kind=summarizer.KIND_ACTIVE))
        split = (
            GOOD_RESPONSE
            + "\n"
            + json.dumps(
                {"rooted": True, "evidence": "파일 수정 동반"}, ensure_ascii=False
            )
        )
        run = RunRecorder([split])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 1
        score, evidence = self._mixing(project)
        assert score == 1
        assert evidence == ["파일 수정 동반"]
        session = SessionStore(project).load_session_by_name("work")
        assert session.summary == "라우터 개선 작업을 진행했다."

    def test_missing_rooted_answer_still_saves_summary(
        self, project: Path, transcripts: Path
    ) -> None:
        summarizer.enqueue(project, _rooting_task())
        summarizer.enqueue(project, _task(kind=summarizer.KIND_ACTIVE))
        run = RunRecorder([GOOD_RESPONSE])  # rooted 응답 없음
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 1
        score, _evidence = self._mixing(project)
        assert score == 0
        session = SessionStore(project).load_session_by_name("work")
        assert session.summary == "라우터 개선 작업을 진행했다."
        # Consumed and logged — never retried forever.
        # 소비 후 로그만 — 무한 재시도하지 않는다.
        assert summarizer.load_pending_tasks(project) == []

    def test_reject_refresh_does_not_carry_rooting_check(
        self, project: Path, transcripts: Path
    ) -> None:
        # The refresh enqueued by the rejection itself is too early to
        # judge rooting — the check must wait for the next refresh.
        # 거부 자신이 적재한 갱신은 정착 판정에 너무 이르다 — 확인은 다음
        # 갱신을 기다려야 한다.
        summarizer.enqueue(project, _rooting_task())
        reject_refresh = summarizer.SummaryTask(
            session_name="work",
            conversation_id=CONV_ID,
            kind=summarizer.KIND_ACTIVE,
            extra={summarizer.EXTRA_FROM_REJECT: True},
        )
        summarizer.enqueue(project, reject_refresh)
        run = RunRecorder([GOOD_RESPONSE])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 1
        assert "추가 질문" not in run.prompts[0]
        pending = summarizer.load_pending_tasks(project)
        assert [t.kind for _, t in pending] == [summarizer.KIND_ROOTING_CHECK]

    def test_departed_refresh_does_not_carry_rooting_check(
        self, project: Path, transcripts: Path
    ) -> None:
        summarizer.enqueue(project, _rooting_task())
        summarizer.enqueue(project, _task(kind=summarizer.KIND_DEPARTED))
        run = RunRecorder([GOOD_RESPONSE])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 1
        assert "추가 질문" not in run.prompts[0]
        pending = summarizer.load_pending_tasks(project)
        assert [t.kind for _, t in pending] == [summarizer.KIND_ROOTING_CHECK]

    def test_failed_ride_restores_rooting_check(
        self, project: Path, transcripts: Path
    ) -> None:
        summarizer.enqueue(project, _rooting_task())
        summarizer.enqueue(project, _task(kind=summarizer.KIND_ACTIVE))
        run = RunRecorder([None])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 0
        # The active task is marked failed; the check returns to pending
        # to ride a later refresh.
        # active 작업은 실패 마킹, 확인은 대기로 복원되어 이후 갱신에 편승.
        assert [r["error"] for r in _failed_records(project)] == [
            "headless_call_failed"
        ]
        pending = summarizer.load_pending_tasks(project)
        assert [t.kind for _, t in pending] == [summarizer.KIND_ROOTING_CHECK]

    def test_topicless_check_left_for_sweep(
        self, project: Path, transcripts: Path
    ) -> None:
        broken = summarizer.SummaryTask(
            session_name="work",
            conversation_id=CONV_ID,
            kind=summarizer.KIND_ROOTING_CHECK,
            extra={},
        )
        summarizer.enqueue(project, broken)
        summarizer.enqueue(project, _task(kind=summarizer.KIND_ACTIVE))
        run = RunRecorder([GOOD_RESPONSE])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 1
        assert "추가 질문" not in run.prompts[0]
        # Unusable check is not claimed — the stale sweep collects it.
        # 사용 불가 확인은 선점하지 않는다 — 오래된 파일 sweep 이 수거.
        pending = summarizer.load_pending_tasks(project)
        assert [t.kind for _, t in pending] == [summarizer.KIND_ROOTING_CHECK]


# ---- run_headless_summary (subprocess envelope handling) -----------------


class _FakeProc:
    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


class TestRunHeadlessSummary:
    @pytest.fixture(autouse=True)
    def _isolate_junk_sweep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep the test away from the real ~/.claude junk directory.

        실제 ~/.claude 정크 디렉토리를 건드리지 않게 격리.
        """
        self.sweeps = 0

        def count_sweep() -> None:
            self.sweeps += 1

        monkeypatch.setattr(summarizer, "_sweep_junk_transcripts", count_sweep)

    def _patch_subprocess(
        self, monkeypatch: pytest.MonkeyPatch, result: Any
    ) -> None:
        self.calls: list[dict[str, Any]] = []

        def fake_run(*args: Any, **kwargs: Any) -> Any:
            self.calls.append({"argv": args[0], **kwargs})
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(summarizer.subprocess, "run", fake_run)

    def test_success_returns_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        envelope = json.dumps(
            {"is_error": False, "session_id": "junk-id", "result": GOOD_RESPONSE}
        )
        self._patch_subprocess(monkeypatch, _FakeProc(envelope))
        assert summarizer.run_headless_summary("p") == GOOD_RESPONSE

    def test_prompt_goes_to_stdin_never_argv(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The excerpt must not be visible to ``ps``.

        발췌가 ``ps`` 로 노출되면 안 된다.
        """
        secret = "user: 비밀 대화 내용"
        envelope = json.dumps({"is_error": False, "result": GOOD_RESPONSE})
        self._patch_subprocess(monkeypatch, _FakeProc(envelope))
        summarizer.run_headless_summary(secret)
        call = self.calls[0]
        assert call["input"] == secret
        assert secret not in call["argv"]

    def test_mcp_servers_disabled_and_socket_env_stripped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No MCP load (measured ~23K tokens) and no route back to the wrapper.

        MCP 무로드 (실측 약 23K 토큰) + 래퍼 소켓 접근 경로 차단.
        """
        monkeypatch.setenv("SESSION_MANAGER_SOCKET", "/tmp/should-not-leak.sock")
        envelope = json.dumps({"is_error": False, "result": GOOD_RESPONSE})
        self._patch_subprocess(monkeypatch, _FakeProc(envelope))
        summarizer.run_headless_summary("p")
        call = self.calls[0]
        assert "--strict-mcp-config" in call["argv"]
        assert summarizer._EMPTY_MCP_CONFIG in call["argv"]
        assert "SESSION_MANAGER_SOCKET" not in call["env"]

    def test_junk_swept_before_every_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sweeping before (not after) survives timeouts and parse failures.

        사후가 아닌 사전 sweep — 타임아웃·파싱 실패에서도 정크가 남지 않는다.
        """
        self._patch_subprocess(
            monkeypatch, subprocess.TimeoutExpired(cmd="claude", timeout=1)
        )
        summarizer.run_headless_summary("p")
        assert self.sweeps == 1

    def test_cli_error_envelope_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        envelope = json.dumps(
            {"is_error": True, "session_id": "junk-id", "result": "Prompt is too long"}
        )
        self._patch_subprocess(monkeypatch, _FakeProc(envelope, returncode=1))
        assert summarizer.run_headless_summary("p") is None

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
            store = SessionStore(project)
            deadline = time.monotonic() + 5.0
            session = None
            while time.monotonic() < deadline:
                session = store.load_session_by_name("work")
                if session is not None and session.summary != "이전 요약":
                    break
                time.sleep(0.05)
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


class TestClaimAndSweep:
    """Multi-instance safety (claim) and queue-file retention (sweep).

    다중 인스턴스 안전성 (claim) 과 큐 파일 보존 정책 (sweep).
    """

    def test_claimed_task_hidden_from_other_workers(self, project: Path) -> None:
        """Two ccode instances must not pay for the same summary twice.

        같은 프로젝트의 ccode 두 개가 같은 요약을 두 번 결제하면 안 된다.
        """
        path = summarizer.enqueue(project, _task())
        assert path is not None
        claimed = summarizer._claim(path)
        assert claimed is not None
        assert claimed.name.endswith(summarizer.CLAIM_SUFFIX)
        assert summarizer.load_pending_tasks(project) == []

    def test_claim_loses_race_returns_none(self, project: Path) -> None:
        path = summarizer.enqueue(project, _task())
        assert path is not None
        assert summarizer._claim(path) is not None
        # Second claimant finds the file gone — rename fails, no crash.
        # 두 번째 선점자는 파일이 사라진 것을 본다 — rename 실패, 크래시 없음.
        assert summarizer._claim(path) is None

    def test_concurrent_pass_processes_task_once(
        self, project: Path, transcripts: Path
    ) -> None:
        """A second pass over the same queue must find nothing left to run.

        같은 큐에 대한 두 번째 pass 는 실행할 작업을 찾지 못해야 한다.
        """
        summarizer.enqueue(project, _task())
        pending_snapshot = summarizer.load_pending_tasks(project)
        run = RunRecorder([GOOD_RESPONSE])
        assert summarizer.process_queue(project, run=run, transcript_dir=transcripts) == 1
        # Simulate the racing worker that had already listed the task.
        # 이미 목록을 확보한 채 경합하던 워커를 재현.
        assert all(summarizer._claim(p) is None for p, _ in pending_snapshot)
        assert len(run.prompts) == 1

    def test_sweep_removes_old_files_only(self, project: Path) -> None:
        queue_dir = project / ".session-manager" / "summary-queue"
        queue_dir.mkdir(parents=True, exist_ok=True)
        old = queue_dir / "old.json.processing"
        old.write_text("{}", encoding="utf-8")
        old_ts = time.time() - 31 * 86400
        os.utime(old, (old_ts, old_ts))
        recent = queue_dir / "recent.json"
        recent.write_text("{}", encoding="utf-8")

        assert summarizer.sweep_stale_queue_files(project, 30) == 1
        assert not old.exists()
        assert recent.exists()

    def test_sweep_without_queue_dir(self, tmp_path: Path) -> None:
        assert summarizer.sweep_stale_queue_files(tmp_path, 30) == 0
