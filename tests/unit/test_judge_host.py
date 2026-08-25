"""
Unit tests for the resident judge host.

The real ``claude`` process is replaced by monkeypatched
``_spawn_and_warm`` / ``_round_trip`` — lifecycle, replies, and
degradation paths are what these tests pin down.

상주 판정 호스트 단위 테스트.

실제 ``claude`` 프로세스는 ``_spawn_and_warm`` / ``_round_trip``
monkeypatch 로 대체한다 — 생명주기·회신·완화 경로를 고정하는 테스트다.
"""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import pytest

from session_manager.models.session import PrecedentRecord, SessionMetadata
from session_manager.storage.file_store import SessionStore
from session_manager.wrapper.judge_host import JudgeHost


def _recv_reply(sock: socket.socket) -> dict:
    sock.settimeout(5.0)
    buffer = b""
    while b"\n" not in buffer:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buffer += chunk
    return json.loads(buffer.split(b"\n", 1)[0])


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _seed_sessions(project: Path) -> None:
    store = SessionStore(project)
    store.init_project()
    frontend = SessionMetadata.new(name="frontend", title="차트", summary="차트 리팩토링")
    frontend.claude_conversation_ids = ["conv-1"]
    store.save_session(frontend)
    backend = SessionMetadata.new(name="backend", title="API", summary="JWT 교체 완료")
    # Mixing signal (R3-C2) — raw value plus evidence must reach the prompt.
    # 혼합도 신호 (R3-C2) — 원값과 근거 인용이 프롬프트에 도달해야 한다.
    backend.mixing_score = 2
    backend.mixing_evidence = ["프론트 차트 얘기가 3턴 이어짐"]
    store.save_session(backend)


VERDICT_JSON = (
    '{"action":"SWITCH","target":"backend","confidence":0.9,'
    '"evidence":"JWT 교체 완료","reason":"인증 소관"}'
)


class TestHandleRequest:
    def test_unavailable_before_ready(self, tmp_path: Path) -> None:
        host = JudgeHost(tmp_path)
        ours, theirs = socket.socketpair()
        try:
            assert host.handle_request({"prompt": "x"}, ours) is True
            reply = _recv_reply(theirs)
            assert reply == {"ok": False, "reason": "judge_unavailable"}
            assert theirs.recv(4096) == b""  # 회신 후 닫힘
        finally:
            theirs.close()

    def test_unavailable_when_dead(self, tmp_path: Path) -> None:
        host = JudgeHost(tmp_path)
        host._dead = True
        ours, theirs = socket.socketpair()
        try:
            host.handle_request({"prompt": "x"}, ours)
            assert _recv_reply(theirs)["ok"] is False
        finally:
            theirs.close()


class TestJudgmentFlow:
    @pytest.fixture
    def host(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> JudgeHost:
        _seed_sessions(tmp_path)
        monkeypatch.setattr(
            JudgeHost, "_spawn_and_warm", lambda self: True
        )
        h = JudgeHost(tmp_path)
        yield h
        h.stop()

    def test_verdict_reply_and_rewarm(
        self, host: JudgeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts: list[str] = []

        def fake_round_trip(self, text: str, timeout: float) -> str:
            prompts.append(text)
            return VERDICT_JSON

        monkeypatch.setattr(JudgeHost, "_round_trip", fake_round_trip)
        host.ensure_started()
        assert _wait_until(lambda: host._ready)

        ours, theirs = socket.socketpair()
        try:
            assert (
                host.handle_request(
                    {
                        "prompt": "로그인 API가 500을 뱉는다",
                        "session_id": "conv-1",
                        "transcript_path": None,
                    },
                    ours,
                )
                is True
            )
            reply = _recv_reply(theirs)
            assert reply["ok"] is True
            assert reply["verdict"]["action"] == "SWITCH"
            assert reply["verdict"]["target"] == "backend"
        finally:
            theirs.close()

        # 판정 프롬프트에 세션 목록·현재 세션 표시·새 프롬프트가 들어간다
        judge_prompt = prompts[-1]
        assert "로그인 API가 500을 뱉는다" in judge_prompt
        assert "frontend (현재 세션)" in judge_prompt
        assert "backend" in judge_prompt
        # Raw mixing signal reaches the judge input untouched (R3-C2).
        # 혼합도 원신호가 가공 없이 판정 입력에 도달한다 (R3-C2).
        assert "mixing_score: 2" in judge_prompt
        assert "프론트 차트 얘기가 3턴 이어짐" in judge_prompt

        # 판정 1회 후 재웜업되어 다시 가용 상태가 된다
        assert _wait_until(lambda: host._ready)

    def test_timeout_replies_stay(
        self, host: JudgeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            JudgeHost, "_round_trip", lambda self, text, timeout: None
        )
        host.ensure_started()
        assert _wait_until(lambda: host._ready)

        ours, theirs = socket.socketpair()
        try:
            host.handle_request({"prompt": "질문"}, ours)
            reply = _recv_reply(theirs)
            assert reply["ok"] is True
            assert reply["verdict"]["action"] == "STAY"
            assert reply["verdict"]["reason"] == "judge_timeout"
        finally:
            theirs.close()

    def test_unparsable_replies_stay(
        self, host: JudgeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            JudgeHost,
            "_round_trip",
            lambda self, text, timeout: "판정할 수 없습니다",
        )
        host.ensure_started()
        assert _wait_until(lambda: host._ready)

        ours, theirs = socket.socketpair()
        try:
            host.handle_request({"prompt": "질문"}, ours)
            reply = _recv_reply(theirs)
            assert reply["verdict"]["action"] == "STAY"
            assert reply["verdict"]["reason"] == "judge_unparsable"
        finally:
            theirs.close()

    def test_empty_prompt_rejected(
        self, host: JudgeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            JudgeHost, "_round_trip", lambda self, text, timeout: VERDICT_JSON
        )
        host.ensure_started()
        assert _wait_until(lambda: host._ready)

        ours, theirs = socket.socketpair()
        try:
            host.handle_request({"prompt": "   "}, ours)
            reply = _recv_reply(theirs)
            assert reply == {"ok": False, "reason": "empty_prompt"}
        finally:
            theirs.close()

    def test_second_request_while_busy_gets_unavailable(
        self, host: JudgeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def slow_round_trip(self, text: str, timeout: float) -> str:
            time.sleep(0.3)
            return VERDICT_JSON

        monkeypatch.setattr(JudgeHost, "_round_trip", slow_round_trip)
        host.ensure_started()
        assert _wait_until(lambda: host._ready)

        first_ours, first_theirs = socket.socketpair()
        second_ours, second_theirs = socket.socketpair()
        try:
            host.handle_request({"prompt": "첫 번째"}, first_ours)
            # 접수 즉시 비가용 — 두 번째 요청은 대기하지 않고 통과한다
            host.handle_request({"prompt": "두 번째"}, second_ours)
            second_reply = _recv_reply(second_theirs)
            assert second_reply["ok"] is False

            first_reply = _recv_reply(first_theirs)
            assert first_reply["ok"] is True
        finally:
            first_theirs.close()
            second_theirs.close()


class TestPrecedentGate:
    """Deterministic precedent suppression (R3-FIX2).

    결정적 판례 억제 (R3-FIX2) — 거부된 대상으로의 SWITCH 는 모델 판단이
    아니라 코드가 강등한다. 판례는 판정 프롬프트에 들어가지 않는다.
    """

    @pytest.fixture
    def host(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> JudgeHost:
        store = SessionStore(tmp_path)
        store.init_project()
        frontend = SessionMetadata.new(
            name="frontend", title="차트", summary="차트 리팩토링"
        )
        frontend.claude_conversation_ids = ["conv-1"]
        frontend.precedents = [
            PrecedentRecord.new(
                prompt_gist="로그인 오류 조사",
                kept_in="frontend",
                rejected="backend",
            )
        ]
        store.save_session(frontend)
        store.save_session(
            SessionMetadata.new(name="backend", title="API", summary="JWT 교체 완료")
        )
        store.save_session(
            SessionMetadata.new(name="infra", title="배포", summary="CI 구성")
        )
        monkeypatch.setattr(JudgeHost, "_spawn_and_warm", lambda self: True)
        h = JudgeHost(tmp_path)
        yield h
        h.stop()

    def _judge(self, host: JudgeHost, monkeypatch, verdict_json: str) -> dict:
        prompts: list[str] = []

        def fake_round_trip(self, text: str, timeout: float) -> str:
            prompts.append(text)
            return verdict_json

        monkeypatch.setattr(JudgeHost, "_round_trip", fake_round_trip)
        host.ensure_started()
        assert _wait_until(lambda: host._ready)
        ours, theirs = socket.socketpair()
        try:
            host.handle_request(
                {"prompt": "로그인 오류 관련해서 봐줘", "session_id": "conv-1"},
                ours,
            )
            reply = _recv_reply(theirs)
        finally:
            theirs.close()
        reply["_prompts"] = prompts
        return reply

    def test_switch_to_rejected_target_suppressed(
        self, host: JudgeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reply = self._judge(host, monkeypatch, VERDICT_JSON)  # SWITCH backend
        assert reply["ok"] is True
        assert reply["verdict"]["action"] == "STAY"
        assert reply["verdict"]["reason"] == "suppressed_by_precedent: backend"

    def test_switch_to_other_target_unaffected(
        self, host: JudgeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        other = (
            '{"action":"SWITCH","target":"infra","confidence":0.9,'
            '"evidence":"CI 구성","reason":"배포 소관"}'
        )
        reply = self._judge(host, monkeypatch, other)
        assert reply["verdict"]["action"] == "SWITCH"
        assert reply["verdict"]["target"] == "infra"

    def test_precedents_never_enter_the_prompt(
        self, host: JudgeHost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The model must not even see the precedent (measured 3/3: it
        # inverted the meaning when asked to weigh it).
        # 모델은 판례를 보지도 않아야 한다 (실측 3/3 의미 반전).
        reply = self._judge(host, monkeypatch, VERDICT_JSON)
        assert "로그인 오류 조사" not in reply["_prompts"][-1]
        assert "판례" not in reply["_prompts"][-1]

    def test_unknown_conversation_skips_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without a resolvable current session there is no kept_in side —
        # the verdict passes through untouched.
        # 현재 세션을 해석할 수 없으면 kept_in 쪽이 없다 — 판정 그대로 통과.
        _seed_sessions(tmp_path)
        monkeypatch.setattr(JudgeHost, "_spawn_and_warm", lambda self: True)
        host = JudgeHost(tmp_path)
        monkeypatch.setattr(
            JudgeHost, "_round_trip", lambda self, text, timeout: VERDICT_JSON
        )
        host.ensure_started()
        assert _wait_until(lambda: host._ready)
        ours, theirs = socket.socketpair()
        try:
            host.handle_request(
                {"prompt": "질문", "session_id": "conv-없음"}, ours
            )
            reply = _recv_reply(theirs)
        finally:
            theirs.close()
        host.stop()
        assert reply["verdict"]["action"] == "SWITCH"


class TestSpawnFailure:
    def test_two_failures_mark_dead(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # conftest 의 기본 가드 (_spawn_and_warm → False) 를 그대로 사용
        host = JudgeHost(tmp_path)
        host.ensure_started()
        assert _wait_until(lambda: host._dead)
        assert host._ready is False

        ours, theirs = socket.socketpair()
        try:
            host.handle_request({"prompt": "x"}, ours)
            assert _recv_reply(theirs)["ok"] is False
        finally:
            theirs.close()
        host.stop()
