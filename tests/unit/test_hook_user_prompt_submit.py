"""
Unit tests for the UserPromptSubmit routing hook.

Covers the graceful-degradation contract (any malformed input → exit 0)
and each deterministic prefilter rule.

UserPromptSubmit 라우팅 hook 단위 테스트.

graceful degradation 계약 (어떤 오염 입력도 exit 0)과 결정적 프리필터
규칙 각각을 검증한다.
"""

from __future__ import annotations

import json
import socket
import threading
import uuid
from pathlib import Path

import pytest

from session_manager import handoff_store
from session_manager.hooks import user_prompt_submit as hook
from session_manager.routing import decision_log


def _write_session(sessions_dir: Path, name: str, **extra: object) -> None:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {"session_id": name, "name": name, **extra}
    (sessions_dir / f"{name}.json").write_text(
        json.dumps(data), encoding="utf-8"
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return tmp_path


def _payload(project: Path, **overrides: object) -> str:
    data: dict[str, object] = {
        "session_id": "conv-1",
        "transcript_path": str(project / "conv-1.jsonl"),
        "cwd": str(project),
        "permission_mode": "default",
        "hook_event_name": "UserPromptSubmit",
        "prompt": "다음 작업을 진행해",
        **overrides,
    }
    return json.dumps(data)


class TestGracefulDegradation:
    def test_malformed_stdin_exits_zero(self) -> None:
        assert hook.run("not json at all") == 0

    def test_empty_stdin_exits_zero(self) -> None:
        assert hook.run("") == 0

    def test_non_object_payload_exits_zero(self) -> None:
        assert hook.run("[1, 2, 3]") == 0

    def test_missing_cwd_exits_zero(self, project: Path) -> None:
        payload = json.loads(_payload(project))
        del payload["cwd"]
        assert hook.run(json.dumps(payload)) == 0


class TestPrefilter:
    def test_no_session_manager_dir(self, project: Path) -> None:
        assert (
            hook._prefilter(json.loads(_payload(project)))
            == "no_session_manager_dir"
        )

    def test_zero_sessions(self, project: Path) -> None:
        (project / ".session-manager" / "sessions").mkdir(parents=True)
        assert (
            hook._prefilter(json.loads(_payload(project)))
            == "single_active_session"
        )

    def test_single_session(self, project: Path) -> None:
        _write_session(project / ".session-manager" / "sessions", "only-one")
        assert (
            hook._prefilter(json.loads(_payload(project)))
            == "single_active_session"
        )

    def test_two_sessions_pass_to_judge(self, project: Path) -> None:
        sessions = project / ".session-manager" / "sessions"
        _write_session(sessions, "frontend")
        _write_session(sessions, "backend")
        assert hook._prefilter(json.loads(_payload(project))) is None

    def test_retired_sessions_not_counted(self, project: Path) -> None:
        # status 필드는 R4 에서 추가 — 미리 forward 호환 검증: retired 는
        # 활성 수에서 제외되어 세션 1개 취급이어야 한다
        sessions = project / ".session-manager" / "sessions"
        _write_session(sessions, "frontend")
        _write_session(sessions, "old-work", status="retired")
        assert (
            hook._prefilter(json.loads(_payload(project)))
            == "single_active_session"
        )

    def test_corrupt_session_file_skipped(self, project: Path) -> None:
        sessions = project / ".session-manager" / "sessions"
        _write_session(sessions, "frontend")
        _write_session(sessions, "backend")
        (sessions / "broken.json").write_text("{oops", encoding="utf-8")
        # 손상 파일은 무시 — 정상 2개로 판정 단계 진행
        assert hook._prefilter(json.loads(_payload(project))) is None

    def test_plan_mode_passes_through(self, project: Path) -> None:
        sessions = project / ".session-manager" / "sessions"
        _write_session(sessions, "frontend")
        _write_session(sessions, "backend")
        payload = json.loads(_payload(project, permission_mode="plan"))
        assert hook._prefilter(payload) == "plan_mode"

    def test_routing_off_passes_through(self, project: Path) -> None:
        sessions = project / ".session-manager" / "sessions"
        _write_session(sessions, "frontend")
        _write_session(sessions, "backend")
        (project / ".session-manager" / "config.json").write_text(
            json.dumps({"routing_mode": "off"}), encoding="utf-8"
        )
        assert hook._prefilter(json.loads(_payload(project))) == "routing_off"


class TestRoutingModeLoad:
    def test_missing_config_defaults_to_confirm(self, project: Path) -> None:
        root = project / ".session-manager"
        root.mkdir()
        assert hook._load_routing_mode(root) == "confirm"

    def test_config_without_key_defaults_to_confirm(
        self, project: Path
    ) -> None:
        root = project / ".session-manager"
        root.mkdir()
        (root / "config.json").write_text(
            json.dumps({"socket_path": "/tmp/x.sock"}), encoding="utf-8"
        )
        assert hook._load_routing_mode(root) == "confirm"

    def test_corrupt_config_defaults_to_confirm(self, project: Path) -> None:
        root = project / ".session-manager"
        root.mkdir()
        (root / "config.json").write_text("{oops", encoding="utf-8")
        assert hook._load_routing_mode(root) == "confirm"


class TestRequestJudgment:
    @pytest.fixture
    def hook_socket(self) -> str:
        return f"/tmp/test-hook-{uuid.uuid4().hex[:8]}.sock"

    def _serve_once(self, path: str, reply: bytes) -> threading.Thread:
        """
        Minimal one-shot server standing in for the wrapper socket.

        래퍼 소켓을 대신하는 최소 단발 서버.
        """
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(path)
        server.listen(1)

        def serve() -> None:
            conn, _ = server.accept()
            buffer = b""
            while b"\n" not in buffer:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buffer += chunk
            if reply:
                conn.sendall(reply)
            conn.close()
            server.close()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        return thread

    def test_round_trip_returns_reply(
        self, hook_socket: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thread = self._serve_once(
            hook_socket, b'{"ok": true, "verdict": {"action": "STAY"}}\n'
        )
        monkeypatch.setenv("SESSION_MANAGER_SOCKET", hook_socket)
        reply = hook._request_judgment(
            {"prompt": "p", "session_id": "c", "transcript_path": "t", "cwd": "/x"}
        )
        thread.join(timeout=2)
        assert reply == {"ok": True, "verdict": {"action": "STAY"}}

    def test_no_env_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SESSION_MANAGER_SOCKET", raising=False)
        assert hook._request_judgment({"prompt": "p"}) is None

    def test_no_server_returns_none(
        self, hook_socket: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SESSION_MANAGER_SOCKET", hook_socket)
        assert hook._request_judgment({"prompt": "p"}) is None

    def test_server_closes_without_reply_returns_none(
        self, hook_socket: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thread = self._serve_once(hook_socket, b"")
        monkeypatch.setenv("SESSION_MANAGER_SOCKET", hook_socket)
        assert hook._request_judgment({"prompt": "p"}) is None
        thread.join(timeout=2)

    def test_malformed_reply_returns_none(
        self, hook_socket: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        thread = self._serve_once(hook_socket, b"not json\n")
        monkeypatch.setenv("SESSION_MANAGER_SOCKET", hook_socket)
        assert hook._request_judgment({"prompt": "p"}) is None
        thread.join(timeout=2)


class TestRouteExecution:
    """Acting on the verdict: confirm context, auto block, pass-through.

    판정에 따른 실행 테스트: confirm 컨텍스트, auto 차단, 통과.
    """

    def _reply(self, **verdict: object) -> dict:
        return {"ok": True, "verdict": verdict}

    def _routable_payload(self, project: Path, mode: str | None = None) -> dict:
        (project / ".session-manager").mkdir(exist_ok=True)
        if mode is not None:
            (project / ".session-manager" / "config.json").write_text(
                json.dumps({"routing_mode": mode}), encoding="utf-8"
            )
        return json.loads(_payload(project))

    def test_stay_passes_silently(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            hook,
            "_request_judgment",
            lambda _p, _g=None: self._reply(action="STAY", reason="연속 작업"),
        )
        hook._route(self._routable_payload(project))
        assert capsys.readouterr().out == ""

    def test_ask_passes_silently(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            hook,
            "_request_judgment",
            lambda _p, _g=None: self._reply(action="ASK", reason="후보 다수"),
        )
        hook._route(self._routable_payload(project))
        assert capsys.readouterr().out == ""

    def test_failed_reply_passes_silently(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        monkeypatch.setattr(hook, "_request_judgment", lambda _p, _g=None: None)
        hook._route(self._routable_payload(project))
        monkeypatch.setattr(
            hook,
            "_request_judgment",
            lambda _p, _g=None: {"ok": False, "reason": "judge_unavailable"},
        )
        hook._route(self._routable_payload(project))
        assert capsys.readouterr().out == ""

    def test_switch_confirm_emits_additional_context(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            hook,
            "_request_judgment",
            lambda _p, _g=None: self._reply(
                action="SWITCH",
                target="backend",
                confidence=0.9,
                evidence="JWT 교체 완료",
            ),
        )
        hook._route(self._routable_payload(project))
        out = json.loads(capsys.readouterr().out)
        context = out["hookSpecificOutput"]["additionalContext"]
        assert out["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        assert "backend" in context
        assert "JWT 교체 완료" in context
        assert "session_switch" in context
        assert "AskUserQuestion" in context

    def test_switch_confirm_instructs_reject_switch_on_keep(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        # R3-C1: the "keep" choice must call reject_switch with the
        # rejected target so the precedent gets recorded.
        # R3-C1 — "유지" 선택은 판례가 기록되도록 거부 대상을 담아
        # reject_switch 를 호출해야 한다.
        monkeypatch.setattr(
            hook,
            "_request_judgment",
            lambda _p, _g=None: self._reply(
                action="SWITCH",
                target="backend",
                confidence=0.9,
                evidence="JWT 교체 완료",
            ),
        )
        hook._route(self._routable_payload(project))
        context = json.loads(capsys.readouterr().out)["hookSpecificOutput"][
            "additionalContext"
        ]
        assert "reject_switch(rejected_target='backend'" in context
        assert "prompt_gist" in context

    def test_new_confirm_keep_does_not_mention_reject_switch(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        # NEW has no rejected target session, so no precedent tool call.
        # NEW 는 거부 대상 세션이 없으므로 판례 도구 호출도 없다.
        monkeypatch.setattr(
            hook,
            "_request_judgment",
            lambda _p, _g=None: self._reply(action="NEW", reason="새 주제"),
        )
        hook._route(self._routable_payload(project))
        context = json.loads(capsys.readouterr().out)["hookSpecificOutput"][
            "additionalContext"
        ]
        assert "reject_switch" not in context

    def test_new_confirm_emits_session_create_instruction(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            hook,
            "_request_judgment",
            lambda _p, _g=None: self._reply(action="NEW", reason="새 주제"),
        )
        hook._route(self._routable_payload(project))
        context = json.loads(capsys.readouterr().out)["hookSpecificOutput"][
            "additionalContext"
        ]
        assert "session_create" in context
        assert "새 주제" in context

    def _auto_reply(self, refute: dict | None = None, **verdict: object) -> dict:
        reply = self._reply(**verdict)
        if refute is not None:
            reply["refute"] = refute
        return reply

    def test_auto_without_calibration_falls_back_to_confirm(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        # 보정 데이터가 없는 auto 모드는 confirm 으로 완화된다 (규칙 8)
        monkeypatch.setattr(
            hook,
            "_request_judgment",
            lambda _p, _g=None: self._reply(
                action="SWITCH", target="backend", confidence=0.99, evidence="e"
            ),
        )
        hook._route(self._routable_payload(project, mode="auto"))
        out = capsys.readouterr().out
        assert "decision" not in out
        assert "session_switch" in out

    def test_auto_with_calibration_blocks_and_delegates(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        sent: list[dict] = []
        gates: list[dict | None] = []

        def fake_round_trip(request: dict, timeout: float = 0.0) -> dict:
            sent.append(request)
            return {"type": "ack", "ok": True}

        def fake_judgment(_payload: dict, auto_gate: dict | None = None) -> dict:
            gates.append(auto_gate)
            return self._auto_reply(
                refute={"refuted": False, "reason": "타당"},
                action="SWITCH",
                target="backend",
                confidence=0.95,
                evidence="e",
            )

        monkeypatch.setattr(hook, "_request_judgment", fake_judgment)
        monkeypatch.setattr(hook, "_calibrated_auto_threshold", lambda _root: 0.9)
        monkeypatch.setattr(hook, "_socket_round_trip", fake_round_trip)

        hook._route(self._routable_payload(project, mode="auto"))

        out = json.loads(capsys.readouterr().out)
        assert out["decision"] == "block"
        assert "backend" in out["reason"]
        # The gate travelled with the judgment request (refute runs
        # wrapper-side in the same warm process).
        # 게이트가 판정 요청과 함께 전달됐다 (반박은 래퍼 측 동일 웜
        # 프로세스에서 돈다).
        assert gates == [{"threshold": 0.9}]
        assert len(sent) == 1
        assert sent[0]["action"] == "route_switch"
        assert sent[0]["client"] == "hook"
        assert sent[0]["target"] == "backend"
        assert sent[0]["user_prompt"] == "다음 작업을 진행해"

    def test_auto_refuted_falls_back_to_confirm(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        # 2차 반박이 성공하면 (refuted=true) confirm 으로 강등된다
        monkeypatch.setattr(
            hook,
            "_request_judgment",
            lambda _p, _g=None: self._auto_reply(
                refute={"refuted": True, "reason": "근거 빈약"},
                action="SWITCH",
                target="backend",
                confidence=0.95,
                evidence="e",
            ),
        )
        monkeypatch.setattr(hook, "_calibrated_auto_threshold", lambda _root: 0.9)
        hook._route(self._routable_payload(project, mode="auto"))
        out = capsys.readouterr().out
        assert "decision" not in out
        assert "session_switch" in out

    def test_auto_missing_refute_falls_back_to_confirm(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        # 반박 결과가 회신에 없으면 (미검증 전환) 자동 실행하지 않는다
        monkeypatch.setattr(
            hook,
            "_request_judgment",
            lambda _p, _g=None: self._reply(
                action="SWITCH", target="backend", confidence=0.95, evidence="e"
            ),
        )
        monkeypatch.setattr(hook, "_calibrated_auto_threshold", lambda _root: 0.9)
        hook._route(self._routable_payload(project, mode="auto"))
        out = capsys.readouterr().out
        assert "decision" not in out
        assert "session_switch" in out

    def test_auto_below_threshold_falls_back_to_confirm(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            hook,
            "_request_judgment",
            lambda _p, _g=None: self._reply(
                action="SWITCH", target="backend", confidence=0.5, evidence="e"
            ),
        )
        monkeypatch.setattr(hook, "_calibrated_auto_threshold", lambda _root: 0.9)
        hook._route(self._routable_payload(project, mode="auto"))
        out = capsys.readouterr().out
        assert "decision" not in out
        assert "session_switch" in out

    def test_auto_socket_failure_falls_back_to_confirm(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        # 래퍼가 ack 하지 않으면 절대 block 하지 않는다 — 프롬프트를
        # 재주입할 주체가 없기 때문
        monkeypatch.setattr(
            hook,
            "_request_judgment",
            lambda _p, _g=None: self._auto_reply(
                refute={"refuted": False, "reason": "타당"},
                action="SWITCH",
                target="backend",
                confidence=0.95,
                evidence="e",
            ),
        )
        monkeypatch.setattr(hook, "_calibrated_auto_threshold", lambda _root: 0.9)
        monkeypatch.setattr(
            hook, "_socket_round_trip", lambda _r, timeout=0.0: None
        )
        hook._route(self._routable_payload(project, mode="auto"))
        out = capsys.readouterr().out
        assert "decision" not in out
        assert "session_switch" in out

    def test_auto_new_never_blocks(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            hook,
            "_request_judgment",
            lambda _p, _g=None: self._reply(
                action="NEW", confidence=0.99, reason="새 주제"
            ),
        )
        monkeypatch.setattr(hook, "_calibrated_auto_threshold", lambda _root: 0.9)
        hook._route(self._routable_payload(project, mode="auto"))
        out = capsys.readouterr().out
        assert "decision" not in out
        assert "session_create" in out

    def test_confirm_switch_records_proposal(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        # confirm SWITCH 제안이 보정 로그에 기록된다 (R3-C4)
        monkeypatch.setattr(
            hook,
            "_request_judgment",
            lambda _p, _g=None: self._reply(
                action="SWITCH", target="backend", confidence=0.9, evidence="e"
            ),
        )
        hook._route(self._routable_payload(project))
        capsys.readouterr()
        events = decision_log.load_events(project)
        assert len(events) == 1
        assert events[0]["type"] == "proposal"
        assert events[0]["target"] == "backend"
        assert events[0]["confidence"] == 0.9
        assert events[0]["mode"] == "confirm"

    def test_auto_switch_records_auto_proposal(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            hook,
            "_request_judgment",
            lambda _p, _g=None: self._auto_reply(
                refute={"refuted": False, "reason": "타당"},
                action="SWITCH",
                target="backend",
                confidence=0.95,
                evidence="e",
            ),
        )
        monkeypatch.setattr(hook, "_calibrated_auto_threshold", lambda _root: 0.9)
        monkeypatch.setattr(
            hook,
            "_socket_round_trip",
            lambda _r, timeout=0.0: {"type": "ack", "ok": True},
        )
        hook._route(self._routable_payload(project, mode="auto"))
        capsys.readouterr()
        events = decision_log.load_events(project)
        assert [e["mode"] for e in events] == ["auto"]

    def test_new_verdict_records_no_proposal(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        monkeypatch.setattr(
            hook,
            "_request_judgment",
            lambda _p, _g=None: self._reply(action="NEW", reason="새 주제"),
        )
        hook._route(self._routable_payload(project))
        capsys.readouterr()
        assert decision_log.load_events(project) == []


class TestCalibratedAutoThreshold:
    """_calibrated_auto_threshold over a real decision log.

    실제 결정 로그 위에서의 _calibrated_auto_threshold.
    """

    def test_empty_log_returns_none(self, project: Path) -> None:
        root = project / ".session-manager"
        root.mkdir(exist_ok=True)
        assert hook._calibrated_auto_threshold(root) is None

    def test_sufficient_accepts_yield_threshold(self, project: Path) -> None:
        root = project / ".session-manager"
        root.mkdir(exist_ok=True)
        # 60 accepted proposals at confidence 0.9 — the one-sided 95%
        # Wilson lower bound of 60/60 is 0.957 ≥ 0.95 target.
        # confidence 0.9 수용 60건 — 60/60 의 단측 95% Wilson 하한은
        # 0.957 로 목표 0.95 이상.
        for _ in range(60):
            decision_log.append_proposal(project, "backend", 0.9, mode="confirm")
            decision_log.append_label(
                project, "backend", decision_log.LABEL_ACCEPT, source="session_switch"
            )
        assert hook._calibrated_auto_threshold(root) == 0.9

    def test_insufficient_samples_return_none(self, project: Path) -> None:
        root = project / ".session-manager"
        root.mkdir(exist_ok=True)
        # 10/10 accepts: Wilson lower bound 1/(1+z²/10) ≈ 0.787 < 0.95.
        # 10/10 수용 — Wilson 하한 약 0.787 로 목표 미달.
        for _ in range(10):
            decision_log.append_proposal(project, "backend", 0.9, mode="confirm")
            decision_log.append_label(
                project, "backend", decision_log.LABEL_ACCEPT, source="session_switch"
            )
        assert hook._calibrated_auto_threshold(root) is None


class TestRun:
    def test_prefiltered_prompt_exits_zero(self, project: Path) -> None:
        assert hook.run(_payload(project)) == 0

    def test_judge_stage_reached_and_exits_zero(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sessions = project / ".session-manager" / "sessions"
        _write_session(sessions, "frontend")
        _write_session(sessions, "backend")

        routed: list[dict] = []
        monkeypatch.setattr(hook, "_route", routed.append)
        assert hook.run(_payload(project)) == 0
        assert len(routed) == 1
        assert routed[0]["prompt"] == "다음 작업을 진행해"

    def test_route_exception_still_exits_zero(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sessions = project / ".session-manager" / "sessions"
        _write_session(sessions, "frontend")
        _write_session(sessions, "backend")

        def boom(_payload: dict) -> None:
            raise RuntimeError("judge crashed")

        monkeypatch.setattr(hook, "_route", boom)
        assert hook.run(_payload(project)) == 0


class TestPendingHandoffDelivery:
    """Transition trigger: consume the pending file, inject, skip routing.

    전환 트리거 — pending 파일 소비·주입·라우팅 skip.
    """

    def _trigger_payload(self, project: Path) -> str:
        return _payload(project, prompt=handoff_store.TRIGGER_PROMPT)

    def test_trigger_delivers_handoff_as_context(
        self, project: Path, capsys: pytest.CaptureFixture
    ) -> None:
        handoff_store.write_pending(
            project,
            target="backend",
            handoff={"from": "frontend", "message": "이전 세션 요약"},
            user_prompt="원래 사용자 프롬프트",
        )
        assert hook.run(self._trigger_payload(project)) == 0
        out = json.loads(capsys.readouterr().out)
        context = out["hookSpecificOutput"]["additionalContext"]
        assert context.startswith("[handoff]")
        assert "이전 세션 요약" in context
        assert context.endswith("원래 사용자 프롬프트")
        # Consumed — a second trigger passes silently.
        # 소비됨 — 두 번째 트리거는 조용히 통과.
        assert handoff_store.take_pending(project) is None

    def test_trigger_skips_routing_even_with_sessions(
        self,
        project: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        # A transition trigger must never be re-routed by the judge.
        # 전환 트리거가 판정기로 재라우팅되면 안 된다.
        sessions = project / ".session-manager" / "sessions"
        _write_session(sessions, "frontend")
        _write_session(sessions, "backend")
        handoff_store.write_pending(project, "backend", {}, "p")
        called: list = []
        monkeypatch.setattr(hook, "_route", called.append)
        assert hook.run(self._trigger_payload(project)) == 0
        assert called == []
        capsys.readouterr()

    def test_trigger_without_file_passes_silently(
        self, project: Path, capsys: pytest.CaptureFixture
    ) -> None:
        assert hook.run(self._trigger_payload(project)) == 0
        assert capsys.readouterr().out == ""

    def test_normal_prompt_leaves_pending_file_alone(
        self, project: Path, capsys: pytest.CaptureFixture
    ) -> None:
        # Only the exact trigger consumes the file — a stale file must
        # not be slurped by an unrelated prompt.
        # 정확한 트리거만 파일을 소비한다 — 무관한 프롬프트가 낡은 파일을
        # 삼키면 안 된다.
        handoff_store.write_pending(project, "backend", {}, "p")
        assert hook.run(_payload(project)) == 0
        capsys.readouterr()
        assert handoff_store.take_pending(project) is not None
