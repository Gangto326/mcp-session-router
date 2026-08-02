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

from session_manager.hooks import user_prompt_submit as hook


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
