"""
Unit tests for the ccode entry point.

ccode 진입점 단위 테스트.
"""

from __future__ import annotations

import sys

import pytest

from session_manager.wrapper.main import _resolve_socket_path, main


class TestResolveSocketPath:
    def test_starts_with_tmp_prefix(self) -> None:
        path = _resolve_socket_path("/some/project")
        assert path.startswith("/tmp/session-manager-")
        assert path.endswith(".sock")

    def test_deterministic_for_same_input(self) -> None:
        a = _resolve_socket_path("/some/project")
        b = _resolve_socket_path("/some/project")
        assert a == b

    def test_different_for_different_inputs(self) -> None:
        a = _resolve_socket_path("/project/a")
        b = _resolve_socket_path("/project/b")
        assert a != b

    def test_within_af_unix_path_limit(self) -> None:
        # AF_UNIX 경로는 보통 108바이트 한계 — 깊은 경로에서도 여유
        path = _resolve_socket_path(
            "/very/deep/nested/path/with/many/segments/and/more/levels"
        )
        assert len(path) < 108

    def test_hash_length_12(self) -> None:
        path = _resolve_socket_path("/x")
        hash_part = path.removeprefix("/tmp/session-manager-").removesuffix(".sock")
        assert len(hash_part) == 12


class TestClaudeArgsPassthrough:
    """ccode passes the user's arguments through untouched.

    ccode 는 사용자 인자를 손대지 않고 그대로 전달한다.

    The wrapper used to prepend an experimental-channels development flag,
    which forced a confirmation prompt on every start and shut out API-key
    users. Nothing depends on channels any more.
    예전에는 experimental channels 개발 플래그를 앞에 붙여 매 시작마다 확인
    창이 떴고 API key 사용자는 쓸 수 없었다. 이제 channels 의존은 없다.
    """

    def _run_main(
        self,
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
        registration_calls: list | None = None,
    ) -> list[str]:
        captured: dict[str, list[str]] = {}

        class _FakeWrapper:
            def __init__(self, **kwargs: object) -> None:
                captured["args"] = list(kwargs["claude_args"])  # type: ignore[arg-type]

            def start(self) -> None:
                return None

        monkeypatch.setattr(
            "session_manager.wrapper.main.SessionManagerWrapper", _FakeWrapper
        )
        # Registration is unit-tested separately; keep main() hermetic.
        # 등록 로직은 별도 단위 테스트 — main() 은 밀폐 상태로 유지.
        monkeypatch.setattr(
            "session_manager.wrapper.main.ensure_hook_registered",
            lambda project: (
                registration_calls.append(project)
                if registration_calls is not None
                else None
            ),
        )
        monkeypatch.setattr(sys, "argv", ["ccode", *argv])
        main()
        return captured["args"]

    def test_no_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._run_main([], monkeypatch) == []

    def test_user_args_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._run_main(["--resume", "foo"], monkeypatch) == [
            "--resume",
            "foo",
        ]

    def test_no_channels_flag_injected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args = self._run_main(["--model", "opus"], monkeypatch)
        assert not any("channel" in a for a in args)


class TestNoHooksFlag:
    """--no-hooks is a ccode-only flag: stripped and skips registration.

    --no-hooks 는 ccode 전용 플래그 — claude 인자에서 제거되고 등록
    검사를 건너뛴다.
    """

    def test_flag_stripped_and_registration_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list = []
        args = TestClaudeArgsPassthrough()._run_main(
            ["--no-hooks", "--model", "opus"], monkeypatch, registration_calls=calls
        )
        assert args == ["--model", "opus"]
        assert calls == []

    def test_registration_runs_without_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list = []
        args = TestClaudeArgsPassthrough()._run_main(
            ["--model", "opus"], monkeypatch, registration_calls=calls
        )
        assert args == ["--model", "opus"]
        assert len(calls) == 1

