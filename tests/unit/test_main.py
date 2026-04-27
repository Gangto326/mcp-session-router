"""
Unit tests for the ccode entry point.

ccode 진입점 단위 테스트.
"""

from __future__ import annotations

from session_manager.wrapper.main import _ensure_channels_flag, _resolve_socket_path


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


class TestEnsureChannelsFlag:
    """_ensure_channels_flag prepends the dev flag when the user hasn't set one.
    사용자가 channels 플래그를 직접 주지 않았으면 dev 플래그를 앞에 추가한다.
    """

    def test_empty_args_gets_dev_flag(self) -> None:
        assert _ensure_channels_flag([]) == [
            "--dangerously-load-development-channels",
            "server:session-manager",
        ]

    def test_user_args_preserved_after_dev_flag(self) -> None:
        result = _ensure_channels_flag(["--resume", "foo"])
        assert result == [
            "--dangerously-load-development-channels",
            "server:session-manager",
            "--resume",
            "foo",
        ]

    def test_user_dev_flag_preserved_as_is(self) -> None:
        """If the user already passed the dev flag, leave args alone.
        사용자가 이미 dev 플래그를 줬으면 그대로 둔다.
        """
        args = ["--dangerously-load-development-channels", "server:custom"]
        assert _ensure_channels_flag(args) == args

    def test_user_dev_flag_with_equals_preserved(self) -> None:
        args = ["--dangerously-load-development-channels=server:custom"]
        assert _ensure_channels_flag(args) == args

    def test_user_channels_flag_preserved(self) -> None:
        """``--channels`` (production allowlist) is also a user opt-in.
        ``--channels``도 사용자 의사로 간주.
        """
        args = ["--channels", "plugin:foo@bar"]
        assert _ensure_channels_flag(args) == args

    def test_user_channels_with_equals_preserved(self) -> None:
        args = ["--channels=plugin:foo@bar"]
        assert _ensure_channels_flag(args) == args

    def test_input_list_not_mutated(self) -> None:
        """Returned list is a new list (defensive copy).
        반환 리스트는 새 리스트 (방어적 복사).
        """
        original = ["--resume", "x"]
        result = _ensure_channels_flag(original)
        assert result is not original
        assert original == ["--resume", "x"]
