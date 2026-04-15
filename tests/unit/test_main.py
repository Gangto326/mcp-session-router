"""
Unit tests for the ccode entry point.

ccode 진입점 단위 테스트.
"""

from __future__ import annotations

from session_manager.wrapper.main import _resolve_socket_path


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
