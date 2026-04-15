"""
Unit tests for the wrapper-side Unix socket server.

래퍼-MCP 통신용 Unix 소켓 서버 단위 테스트. 실제 AF_UNIX 소켓을 임시
디렉토리에 생성해 검증한다 (mock 보다 실증력이 큼).
"""

from __future__ import annotations

import os
import socket
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from session_manager.wrapper.socket_server import WrapperSocketServer


@pytest.fixture
def socket_path() -> Iterator[str]:
    # macOS pytest tmp_path (예: /private/var/folders/.../) 는 AF_UNIX 108B 한계
    # 초과 가능. /tmp 에 짧은 경로로 직접 만든다.
    path = f"/tmp/test-sock-{uuid.uuid4().hex[:8]}.sock"
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def _connect(path: str) -> socket.socket:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(path)
    return sock


class TestStartStop:
    def test_start_creates_socket_file(self, socket_path: str) -> None:
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        try:
            assert Path(socket_path).exists()
        finally:
            server.stop()

    def test_stop_removes_socket_file(self, socket_path: str) -> None:
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        server.stop()
        assert not Path(socket_path).exists()

    def test_start_cleans_stale_socket_file(self, socket_path: str) -> None:
        # 사전에 stale 파일 만듦 — 시작 시 자동 정리되어야 함
        Path(socket_path).touch()
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        try:
            assert Path(socket_path).exists()
        finally:
            server.stop()

    def test_listen_fileno_negative_before_start(self, socket_path: str) -> None:
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        assert server.listen_fileno == -1

    def test_client_fileno_negative_before_connection(
        self, socket_path: str
    ) -> None:
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        try:
            assert server.client_fileno == -1
            assert server.has_client() is False
        finally:
            server.stop()


class TestClientConnection:
    def test_accepts_first_client(self, socket_path: str) -> None:
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        try:
            client = _connect(socket_path)
            server.handle_listen_readable()
            assert server.has_client()
            client.close()
        finally:
            server.stop()

    def test_rejects_second_client(self, socket_path: str) -> None:
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        try:
            client1 = _connect(socket_path)
            server.handle_listen_readable()
            first_client_fd = server.client_fileno
            assert first_client_fd >= 0

            client2 = _connect(socket_path)
            server.handle_listen_readable()
            # 단일 클라이언트 정책 — 새 연결은 거부, 기존 client fd 유지
            assert server.client_fileno == first_client_fd

            client1.close()
            client2.close()
        finally:
            server.stop()


class TestMessageReceive:
    def test_receives_json_message(self, socket_path: str) -> None:
        received: list[dict] = []
        server = WrapperSocketServer(socket_path, on_message=received.append)
        server.start()
        try:
            client = _connect(socket_path)
            server.handle_listen_readable()

            client.sendall(b'{"type":"handshake_request"}\n')
            server.handle_client_readable()

            assert received == [{"type": "handshake_request"}]
            client.close()
        finally:
            server.stop()

    def test_receives_multiple_messages_in_one_chunk(
        self, socket_path: str
    ) -> None:
        received: list[dict] = []
        server = WrapperSocketServer(socket_path, on_message=received.append)
        server.start()
        try:
            client = _connect(socket_path)
            server.handle_listen_readable()

            client.sendall(b'{"a":1}\n{"b":2}\n')
            server.handle_client_readable()

            assert received == [{"a": 1}, {"b": 2}]
            client.close()
        finally:
            server.stop()

    def test_partial_message_buffered(self, socket_path: str) -> None:
        received: list[dict] = []
        server = WrapperSocketServer(socket_path, on_message=received.append)
        server.start()
        try:
            client = _connect(socket_path)
            server.handle_listen_readable()

            client.sendall(b'{"type":"hand')
            server.handle_client_readable()
            assert received == []

            client.sendall(b'shake_request"}\n')
            server.handle_client_readable()
            assert received == [{"type": "handshake_request"}]
            client.close()
        finally:
            server.stop()

    def test_malformed_json_ignored(self, socket_path: str) -> None:
        received: list[dict] = []
        server = WrapperSocketServer(socket_path, on_message=received.append)
        server.start()
        try:
            client = _connect(socket_path)
            server.handle_listen_readable()

            client.sendall(b'not json at all\n{"valid":true}\n')
            server.handle_client_readable()

            assert received == [{"valid": True}]
            client.close()
        finally:
            server.stop()

    def test_client_eof_closes_connection(self, socket_path: str) -> None:
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        try:
            client = _connect(socket_path)
            server.handle_listen_readable()
            assert server.has_client()

            client.close()
            server.handle_client_readable()
            assert not server.has_client()
        finally:
            server.stop()


class TestSend:
    def test_sends_json_with_newline(self, socket_path: str) -> None:
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        try:
            client = _connect(socket_path)
            server.handle_listen_readable()

            assert server.send({"current_session_name": "foo"}) is True

            data = client.recv(4096)
            assert data == b'{"current_session_name": "foo"}\n'
            client.close()
        finally:
            server.stop()

    def test_send_without_client_returns_false(self, socket_path: str) -> None:
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        try:
            assert server.send({"x": 1}) is False
        finally:
            server.stop()

    def test_send_korean_not_ascii_escaped(self, socket_path: str) -> None:
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        try:
            client = _connect(socket_path)
            server.handle_listen_readable()

            server.send({"msg": "한글"})
            data = client.recv(4096)
            assert "한글".encode() in data
            assert b"\\u" not in data
            client.close()
        finally:
            server.stop()
