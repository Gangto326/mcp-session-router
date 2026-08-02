"""
Unit tests for the wrapper-side Unix socket server.

Covers both connection types: the resident MCP connection (promoted from
pending on its first non-hook message) and short-lived hook connections
(one message → ack → close).

래퍼-MCP 통신용 Unix 소켓 서버 단위 테스트. 실제 AF_UNIX 소켓을 임시
디렉토리에 생성해 검증한다 (mock 보다 실증력이 큼).

두 연결 유형을 모두 다룬다: 상주 MCP 연결 (첫 비-hook 메시지에서 pending
으로부터 승격)과 단발 hook 연결 (메시지 1건 → ack → 종료).
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


def data_is_ack(data: bytes) -> bool:
    return data == b'{"type": "ack", "ok": true}\n'


def _connect_resident(
    server: WrapperSocketServer, path: str
) -> socket.socket:
    """
    Connect and promote to the resident slot with a handshake message.

    접속 후 handshake 메시지로 상주 자리 승격까지 완료한다.
    """
    sock = _connect(path)
    server.handle_listen_readable()
    sock.sendall(b'{"type":"handshake_request"}\n')
    server.handle_pending_readable(server.pending_filenos[0])
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
            assert server.pending_filenos == []
        finally:
            server.stop()

    def test_stop_closes_pending_connections(self, socket_path: str) -> None:
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        client = _connect(socket_path)
        server.handle_listen_readable()
        assert len(server.pending_filenos) == 1

        server.stop()
        assert server.pending_filenos == []
        # 서버가 닫았으므로 클라이언트는 EOF 를 받는다
        assert client.recv(4096) == b""
        client.close()


class TestClientConnection:
    def test_accept_holds_connection_as_pending(self, socket_path: str) -> None:
        # accept 만으로는 유형 미확정 — 상주 자리를 차지하지 않는다
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        try:
            client = _connect(socket_path)
            server.handle_listen_readable()
            assert not server.has_client()
            assert len(server.pending_filenos) == 1
            client.close()
        finally:
            server.stop()

    def test_first_message_promotes_to_resident(self, socket_path: str) -> None:
        received: list[dict] = []
        server = WrapperSocketServer(socket_path, on_message=received.append)
        server.start()
        try:
            client = _connect_resident(server, socket_path)
            assert server.has_client()
            assert server.pending_filenos == []
            assert received == [{"type": "handshake_request"}]
            client.close()
        finally:
            server.stop()

    def test_rejects_second_resident(self, socket_path: str) -> None:
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        try:
            client1 = _connect_resident(server, socket_path)
            first_client_fd = server.client_fileno
            assert first_client_fd >= 0

            client2 = _connect(socket_path)
            server.handle_listen_readable()
            client2.sendall(b'{"type":"handshake_request"}\n')
            server.handle_pending_readable(server.pending_filenos[0])
            # 단일 상주 정책 — 새 상주 시도는 거부, 기존 client fd 유지
            assert server.client_fileno == first_client_fd
            assert server.pending_filenos == []
            # 거부된 쪽은 연결이 닫혀 EOF 를 받는다
            assert client2.recv(4096) == b""

            client1.close()
            client2.close()
        finally:
            server.stop()

    def test_pending_eof_before_message_cleans_up(self, socket_path: str) -> None:
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        try:
            client = _connect(socket_path)
            server.handle_listen_readable()
            fd = server.pending_filenos[0]

            client.close()
            server.handle_pending_readable(fd)
            assert server.pending_filenos == []
            assert not server.has_client()
        finally:
            server.stop()


class TestHookConnection:
    def test_hook_message_dispatched_acked_and_closed(
        self, socket_path: str
    ) -> None:
        received: list[dict] = []
        server = WrapperSocketServer(socket_path, on_message=received.append)
        server.start()
        try:
            hook = _connect(socket_path)
            server.handle_listen_readable()
            hook.sendall(b'{"client":"hook","action":"route_check"}\n')
            server.handle_pending_readable(server.pending_filenos[0])

            assert received == [{"client": "hook", "action": "route_check"}]
            # ack 수신 후 서버 측이 연결을 닫으므로 ack 다음에 EOF
            data = hook.recv(4096)
            assert data == b'{"type": "ack", "ok": true}\n'
            assert hook.recv(4096) == b""
            assert server.pending_filenos == []
            hook.close()
        finally:
            server.stop()

    def test_hook_does_not_claim_resident_slot(self, socket_path: str) -> None:
        # 상주(MCP) 부재 중 hook 이 먼저 접속해도 상주 자리는 빈 채로 남는다
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        try:
            hook = _connect(socket_path)
            server.handle_listen_readable()
            hook.sendall(b'{"client":"hook"}\n')
            server.handle_pending_readable(server.pending_filenos[0])
            assert not server.has_client()
            hook.close()

            # 이후 상주 연결이 정상 승격되는지 확인
            client = _connect_resident(server, socket_path)
            assert server.has_client()
            client.close()
        finally:
            server.stop()

    def test_hook_works_while_resident_connected(self, socket_path: str) -> None:
        received: list[dict] = []
        server = WrapperSocketServer(socket_path, on_message=received.append)
        server.start()
        try:
            client = _connect_resident(server, socket_path)
            resident_fd = server.client_fileno

            hook = _connect(socket_path)
            server.handle_listen_readable()
            hook.sendall(b'{"client":"hook","n":1}\n')
            server.handle_pending_readable(server.pending_filenos[0])

            assert {"client": "hook", "n": 1} in received
            assert server.client_fileno == resident_fd
            assert data_is_ack(hook.recv(4096))
            hook.close()
            client.close()
        finally:
            server.stop()


class TestMessageReceive:
    def test_receives_json_message(self, socket_path: str) -> None:
        received: list[dict] = []
        server = WrapperSocketServer(socket_path, on_message=received.append)
        server.start()
        try:
            client = _connect_resident(server, socket_path)
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
            client = _connect_resident(server, socket_path)
            received.clear()

            client.sendall(b'{"a":1}\n{"b":2}\n')
            server.handle_client_readable()

            assert received == [{"a": 1}, {"b": 2}]
            client.close()
        finally:
            server.stop()

    def test_promotion_carries_trailing_messages(self, socket_path: str) -> None:
        # 첫 메시지와 같은 chunk 에 도착한 후속 프레임이 승격 시 유실되지
        # 않아야 한다
        received: list[dict] = []
        server = WrapperSocketServer(socket_path, on_message=received.append)
        server.start()
        try:
            client = _connect(socket_path)
            server.handle_listen_readable()

            client.sendall(b'{"type":"handshake_request"}\n{"a":1}\n')
            server.handle_pending_readable(server.pending_filenos[0])

            assert received == [{"type": "handshake_request"}, {"a": 1}]
            assert server.has_client()
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
            fd = server.pending_filenos[0]

            client.sendall(b'{"type":"hand')
            server.handle_pending_readable(fd)
            assert received == []
            assert not server.has_client()

            client.sendall(b'shake_request"}\n')
            server.handle_pending_readable(fd)
            assert received == [{"type": "handshake_request"}]
            assert server.has_client()
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
            server.handle_pending_readable(server.pending_filenos[0])

            assert received == [{"valid": True}]
            client.close()
        finally:
            server.stop()

    def test_client_eof_closes_connection(self, socket_path: str) -> None:
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        try:
            client = _connect_resident(server, socket_path)
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
            client = _connect_resident(server, socket_path)

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

    def test_send_to_pending_returns_false(self, socket_path: str) -> None:
        # pending 은 아직 상주가 아니다 — 승격 전에는 송신 대상이 없다
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        try:
            client = _connect(socket_path)
            server.handle_listen_readable()
            assert server.send({"x": 1}) is False
            client.close()
        finally:
            server.stop()

    def test_send_korean_not_ascii_escaped(self, socket_path: str) -> None:
        server = WrapperSocketServer(socket_path, on_message=lambda _: None)
        server.start()
        try:
            client = _connect_resident(server, socket_path)

            server.send({"msg": "한글"})
            data = client.recv(4096)
            assert "한글".encode() in data
            assert b"\\u" not in data
            client.close()
        finally:
            server.stop()
