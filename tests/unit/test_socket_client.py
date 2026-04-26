"""Tests for WrapperSocketClient.recv_loop (async push receive).
WrapperSocketClient.recv_loop (비동기 push 수신) 단위 테스트.

Uses a real AF_UNIX socketpair to exercise the full I/O path: writes
go from the "wrapper side" socket and `recv_loop` parses them line-by-line
into the on_message callback.

실제 AF_UNIX socketpair로 I/O 경로 전체를 검증한다 — "래퍼 쪽" 소켓에 쓰면
recv_loop가 라인 단위로 파싱해 on_message 콜백에 전달.
"""

from __future__ import annotations

import asyncio
import json
import socket

import pytest

from session_manager.wrapper.socket_client import WrapperSocketClient


def _make_pair() -> tuple[WrapperSocketClient, socket.socket]:
    """Build a connected WrapperSocketClient + the wrapper-side counterpart.
    연결된 WrapperSocketClient + 래퍼 쪽 짝 소켓을 만든다.
    """
    server_sock, client_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    client = WrapperSocketClient("/dev/null")
    client._sock = client_sock
    return client, server_sock


@pytest.mark.asyncio
async def test_recv_loop_dispatches_single_message():
    """Single JSON line from wrapper → on_message called once with parsed dict.
    래퍼가 보낸 JSON 한 줄 → on_message 1회 호출 (dict 파싱됨).
    """
    client, wrapper_sock = _make_pair()
    received: list[dict] = []

    async def on_msg(msg: dict) -> None:
        received.append(msg)

    task = asyncio.create_task(client.recv_loop(on_msg))
    # Give the loop a tick to switch to non-blocking mode
    # 루프가 non-blocking으로 전환할 시간을 줌
    await asyncio.sleep(0.01)

    wrapper_sock.sendall(b'{"action":"intercept","command":"resume","args":"foo"}\n')
    # Yield until the loop processes it
    # 루프가 처리할 때까지 양보
    for _ in range(50):
        await asyncio.sleep(0.01)
        if received:
            break

    wrapper_sock.close()
    await asyncio.wait_for(task, timeout=1.0)

    assert received == [
        {"action": "intercept", "command": "resume", "args": "foo"}
    ]


@pytest.mark.asyncio
async def test_recv_loop_dispatches_multiple_messages():
    """Multiple lines in one chunk → multiple on_message calls in order.
    한 chunk에 여러 줄 → 순서대로 on_message 여러 번.
    """
    client, wrapper_sock = _make_pair()
    received: list[dict] = []

    async def on_msg(msg: dict) -> None:
        received.append(msg)

    task = asyncio.create_task(client.recv_loop(on_msg))
    await asyncio.sleep(0.01)

    wrapper_sock.sendall(b'{"a":1}\n{"a":2}\n{"a":3}\n')
    for _ in range(50):
        await asyncio.sleep(0.01)
        if len(received) >= 3:
            break

    wrapper_sock.close()
    await asyncio.wait_for(task, timeout=1.0)

    assert received == [{"a": 1}, {"a": 2}, {"a": 3}]


@pytest.mark.asyncio
async def test_recv_loop_handles_split_chunks():
    """A JSON line split across two writes → reassembled correctly.
    JSON 한 줄이 두 chunk에 나뉘어도 재조립.
    """
    client, wrapper_sock = _make_pair()
    received: list[dict] = []

    async def on_msg(msg: dict) -> None:
        received.append(msg)

    task = asyncio.create_task(client.recv_loop(on_msg))
    await asyncio.sleep(0.01)

    wrapper_sock.sendall(b'{"hel')
    await asyncio.sleep(0.05)
    wrapper_sock.sendall(b'lo":"world"}\n')
    for _ in range(50):
        await asyncio.sleep(0.01)
        if received:
            break

    wrapper_sock.close()
    await asyncio.wait_for(task, timeout=1.0)

    assert received == [{"hello": "world"}]


@pytest.mark.asyncio
async def test_recv_loop_skips_malformed_json():
    """Malformed JSON lines are silently skipped; subsequent valid lines work.
    깨진 JSON은 조용히 skip, 이후 정상 라인은 처리됨.
    """
    client, wrapper_sock = _make_pair()
    received: list[dict] = []

    async def on_msg(msg: dict) -> None:
        received.append(msg)

    task = asyncio.create_task(client.recv_loop(on_msg))
    await asyncio.sleep(0.01)

    wrapper_sock.sendall(b'not json\n{"valid":true}\n')
    for _ in range(50):
        await asyncio.sleep(0.01)
        if received:
            break

    wrapper_sock.close()
    await asyncio.wait_for(task, timeout=1.0)

    assert received == [{"valid": True}]


@pytest.mark.asyncio
async def test_recv_loop_ignores_non_dict_messages():
    """JSON arrays/scalars are skipped — only dicts get dispatched.
    JSON array/scalar는 skip, dict만 dispatch.
    """
    client, wrapper_sock = _make_pair()
    received: list[dict] = []

    async def on_msg(msg: dict) -> None:
        received.append(msg)

    task = asyncio.create_task(client.recv_loop(on_msg))
    await asyncio.sleep(0.01)

    wrapper_sock.sendall(b'[1,2,3]\n"a string"\n42\n{"ok":true}\n')
    for _ in range(50):
        await asyncio.sleep(0.01)
        if received:
            break

    wrapper_sock.close()
    await asyncio.wait_for(task, timeout=1.0)

    assert received == [{"ok": True}]


@pytest.mark.asyncio
async def test_recv_loop_continues_after_callback_exception():
    """An exception from on_message must not kill the loop.
    on_message 예외가 루프를 죽이면 안 됨.
    """
    client, wrapper_sock = _make_pair()
    received: list[dict] = []
    call_count = 0

    async def on_msg(msg: dict) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("boom")
        received.append(msg)

    task = asyncio.create_task(client.recv_loop(on_msg))
    await asyncio.sleep(0.01)

    wrapper_sock.sendall(b'{"first":1}\n{"second":2}\n')
    for _ in range(50):
        await asyncio.sleep(0.01)
        if received:
            break

    wrapper_sock.close()
    await asyncio.wait_for(task, timeout=1.0)

    assert call_count == 2  # 둘 다 시도됨
    assert received == [{"second": 2}]  # 두 번째만 성공 적재


@pytest.mark.asyncio
async def test_recv_loop_returns_on_eof():
    """When the wrapper closes its end, recv_loop returns cleanly.
    래퍼가 자기 쪽 닫으면 recv_loop는 깔끔하게 return.
    """
    client, wrapper_sock = _make_pair()

    async def on_msg(msg: dict) -> None:
        pass

    task = asyncio.create_task(client.recv_loop(on_msg))
    await asyncio.sleep(0.01)

    wrapper_sock.close()
    # Loop should exit on EOF
    # EOF에 루프 종료
    await asyncio.wait_for(task, timeout=1.0)
    assert task.done()


@pytest.mark.asyncio
async def test_recv_loop_no_op_when_not_connected():
    """recv_loop is a silent no-op when no socket is connected.
    소켓 연결 없을 때 recv_loop는 silent no-op.
    """
    client = WrapperSocketClient("/dev/null")
    # Note: not connected
    # 연결 안 함

    async def on_msg(msg: dict) -> None:
        pytest.fail("should not be called")

    # Should return immediately
    # 즉시 return
    await asyncio.wait_for(client.recv_loop(on_msg), timeout=0.5)


@pytest.mark.asyncio
async def test_recv_loop_can_be_cancelled():
    """asyncio.Task.cancel() exits the loop cleanly.
    Task.cancel()로 루프 정상 종료.
    """
    client, wrapper_sock = _make_pair()

    async def on_msg(msg: dict) -> None:
        pass

    task = asyncio.create_task(client.recv_loop(on_msg))
    await asyncio.sleep(0.05)

    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except asyncio.CancelledError:
        pass

    wrapper_sock.close()
    assert task.done()


def test_recv_loop_round_trip_serialization():
    """Sanity: the format we expect from the wrapper is what we serialize.
    래퍼로부터 받을 형식이 우리 직렬화와 호환되는지.
    """
    payload = {"action": "intercept", "command": "resume", "args": "foo"}
    line = (json.dumps(payload) + "\n").encode("utf-8")
    parsed = json.loads(line.split(b"\n", 1)[0].decode("utf-8"))
    assert parsed == payload
