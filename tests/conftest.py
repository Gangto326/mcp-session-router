"""
Shared test fixtures.

공용 테스트 픽스처.
"""

from __future__ import annotations

import pytest

from session_manager.wrapper.judge_host import JudgeHost


@pytest.fixture(autouse=True)
def _no_real_judge_spawn(monkeypatch: pytest.MonkeyPatch):
    """
    Block the judge host from spawning a real ``claude`` subprocess.

    The routing judge warms up a real CLI process (seconds of latency,
    real API cost). No test should trigger that implicitly just because
    its fixture project happens to satisfy the judge start conditions.
    Tests that exercise the spawn path override this with their own
    monkeypatch.

    판정 호스트가 실제 ``claude`` subprocess 를 spawn 하지 못하게 막는다.

    라우팅 판정기는 실제 CLI 프로세스를 웜업한다 (수 초 지연 + 실제 API
    비용). 픽스처 프로젝트가 우연히 판정기 시작 조건을 만족한다는 이유로
    테스트가 암묵적으로 그걸 유발해서는 안 된다. spawn 경로를 검증하는
    테스트는 자체 monkeypatch 로 이 가드를 덮어쓴다.
    """
    monkeypatch.setattr(JudgeHost, "_spawn_and_warm", lambda self: False)
