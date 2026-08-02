"""
Routing — deciding which session a new user prompt belongs to.

``judge.py`` holds the pure logic (prompt assembly, verdict parsing,
timeout constants); the process management lives in
``wrapper/judge_host.py``.

라우팅 — 새 사용자 프롬프트가 어느 세션 소관인지 판정하는 계층.

``judge.py``가 순수 로직(프롬프트 조립, 판정 파싱, 타임아웃 상수)을
담당하고, 프로세스 관리는 ``wrapper/judge_host.py``에 있다.
"""
