---
name: sessions
description: 이 프로젝트의 세션 목록을 보여준다. 어떤 세션들이 있고 현재 어디에 있는지 확인하고 싶을 때 사용.
---

ccode 래퍼 안에서 `/sessions` 제출은 보통 래퍼가 PTY 수준에서 가로채 LLM
왕복 없이 즉시 목록을 그린다 — 동작의 단일 출처는
`src/session_manager/wrapper/pty_wrapper.py` (`_handle_sessions_command`) +
`wrapper/notice.py` 다. 이 스킬 본문이 실행됐다면 가로채기를 거치지 않은
경로다 (맨몸 claude, 또는 팝업 선택 등 래퍼가 입력란 전문을 읽지 못한
제출). 어느 쪽인지 단정하지 말고 폴백으로 목록만 보여준다:

1. `.session-manager/sessions/` 아래 `*.json` 파일을 직접 읽는다 — MCP
   도구를 쓰지 마라 (이 경로는 도구 없이도 동작해야 한다). 파일이
   없으면 "등록된 세션 없음" 한 줄.
2. `status` 가 `active` 인 세션만 (필드 부재는 active 로 간주), 세션당
   한 줄: `이름 · 제목 · 마지막 접근(last_accessed)`.
3. 마지막에 "세션 전환·자동 라우팅은 `ccode` 실행 중에 동작합니다" 한
   줄을 덧붙인다.
4. 다른 작업 금지 — 목록 표시 외에 아무것도 바꾸지 마라.
