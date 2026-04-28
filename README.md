# mcp-session-router

Claude Code에서 여러 작업 주제를 **세션 단위로 분리**하여 컨텍스트 오염을 방지하는 MCP 서버 + PTY 래퍼.

하나의 Claude Code 프로세스 안에서 프론트엔드·백엔드·인프라 등 주제가 섞이면 LLM이 맥락을 혼동한다. mcp-session-router는 각 주제를 독립된 세션으로 관리하고, 주제가 바뀌면 자동으로 세션을 전환하여 이전 맥락을 보존한다.

## 요구 사항

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) 패키지 매니저
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) v2.1.80 이상
- **claude.ai 계정** (Pro/Max/Free) 로 인증된 Claude Code — API key (`ANTHROPIC_API_KEY`) 인증은 슬래시 명령 가로채기에 쓰이는 `claude/channel` capability를 지원하지 않아 사용 불가

## 설치

```bash
git clone https://github.com/Gangto326/mcp-session-router.git
cd mcp-session-router
uv sync
```

## 설정

### MCP 서버 등록

Claude Code에 MCP 서버를 등록한다. `--scope user`로 전역 등록하면 모든 프로젝트에서 사용할 수 있다.

```bash
claude mcp add session-manager --scope user -- \
  uv run --project /path/to/mcp-session-router \
  python -m session_manager.server
```

`/path/to/mcp-session-router`를 실제 클론한 경로로 교체한다.

## 사용법

작업할 프로젝트 디렉토리에서 `claude` 대신 아래 명령을 실행한다. Claude Code가 실행된 뒤의 사용법은 평소와 완전히 같으며, 세션 전환은 LLM이 자동으로 처리한다.

```bash
# 기존: claude → uv run ccode로 바꿔서 시작
uv run --project /path/to/mcp-session-router ccode

# claude에 전달하던 인자도 그대로 사용 가능
uv run --project /path/to/mcp-session-router ccode --model sonnet

# 이전 대화 기록까지 이어서 하려면 --resume 사용
uv run --project /path/to/mcp-session-router ccode --resume <session-name>
```

매번 경로를 입력하기 번거로우면 셸 alias를 등록한다:

```bash
# ~/.zshrc 또는 ~/.bashrc
alias ccode='uv run --project /path/to/mcp-session-router ccode'
```

이후 아무 디렉토리에서든 `ccode`로 실행할 수 있다.

세션 데이터는 프로젝트 루트의 `.session-manager/` 디렉토리에 JSON 파일로 저장된다.

## 작동 방식

```
┌─────────────────────────────────────────────────┐
│ ccode (PTY 래퍼)                                 │
│  ├─ Claude Code를 PTY 위에 spawn                  │
│  ├─ Unix Socket 서버 (MCP 서버가 여기에 접속)         │
│  └─ 세션 전환 시 /resume, /exit 등 명령 자동 주입      │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │ Claude Code                              │    │
│  │  └─ MCP 서버 (session-manager)            │    │
│  │      ├─ 래퍼에 접속하여 핸드셰이크              │    │
│  │      ├─ 세션 전환 시 래퍼에 신호 전송           │    │
│  │      └─ 세션 메타데이터 관리 (디스크 저장)       │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

1. 사용자가 메시지를 입력하면 LLM이 현재 세션 주제와 비교한다.
2. 주제가 다르면 LLM이 서브 에이전트를 spawn하여 `check_session`으로 세션 목록의 summary를 읽고 어느 세션으로 보낼지 판단한다.
3. 판단 결과에 따라:
   - **STAY**: 현재 세션에서 그대로 처리.
   - **SWITCH**: 사용자에게 확인 → `session_switch` → 래퍼가 `/resume <target>` 주입 → 대상 세션으로 전환.
   - **NEW**: 사용자에게 확인 → `session_create` → 래퍼가 `/exit` + 새 Claude Code 재시작 → 새 세션 시작.
4. 전환 시 나가는 세션의 summary가 저장되고, 들어오는 세션에 handoff 블록(이전 맥락 요약 + 읽어야 할 파일 목록)이 전달된다.

### 사용자 직접 슬래시 명령 가로채기

사용자가 `/resume`·`/exit`·`/rename`·`/new`를 직접 타이핑하고 Enter를 누르면 `ccode`가 다음 흐름을 자동 실행한다:

1. `ccode`가 Enter(`\r`)를 가로채고 입력란에 표시된 명령을 가상 터미널에서 추출
2. MCP 서버가 LLM 컨텍스트로 channel 메시지 push (`<channel source="session-manager" command="..." args="...">`)
3. LLM이 instructions에 따라 `session_end` 도구를 호출 → 현재 세션 summary 갱신
4. `ccode`가 보관해둔 `\r`을 PTY로 forward → 사용자가 친 명령 실행

이 메커니즘은 Claude Code의 `claude/channel` capability를 사용하므로 `ccode`가 spawn 시 자동으로 `--dangerously-load-development-channels server:session-manager` 플래그를 추가한다. 매 실행마다 뜨는 channels dev 경고와 첫 실행 시 MCP server 등록 prompt는 모두 자동 승인된다 — 사용자가 별도로 키를 누를 필요 없음.

15초 안에 LLM이 응답하지 않으면 graceful degradation으로 `\r`을 그대로 forward하고 안내 메시지를 출력한다 (summary는 갱신되지 못한 채). 가로채기 중 Ctrl+C를 누르면 명령 자체가 취소 (LLM 응답 중단 + 명령 실행 안 함).

## 사용 시나리오

### 세션 전환

```
[세션: frontend-ui]
사용자: "로그인 폼에 비밀번호 유효성 검사 추가해줘"
→ 현재 세션 주제와 일치 → STAY → 바로 작업 수행

사용자: "/api/users 엔드포인트의 페이지네이션 어떻게 되어 있어?"
→ 서브 에이전트 판단: 백엔드 주제 → SWITCH to backend-api

Claude: "이 질문은 backend-api 세션에서 다루던 주제입니다.
         세션을 전환할까요?"
  [Yes]  → session_switch → 래퍼가 /resume backend-api 주입 → 전환
  [No]   → 현재 세션에서 그대로 처리
  [입력] → 사용자가 직접 대상 세션을 지정
```

### 새 세션 생성

```
[세션: frontend-ui]
사용자: "GitHub Actions CI/CD 파이프라인 설정해줘"
→ 서브 에이전트 판단: 기존 세션 어디에도 해당 없음 → NEW

Claude: "기존 세션에 해당하지 않는 새 주제입니다.
         새 세션을 만들까요?"
  [Yes]  → session_create → 래퍼가 현재 세션 종료 후 새 Claude Code 시작
  [No]   → 현재 세션에서 그대로 처리
  [입력] → 사용자가 세션 이름이나 방향을 직접 지정
```

### 모호한 주제

```
[세션: frontend-ui]
사용자: "테스트 고쳐줘"
→ 서브 에이전트 판단: 여러 세션이 후보 → ASK_USER

Claude: "다음 세션 중 어디에서 작업할까요?
         1. test-refactor — pytest 픽스처 마이그레이션, 3개 파일 남음
         2. frontend-ui (현재) — 로그인 폼 UI 작업 중"
  [입력] → 사용자가 번호나 세션 이름으로 선택
```

### 세션 복귀

```bash
# --resume 없이 시작하면 새 대화로 시작된다.
# MCP 서버는 가장 최근 세션의 메타데이터(이름, summary)를 기억하지만,
# Claude Code의 대화 기록은 새로 시작된다.
ccode

# 이전 대화 기록까지 이어서 하려면 --resume 사용
ccode --resume backend-api
```

(위 예시는 셸 alias 등록 후 기준. alias 없이는 `uv run --project /path/to/mcp-session-router ccode ...`)

## MCP 도구 목록

| 도구 | 설명 |
|------|------|
| `check_session` | 현재 세션과 전체 세션 목록(이름, 제목, summary, 상태) 조회 |
| `session_register` | 새 세션 등록 (부트스트랩 시 호출) |
| `session_switch` | 기존 세션으로 전환 — summary 저장 + 래퍼에 SWITCH 신호 |
| `session_create` | 새 세션 생성 — Claude Code 재시작 + 래퍼에 NEW 신호 |
| `session_end` | 세션 종료 + ARCHIVED 상태로 변경 |
| `update_static` | 프로젝트 전역 공유 정보(환경, 컨벤션 등) 부분 갱신 |
| `init_project` | project-context.md 초기 생성 (이미 존재하면 no-op) |
| `reinit_project` | project-context.md 전체 재작성 |
| `update_project_context` | project-context.md 내용 교체 |

## 현재 버전 한계

### 사용자 직접 전환 시 summary 갱신은 LLM 신뢰도 의존

가로채기 흐름은 LLM이 channel 메시지를 보고 `session_end`를 호출해야 작동한다. LLM이 instructions를 따르지 않으면 15초 timeout 후 명령은 그대로 실행되지만 summary는 stale로 남는다. 즉 가로채기는 "summary 갱신을 시도"이지 "보장"이 아니다.

또한 가로채기 동안:
- 사용자 화면에 channel 메시지 (`← session-manager: ...`)가 1-3초간 visible
- 가로채기 중 추가 키 입력은 무시됨 (Ctrl+C만 명령 취소로 처리)
- 사용자가 빠르게 타이핑할 때 일부 키가 가상 화면에 누락될 가능성이 의심됨 (race) — 매칭 실패 시 명령은 그대로 실행되고 summary만 stale로 남는다

### Channels research preview 의존

`claude/channel` capability는 Anthropic의 research preview 단계라 미래 버전에서 protocol 변경 가능. Claude Code 업데이트 후 가로채기가 깨질 수 있으니 변경 시 issue 등록 부탁한다.

### `--dangerously-load-development-channels` 자동 사용

본 서버는 Anthropic 공식 channel allowlist에 등재되지 않았으므로 `ccode`가 위 development 플래그를 자동 추가한다. 이 플래그는 보안 경계를 약화시킨다는 의미를 가지며, 매 실행 시 뜨는 channels dev 경고 prompt도 자동 우회된다 — 사용자가 모르게 보안 prompt를 넘긴다는 의미. 신뢰하지 않는 환경에서는 사용을 권하지 않는다.

### `/clear` 후 stale summary

`/clear`는 LLM 컨텍스트만 비우고 세션 ID와 MCP 서버는 유지한다. `/clear` 후에도 세션의 summary는 마지막 전환 시점의 값 그대로이므로, 이후 서브 에이전트 매칭에서 부정확한 판단이 있을 수 있다.

### 가로채기 매칭의 false positive / negative

- **False positive**: 사용자가 슬래시 명령 인자에 `[...]` 형태의 텍스트를 포함하면 placeholder hint로 오인되어 잘릴 수 있다 (예: `/rename foo [bar]` → `/rename foo`).
- **False negative**: 매칭 화이트리스트는 `/resume`·`/exit`·`/rename`·`/new`로 고정. Claude Code가 새 세션 lifecycle 명령을 추가하면 가로채기 대상에서 누락된다.

### Claude Code 입력란 디자인 의존

가로채기는 입력란 마커(`❯`)와 부분 갱신 ANSI 패턴에 의존한다. Claude Code 입력란 디자인이 변경되면 가로채기가 깨질 수 있다.


## 개발

```bash
# 테스트
uv run pytest tests/unit -v
uv run pytest tests/integration -v

# 린트
uv run ruff check src/ tests/
```

## 라이선스

MIT
