"""
Entry point for the `ccode` command.

Resolves a per-project Unix socket path, exports it via environment so the
MCP server (spawned as a child of Claude Code) can find the wrapper, then
hands control to SessionManagerWrapper which spawns Claude Code on a PTY
and runs the I/O loop until exit.

`ccode` 명령의 진입점.

프로젝트별로 고유한 Unix 소켓 경로를 결정하고, MCP 서버(Claude Code 자식
프로세스로 spawn 됨) 가 래퍼를 찾을 수 있도록 환경 변수로 노출한다.
이후 SessionManagerWrapper 가 Claude Code 를 PTY 에 띄우고 종료까지
I/O 루프를 돌린다.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

from session_manager import debug_log
from session_manager.hooks.registration import (
    ensure_hook_registered,
    ensure_statusline_registered,
)
from session_manager.routing import stats as routing_stats
from session_manager.wrapper.pty_wrapper import SessionManagerWrapper

SOCKET_ENV_VAR = "SESSION_MANAGER_SOCKET"

# ccode-only flag: skip the routing-hook registration check. Stripped
# from the args passed on to Claude Code.
# ccode 전용 플래그 — 라우팅 hook 등록 검사를 건너뛴다. Claude Code 에
# 넘기는 인자에서는 제거된다.
NO_HOOKS_FLAG = "--no-hooks"


# AF_UNIX sun_path limit is 104 bytes on macOS (108 on Linux). Engineering
# parameter: use the smaller platform limit minus a 4-byte margin.
# AF_UNIX sun_path 한계는 macOS 104바이트 (Linux 108). 공학 파라미터 —
# 더 작은 플랫폼 한계에서 4바이트 여유를 뺀 값.
_SOCKET_PATH_MAX_BYTES = 100


def _resolve_socket_path(project_path: str) -> str:
    # Short hash keeps the path short while giving a per-project namespace.
    # The socket lives under the user's home (F17): /tmp is world-readable,
    # so on multi-user machines any user could connect and drive the
    # wrapper (switch signals, judge requests). A 0700 run dir plus the
    # socket's own 0600 (set at bind) closes that. Pathologically long
    # home paths fall back to /tmp — the 0600 socket mode still applies.
    # 짧은 해시로 프로젝트별 네임스페이스를 확보한다. 소켓은 사용자 홈
    # 아래에 둔다 (F17): /tmp 는 모두가 접근 가능해 다중 사용자 머신에서
    # 타 사용자가 connect 해 래퍼를 조종(전환 신호·판정 요청)할 수 있다.
    # 0700 run 디렉토리 + 소켓 자체 0600 (bind 시 설정) 으로 차단한다.
    # 비정상적으로 긴 홈 경로만 /tmp 로 폴백 — 그 경우에도 0600 은 적용.
    project_hash = hashlib.md5(project_path.encode("utf-8")).hexdigest()[:12]
    sock_name = f"session-manager-{project_hash}.sock"
    run_dir = Path.home() / ".session-manager" / "run"
    candidate = run_dir / sock_name
    if len(str(candidate).encode("utf-8")) <= _SOCKET_PATH_MAX_BYTES:
        run_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(run_dir, 0o700)
        return str(candidate)
    return f"/tmp/{sock_name}"


def main() -> int:
    # `ccode --stats [--json]` is a read-only report, handled before any
    # wrapper machinery: no run id, no socket, no hook check, and the
    # flag must never reach `claude` (unknown option there).
    # `ccode --stats [--json]` 은 읽기 전용 보고 — 래퍼 기동 전부 전에
    # 처리한다: run id·소켓·hook 검사 없음. 이 플래그가 `claude` 에
    # 흘러가면 unknown option 이므로 여기서 반드시 가로챈다.
    if routing_stats.STATS_FLAG in sys.argv[1:]:
        print(
            routing_stats.run_stats(
                Path(os.getcwd()),
                debug_log.get_log_dir(),
                as_json=routing_stats.JSON_FLAG in sys.argv[1:],
            )
        )
        return 0

    # Tag this process and seed the run id BEFORE anything else so the
    # MCP server (grandchild) inherits the same id via env and all events
    # land in one NDJSON file.
    # 다른 동작보다 먼저 이 프로세스를 태깅하고 run id를 seed — MCP 서버
    # (손자 프로세스) 가 환경 변수로 같은 id를 상속해 모든 이벤트가 한
    # NDJSON 파일에 모이도록 한다.
    debug_log.set_proc_label("wrapper")
    debug_log.get_run_id()

    project_path = os.getcwd()
    socket_path = _resolve_socket_path(project_path)

    # Export to env so the MCP server (a grandchild process spawned by
    # Claude Code) inherits and can connect back.
    # MCP 서버가 손자 프로세스로 spawn 되며 환경 변수를 상속해 래퍼로
    # 다시 connect 할 수 있도록 노출.
    os.environ[SOCKET_ENV_VAR] = socket_path

    # Claude Code is launched with exactly the arguments the user gave.
    # The wrapper used to prepend an experimental-channels development flag,
    # which forced a scary confirmation prompt on every start and shut out
    # API-key users entirely; nothing depends on channels any more.
    # Claude Code 는 사용자가 준 인자 그대로 실행된다. 예전에는 래퍼가
    # experimental channels 개발 플래그를 앞에 붙였고, 그 탓에 매 시작마다
    # 경고 확인 창이 떴으며 API key 사용자는 아예 쓸 수 없었다. 이제 channels
    # 에 의존하는 것은 없다.
    claude_args = sys.argv[1:]
    no_hooks = NO_HOOKS_FLAG in claude_args
    if no_hooks:
        claude_args = [a for a in claude_args if a != NO_HOOKS_FLAG]
    debug_log.log(
        "WRAPPER_BOOT",
        "SYSTEM",
        {
            "project_path": project_path,
            "socket_path": socket_path,
            "claude_args": claude_args,
            "no_hooks": no_hooks,
            "env": debug_log.mask_env(),
        },
    )

    # Routing-hook registration check runs before the PTY takes over the
    # terminal, while a plain input() prompt is still possible. The
    # statusline collector (R4-C1 context detection) follows the same
    # window and the same --no-hooks opt-out.
    # 라우팅 hook 등록 검사는 PTY 가 터미널을 점유하기 전 — 평범한
    # input() 프롬프트가 아직 가능한 시점 — 에 수행한다. statusline
    # 수집기 (R4-C1 컨텍스트 감지) 도 같은 시점·같은 --no-hooks 옵트아웃.
    if not no_hooks:
        ensure_hook_registered(Path(project_path))
        ensure_statusline_registered(Path(project_path))

    wrapper = SessionManagerWrapper(
        socket_path=socket_path,
        claude_args=claude_args,
        project_path=project_path,
    )
    wrapper.start()
    debug_log.log("WRAPPER_EXIT", "SYSTEM", {})
    return 0


if __name__ == "__main__":
    sys.exit(main())
