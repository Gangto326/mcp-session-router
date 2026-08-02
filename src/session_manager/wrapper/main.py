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
from session_manager.hooks.registration import ensure_hook_registered
from session_manager.wrapper.pty_wrapper import SessionManagerWrapper

SOCKET_ENV_VAR = "SESSION_MANAGER_SOCKET"

# ccode-only flag: skip the routing-hook registration check. Stripped
# from the args passed on to Claude Code.
# ccode 전용 플래그 — 라우팅 hook 등록 검사를 건너뛴다. Claude Code 에
# 넘기는 인자에서는 제거된다.
NO_HOOKS_FLAG = "--no-hooks"


def _resolve_socket_path(project_path: str) -> str:
    # Short hash keeps the path well under the AF_UNIX 108-byte limit while
    # still giving a per-project namespace.
    # 짧은 해시로 프로젝트별 네임스페이스를 확보하면서도 AF_UNIX 의 108바이트
    # 경로 제한을 여유 있게 지킨다.
    project_hash = hashlib.md5(project_path.encode("utf-8")).hexdigest()[:12]
    return f"/tmp/session-manager-{project_hash}.sock"


def main() -> int:
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
    # terminal, while a plain input() prompt is still possible.
    # 라우팅 hook 등록 검사는 PTY 가 터미널을 점유하기 전 — 평범한
    # input() 프롬프트가 아직 가능한 시점 — 에 수행한다.
    if not no_hooks:
        ensure_hook_registered(Path(project_path))

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
