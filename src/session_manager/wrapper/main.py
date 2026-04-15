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

from session_manager.wrapper.pty_wrapper import SessionManagerWrapper

SOCKET_ENV_VAR = "SESSION_MANAGER_SOCKET"


def _resolve_socket_path(project_path: str) -> str:
    # Short hash keeps the path well under the AF_UNIX 108-byte limit while
    # still giving a per-project namespace.
    # 짧은 해시로 프로젝트별 네임스페이스를 확보하면서도 AF_UNIX 의 108바이트
    # 경로 제한을 여유 있게 지킨다.
    project_hash = hashlib.md5(project_path.encode("utf-8")).hexdigest()[:12]
    return f"/tmp/session-manager-{project_hash}.sock"


def main() -> int:
    project_path = os.getcwd()
    socket_path = _resolve_socket_path(project_path)

    # Export to env so the MCP server (a grandchild process spawned by
    # Claude Code) inherits and can connect back.
    # MCP 서버가 손자 프로세스로 spawn 되며 환경 변수를 상속해 래퍼로
    # 다시 connect 할 수 있도록 노출.
    os.environ[SOCKET_ENV_VAR] = socket_path

    wrapper = SessionManagerWrapper(
        socket_path=socket_path,
        claude_args=sys.argv[1:],
        project_path=project_path,
    )
    wrapper.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
