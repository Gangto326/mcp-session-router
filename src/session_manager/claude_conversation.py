"""Helper for resolving the currently active Claude Code conversation id.

활성 Claude Code conversation id 추출 헬퍼.

Claude Code persists every conversation as
``~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`` where the filename
itself is the conversation id and the directory is the cwd with all
non-alphanumeric characters replaced by ``-`` (per the official Sessions
docs at https://code.claude.com/docs/en/agent-sdk/sessions).

Each user/assistant message appends a line to the active conversation's
file, refreshing its mtime. Therefore the jsonl file with the newest
mtime under the project directory points at the conversation that
received the most recent message — i.e. the active one.

Claude Code는 conversation을 ``~/.claude/projects/<encoded-cwd>/<session-id>.jsonl``
형태로 영속화한다. 파일명이 곧 conversation id이고 디렉토리는 cwd의 비-알파넘
문자를 ``-``로 치환한 것 (공식 Sessions docs 기준).

메시지가 추가될 때마다 해당 conversation의 jsonl 파일에 한 줄 append되어
mtime이 갱신되므로, 프로젝트 디렉토리 안에서 mtime이 가장 최근인 파일이
곧 가장 최근에 메시지를 받은 conversation = 활성 conversation.
"""

from __future__ import annotations

import re
from pathlib import Path

# Claude Code's project-directory naming rule: every non-alphanumeric
# character of the absolute cwd is replaced by '-'.
# Claude Code 프로젝트 디렉토리 명명 규칙 — 절대 cwd의 비-알파넘 문자를
# 모두 '-'로 치환.
_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]")


def encode_cwd(cwd: Path) -> str:
    """Encode a cwd path to Claude Code's project-directory naming.

    cwd 경로를 Claude Code 프로젝트 디렉토리 명명 규칙으로 인코딩.
    """
    return _NON_ALNUM_RE.sub("-", str(cwd.resolve()))


def get_active_conversation_id(cwd: Path) -> str | None:
    """Return the active Claude Code conversation id for *cwd*, or None.

    *cwd*에 해당하는 활성 Claude Code conversation id를 반환. 없으면 None.

    "Active" is defined as the conversation whose jsonl file has the most
    recent mtime — i.e. the one that just received a message. Returns
    None when the project directory does not exist (fresh project) or
    contains no jsonl files yet.

    "활성"의 정의 — jsonl 파일 mtime이 가장 최근인 conversation. 즉 방금
    메시지를 받은 conversation. 프로젝트 디렉토리가 없거나 jsonl 파일이
    아직 하나도 없으면 None 반환.
    """
    project_dir = Path.home() / ".claude" / "projects" / encode_cwd(cwd)
    if not project_dir.is_dir():
        return None
    jsonls = list(project_dir.glob("*.jsonl"))
    if not jsonls:
        return None
    latest = max(jsonls, key=lambda p: p.stat().st_mtime)
    return latest.stem
