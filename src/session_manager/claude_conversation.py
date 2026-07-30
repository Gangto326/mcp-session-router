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

import datetime
import re
from collections.abc import Iterable
from pathlib import Path

from session_manager import debug_log

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


def get_conversation_activity(
    cwd: Path, conversation_ids: Iterable[str]
) -> datetime.datetime | None:
    """Return the newest transcript mtime among *conversation_ids*, or None.

    *conversation_ids* 중 가장 최근의 transcript mtime 을 반환. 없으면 None.

    This is the only trustworthy "when was this session actually used"
    signal we have. Session metadata's ``last_accessed`` is written only by
    tool calls that happen to touch the session (a switch touches the
    session being left, not the one being entered), so a session used all
    day may carry a week-old ``last_accessed``. The transcript file, by
    contrast, gets appended on every single message.

    이것이 "이 세션이 실제로 언제 쓰였나" 에 대한 유일하게 신뢰할 수 있는
    신호다. 메타데이터의 ``last_accessed`` 는 세션을 건드리는 도구 호출이
    있을 때만 기록되므로 (전환은 떠나는 세션만 touch 하고 들어가는 세션은
    하지 않는다), 하루 종일 쓴 세션의 ``last_accessed`` 가 일주일 전일 수
    있다. 반면 transcript 파일은 메시지마다 append 된다.

    Returns None when no listed transcript exists (never linked, or the
    files were removed by Claude Code's own cleanup) — callers must then
    fall back to metadata timestamps rather than assuming inactivity.

    나열된 transcript 가 하나도 없으면 (연결된 적 없거나 Claude Code 자체
    정리로 삭제됨) None — 호출자는 "비활성" 으로 단정하지 말고 메타데이터
    타임스탬프로 fallback 해야 한다.
    """
    project_dir = Path.home() / ".claude" / "projects" / encode_cwd(cwd)
    newest: float | None = None
    for conv_id in conversation_ids:
        if not conv_id:
            continue
        try:
            mtime = (project_dir / f"{conv_id}.jsonl").stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest:
            newest = mtime
    if newest is None:
        return None
    return datetime.datetime.fromtimestamp(newest, tz=datetime.UTC)


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
        debug_log.log(
            "CONV_QUERY",
            "SYSTEM",
            {
                "result": None,
                "reason": "project_dir_missing",
                "project_dir": str(project_dir),
            },
        )
        return None
    jsonls = list(project_dir.glob("*.jsonl"))
    if not jsonls:
        debug_log.log(
            "CONV_QUERY",
            "SYSTEM",
            {
                "result": None,
                "reason": "no_jsonl_files",
                "project_dir": str(project_dir),
            },
        )
        return None
    latest = max(jsonls, key=lambda p: p.stat().st_mtime)
    # Top 3 by mtime so the log shows alternatives when the wrong jsonl
    # is suspected of getting picked (helps diagnose stale-conv issues).
    # mtime 상위 3개를 함께 기록 — 잘못된 jsonl 이 골라졌다고 의심될 때
    # 대안을 로그에서 확인 (옛 conversation 잔존 문제 진단에 도움).
    top3 = sorted(jsonls, key=lambda p: p.stat().st_mtime, reverse=True)[:3]
    debug_log.log(
        "CONV_QUERY",
        "SYSTEM",
        {
            "result": latest.stem,
            "project_dir": str(project_dir),
            "candidates": len(jsonls),
            "top3": [
                {"id": p.stem, "mtime": p.stat().st_mtime} for p in top3
            ],
        },
        conv_id=latest.stem,
    )
    return latest.stem
