"""TTL-based cleanup — delete sessions whose last_accessed exceeds
the configured retention period.

TTL 기반 정리 — last_accessed가 보존 기간을 초과한 세션을 삭제한다.
Claude Code의 cleanupPeriodDays 설정과 동기화하여 동일한 기준을 적용한다.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

from session_manager import debug_log
from session_manager.claude_conversation import get_conversation_activity
from session_manager.storage import SessionStore

logger = logging.getLogger(__name__)

_DEFAULT_CLEANUP_PERIOD_DAYS = 30
_MIN_CLEANUP_PERIOD_DAYS = 1


def get_cleanup_period_days() -> int:
    """Read ``cleanupPeriodDays`` from ``~/.claude/settings.json``.

    ``~/.claude/settings.json`` 에서 ``cleanupPeriodDays`` 값을 읽는다.
    파일이 없거나 키가 없으면 기본값 30을 반환한다.
    """
    settings_path = Path.home() / ".claude" / "settings.json"
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        value = data.get("cleanupPeriodDays", _DEFAULT_CLEANUP_PERIOD_DAYS)
        if not isinstance(value, int) or value < _MIN_CLEANUP_PERIOD_DAYS:
            return _DEFAULT_CLEANUP_PERIOD_DAYS
        return value
    except (OSError, json.JSONDecodeError, TypeError):
        return _DEFAULT_CLEANUP_PERIOD_DAYS


def cleanup_expired_sessions(
    store: SessionStore,
    period_days: int,
    project_path: Path | None = None,
) -> list[str]:
    """Delete sessions untouched for longer than *period_days*.

    *period_days* 보다 오래 사용되지 않은 세션을 삭제한다.

    Activity is ``max(last_accessed, newest linked transcript mtime)``.
    Metadata alone is not enough: ``last_accessed`` is only written by tool
    calls that touch the session, so someone working in a single session
    for weeks keeps a stale ``last_accessed`` and their in-use session
    would be deleted here. The transcript mtime is the real usage signal.
    *project_path* is required to locate transcripts; without it the
    function degrades to the metadata-only behaviour.

    활동 시각은 ``max(last_accessed, 연결된 transcript 중 최신 mtime)``.
    메타데이터만으로는 부족하다 — ``last_accessed`` 는 세션을 건드리는 도구
    호출이 있을 때만 기록되므로, 한 세션에서 몇 주간 작업하는 사용자는
    ``last_accessed`` 가 낡은 채로 남아 사용 중인 세션이 여기서 삭제된다.
    transcript mtime 이 실제 사용 신호다. *project_path* 가 없으면 기존
    메타데이터 전용 동작으로 degrade 한다.

    The predicate is intentionally status-agnostic: archived sessions
    age out under the same TTL as active ones — ending a session removes
    it from routing, not from storage.

    술어는 의도적으로 status 무관이다 — archived 세션도 active 와 같은
    TTL 로 삭제된다. 끝냄은 라우팅에서 빼는 것이지 저장소에서 빼는 것이
    아니다.

    삭제된 세션 이름 목록을 반환한다.
    """
    now = datetime.datetime.now(datetime.UTC)
    cutoff = now - datetime.timedelta(days=period_days)
    deleted: list[str] = []
    debug_log.log(
        "CLEANUP_START",
        "SYSTEM",
        {"period_days": period_days, "cutoff": cutoff.isoformat()},
    )

    for session in store.list_sessions():
        try:
            accessed = datetime.datetime.fromisoformat(session.last_accessed)
        except (ValueError, TypeError):
            # Malformed timestamp — skip, don't delete.
            # 잘못된 타임스탬프 — 건너뛰고 삭제하지 않는다.
            logger.warning(
                "Skipping session %s — malformed last_accessed: %r",
                session.name,
                session.last_accessed,
            )
            continue

        transcript_activity = (
            get_conversation_activity(project_path, session.claude_conversation_ids)
            if project_path is not None
            else None
        )
        if transcript_activity is not None and transcript_activity > accessed:
            accessed = transcript_activity

        if accessed < cutoff:
            store.delete_session(session.session_id)
            deleted.append(session.name)
            logger.info(
                "Cleaned up expired session: %s (last activity=%s)",
                session.name,
                accessed.isoformat(),
            )
            debug_log.log(
                "CLEANUP_DELETE",
                "SYSTEM",
                {
                    "session": session.name,
                    "session_id": session.session_id,
                    "last_accessed": session.last_accessed,
                    "transcript_activity": (
                        transcript_activity.isoformat()
                        if transcript_activity is not None
                        else None
                    ),
                },
                session=session.name,
            )

    debug_log.log(
        "CLEANUP_DONE",
        "SYSTEM",
        {"deleted_count": len(deleted), "deleted": deleted},
    )
    return deleted
