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
) -> list[str]:
    """Delete sessions whose ``last_accessed`` is older than *period_days*.

    ``last_accessed`` 가 *period_days* 보다 오래된 세션을 삭제한다.
    ACTIVE 상태인 세션만 만료 삭제하며, 이미 ARCHIVED/EXPIRED인 세션도
    동일 기준으로 삭제한다.  삭제된 세션 이름 목록을 반환한다.
    """
    now = datetime.datetime.now(datetime.UTC)
    cutoff = now - datetime.timedelta(days=period_days)
    deleted: list[str] = []

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

        if accessed < cutoff:
            store.delete_session(session.session_id)
            deleted.append(session.name)
            logger.info(
                "Cleaned up expired session: %s (last_accessed=%s)",
                session.name,
                session.last_accessed,
            )

    return deleted
