"""
Hook auto-registration at ccode boot.

The routing pipeline only fires if the UserPromptSubmit hook is listed
in the project's ``.claude/settings.json``. This module checks that at
ccode start and — with the user's consent — registers it in the
measured format (docs/poc/R2-hook.md §5). Principles:

- Never modify a settings.json we cannot parse (the user's file is
  sacred; a broken file gets a warning, not a rewrite).
- Register an absolute command path — a bare name would depend on the
  PATH of whatever process runs the hook later.
- A decline is recorded in ``.session-manager/config.json`` so the
  user is not re-asked on every boot; ``--no-hooks`` skips everything.

ccode 부팅 시 hook 자동 등록.

라우팅 파이프라인은 UserPromptSubmit hook 이 프로젝트
``.claude/settings.json`` 에 등록되어야만 발동한다. 이 모듈은 ccode
시작 시 그것을 검사하고 — 사용자 동의 하에 — 실측 형식
(docs/poc/R2-hook.md §5)으로 등록한다. 원칙:

- 파싱할 수 없는 settings.json 은 절대 수정하지 않는다 (사용자의
  파일은 불가침 — 깨진 파일에는 경고만, 재작성은 없다).
- 명령은 절대 경로로 등록한다 — 이름만 쓰면 나중에 hook 을 실행하는
  프로세스의 PATH 에 의존하게 된다.
- 거절은 ``.session-manager/config.json`` 에 기록해 매 부팅 재질문을
  막는다. ``--no-hooks`` 는 전체를 건너뛴다.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from session_manager import debug_log
from session_manager.storage.file_store import (
    _CONFIG_FILENAME,
    _SESSION_MANAGER_DIRNAME,
    _atomic_write_text,
)

# Console-script name registered in pyproject [project.scripts].
# pyproject [project.scripts] 에 등록된 콘솔 스크립트 이름.
HOOK_SCRIPT_NAME = "ccode-hook-user-prompt-submit"
PRE_TOOL_USE_SCRIPT_NAME = "ccode-hook-pre-tool-use"

# All hooks ccode manages: (event name, matcher or None, script name).
# One consent covers the set; a partially-registered project only gets
# the missing entries appended.
# ccode 가 관리하는 hook 전체: (이벤트명, matcher 또는 None, 스크립트명).
# 동의 한 번이 세트 전체를 커버하고, 일부만 등록된 프로젝트에는 빠진
# 항목만 추가된다.
MANAGED_HOOKS: tuple[tuple[str, str | None, str], ...] = (
    ("UserPromptSubmit", None, HOOK_SCRIPT_NAME),
    ("PreToolUse", "Read|Bash", PRE_TOOL_USE_SCRIPT_NAME),
)

_SETTINGS_RELPATH = Path(".claude") / "settings.json"
_DECLINED_KEY = "hook_registration_declined"

_CONSENT_PROMPT = (
    "session-manager: 프롬프트 라우팅과 transcript 가드를 위해 hook 을 "
    ".claude/settings.json 에 등록할까요? [y/N] "
)


def _log(result: str, **extra: Any) -> None:
    debug_log.log("HOOK_REGISTRATION", "WRAPPER", {"result": result, **extra})


def _load_settings(path: Path) -> dict[str, Any] | None:
    """Return parsed settings, {} if missing, None if unparsable.

    settings 를 파싱해 반환. 파일 부재는 {}, 파싱 불가는 None.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _event_has_script(
    settings: dict[str, Any], event: str, script_name: str
) -> bool:
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return False
    entries = hooks.get(event)
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for hook in entry.get("hooks", []) or []:
            command = hook.get("command") if isinstance(hook, dict) else None
            if isinstance(command, str) and script_name in command:
                return True
    return False


def _missing_hooks(
    settings: dict[str, Any],
) -> list[tuple[str, str | None, str]]:
    """Return the managed hooks not yet present in settings.

    settings 에 아직 없는 관리 대상 hook 목록을 반환한다.
    """
    return [
        spec
        for spec in MANAGED_HOOKS
        if not _event_has_script(settings, spec[0], spec[2])
    ]


def _config_path(project_path: Path) -> Path:
    return project_path / _SESSION_MANAGER_DIRNAME / _CONFIG_FILENAME


def _was_declined(project_path: Path) -> bool:
    try:
        data = json.loads(_config_path(project_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(data, dict) and data.get(_DECLINED_KEY) is True


def _record_declined(project_path: Path) -> None:
    """
    Raw read-modify-write that preserves every key the user may have
    hand-written (routing_mode etc.) — the Config model is deliberately
    not used here, as it would drop unknown keys.

    사용자가 손으로 써 둔 키 (routing_mode 등) 를 전부 보존하는 raw
    read-modify-write. Config 모델은 의도적으로 쓰지 않는다 — 모델
    경유 저장은 모르는 키를 유실시킨다.
    """
    path = _config_path(project_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError):
        # Unparsable config: don't compound the damage by rewriting it.
        # 파싱 불가 config 는 재작성으로 손상을 키우지 않는다.
        return
    data[_DECLINED_KEY] = True
    try:
        _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))
    except OSError:
        pass


def _register(
    settings: dict[str, Any],
    missing: list[tuple[str, str | None, str]],
    commands: dict[str, str],
    path: Path,
) -> None:
    """Append the missing hook entries (measured format, PoC §5/§10) and write.

    빠진 hook 항목들을 실측 형식 (PoC §5/§10) 으로 추가하고 저장한다.
    기존 항목은 전부 보존된다.
    """
    hooks = settings.setdefault("hooks", {})
    for event, matcher, script_name in missing:
        entries = hooks.setdefault(event, [])
        entry: dict[str, Any] = {
            "hooks": [{"type": "command", "command": commands[script_name]}]
        }
        if matcher is not None:
            entry = {"matcher": matcher, **entry}
        entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, json.dumps(settings, ensure_ascii=False, indent=2))


def ensure_hook_registered(
    project_path: Path,
    ask_user: Callable[[str], str] | None = None,
) -> str:
    """
    Check and (with consent) register the routing hook. Returns a status
    string for logging/tests; never raises.

    라우팅 hook 등록을 검사하고 (동의 시) 등록한다. 로깅·테스트용 상태
    문자열을 반환하며 예외를 던지지 않는다.
    """
    try:
        settings_path = project_path / _SETTINGS_RELPATH
        settings = _load_settings(settings_path)
        if settings is None:
            print(
                "session-manager: .claude/settings.json 을 파싱할 수 없어 "
                "라우팅 hook 등록을 건너뜁니다 (파일은 수정하지 않았습니다).",
                file=sys.stderr,
            )
            _log("broken_settings")
            return "broken_settings"
        missing = _missing_hooks(settings)
        if not missing:
            _log("already_registered")
            return "already_registered"
        if _was_declined(project_path):
            _log("declined_previously")
            return "declined_previously"

        # All managed scripts install together — one unresolvable script
        # signals a broken install, so skip the whole set.
        # 관리 스크립트들은 함께 설치된다 — 하나라도 해석 불가면 설치가
        # 깨진 것이므로 세트 전체를 건너뛴다.
        commands: dict[str, str] = {}
        for _event, _matcher, script_name in missing:
            command = shutil.which(script_name)
            if command is None:
                print(
                    f"session-manager: '{script_name}' 스크립트를 PATH 에서 "
                    "찾지 못해 hook 등록을 건너뜁니다.",
                    file=sys.stderr,
                )
                _log("script_not_found", script=script_name)
                return "script_not_found"
            commands[script_name] = command

        if ask_user is None:
            if not sys.stdin.isatty():
                _log("non_interactive")
                return "non_interactive"
            ask_user = input

        answer = ask_user(_CONSENT_PROMPT).strip().lower()
        if answer not in ("y", "yes"):
            _record_declined(project_path)
            print(
                "session-manager: 등록하지 않았습니다. 나중에 등록하려면 "
                ".session-manager/config.json 의 "
                f"{_DECLINED_KEY} 를 지우고 ccode 를 재시작하세요."
            )
            _log("declined")
            return "declined"

        _register(settings, missing, commands, settings_path)
        print("session-manager: hook 을 등록했습니다.")
        _log("registered", events=[m[0] for m in missing])
        return "registered"
    except Exception as exc:
        # Registration is a convenience — a bug here must never stop ccode.
        # 등록은 편의 기능 — 여기서의 버그가 ccode 를 멈춰선 안 된다.
        _log("error", error=str(exc))
        return "error"
