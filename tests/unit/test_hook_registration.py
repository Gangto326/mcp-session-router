"""
Unit tests for the boot-time hook auto-registration.

Focus: the user's settings.json is never damaged, consent is honored
and remembered, and every failure degrades to a skip.

부팅 시 hook 자동 등록 단위 테스트.

초점: 사용자의 settings.json 을 절대 손상시키지 않고, 동의를 존중·기억
하며, 모든 실패가 skip 으로 완화되는지.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from session_manager.hooks import registration

FAKE_COMMAND = "/opt/bin/ccode-hook-user-prompt-submit"
FAKE_GUARD_COMMAND = "/opt/bin/ccode-hook-pre-tool-use"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def which_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        registration.shutil, "which", lambda name: f"/opt/bin/{name}"
    )


def _settings_path(project: Path) -> Path:
    return project / ".claude" / "settings.json"


def _read_settings(project: Path) -> dict:
    return json.loads(_settings_path(project).read_text(encoding="utf-8"))


def _fail_ask(_prompt: str) -> str:
    raise AssertionError("ask_user must not be called")


class TestRegister:
    def test_registers_with_consent(
        self, project: Path, which_found: None
    ) -> None:
        status = registration.ensure_hook_registered(
            project, ask_user=lambda _p: "y"
        )
        assert status == "registered"
        settings = _read_settings(project)
        # P2-e·§10 실측 형식 (docs/poc/R2-hook.md)
        assert settings["hooks"]["UserPromptSubmit"] == [
            {"hooks": [{"type": "command", "command": FAKE_COMMAND}]}
        ]
        assert settings["hooks"]["PreToolUse"] == [
            {
                "matcher": "Read|Bash",
                "hooks": [{"type": "command", "command": FAKE_GUARD_COMMAND}],
            }
        ]

    def test_partial_registration_adds_only_missing(
        self, project: Path, which_found: None
    ) -> None:
        # UserPromptSubmit 만 등록된 상태 → PreToolUse 만 추가돼야 한다
        _settings_path(project).parent.mkdir(parents=True)
        existing = {
            "hooks": {
                "UserPromptSubmit": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": f"/x/{registration.HOOK_SCRIPT_NAME}",
                            }
                        ]
                    }
                ]
            }
        }
        _settings_path(project).write_text(
            json.dumps(existing), encoding="utf-8"
        )

        status = registration.ensure_hook_registered(
            project, ask_user=lambda _p: "y"
        )
        assert status == "registered"
        settings = _read_settings(project)
        assert len(settings["hooks"]["UserPromptSubmit"]) == 1
        assert settings["hooks"]["PreToolUse"][0]["matcher"] == "Read|Bash"

    def test_preserves_existing_settings(
        self, project: Path, which_found: None
    ) -> None:
        _settings_path(project).parent.mkdir(parents=True)
        existing = {
            "permissions": {"allow": ["Bash(ls:*)"]},
            "hooks": {
                "PreToolUse": [{"hooks": [{"type": "command", "command": "x"}]}],
                "UserPromptSubmit": [
                    {"hooks": [{"type": "command", "command": "other-hook"}]}
                ],
            },
        }
        _settings_path(project).write_text(
            json.dumps(existing), encoding="utf-8"
        )

        status = registration.ensure_hook_registered(
            project, ask_user=lambda _p: "yes"
        )
        assert status == "registered"
        settings = _read_settings(project)
        assert settings["permissions"] == {"allow": ["Bash(ls:*)"]}
        commands = [
            h["command"]
            for entry in settings["hooks"]["UserPromptSubmit"]
            for h in entry["hooks"]
        ]
        assert commands == ["other-hook", FAKE_COMMAND]
        # 기존 PreToolUse 항목은 보존되고 우리 가드가 뒤에 추가된다
        guard_commands = [
            h["command"]
            for entry in settings["hooks"]["PreToolUse"]
            for h in entry["hooks"]
        ]
        assert guard_commands == ["x", FAKE_GUARD_COMMAND]

    def test_already_registered_leaves_file_untouched(
        self, project: Path, which_found: None
    ) -> None:
        _settings_path(project).parent.mkdir(parents=True)
        original = json.dumps(
            {
                "hooks": {
                    "UserPromptSubmit": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": f"/x/{registration.HOOK_SCRIPT_NAME}",
                                }
                            ]
                        }
                    ],
                    "PreToolUse": [
                        {
                            "matcher": "Read|Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": (
                                        f"/x/{registration.PRE_TOOL_USE_SCRIPT_NAME}"
                                    ),
                                }
                            ],
                        }
                    ],
                }
            }
        )
        _settings_path(project).write_text(original, encoding="utf-8")

        status = registration.ensure_hook_registered(project, ask_user=_fail_ask)
        assert status == "already_registered"
        assert _settings_path(project).read_text(encoding="utf-8") == original


class TestDecline:
    def test_decline_recorded_and_not_registered(
        self, project: Path, which_found: None
    ) -> None:
        status = registration.ensure_hook_registered(
            project, ask_user=lambda _p: "n"
        )
        assert status == "declined"
        assert not _settings_path(project).exists()
        config = json.loads(
            (project / ".session-manager" / "config.json").read_text(
                encoding="utf-8"
            )
        )
        assert config["hook_registration_declined"] is True

    def test_declined_previously_not_reasked(
        self, project: Path, which_found: None
    ) -> None:
        registration.ensure_hook_registered(project, ask_user=lambda _p: "n")
        status = registration.ensure_hook_registered(project, ask_user=_fail_ask)
        assert status == "declined_previously"

    def test_decline_preserves_existing_config_keys(
        self, project: Path, which_found: None
    ) -> None:
        config_dir = project / ".session-manager"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(
            json.dumps({"routing_mode": "off"}), encoding="utf-8"
        )
        registration.ensure_hook_registered(project, ask_user=lambda _p: "n")
        config = json.loads(
            (config_dir / "config.json").read_text(encoding="utf-8")
        )
        assert config["routing_mode"] == "off"
        assert config["hook_registration_declined"] is True

    def test_empty_answer_means_decline(
        self, project: Path, which_found: None
    ) -> None:
        assert (
            registration.ensure_hook_registered(project, ask_user=lambda _p: "")
            == "declined"
        )


class TestSkips:
    def test_broken_settings_untouched(
        self, project: Path, which_found: None
    ) -> None:
        _settings_path(project).parent.mkdir(parents=True)
        _settings_path(project).write_text("{oops", encoding="utf-8")

        status = registration.ensure_hook_registered(project, ask_user=_fail_ask)
        assert status == "broken_settings"
        assert _settings_path(project).read_text(encoding="utf-8") == "{oops"

    def test_script_not_found_skips(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(registration.shutil, "which", lambda _n: None)
        status = registration.ensure_hook_registered(project, ask_user=_fail_ask)
        assert status == "script_not_found"
        assert not _settings_path(project).exists()

    def test_non_interactive_skips(
        self, project: Path, which_found: None
    ) -> None:
        # pytest 의 stdin 은 tty 가 아니다 — ask_user 미지정 시 질문 없이 skip
        status = registration.ensure_hook_registered(project, ask_user=None)
        assert status == "non_interactive"

    def test_internal_error_returns_error(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(_path: Path) -> dict | None:
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(registration, "_load_settings", boom)
        assert (
            registration.ensure_hook_registered(project, ask_user=_fail_ask)
            == "error"
        )


class TestStatuslineRegistration:
    """R4-C1: statusline collector registration.

    R4-C1: statusline 수집기 등록.
    """

    def test_registers_with_consent(
        self, project: Path, which_found: None
    ) -> None:
        status = registration.ensure_statusline_registered(
            project, ask_user=lambda _p: "y"
        )
        assert status == "registered"
        settings = _read_settings(project)
        assert settings["statusLine"] == {
            "type": "command",
            "command": "/opt/bin/ccode-statusline",
        }

    def test_preserves_existing_settings_keys(
        self, project: Path, which_found: None
    ) -> None:
        path = _settings_path(project)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"env": {"FOO": "1"}}), encoding="utf-8")
        registration.ensure_statusline_registered(
            project, ask_user=lambda _p: "y"
        )
        settings = _read_settings(project)
        assert settings["env"] == {"FOO": "1"}
        assert "statusLine" in settings

    def test_own_registration_is_idempotent(
        self, project: Path, which_found: None
    ) -> None:
        registration.ensure_statusline_registered(
            project, ask_user=lambda _p: "y"
        )
        status = registration.ensure_statusline_registered(
            project, ask_user=_fail_ask
        )
        assert status == "already_registered"

    def test_foreign_statusline_is_never_touched(
        self, project: Path, which_found: None
    ) -> None:
        # The user's own statusline always wins — detection falls back
        # to the model mapping instead.
        # 사용자 자신의 statusline 이 항상 우선 — 감지는 모델 매핑
        # 폴백으로 동작한다.
        path = _settings_path(project)
        path.parent.mkdir(parents=True)
        original = {"statusLine": {"type": "command", "command": "/my/own.sh"}}
        path.write_text(json.dumps(original), encoding="utf-8")
        status = registration.ensure_statusline_registered(
            project, ask_user=_fail_ask
        )
        assert status == "foreign_statusline"
        assert _read_settings(project) == original

    def test_decline_is_recorded_and_remembered(
        self, project: Path, which_found: None
    ) -> None:
        status = registration.ensure_statusline_registered(
            project, ask_user=lambda _p: "n"
        )
        assert status == "declined"
        config = json.loads(
            (project / ".session-manager" / "config.json").read_text(
                encoding="utf-8"
            )
        )
        assert config["statusline_registration_declined"] is True
        status = registration.ensure_statusline_registered(
            project, ask_user=_fail_ask
        )
        assert status == "declined_previously"

    def test_statusline_decline_does_not_block_hooks(
        self, project: Path, which_found: None
    ) -> None:
        # The two consents are independent keys in config.json.
        # 두 동의는 config.json 의 독립 키다.
        registration.ensure_statusline_registered(
            project, ask_user=lambda _p: "n"
        )
        status = registration.ensure_hook_registered(
            project, ask_user=lambda _p: "y"
        )
        assert status == "registered"

    def test_broken_settings_skips(self, project: Path) -> None:
        path = _settings_path(project)
        path.parent.mkdir(parents=True)
        path.write_text("{broken", encoding="utf-8")
        status = registration.ensure_statusline_registered(
            project, ask_user=_fail_ask
        )
        assert status == "broken_settings"
        assert path.read_text(encoding="utf-8") == "{broken"

    def test_script_not_found_skips(
        self, project: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(registration.shutil, "which", lambda _n: None)
        status = registration.ensure_statusline_registered(
            project, ask_user=_fail_ask
        )
        assert status == "script_not_found"
        assert not _settings_path(project).exists()

    def test_non_interactive_skips(
        self, project: Path, which_found: None
    ) -> None:
        status = registration.ensure_statusline_registered(
            project, ask_user=None
        )
        assert status == "non_interactive"
