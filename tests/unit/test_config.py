"""Tests for Config model."""

from __future__ import annotations

import pytest

from session_manager.models import Config
from session_manager.models.config import (
    DEFAULT_AUTO_ERROR_TOLERANCE,
    DEFAULT_CLEANUP_PERIOD_DAYS,
    DEFAULT_ROUTING_MODE,
    ROUTING_MODES,
)


class TestConfigDefaults:
    def test_default_cleanup_period_is_thirty_days(self) -> None:
        assert DEFAULT_CLEANUP_PERIOD_DAYS == 30

    def test_config_uses_default_cleanup_period(self) -> None:
        config = Config(socket_path="/tmp/s.sock")
        assert config.cleanup_period_days == 30

    def test_cleanup_period_can_be_overridden(self) -> None:
        config = Config(socket_path="/tmp/s.sock", cleanup_period_days=7)
        assert config.cleanup_period_days == 7

    def test_default_routing_mode_is_confirm(self) -> None:
        # Plan §1.4 — 기본은 제안만 하는 confirm
        assert DEFAULT_ROUTING_MODE == "confirm"
        assert DEFAULT_ROUTING_MODE in ROUTING_MODES
        assert Config(socket_path="/tmp/s.sock").routing_mode == "confirm"

    def test_hook_default_shares_config_default(self) -> None:
        # hook 이 raw 읽기에서 쓰는 기본값은 Config 모델과 단일 출처다
        from session_manager.hooks import user_prompt_submit

        assert user_prompt_submit.DEFAULT_ROUTING_MODE is DEFAULT_ROUTING_MODE


class TestConfigRoundtrip:
    def test_roundtrip_preserves_fields(self) -> None:
        config = Config(
            socket_path="/tmp/session-manager-abc.sock",
            cleanup_period_days=14,
            routing_mode="off",
        )
        restored = Config.from_dict(config.to_dict())
        assert restored == config

    def test_from_dict_missing_routing_mode_uses_default(self) -> None:
        restored = Config.from_dict({"socket_path": "/tmp/s.sock"})
        assert restored.routing_mode == "confirm"

    def test_from_dict_missing_cleanup_period_uses_default(self) -> None:
        restored = Config.from_dict({"socket_path": "/tmp/s.sock"})
        assert restored.cleanup_period_days == 30

    def test_from_dict_missing_socket_path_raises(self) -> None:
        with pytest.raises(KeyError):
            Config.from_dict({"cleanup_period_days": 10})


class TestAutoErrorTolerance:
    """R3-C4 policy parameter: auto_error_tolerance.

    R3-C4 정책 파라미터 — auto_error_tolerance.
    """

    def test_default_is_five_percent(self) -> None:
        assert DEFAULT_AUTO_ERROR_TOLERANCE == 0.05
        assert Config(socket_path="/s").auto_error_tolerance == 0.05

    def test_roundtrip_preserves_override(self) -> None:
        config = Config(socket_path="/s", auto_error_tolerance=0.1)
        assert Config.from_dict(config.to_dict()).auto_error_tolerance == 0.1

    def test_from_dict_missing_key_uses_default(self) -> None:
        config = Config.from_dict({"socket_path": "/s"})
        assert config.auto_error_tolerance == DEFAULT_AUTO_ERROR_TOLERANCE
