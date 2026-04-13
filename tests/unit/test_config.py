"""Tests for Config model."""

from __future__ import annotations

import pytest

from session_manager.models import Config
from session_manager.models.config import DEFAULT_CLEANUP_PERIOD_DAYS


class TestConfigDefaults:
    def test_default_cleanup_period_is_thirty_days(self) -> None:
        assert DEFAULT_CLEANUP_PERIOD_DAYS == 30

    def test_config_uses_default_cleanup_period(self) -> None:
        config = Config(socket_path="/tmp/s.sock")
        assert config.cleanup_period_days == 30

    def test_cleanup_period_can_be_overridden(self) -> None:
        config = Config(socket_path="/tmp/s.sock", cleanup_period_days=7)
        assert config.cleanup_period_days == 7


class TestConfigRoundtrip:
    def test_roundtrip_preserves_fields(self) -> None:
        config = Config(socket_path="/tmp/session-manager-abc.sock", cleanup_period_days=14)
        restored = Config.from_dict(config.to_dict())
        assert restored == config

    def test_from_dict_missing_cleanup_period_uses_default(self) -> None:
        restored = Config.from_dict({"socket_path": "/tmp/s.sock"})
        assert restored.cleanup_period_days == 30

    def test_from_dict_missing_socket_path_raises(self) -> None:
        with pytest.raises(KeyError):
            Config.from_dict({"cleanup_period_days": 10})
