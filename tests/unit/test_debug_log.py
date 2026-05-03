"""
Unit tests for the debug logging infrastructure.

Verifies activation gating, run id propagation, file output format,
masking helpers, and large-payload spill behaviour.

디버그 로깅 인프라 단위 테스트.

활성화 gating, run id 상속, 파일 출력 포맷, 마스킹 헬퍼, 큰 payload
spill 동작을 검증한다.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from session_manager import debug_log


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reset process-wide state before every test.

    Each test gets a fresh tmp log dir, no inherited run id, debug off
    by default. Module-level globals (proc label, spill counter) reset
    to module defaults via reload.

    매 테스트마다 프로세스 전역 상태 리셋.

    각 테스트는 새 tmp 로그 디렉토리, 상속 run id 없음, 디버그 비활성
    상태로 시작. 모듈 전역 변수 (proc label, spill counter) 는 reload
    로 모듈 기본값으로 되돌린다.
    """
    monkeypatch.delenv("SESSION_MANAGER_DEBUG", raising=False)
    monkeypatch.delenv("SESSION_MANAGER_RUN_ID", raising=False)
    monkeypatch.delenv("SESSION_MANAGER_LOG_RAW_STDIN", raising=False)
    monkeypatch.setenv("SESSION_MANAGER_LOG_DIR", str(tmp_path))
    importlib.reload(debug_log)


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SESSION_MANAGER_DEBUG", "1")


def _read_lines() -> list[dict]:
    path = debug_log.get_log_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---- is_enabled ----------------------------------------------------------


class TestIsEnabled:
    def test_default_disabled(self) -> None:
        assert debug_log.is_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes", "on"])
    def test_truthy_values_enable(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("SESSION_MANAGER_DEBUG", value)
        assert debug_log.is_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "FALSE", "no", "off"])
    def test_falsy_values_disable(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("SESSION_MANAGER_DEBUG", value)
        assert debug_log.is_enabled() is False


# ---- get_run_id ----------------------------------------------------------


class TestGetRunId:
    def test_generates_when_missing_and_exports_to_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SESSION_MANAGER_RUN_ID", raising=False)
        rid = debug_log.get_run_id()
        assert isinstance(rid, str) and len(rid) == 16
        # Env should now carry the seeded id so child processes inherit.
        # 환경 변수에 seed된 id가 들어가야 자식 프로세스가 상속한다.
        import os

        assert os.environ["SESSION_MANAGER_RUN_ID"] == rid

    def test_inherits_existing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SESSION_MANAGER_RUN_ID", "deadbeef00112233")
        assert debug_log.get_run_id() == "deadbeef00112233"

    def test_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SESSION_MANAGER_RUN_ID", raising=False)
        rid1 = debug_log.get_run_id()
        rid2 = debug_log.get_run_id()
        assert rid1 == rid2


# ---- get_log_dir / get_log_path ------------------------------------------


class TestLogPaths:
    def test_log_dir_created(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        nested = tmp_path / "nested" / "deeper"
        monkeypatch.setenv("SESSION_MANAGER_LOG_DIR", str(nested))
        result = debug_log.get_log_dir()
        assert result == nested
        assert nested.is_dir()

    def test_log_path_uses_run_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("SESSION_MANAGER_RUN_ID", "abcdef1234567890")
        path = debug_log.get_log_path()
        assert path.parent == tmp_path
        assert path.name == "abcdef1234567890.ndjson"


# ---- mask_env ------------------------------------------------------------


class TestMaskEnv:
    def test_whitelist_kept_verbatim(self) -> None:
        env = {"PATH": "/usr/bin", "HOME": "/home/u", "PWD": "/tmp"}
        out = debug_log.mask_env(env)
        assert out == env

    def test_sensitive_pattern_masked(self) -> None:
        env = {
            "ANTHROPIC_API_KEY": "sk-ant-12345",
            "MY_AUTH_TOKEN": "token-xyz",
            "DB_PASSWORD": "p@ssw0rd",
            "CLIENT_SECRET": "shh",
            "SOME_CREDENTIAL": "value",
        }
        out = debug_log.mask_env(env)
        for key in env:
            assert key in out
            assert "<masked" in out[key]
            assert str(len(env[key])) in out[key]

    def test_unrelated_dropped(self) -> None:
        env = {"FOO_VAR": "bar", "RANDOM_ENV": "value"}
        out = debug_log.mask_env(env)
        assert out == {}

    def test_default_uses_os_environ(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PWD", "/marker-pwd")
        out = debug_log.mask_env()
        assert out.get("PWD") == "/marker-pwd"


# ---- mask_text -----------------------------------------------------------


class TestMaskText:
    def test_none(self) -> None:
        assert debug_log.mask_text(None) == {"len": 0, "preview": None}

    def test_short(self) -> None:
        out = debug_log.mask_text("hello")
        assert out == {"len": 5, "preview": "hello"}

    def test_preview_truncated_at_default(self) -> None:
        text = "x" * 500
        out = debug_log.mask_text(text)
        assert out["len"] == 500
        assert out["preview"] == "x" * 200
        assert "text_ref" not in out

    def test_custom_prefix_len(self) -> None:
        out = debug_log.mask_text("abcdefghij", prefix_len=3)
        assert out["preview"] == "abc"

    def test_spills_when_above_threshold(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _enable(monkeypatch)
        big = "Z" * (debug_log.SPILL_THRESHOLD_CHARS + 100)
        out = debug_log.mask_text(big, prefix_len=10)
        assert out["len"] == len(big)
        assert out["preview"] == "Z" * 10
        assert "text_ref" in out
        # The spill file should exist and contain the full body.
        # spill 파일이 존재하고 본문 전체를 담고 있어야 한다.
        assert (tmp_path / out["text_ref"]).read_text() == big


# ---- mask_stdin_chunk ----------------------------------------------------


class TestMaskStdinChunk:
    def test_redacted_by_default(self) -> None:
        out = debug_log.mask_stdin_chunk(b"hello world")
        assert out["len"] == 11
        assert out["head_hex"] == b"hello wo"[:8].hex()
        assert out["tail_hex"] == b"lo world"[-8:].hex()
        assert "raw" not in out
        assert len(out["sha256_prefix"]) == 16

    def test_short_chunk_no_tail(self) -> None:
        out = debug_log.mask_stdin_chunk(b"hi")
        assert out["len"] == 2
        assert out["head_hex"] == b"hi".hex()
        assert out["tail_hex"] == ""

    def test_raw_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SESSION_MANAGER_LOG_RAW_STDIN", "1")
        out = debug_log.mask_stdin_chunk(b"plain")
        assert out["raw"] == "plain"

    def test_raw_handles_invalid_utf8(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SESSION_MANAGER_LOG_RAW_STDIN", "1")
        out = debug_log.mask_stdin_chunk(b"\xff\xfe\xfd")
        # Lossy decode falls back to replacement chars rather than crashing.
        # 손실 디코드 시 replacement 문자로 fallback (예외 없이).
        assert "raw" in out


# ---- mask_dict_keys_only -------------------------------------------------


class TestMaskDictKeysOnly:
    def test_none_and_empty(self) -> None:
        assert debug_log.mask_dict_keys_only(None) == {}
        assert debug_log.mask_dict_keys_only({}) == {}

    def test_keys_kept_values_replaced_with_lengths(self) -> None:
        out = debug_log.mask_dict_keys_only({"a": "abc", "b": 1234})
        assert set(out.keys()) == {"a", "b"}
        assert out["a"] == {"len": 3}
        assert out["b"] == {"len": 4}


# ---- new_event_id --------------------------------------------------------


class TestNewEventId:
    def test_eight_hex_chars(self) -> None:
        eid = debug_log.new_event_id()
        assert len(eid) == 8
        int(eid, 16)  # parses as hex / 16진수로 파싱 가능

    def test_unique(self) -> None:
        ids = {debug_log.new_event_id() for _ in range(100)}
        assert len(ids) == 100


# ---- set_proc_label ------------------------------------------------------


class TestSetProcLabel:
    def test_label_appears_in_log_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable(monkeypatch)
        debug_log.set_proc_label("mcp")
        debug_log.log("MCP_TOOL_CALL", "MCP_TOOL")
        records = _read_lines()
        assert records and records[-1]["proc"] == "mcp"


# ---- log -----------------------------------------------------------------


class TestLog:
    def test_disabled_is_noop(self, tmp_path: Path) -> None:
        # Without SESSION_MANAGER_DEBUG the log file must not be created.
        # SESSION_MANAGER_DEBUG 없으면 로그 파일이 생성되어선 안 된다.
        debug_log.log("USER_KEY", "USER", {"len": 1})
        assert not any(tmp_path.iterdir())

    def test_enabled_writes_one_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable(monkeypatch)
        debug_log.log("USER_KEY", "USER", {"len": 5, "preview": "hello"})
        records = _read_lines()
        assert len(records) == 1
        rec = records[0]
        assert rec["category"] == "USER_KEY"
        assert rec["origin"] == "USER"
        assert rec["payload"] == {"len": 5, "preview": "hello"}
        assert "ts" in rec and rec["ts"].endswith("Z")
        assert isinstance(rec["mono_ns"], int)
        assert rec["pid"] > 0
        assert len(rec["run_id"]) == 16

    def test_optional_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable(monkeypatch)
        debug_log.log(
            "MCP_TOOL_CALL",
            "MCP_TOOL",
            {"tool": "check_session"},
            conv_id="conv-abc",
            session="rag-explainer",
        )
        rec = _read_lines()[-1]
        assert rec["conv_id"] == "conv-abc"
        assert rec["session"] == "rag-explainer"

    def test_korean_payload_not_ascii_escaped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable(monkeypatch)
        debug_log.log("WRAPPER_INJECT", "WRAPPER", {"text": "한국어 본문"})
        raw = debug_log.get_log_path().read_text()
        assert "한국어 본문" in raw
        assert "\\u" not in raw

    def test_serialisation_failure_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable(monkeypatch)

        # Object whose default str() is fine but which raises inside json
        # via a custom __repr__/__class__ is hard to craft; use a set,
        # which json.dumps cannot serialise. The default=str fallback in
        # log() should rescue it — verify NDJSON stays parseable.
        #
        # set은 json.dumps가 직렬화 못 하지만 default=str 로 구제됨.
        # NDJSON 라인이 여전히 파싱 가능한지 검증.
        debug_log.log("MISC", "SYSTEM", {"value": {1, 2, 3}})
        records = _read_lines()
        assert len(records) == 1
        # Either the original event (rescued by default=str) or the
        # LOG_ERROR fallback shape — both are valid recoveries.
        # 원본 이벤트 (default=str로 구제) 든 LOG_ERROR fallback 이든 둘 다
        # 유효한 복구 결과.
        assert records[0]["category"] in ("MISC", "LOG_ERROR")

    def test_appends_multiple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable(monkeypatch)
        for i in range(5):
            debug_log.log("USER_KEY", "USER", {"i": i})
        assert len(_read_lines()) == 5

    def test_run_id_consistent_across_records(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable(monkeypatch)
        debug_log.log("A", "SYSTEM")
        debug_log.log("B", "SYSTEM")
        records = _read_lines()
        assert records[0]["run_id"] == records[1]["run_id"]


# ---- spill ---------------------------------------------------------------


class TestSpill:
    def test_writes_file_and_returns_basename(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _enable(monkeypatch)
        name = debug_log.spill("body content")
        path = tmp_path / name
        assert path.exists()
        assert path.read_text() == "body content"

    def test_sequential_filenames(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _enable(monkeypatch)
        names = [debug_log.spill(f"body-{i}") for i in range(3)]
        # Each spill uses a fresh sequence — names must be distinct.
        # spill 마다 새 시퀀스 — 이름이 모두 달라야 한다.
        assert len(set(names)) == 3
