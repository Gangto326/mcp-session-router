"""Unit tests for Claude Code conversation-directory encoding.

Claude Code 대화 디렉토리 인코딩 단위 테스트.

Focus: the project-directory name must match what Claude Code writes
regardless of the Unicode normalisation form the filesystem returns
(measured: Claude Code writes NFC; macOS APFS preserves NFD for
Finder-created Korean folder names).

초점: 파일시스템이 돌려주는 유니코드 정규화 형식과 무관하게 Claude Code
가 쓰는 디렉토리명과 일치해야 한다 (실측: Claude Code 는 NFC, macOS APFS
는 Finder 로 만든 한글 폴더명을 NFD 로 보존).
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

from session_manager.claude_conversation import encode_cwd, get_active_conversation_id


class TestEncodeCwd:
    def test_ascii_path(self, tmp_path: Path) -> None:
        p = tmp_path / "my.project_v2"
        p.mkdir()
        expected = "-" + str(p.resolve())[1:].replace("/", "-")
        expected = expected.replace(".", "-").replace("_", "-")
        assert encode_cwd(p) == expected

    def test_nfd_and_nfc_paths_encode_identically(self, tmp_path: Path) -> None:
        # The same folder reached through an NFD spelling must produce the
        # NFC-based name Claude Code uses: "/" + 3 syllables → 4 dashes
        # (measured: ``-Users-kimgangto-Desktop----``), never 7.
        # NFD 철자로 도달한 같은 폴더도 Claude Code 가 쓰는 NFC 기반 이름이
        # 나와야 한다: "/" + 음절 3개 → 대시 4개 (실측:
        # ``-Users-kimgangto-Desktop----``), 7개가 아니라.
        name_nfc = unicodedata.normalize("NFC", "자소서")
        name_nfd = unicodedata.normalize("NFD", "자소서")
        assert len(name_nfc) == 3 and len(name_nfd) == 6
        (tmp_path / name_nfd).mkdir()
        enc_nfc = encode_cwd(tmp_path / name_nfc)
        enc_nfd = encode_cwd(tmp_path / name_nfd)
        assert enc_nfc == enc_nfd
        assert enc_nfc.endswith("----")
        assert not enc_nfc.endswith("-----")

    def test_nfd_path_finds_claude_code_directory(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # End-to-end shape of the bug: Claude Code's directory is named
        # from the NFC form; an NFD cwd must still resolve to it.
        # 버그의 실제 모양: Claude Code 디렉토리는 NFC 로 이름 붙고, NFD
        # cwd 도 그 디렉토리를 찾아야 한다.
        home = tmp_path / "home"
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        project = tmp_path / unicodedata.normalize("NFD", "자소서")
        project.mkdir()
        nfc_name = "-" + unicodedata.normalize("NFC", str(project.resolve()))[1:]
        import re

        nfc_name = re.sub(r"[^a-zA-Z0-9]", "-", nfc_name)
        conv_dir = home / ".claude" / "projects" / nfc_name
        conv_dir.mkdir(parents=True)
        (conv_dir / "abc-123.jsonl").write_text("{}", encoding="utf-8")
        assert get_active_conversation_id(project) == "abc-123"
