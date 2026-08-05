"""Unit tests for command_matcher.
command_matcher 단위 테스트.
"""

from session_manager.wrapper.command_matcher import (
    InterceptedCommand,
    match_back_command,
    match_intercept_command,
)


class TestPositiveMatches:
    """Inputs that SHOULD trigger interception.
    가로채기를 trigger 해야 하는 입력.
    """

    def test_resume_with_arg(self):
        assert match_intercept_command("/resume foo") == InterceptedCommand(
            "resume", "foo"
        )

    def test_resume_with_multi_word_arg(self):
        assert match_intercept_command("/resume foo bar baz") == InterceptedCommand(
            "resume", "foo bar baz"
        )

    def test_resume_with_extra_whitespace(self):
        """Multiple spaces between command and arg, plus trailing whitespace.
        명령과 인자 사이 다중 공백, trailing 공백.
        """
        assert match_intercept_command("/resume   foo  ") == InterceptedCommand(
            "resume", "foo"
        )

    def test_exit_no_arg(self):
        assert match_intercept_command("/exit") == InterceptedCommand("exit", "")

    def test_exit_with_trailing_whitespace(self):
        assert match_intercept_command("/exit   ") == InterceptedCommand("exit", "")

    def test_rename_with_arg(self):
        assert match_intercept_command("/rename new-name") == InterceptedCommand(
            "rename", "new-name"
        )

    def test_new_no_arg(self):
        assert match_intercept_command("/new") == InterceptedCommand("new", "")

    def test_resume_strips_placeholder(self):
        """Ink placeholder hint after argument is stripped.
        인자 뒤의 Ink placeholder hint 제거.
        """
        assert match_intercept_command(
            "/resume foo  [conversation id or search term]"
        ) == InterceptedCommand("resume", "foo")

    def test_bare_resume_intercepted(self):
        """Bare /resume (no argument) is still intercepted so the leaving
        session's summary gets force-updated before the picker opens.
        The intercept's purpose is summary-update for routing accuracy,
        not archive intent — letting bare /resume slip past would leave
        the leaving session's summary stale.
        인자 없는 /resume 도 가로채기 — picker 가 등장하기 전 떠나는 세션의
        summary 를 강제 갱신해 라우팅 정확도를 보호한다. 가로채기의 목적은
        archive 의도가 아니라 summary 갱신이고, 빈 /resume 을 흘려보내면
        떠나는 세션 summary 가 stale 로 남아 라우팅이 부정확해진다.
        """
        assert match_intercept_command("/resume") == InterceptedCommand(
            "resume", ""
        )
        assert match_intercept_command("/resume   ") == InterceptedCommand(
            "resume", ""
        )

    def test_resume_only_placeholder(self):
        """Empty argument with only Ink placeholder visible — still bare
        /resume, still intercepted.
        Ink 가 placeholder 만 그린 빈 입력란도 본질은 인자 없는 /resume 이므로
        가로채기 발동.
        """
        assert match_intercept_command(
            "/resume  [conversation id or search term]"
        ) == InterceptedCommand("resume", "")


class TestNegativeMatches:
    """Inputs that must NOT trigger interception.
    가로채기를 trigger 하면 안 되는 입력.
    """

    def test_none_input(self):
        assert match_intercept_command(None) is None

    def test_empty_string(self):
        assert match_intercept_command("") is None

    def test_whitespace_only(self):
        assert match_intercept_command("   ") is None

    def test_help_not_intercepted(self):
        """/help is an information command — not in whitelist.
        /help는 정보 명령 — 화이트리스트 외.
        """
        assert match_intercept_command("/help") is None

    def test_cost_not_intercepted(self):
        assert match_intercept_command("/cost") is None

    def test_model_not_intercepted(self):
        assert match_intercept_command("/model sonnet") is None

    def test_clear_not_intercepted(self):
        """/clear keeps session ID — no session_end needed.
        /clear는 세션 ID 유지 — session_end 불필요.
        """
        assert match_intercept_command("/clear") is None

    def test_uppercase_rejected(self):
        """Strict case match — /RESUME does not match /resume.
        엄격 대소문자 매칭 — /RESUME은 /resume과 다름.
        """
        assert match_intercept_command("/RESUME foo") is None

    def test_partial_command_rejected(self):
        """`/res` is not a prefix-match for `/resume`.
        `/res`는 `/resume`의 prefix-match 대상이 아님.
        """
        assert match_intercept_command("/res") is None

    def test_no_slash_prefix(self):
        assert match_intercept_command("resume foo") is None

    def test_command_in_middle_of_text(self):
        """Command appearing in the middle of normal text.
        일반 텍스트 중간에 명령이 있는 경우.
        """
        assert match_intercept_command("hello /resume foo") is None

    def test_path_like_rejected(self):
        """`/path/to/file` looks like a path — not a command.
        `/path/to/file`은 path 형태 — 명령 아님.
        """
        assert match_intercept_command("/path/to/file") is None

    def test_korean_text(self):
        """일반 한글 텍스트는 명령 아님."""
        assert match_intercept_command("안녕하세요") is None

    def test_unknown_slash_command(self):
        """`/foo` is not in whitelist.
        `/foo`는 화이트리스트 외.
        """
        assert match_intercept_command("/foo bar") is None

    def test_command_with_immediate_text(self):
        """`/resumefoo` (no space) is not `/resume foo`.
        `/resumefoo` (공백 없음)은 `/resume foo`와 다름.
        """
        assert match_intercept_command("/resumefoo") is None


class TestEdgeCases:
    """Boundary cases that need explicit verification.
    명시적 검증이 필요한 경계 케이스.
    """

    def test_resume_with_quoted_arg(self):
        """Quoted argument is preserved as-is.
        따옴표 인자 그대로 보존.
        """
        assert match_intercept_command(
            '/resume "session with spaces"'
        ) == InterceptedCommand("resume", '"session with spaces"')

    def test_rename_with_special_chars(self):
        """Hyphens, underscores, dots in session names.
        세션 이름에 하이픈/언더스코어/점.
        """
        assert match_intercept_command(
            "/rename my_new-session.v2"
        ) == InterceptedCommand("rename", "my_new-session.v2")

    def test_argument_containing_brackets_loses_placeholder_part(self):
        """KNOWN LIMITATION: trailing `[...]` in user arg is stripped.
        알려진 한계 — 사용자 인자 끝에 `[...]`가 있으면 잘림.
        """
        result = match_intercept_command("/rename foo [bar]")
        assert result == InterceptedCommand("rename", "foo")


class TestMatchBackCommand:
    """/back matching (R3-C3) — wrapper-native, argument-free.
    /back 매칭 (R3-C3) — 래퍼 자체 명령, 인자 없음.
    """

    def test_bare_back(self):
        assert match_back_command("/back") is True

    def test_trailing_whitespace(self):
        assert match_back_command("/back   ") is True

    def test_back_with_argument_not_matched(self):
        # "/back x" is not an undo request — falls through to the TUI.
        # "/back x" 는 undo 요청이 아니다 — TUI 로 흘러간다.
        assert match_back_command("/back x") is False

    def test_backup_not_matched(self):
        assert match_back_command("/backup") is False

    def test_not_intercept_whitelisted(self):
        # /back must not enter the observe-and-forward path.
        # /back 은 관찰 후 통과 경로에 들어가면 안 된다.
        assert match_intercept_command("/back") is None

    def test_none_and_empty(self):
        assert match_back_command(None) is False
        assert match_back_command("") is False
        assert match_back_command("   ") is False
