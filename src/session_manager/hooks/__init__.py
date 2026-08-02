"""
Claude Code hook entrypoints.

Each module in this package is a standalone script Claude Code invokes
via the project's ``.claude/settings.json`` hooks configuration. Hooks
run as short-lived processes: read one JSON payload from stdin, act,
exit. A hook failure must never block the user's conversation.

Claude Code hook 진입점 패키지.

이 패키지의 각 모듈은 프로젝트 ``.claude/settings.json`` hooks 설정을
통해 Claude Code가 실행하는 독립 스크립트다. hook은 단명 프로세스로
동작한다: stdin에서 JSON 페이로드 1건을 읽고, 처리하고, 종료한다.
hook의 실패가 사용자의 대화를 막아서는 절대 안 된다.
"""
