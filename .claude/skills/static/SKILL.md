---
name: static
description: 프로젝트 공유 static 필드(환경·컨벤션·프로젝트 맵·변수)를 보고 고친다. 사용자가 공유 정보를 등록·조회·수정·삭제·되돌리기 하고 싶을 때 사용.
---

static 필드는 세션 간 공유되는 프로젝트 정보다. 필드는 4개:
`project_context`(문자열) · `conventions`(문자열) · `project_map`(경로→설명
dict) · `variables`(이름→값 dict, **비밀 가능**). 항목마다 출처
(`source: auto|user`)·갱신 시각·직전 값이 기록된다.

비밀 취급 원칙: `variables` 값은 도구(`get_static_all`)가 애초에 형태
태그(`<str, 12자>`)로 가려서 반환한다 — 마스킹은 서버의 보장이며 네
재량이 아니다. 너는 어떤 경우에도 variables 값을 응답에 만들어 내거나,
이전 대화에서 기억하는 값을 다시 쓰지 마라. 값 확인이 필요한 사용자에게는
`.session-manager/static-field.json` 파일을 직접 열어 보라고 안내하라.

## /static <내용> (인자가 있을 때)

1. 내용을 위 4개 필드 중 하나로 분류한다 (키=값 꼴이나 접속 정보·토큰은
   variables, 경로 설명은 project_map, 규칙·스타일은 conventions, 나머지
   프로젝트 서술은 project_context).
2. `update_static` 을 **source="user"** 로 호출한다 (dict 필드는 해당
   키만 담아 보낸다 — 키별 병합이므로 나머지는 유지된다).
3. 한 줄로 확인한다. variables 는 키 이름만 (`static 갱신:
   variables.DB_HOST`), 다른 필드는 값을 보여도 된다.

## /static (인자가 없을 때)

1. `get_static_all` 을 호출해 전체를 markdown 으로 렌더링한다. 항목마다
   출처 태그와 갱신 시각을 붙인다 (예: `DB_HOST  <str, 9자>  [user,
   08-24]`). variables 에 `prev_updated_at` 이 있으면 "(직전 덮어씀:
   {시각})" 을 병기한다.
2. AskUserQuestion 으로 [항목 수정 / 항목 삭제 / 되돌리기 / 닫기] 를
   묻는다.
3. 항목 지정은 2단이다: 필드(카테고리) 선택 → 항목 선택. **질문당
   옵션은 최대 4개** — 항목이 4개를 넘으면 3개씩 보여주고 네 번째
   옵션을 "다음 페이지" 로 쓴다.
4. 동작별:
   - **수정**: 항목 선택 → "새 값을 입력해 주세요" 로 사용자 타이핑을
     받아 `update_static(source="user")`. **현재 값은 보여 주지 않는다**
     (variables 는 서버가 가려서 애초에 없고, 다른 필드도 새 값 입력에
     옛 값 표시가 필요하지 않다 — 확인이 필요하면 파일 직접 열람 안내).
     인라인 텍스트 편집 UI 는 플랫폼상 불가 — 선택 + 타이핑이 상한이다.
   - **삭제**: 필드 선택 → 항목 multiSelect 체크 → **삭제는 이 시스템
     유일의 복구 불가 연산이다** — 실행 전에 선택 항목을 나열하고
     AskUserQuestion 으로 한 번 더 확인받은 뒤, 항목마다
     `delete_static_entry(field, key)` 호출 → 결과 요약 한 줄. 삭제는
     직전 값(prev_value)까지 함께 지운다.
   - **되돌리기**: 필드 선택지에서 **variables 는 제외한다** — 비밀
     정책상 값 이력이 없어 되돌릴 수 없다 (목록의 "직전 덮어씀" 시각이
     전부이며, 필요하면 새 값 재입력을 안내). 나머지 필드는 항목 선택 →
     `revert_static_entry(field, key)`. `reverted` 면 한 줄 확인,
     `no_history` 면 "변경 이력 없음" 안내.
5. "닫기" 선택 시 아무 도구도 호출하지 않는다. 위 동작 외 다른 작업 금지.

참고: 저장 파일은 `.session-manager/static-field.json` 이며 스키마가 이
코드보다 새 버전이면 도구가 `unsupported_schema` 를 반환한다 — 그 경우
파일을 건드리지 말고 사용자에게 버전 불일치를 알려라.
