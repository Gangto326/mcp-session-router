"""Static Field model (project-wide shared context + user environment).

Static Field 모델 (프로젝트 전역 공유 컨텍스트 + 사용자 환경).

Since R5-C4 every entry carries provenance: ``{value, source, updated_at,
prev_value}``. ``source`` says who wrote it (``auto`` = the LLM decided
on its own, ``user`` = the user explicitly asked), ``updated_at`` when,
and ``prev_value`` what it replaced — the display and revert base for
the ``/static`` skill (R5-C5). String fields (project_context,
conventions) are one entry each; dict fields (project_map, variables)
are one entry per key.

R5-C4 부터 모든 항목이 출처를 갖는다: ``{value, source, updated_at,
prev_value}``. ``source`` 는 누가 썼는지 (``auto`` = LLM 자체 판단,
``user`` = 사용자의 명시 지시), ``updated_at`` 은 언제, ``prev_value``
는 직전 값 — ``/static`` 스킬 (R5-C5) 의 표시·되돌리기 기반이다. 문자열
필드 (project_context, conventions) 는 필드 전체가 1항목, dict 필드
(project_map, variables) 는 키마다 1항목이다.

File format / 파일 형식: the file carries ``"schema": 2``. A file
without that marker is the pre-R5-C4 flat format and is migrated on
load — each flat value becomes an entry with ``source="auto"`` (every
writer before R5-C4 was an LLM ``update_static`` call), ``updated_at``
= the file-level timestamp, ``prev_value=None``. The marker makes the
format decision deterministic — guessing from value shapes would break
on variables whose legacy value happens to be a dict with a "value" key.

파일에는 ``"schema": 2`` 마커가 있다. 마커 없는 파일은 R5-C4 이전의
평면 형식이며 로드 시 마이그레이션된다 — 평면 값마다 ``source="auto"``
(R5-C4 이전의 쓰기는 전부 LLM 의 ``update_static`` 호출이었다),
``updated_at`` = 파일 타임스탬프, ``prev_value=None`` 인 항목이 된다.
마커 덕에 형식 판별이 결정적이다 — 값 모양으로 추측하면 우연히
{"value": ...} 꼴 dict 를 가진 구식 variables 에서 깨진다.

Unknown schema / 미지 스키마 (R5-C4 검증에서 추가): a marker that is
present but not this version (a future 3, a corrupted value) is NOT
treated as legacy — re-wrapping a newer format would silently destroy
its provenance and freeze the damage on the next save. ``from_dict``
raises :class:`UnsupportedStaticSchemaError` instead and the caller
must leave the file untouched. Inside a schema-2 file, a value that is
not an entry dict (hand-edited flat value) is preserved by wrapping it
as a migrated entry — never silently replaced with an empty one.

존재하지만 이 버전이 아닌 마커 (미래의 3, 오염된 값) 는 구식으로
취급하지 **않는다** — 더 새로운 형식을 재포장하면 출처가 조용히
파괴되고 다음 저장에서 손상이 굳는다. ``from_dict`` 는
:class:`UnsupportedStaticSchemaError` 를 던지고 호출자는 파일을 건드리지
않아야 한다. schema 2 파일 안의 entry dict 아닌 값 (손편집 평면값) 은
이행 항목으로 감싸 보존한다 — 빈 항목으로 조용히 교체하지 않는다.

Secrets / 비밀 (R5-C4 검증에서 추가): ``variables`` may hold secrets
(update_static already masks them in logs), so their entries record
``prev_updated_at`` but NOT ``prev_value`` — before R5-C4 an overwrite
scrubbed the old secret from the file, and keeping it would re-expose
it to every later conversation through the boot/rollover file-read
instructions. Non-secret fields keep full ``prev_value`` for revert.

``variables`` 는 비밀을 담을 수 있으므로 (update_static 이 로그에서
이미 마스킹한다) 그 항목은 ``prev_updated_at`` 만 기록하고
``prev_value`` 는 남기지 않는다 — R5-C4 이전에는 덮어쓰기가 곧 옛
비밀의 파기였고, 보존하면 부팅·롤오버의 파일 읽기 지시를 타고 이후
모든 대화에 재노출된다. 비밀 아닌 필드는 되돌리기용으로 ``prev_value``
를 온전히 보존한다.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

SOURCE_AUTO = "auto"
SOURCE_USER = "user"
VALID_SOURCES = (SOURCE_AUTO, SOURCE_USER)

# Field names addressable by the /static entry tools (R5-C5).
# /static 항목 도구 (R5-C5) 가 지칭할 수 있는 필드 이름들.
TEXT_FIELDS = ("project_context", "conventions")
DICT_FIELDS = ("project_map", "variables")
STATIC_FIELDS = TEXT_FIELDS + DICT_FIELDS

# revert_entry outcomes / revert_entry 결과.
REVERT_OK = "reverted"
REVERT_NO_HISTORY = "no_history"
REVERT_SECRET = "secret_no_history"
REVERT_NOT_FOUND = "not_found"

_SCHEMA_VERSION = 2


class UnsupportedStaticSchemaError(ValueError):
    """The static-field file carries a schema this code does not know.

    static-field 파일의 schema 를 이 코드가 모른다. 호출자는 파일을
    덮어쓰지 말아야 한다 (모듈 docstring 참조).
    """

    def __init__(self, schema: Any) -> None:
        super().__init__(f"unsupported static-field schema: {schema!r}")
        self.schema = schema


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _shape_tag(value: Any) -> str:
    """Type-and-size placeholder for a masked secret value.

    마스킹된 비밀값의 형태 태그 — 문자열은 길이까지, 그 외는 타입만.
    """
    if isinstance(value, str):
        return f"<str, {len(value)}자>"
    return f"<{type(value).__name__}>"


@dataclass
class StaticEntry:
    """One provenance-tracked value.

    출처가 붙은 값 1개.
    """

    value: Any = ""
    source: str = SOURCE_AUTO
    updated_at: str = ""
    prev_value: Any = None
    prev_updated_at: str = ""

    def set(self, value: Any, source: str, keep_prev: bool = True) -> bool:
        """Overwrite the value, keeping history per *keep_prev*.

        값을 덮어쓴다. keep_prev=True 면 이전 값을 prev_value 로 보존,
        False (비밀 필드 — 모듈 docstring) 면 prev_updated_at 만 남기고
        이전 값은 파기한다. 값이 같으면 아무것도 하지 않는다 —
        prev_value 가 자기 자신으로 덮여 이력이 지워지는 것을 막는다.
        바뀌었을 때만 True.
        """
        if value == self.value:
            return False
        self.prev_value = self.value if keep_prev else None
        self.prev_updated_at = self.updated_at
        self.value = value
        self.source = source
        self.updated_at = _utc_now_iso()
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "updated_at": self.updated_at,
            "prev_value": self.prev_value,
            "prev_updated_at": self.prev_updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StaticEntry:
        return cls(
            value=data.get("value", ""),
            source=data.get("source", SOURCE_AUTO),
            updated_at=data.get("updated_at", ""),
            prev_value=data.get("prev_value"),
            prev_updated_at=data.get("prev_updated_at", ""),
        )

    def revert(self, source: str) -> None:
        """Swap value and prev_value (caller checked history exists).

        value 와 prev_value 를 교환한다 (이력 존재는 호출자가 확인).
        되돌림도 하나의 변경이므로 시각·출처를 갱신하고, 직전 상태
        (되돌리기 전 값) 를 새 prev 로 남겨 재되돌리기가 가능하게 한다.
        """
        self.value, self.prev_value = self.prev_value, self.value
        self.prev_updated_at = self.updated_at
        self.updated_at = _utc_now_iso()
        self.source = source

    @classmethod
    def migrated(cls, value: Any, updated_at: str) -> StaticEntry:
        """Wrap one pre-R5-C4 flat value (see module docstring).

        R5-C4 이전 평면 값 1개를 감싼다 (모듈 docstring 참조).
        """
        return cls(value=value, source=SOURCE_AUTO, updated_at=updated_at)


def _entry_from_value(raw: Any, file_updated_at: str) -> StaticEntry:
    """Parse a schema-2 field value; preserve a hand-edited flat value.

    schema 2 필드 값을 파싱한다. entry dict 아닌 값 (손편집 평면값) 은
    이행 항목으로 감싸 보존한다 (모듈 docstring).
    """
    if isinstance(raw, dict):
        return StaticEntry.from_dict(raw)
    if raw is None:
        return StaticEntry()
    return StaticEntry.migrated(raw, file_updated_at)


def _entry_map_from_dict(
    data: Any, file_updated_at: str, migrate: bool
) -> dict[str, StaticEntry]:
    if not isinstance(data, dict):
        return {}
    if migrate:
        return {
            key: StaticEntry.migrated(value, file_updated_at)
            for key, value in data.items()
        }
    # Schema-2 file: a non-entry value (hand edit) is preserved by
    # wrapping, never silently dropped (module docstring).
    # schema 2 파일 — entry 아닌 값 (손편집) 은 감싸서 보존한다. 조용히
    # 버리지 않는다 (모듈 docstring).
    return {
        key: (
            StaticEntry.from_dict(value)
            if isinstance(value, dict)
            else StaticEntry.migrated(value, file_updated_at)
        )
        for key, value in data.items()
    }


@dataclass
class StaticField:
    project_context: StaticEntry = field(default_factory=StaticEntry)
    conventions: StaticEntry = field(default_factory=StaticEntry)
    project_map: dict[str, StaticEntry] = field(default_factory=dict)
    variables: dict[str, StaticEntry] = field(default_factory=dict)
    updated_at: str = ""

    @classmethod
    def new(cls) -> StaticField:
        return cls(updated_at=_utc_now_iso())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA_VERSION,
            "project_context": self.project_context.to_dict(),
            "conventions": self.conventions.to_dict(),
            "project_map": {k: e.to_dict() for k, e in self.project_map.items()},
            "variables": {k: e.to_dict() for k, e in self.variables.items()},
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StaticField:
        file_updated_at = data.get("updated_at", "")
        schema = data.get("schema")
        if schema is not None and schema != _SCHEMA_VERSION:
            # Present but unknown — refuse rather than destroy (module
            # docstring). 존재하지만 미지 — 파괴 대신 거부 (모듈 docstring).
            raise UnsupportedStaticSchemaError(schema)
        migrate = schema is None
        if migrate:
            # Pre-R5-C4 flat file — see module docstring.
            # R5-C4 이전 평면 파일 — 모듈 docstring 참조.
            project_context = StaticEntry.migrated(
                data.get("project_context", ""), file_updated_at
            )
            conventions = StaticEntry.migrated(
                data.get("conventions", ""), file_updated_at
            )
        else:
            project_context = _entry_from_value(
                data.get("project_context"), file_updated_at
            )
            conventions = _entry_from_value(
                data.get("conventions"), file_updated_at
            )
        return cls(
            project_context=project_context,
            conventions=conventions,
            project_map=_entry_map_from_dict(
                data.get("project_map"), file_updated_at, migrate
            ),
            variables=_entry_map_from_dict(
                data.get("variables"), file_updated_at, migrate
            ),
            updated_at=file_updated_at,
        )

    def touch(self) -> None:
        self.updated_at = _utc_now_iso()

    def to_masked_dict(self) -> dict[str, Any]:
        """to_dict with every variables value replaced by a shape tag.

        variables 값을 형태 태그 (예: "<str, 12자>") 로 바꾼 to_dict —
        /static 목록용. 마스킹이 LLM 지시 순응이 아니라 서버 코드의
        보장이 되도록 여기서 수행한다 (R5-C5 재검토). 값이 필요한
        경우는 없다 — 수정 흐름도 옛 값을 보여 주지 않는다 (파일 직접
        열람이 대안).
        """
        data = self.to_dict()
        data["variables"] = {
            key: {**entry_dict, "value": _shape_tag(entry_dict["value"])}
            for key, entry_dict in data["variables"].items()
        }
        return data

    # ---- Mutators (R5-C4) / 변경자 ----

    def set_project_context(self, value: str, source: str) -> list[str]:
        return ["project_context"] if self.project_context.set(value, source) else []

    def set_conventions(self, value: str, source: str) -> list[str]:
        return ["conventions"] if self.conventions.set(value, source) else []

    def merge_project_map(self, mapping: dict[str, Any], source: str) -> list[str]:
        return self._merge("project_map", self.project_map, mapping, source)

    def merge_variables(self, mapping: dict[str, Any], source: str) -> list[str]:
        # keep_prev=False: variables may hold secrets — no value history
        # (module docstring). 비밀 가능 — 값 이력 미보존 (모듈 docstring).
        return self._merge(
            "variables", self.variables, mapping, source, keep_prev=False
        )

    def _entry_at(self, field_name: str, key: str | None) -> StaticEntry | None:
        """Resolve one addressable entry; None when absent.

        지칭된 항목 1개를 찾는다. 없으면 None.
        """
        if field_name in TEXT_FIELDS:
            return getattr(self, field_name)
        if field_name in DICT_FIELDS and key is not None:
            return getattr(self, field_name).get(key)
        return None

    def delete_entry(self, field_name: str, key: str | None = None) -> bool:
        """Remove one entry entirely — prev_value included (R5-C5).

        항목 1개를 통째로 제거한다 — prev_value 까지 함께 (R5-C5: 되돌리기
        근거로 남은 직전 값도 삭제 의사에 포함된다). 문자열 필드는 빈
        항목으로 리셋. 지워진 것이 있을 때만 True.
        """
        if field_name in TEXT_FIELDS:
            entry: StaticEntry = getattr(self, field_name)
            if entry == StaticEntry():
                return False
            setattr(self, field_name, StaticEntry())
            return True
        if field_name in DICT_FIELDS and key is not None:
            return getattr(self, field_name).pop(key, None) is not None
        return False

    def revert_entry(
        self, field_name: str, key: str | None = None
    ) -> tuple[str, str]:
        """Swap one entry back to its previous value where history exists.

        항목 1개를 직전 값으로 되돌린다. (결과, prev_updated_at) 을
        돌려준다 — variables 는 값 이력을 남기지 않으므로 (모듈
        docstring — 비밀 정책) 항상 REVERT_SECRET 이며, 이때
        prev_updated_at (덮어쓴 시각) 이 호출 쪽 (/static 스킬) 이
        사용자에게 보여 줄 유일한 이력이다.
        """
        entry = self._entry_at(field_name, key)
        if entry is None:
            return REVERT_NOT_FOUND, ""
        if field_name == "variables":
            return REVERT_SECRET, entry.prev_updated_at
        if not entry.prev_updated_at:
            return REVERT_NO_HISTORY, ""
        entry.revert(SOURCE_USER)
        return REVERT_OK, entry.prev_updated_at

    @staticmethod
    def _merge(
        field_name: str,
        entries: dict[str, StaticEntry],
        mapping: dict[str, Any],
        source: str,
        keep_prev: bool = True,
    ) -> list[str]:
        """Merge *mapping* key-by-key; return the changed entry names.

        *mapping* 을 키별로 병합하고 바뀐 항목 이름 목록을 돌려준다.
        전달된 키만 갱신·추가하며 나머지 키는 유지한다 — 통째 교체는
        LLM 이 바뀐 키만 보낼 때 나머지를 지워 버린다 (R5-C4 에서 병합
        의미론으로 확정). 항목 삭제는 R5-C5 의 전용 도구 몫이다.
        """
        changed: list[str] = []
        for key, value in mapping.items():
            entry = entries.setdefault(key, StaticEntry())
            if entry.set(value, source, keep_prev=keep_prev):
                changed.append(f"{field_name}.{key}")
        return changed
