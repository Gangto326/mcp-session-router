---
name: router
description: 세션 라우팅 모드(auto/confirm/off)를 조회하고 변경한다. 사용자가 라우팅 모드·수용률을 묻거나 자동 전환을 켜고 끄고 싶을 때 사용.
---

get_routing_status 도구로 현재 모드와 최근 수용률을 조회해 한 줄로 보여준 뒤,
AskUserQuestion으로 [auto(수용률 기준 권장 표시) / confirm / off / 현재 유지]를
묻고, 선택에 따라 set_routing_mode를 호출하라. 다른 작업 금지.

참고:

- auto 선택지의 권장 표시는 get_routing_status 의 `auto_available` 값을 따른다
  — true 면 "(권장)" 을 붙이고, false 면 "(보정 데이터 부족 — 켜도 confirm 으로
  동작)" 을 붙인다.
- 수용률 한 줄에는 `overall_acceptance_rate` 와 `recent_acceptance_rate`
  (최근 추세) 를 병기한다. 표본이 없으면 (null) "아직 수용/거부 기록 없음" 으로
  표시한다.
- "현재 유지" 선택 시 set_routing_mode 를 호출하지 않는다.
