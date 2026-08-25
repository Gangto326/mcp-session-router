---
name: router
description: 세션 라우팅 모드(confirm/off)를 조회하고 변경한다. 사용자가 라우팅 모드·수용률을 묻거나 라우팅을 켜고 끄고 싶을 때 사용.
---

get_routing_status 도구로 현재 모드와 최근 수용률을 조회해 한 줄로 보여준 뒤,
AskUserQuestion으로 [confirm (제안 후 물어봄) / off / 현재 유지]를 묻고,
선택에 따라 set_routing_mode를 호출하라. 다른 작업 금지.

참고:

- 수용률 한 줄에는 `overall_acceptance_rate` 와 `recent_acceptance_rate`
  (최근 추세) 를 병기한다. 표본이 없으면 (null) "아직 수용/거부 기록 없음" 으로
  표시한다.
- "현재 유지" 선택 시 set_routing_mode 를 호출하지 않는다.
- (auto 모드는 R6-C3 에서 제거 — 낡은 config 의 "auto" 는 confirm 으로
  동작한다.)
