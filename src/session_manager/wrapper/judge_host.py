"""
Resident routing-judge host — manages the warmed disposable judge process.

Architecture (measured, docs/poc/R2-hook.md §9): a naive resident
process reused across judgments saves nothing (context accumulation +
thinking tokens put it back at ~8s). What works is a **warmed disposable
process**: spawn ``claude -p`` in stream-json mode, run one cheap warmup
round (hidden in the background, creates the system-prompt cache), serve
exactly ONE judgment (measured 1.6–3.1s warm), then retire the process
and re-warm a fresh one. Every judgment gets a clean context.

Request flow: the UserPromptSubmit hook sends a ``judge_request`` over
the wrapper socket; the socket server transfers connection ownership
here (deferred reply). If the judge is warm, a worker thread runs the
judgment and replies with the verdict; if it is still warming up (or
dead), the host replies ``judge_unavailable`` immediately — a missed
routing check is harmless (the next prompt retries), but making the
user wait for a warmup is not.

상주 라우팅 판정 호스트 — 웜업된 1회용 판정 프로세스를 관리한다.

아키텍처 (실측, docs/poc/R2-hook.md §9): 판정을 한 프로세스에 계속
흘리는 소박한 상주는 이득이 없다 (컨텍스트 누적 + thinking 토큰으로
~8s). 동작하는 것은 **웜업된 1회용 프로세스**다: stream-json 모드로
``claude -p``를 spawn → 저렴한 웜업 라운드 1회 (백그라운드에 숨김,
시스템 프롬프트 캐시 생성) → 판정 정확히 1회 수행 (실측 warm 1.6~3.1s)
→ 프로세스 은퇴·새 프로세스 재웜업. 모든 판정이 깨끗한 컨텍스트를 얻는다.

요청 흐름: UserPromptSubmit hook 이 래퍼 소켓으로 ``judge_request``를
보내면 소켓 서버가 연결 소유권을 이곳으로 이관한다 (지연 회신). 판정기가
warm 이면 워커 스레드가 판정을 수행해 결과를 회신하고, 웜업 중(또는
사망)이면 즉시 ``judge_unavailable``을 회신한다 — 라우팅 1회 미발동은
무해하지만 (다음 프롬프트에서 재시도), 사용자를 웜업 동안 기다리게 하는
것은 유해하다.
"""

from __future__ import annotations

import json
import os
import queue
import select
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from session_manager import debug_log
from session_manager.claude_conversation import encode_cwd
from session_manager.models.session import SessionMetadata
from session_manager.routing import judge
from session_manager.storage.file_store import SessionStore
from session_manager.transcript_excerpt import extract_dialogue, read_tail_events

# Headless isolation shared with the summarizer (see
# summarizer.run_headless_summary for the measured rationale): no MCP
# servers, socket env stripped, prompt over stdin. Additionally
# MAX_THINKING_TOKENS=0 — thinking tokens were the dominant latency term
# (765–1140 tokens/judgment, docs/poc/R2-hook.md §9.1).
# summarizer 와 공유하는 headless 격리 (실측 근거는
# summarizer.run_headless_summary 참조): MCP 무로드, 소켓 env 제거,
# 프롬프트 stdin 전달. 추가로 MAX_THINKING_TOKENS=0 — thinking 토큰이
# 지연의 지배 항이었다 (판정당 765~1140 토큰, docs/poc/R2-hook.md §9.1).
_EMPTY_MCP_CONFIG = '{"mcpServers":{}}'
_SOCKET_ENV_VAR = "SESSION_MANAGER_SOCKET"
_CHILD_SESSION_ENV_VAR = "CLAUDE_CODE_CHILD_SESSION"
_THINKING_ENV_VAR = "MAX_THINKING_TOKENS"

# Dedicated neutral cwd, separate from the summarizer's headless-tmp:
# the summarizer sweeps its junk-transcript dir before every call, and a
# shared dir would let that sweep delete the *live* judge conversation.
# The judge sweeps its own dir right before each spawn (no live process
# exists at that moment).
# 전용 중립 cwd — summarizer 의 headless-tmp 와 분리한다. summarizer 는
# 매 호출 전 정크 transcript 디렉토리를 쓸어내는데, 공유하면 그 sweep 이
# *살아 있는* 판정 대화를 지울 수 있다. 판정기는 자기 디렉토리를 매 spawn
# 직전 (그 시점엔 살아 있는 프로세스가 없음) 에 직접 쓸어낸다.
_JUDGE_NEUTRAL_CWD = Path.home() / ".session-manager" / "judge-tmp"

# Failure policy mirrors the summarizer: one retry. Two consecutive
# spawn/warmup failures mark the judge dead until the wrapper restarts —
# routing then degrades to pass-through instead of burning quota on a
# broken CLI.
# 실패 정책은 summarizer 와 동일 — 1회 재시도. spawn/웜업 연속 2회 실패
# 시 래퍼 재시작 전까지 판정기를 사망 처리한다 — 라우팅은 통과로
# 완화되고, 고장난 CLI 에 쿼터를 태우지 않는다.
_MAX_CONSECUTIVE_SPAWN_FAILURES = 2

# Read chunk size for the judge process stdout.
# 판정 프로세스 stdout 읽기 chunk 크기.
_READ_CHUNK = 65536


def _stream_user_message(text: str) -> bytes:
    """Encode one stream-json user message line.

    stream-json 사용자 메시지 한 줄을 인코딩한다.
    """
    return (
        json.dumps(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                },
            },
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


class JudgeHost:
    """Owns the judge worker thread and the disposable judge process.

    판정 워커 스레드와 1회용 판정 프로세스를 소유한다.
    """

    def __init__(self, project_path: Path) -> None:
        self._project_path = Path(project_path)
        self._store = SessionStore(self._project_path)
        # Single-slot queue: prompts are serialized by the TUI, so a
        # second request while one is in flight only happens in races —
        # it gets an immediate unavailable reply instead of queueing.
        # 단일 슬롯 큐 — 프롬프트는 TUI 가 직렬화하므로 처리 중 두 번째
        # 요청은 경합에서만 발생하고, 대기 대신 즉시 unavailable 회신.
        self._requests: queue.Queue[tuple[dict[str, Any], socket.socket]] = (
            queue.Queue(maxsize=1)
        )
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ready = False
        self._dead = False
        self._proc: subprocess.Popen[bytes] | None = None
        self._proc_buffer = b""

    # ------------------------------------------------------------ lifecycle
    # 생명주기 -------------------------------------------------------------------

    def ensure_started(self) -> None:
        """Start the worker thread if not already running. Idempotent.

        워커 스레드가 없으면 시작한다. 멱등.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        if self._stop_event.is_set() or self._dead:
            return
        self._thread = threading.Thread(
            target=self._worker_loop, name="judge-host", daemon=True
        )
        self._thread.start()
        debug_log.log("JUDGE", "WRAPPER", {"op": "start"})

    def stop(self) -> None:
        """Stop the worker and terminate any live judge process.

        워커를 멈추고 살아 있는 판정 프로세스를 종료한다.
        """
        self._stop_event.set()
        self._ready = False
        self._retire_process()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        # Drain any queued request so its hook socket isn't leaked.
        # 대기 중 요청의 hook 소켓이 누수되지 않게 배출한다.
        try:
            _message, sock = self._requests.get_nowait()
        except queue.Empty:
            pass
        else:
            self._reply_and_close(sock, {"ok": False, "reason": "shutting_down"})
        debug_log.log("JUDGE", "WRAPPER", {"op": "stop"})

    # ------------------------------------------------------- request intake
    # 요청 수신 (메인 루프 스레드에서 호출) --------------------------------------

    def handle_request(
        self, message: dict[str, Any], sock: socket.socket
    ) -> bool:
        """
        Take ownership of a judge_request connection. Runs on the
        wrapper's I/O-loop thread, so it must not block: either the
        request is handed to the worker, or an unavailable reply goes
        out immediately.

        judge_request 연결의 소유권을 인수한다. 래퍼 I/O 루프 스레드에서
        실행되므로 블로킹 금지 — 요청을 워커에 넘기거나, 즉시
        unavailable 을 회신한다.
        """
        if not self._ready or self._stop_event.is_set() or self._dead:
            self._reply_and_close(
                sock, {"ok": False, "reason": "judge_unavailable"}
            )
            debug_log.log(
                "JUDGE",
                "WRAPPER",
                {"op": "request", "result": "unavailable", "dead": self._dead},
            )
            return True
        try:
            self._requests.put_nowait((message, sock))
        except queue.Full:
            self._reply_and_close(sock, {"ok": False, "reason": "judge_busy"})
            debug_log.log(
                "JUDGE", "WRAPPER", {"op": "request", "result": "busy"}
            )
            return True
        # Claimed: the process serves one judgment then retires, so it
        # stops being available the moment a request is accepted.
        # 접수 즉시 비가용 처리 — 프로세스는 판정 1회 후 은퇴한다.
        self._ready = False
        return True

    # --------------------------------------------------------- worker thread
    # 워커 스레드 ----------------------------------------------------------------

    def _worker_loop(self) -> None:
        spawn_failures = 0
        while not self._stop_event.is_set():
            if self._proc is None:
                if self._spawn_and_warm():
                    spawn_failures = 0
                    self._ready = True
                else:
                    spawn_failures += 1
                    if spawn_failures >= _MAX_CONSECUTIVE_SPAWN_FAILURES:
                        self._dead = True
                        debug_log.log(
                            "JUDGE",
                            "WRAPPER",
                            {"op": "worker", "result": "dead"},
                        )
                        return
                    continue
            try:
                request, sock = self._requests.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                reply = self._serve_request(request)
            except Exception as exc:
                # A judgment bug must not kill the worker; the hook side
                # treats the failure as pass-through.
                # 판정 버그가 워커를 죽여선 안 된다. hook 측은 실패를
                # 통과로 처리한다.
                debug_log.log(
                    "JUDGE",
                    "WRAPPER",
                    {"op": "serve", "result": "exception", "error": str(exc)},
                )
                reply = {"ok": False, "reason": "judge_error"}
            self._reply_and_close(sock, reply)
            # One judgment per process — retire and re-warm.
            # 프로세스당 판정 1회 — 은퇴 후 재웜업.
            self._retire_process()

    def _serve_request(self, request: dict[str, Any]) -> dict[str, Any]:
        prompt = request.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return {"ok": False, "reason": "empty_prompt"}

        excerpt = ""
        transcript_path = request.get("transcript_path")
        if isinstance(transcript_path, str) and transcript_path:
            events = read_tail_events(Path(transcript_path))
            excerpt = extract_dialogue(events)

        sessions = [
            {
                "name": s.name,
                "title": s.title,
                "summary": s.summary,
                "last_accessed": s.last_accessed,
                # Raw mixing signal (R3-C2) — no threshold applied here;
                # the judge weighs the raw value and its evidence.
                # 혼합도 원신호 (R3-C2) — 여기서 임계를 적용하지 않는다.
                # 원값과 근거 인용의 가중은 판정기가 결정한다.
                "mixing_score": s.mixing_score,
                "mixing_evidence": list(s.mixing_evidence),
            }
            for s in self._store.list_sessions()
            if s.status.value == "active"
        ]
        current = self._resolve_current_session(request)
        current_name = current.name if current is not None else None

        judge_prompt = judge.build_judge_prompt(
            prompt=prompt,
            excerpt=excerpt,
            sessions=sessions,
            current_name=current_name,
        )
        t0 = time.monotonic()
        raw = self._round_trip(judge_prompt, judge.JUDGE_TIMEOUT_SECS)
        elapsed = time.monotonic() - t0
        if raw is None:
            verdict = judge.Verdict.stay("judge_timeout")
        else:
            parsed = judge.parse_verdict(raw)
            verdict = (
                parsed if parsed is not None else judge.Verdict.stay("judge_unparsable")
            )

        # Deterministic precedent gate (R3-FIX2): a SWITCH whose target
        # the user has rejected before (live precedent on the current
        # session) is demoted to STAY here, in code — not entrusted to
        # the model (measured 3/3: the model inverted the precedent's
        # meaning when asked to weigh it). Invalidation stays
        # event-based: acceptance, manual move into the target, rollover.
        # The suppression is logged for R5 metrics.
        # 결정적 판례 게이트 (R3-FIX2) — 사용자가 이미 거부한 대상 (현재
        # 세션의 유효 판례) 으로의 SWITCH 는 모델에게 맡기지 않고 여기
        # 코드에서 STAY 로 강등한다 (실측 3/3: 모델은 판례의 의미를 반전
        # 해석했다). 무효화는 이벤트 기반 유지 — 수락·대상으로의 수동
        # 이동·롤오버. 억제는 R5 계측용으로 로그에 남긴다.
        if (
            verdict.action == judge.ACTION_SWITCH
            and current is not None
            and verdict.target is not None
            and any(p.rejected == verdict.target for p in current.precedents)
        ):
            debug_log.log(
                "JUDGE",
                "WRAPPER",
                {
                    "op": "precedent_suppress",
                    "target": verdict.target,
                    "kept_in": current.name,
                    "original_confidence": verdict.confidence,
                },
                conv_id=request.get("session_id"),
                session=current.name,
            )
            verdict = judge.Verdict.stay(
                f"suppressed_by_precedent: {verdict.target}"
            )
        debug_log.log(
            "JUDGE",
            "WRAPPER",
            {
                "op": "verdict",
                "elapsed_s": round(elapsed, 2),
                "verdict": verdict.to_dict(),
                "raw": raw,
            },
            conv_id=request.get("session_id"),
        )
        reply: dict[str, Any] = {"ok": True, "verdict": verdict.to_dict()}
        refute = self._maybe_refute(request, verdict)
        if refute is not None:
            reply["refute"] = refute
        return reply

    def _maybe_refute(
        self, request: dict[str, Any], verdict: judge.Verdict
    ) -> dict[str, Any] | None:
        """Run the second-pass refutation when the verdict qualifies for auto.

        판정이 auto 자격일 때 2차 반박 검증을 수행한다 (R3-C4).

        Runs one extra turn in the SAME warm process right before its
        retirement (rationale measured — judge.REFUTE_PROMPT_TEMPLATE
        comment). Only fires when the hook attached an ``auto_gate`` and
        the verdict is a SWITCH at or above the calibrated threshold —
        so the extra round-trip cost is bounded to imminent auto
        switches. A timeout or unparseable answer reports refuted=true:
        an unverified switch must not run automatically.

        은퇴 직전의 **같은 웜 프로세스**에 1턴을 추가로 돌린다 (근거
        실측 — judge.REFUTE_PROMPT_TEMPLATE 주석). hook 이 ``auto_gate``
        를 동봉했고 판정이 보정 임계 이상의 SWITCH 일 때만 발동 — 추가
        왕복 비용이 임박한 자동 전환에만 한정된다. 타임아웃·파싱 불가는
        refuted=true 로 보고한다: 검증되지 않은 전환을 자동 실행하면
        안 된다.
        """
        auto_gate = request.get("auto_gate")
        if not isinstance(auto_gate, dict):
            return None
        threshold = auto_gate.get("threshold")
        if not isinstance(threshold, int | float):
            return None
        if verdict.action != judge.ACTION_SWITCH or verdict.confidence < threshold:
            return None
        raw = self._round_trip(
            judge.build_refute_prompt(verdict.to_dict()), judge.JUDGE_TIMEOUT_SECS
        )
        parsed = judge.parse_refute(raw) if raw is not None else None
        if parsed is None:
            result = {"refuted": True, "reason": "refute_unavailable"}
        else:
            result = parsed
        debug_log.log(
            "JUDGE",
            "WRAPPER",
            {"op": "refute", "result": result, "raw": raw},
            conv_id=request.get("session_id"),
        )
        return result

    def _resolve_current_session(
        self, request: dict[str, Any]
    ) -> SessionMetadata | None:
        """Map the hook's conversation id to a session, if known.

        hook 의 conversation id 를 세션으로 대응시킨다 (가능할 때).
        """
        conv_id = request.get("session_id")
        if not isinstance(conv_id, str) or not conv_id:
            return None
        for s in self._store.list_sessions():
            if conv_id in s.claude_conversation_ids:
                return s
        return None

    # ------------------------------------------------- judge process plumbing
    # 판정 프로세스 배관 ---------------------------------------------------------

    def _spawn_and_warm(self) -> bool:
        """Spawn a fresh judge process and run the warmup round.

        새 판정 프로세스를 spawn 하고 웜업 라운드를 수행한다.
        """
        self._sweep_junk_transcripts()
        _JUDGE_NEUTRAL_CWD.mkdir(parents=True, exist_ok=True)
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in (_SOCKET_ENV_VAR, _CHILD_SESSION_ENV_VAR)
        }
        env[_THINKING_ENV_VAR] = "0"
        try:
            self._proc = subprocess.Popen(
                [
                    "claude",
                    "-p",
                    "--model",
                    judge.JUDGE_MODEL,
                    "--input-format",
                    "stream-json",
                    "--output-format",
                    "stream-json",
                    # --verbose is the CLI's required companion flag for
                    # stream-json output in print mode.
                    # print 모드 stream-json 출력의 필수 동반 플래그.
                    "--verbose",
                    "--strict-mcp-config",
                    "--mcp-config",
                    _EMPTY_MCP_CONFIG,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=_JUDGE_NEUTRAL_CWD,
                env=env,
            )
        except OSError as exc:
            debug_log.log(
                "JUDGE",
                "WRAPPER",
                {"op": "spawn", "result": "error", "error": str(exc)},
            )
            self._proc = None
            return False
        self._proc_buffer = b""
        warm = self._round_trip(judge.WARMUP_PROMPT, judge.WARMUP_TIMEOUT_SECS)
        if warm is None:
            debug_log.log(
                "JUDGE", "WRAPPER", {"op": "warmup", "result": "timeout"}
            )
            self._retire_process()
            return False
        debug_log.log("JUDGE", "WRAPPER", {"op": "warmup", "result": "ok"})
        return True

    def _round_trip(self, text: str, timeout: float) -> str | None:
        """Send one user message and wait for its result event.

        사용자 메시지 1건을 보내고 result 이벤트를 기다린다. 타임아웃·EOF
        시 None.
        """
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            return None
        try:
            proc.stdin.write(_stream_user_message(text))
            proc.stdin.flush()
        except (OSError, ValueError):
            return None

        fd = proc.stdout.fileno()
        deadline = time.monotonic() + timeout
        while True:
            while b"\n" in self._proc_buffer:
                line, self._proc_buffer = self._proc_buffer.split(b"\n", 1)
                if not line:
                    continue
                try:
                    event = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if isinstance(event, dict) and event.get("type") == "result":
                    result = event.get("result")
                    return result if isinstance(result, str) else None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                readable, _, _ = select.select([fd], [], [], min(0.2, remaining))
            except OSError:
                return None
            if fd not in readable:
                continue
            try:
                chunk = os.read(fd, _READ_CHUNK)
            except OSError:
                return None
            if not chunk:
                return None
            self._proc_buffer += chunk

    def _retire_process(self) -> None:
        proc = self._proc
        self._proc = None
        self._proc_buffer = b""
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.terminate()
            proc.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                proc.kill()
            except OSError:
                pass

    def _sweep_junk_transcripts(self) -> None:
        """Delete junk transcripts under the judge's neutral cwd.

        판정 중립 cwd 아래의 정크 transcript 를 삭제한다. spawn 직전에만
        호출되므로 살아 있는 판정 대화를 지울 일이 없다.
        """
        junk_dir = Path.home() / ".claude" / "projects" / encode_cwd(_JUDGE_NEUTRAL_CWD)
        if not junk_dir.is_dir():
            return
        removed = 0
        for entry in junk_dir.iterdir():
            try:
                if entry.is_dir():
                    shutil.rmtree(entry, ignore_errors=True)
                else:
                    entry.unlink(missing_ok=True)
                removed += 1
            except OSError:
                continue
        if removed:
            debug_log.log(
                "JUDGE",
                "WRAPPER",
                {"op": "sweep_junk", "removed": removed, "dir": str(junk_dir)},
            )

    # --------------------------------------------------------------- replies
    # 회신 -----------------------------------------------------------------------

    @staticmethod
    def _reply_and_close(sock: socket.socket, message: dict[str, Any]) -> None:
        """Send one JSON line on the detached hook socket and close it.

        이관받은 hook 소켓에 JSON 한 줄을 보내고 닫는다. hook 측이 이미
        떠났어도 (자체 타임아웃) 무해하다.
        """
        try:
            # Engineering margin: a local AF_UNIX sendall of <1 KB is
            # sub-millisecond; 1s is ~3 orders of magnitude headroom.
            # 공학 여유 — 로컬 AF_UNIX 로 1KB 미만 sendall 은 밀리초
            # 미만이다. 1초는 약 3자릿수의 여유.
            sock.settimeout(1.0)
            payload = (json.dumps(message, ensure_ascii=False) + "\n").encode(
                "utf-8"
            )
            sock.sendall(payload)
        except OSError:
            pass
        finally:
            try:
                sock.close()
            except OSError:
                pass
