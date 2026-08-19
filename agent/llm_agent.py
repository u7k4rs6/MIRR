import json
import os
import re

from typing import Optional

from groq import Groq

SYSTEM_PROMPT = """You are an on-call engineer responding to a live production incident.

Your goal: identify the root cause service and failure mode, then apply the correct fix.

SERVICES: api-gateway, auth-service, database, cache, worker
FAILURE MODES: crashed, memory_leak, overloaded, bad_deploy

FIX MAP:
- crashed → restart_service (alias: kubectl_rollout_restart, restart_pod)
- memory_leak → restart_service
- overloaded → scale_up (alias: kubectl_scale_deployment, scale_deployment)
- bad_deploy → rollback_deploy (alias: kubectl_rollout_undo)

If last_action_result contains CASCADE or "new primary", the incident changed — investigate again
and you may call diagnose again for the new root cause.

STRATEGY:
1. check_logs on 1-2 suspicious services (high CPU/error_rate); aliases: kubectl_logs, tail_logs
2. diagnose once you're confident (optional "evidence": short string citing logs/metrics)
3. apply the matching fix
4. use defer with optional "reason" if you need one more signal without risky changes
5. confirm recovery

OUTPUT ONLY valid JSON action, nothing else:
{"type": "check_logs", "target": "database"}
{"type": "diagnose", "target": "database", "failure_mode": "memory_leak", "evidence": "heap lines in logs"}
{"type": "restart_service", "target": "database"}
{"type": "kubectl_rollout_undo", "target": "auth-service"}
{"type": "scale_up", "target": "cache"}
{"type": "enable_circuit_breaker", "target": "worker"}
{"type": "defer", "reason": "need one more metric window"}
{"type": "no_op"}
"""


def parse_action(text: str) -> dict:
    """Extract JSON from model response."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find outermost JSON object by brace matching
    start = text.find("{")
    if start == -1:
        return {"type": "no_op"}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    break
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"type": "no_op"}


class LLMAgent:
    # Groq retired the llama-3.1-* line; a stale id 404s even with a valid key.
    # Override with GROQ_MODEL rather than editing this file.
    DEFAULT_MODEL = "openai/gpt-oss-20b"

    def __init__(self, model=None, max_tokens=150):
        self.model = model or os.environ.get("GROQ_MODEL") or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.history = []
        self._client: Optional[Groq] = None

    def _get_client(self) -> Groq:
        if self._client is None:
            key = os.environ.get("GROQ_API_KEY")
            if not key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Add it to your environment or HF Space secrets."
                )
            self._client = Groq(api_key=key)
        return self._client

    def reset(self):
        self.history = []

    def act(self, obs: dict) -> dict:
        user_msg = f"OBSERVATION:\n{json.dumps(obs, indent=2)}\n\nWhat is your next action? Output only JSON."
        self.history.append({"role": "user", "content": user_msg})

        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self.history[-6:],
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        reply = response.choices[0].message.content or ""
        self.history.append({"role": "assistant", "content": reply})
        return parse_action(reply)
