"""Pluggable model & agent adapters for deterministic offline evals and live endpoints."""

from abc import ABC, abstractmethod
import hashlib
import json
import os
import time
import urllib.request
from typing import Any, Dict, List, Optional

from src.schemas import AgentResponse, EvalTask, ToolCall


class BaseAgentAdapter(ABC):
    """Abstract interface for any LLM, coding agent, or tool-calling system under test."""

    name: str = "base"

    @abstractmethod
    def generate(self, task: EvalTask) -> AgentResponse:
        """Execute a single evaluation task and return structured trace + operational stats."""
        raise NotImplementedError


class DeterministicBenchmarkAdapter(BaseAgentAdapter):
    """Reproducible agent simulator modeling realistic failure modes across 3 release tiers.

    Profiles:
    - baseline_v1:           ~78% pass rate (moderate coding, tool-arg & hallucination errors)
    - candidate_v2:          ~95% pass rate (improved tool precision & groundedness, faster p95)
    - candidate_v2_regressed: ~62% pass rate (schema/hallucination regressions & latency spike)
    """

    def __init__(self, profile: str = "baseline_v1"):
        if profile not in {"baseline_v1", "candidate_v2", "candidate_v2_regressed"}:
            raise ValueError(f"Unknown adapter profile: {profile}")
        self.profile = profile
        self.name = profile

    def _task_bucket(self, task_id: str) -> int:
        digest = hashlib.sha256(f"{task_id}".encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % 100

    def _should_fail(self, task: EvalTask) -> bool:
        bucket = self._task_bucket(task.task_id)
        if self.profile == "baseline_v1":
            return bucket < 22
        if self.profile == "candidate_v2":
            return bucket < 5
        # candidate_v2_regressed: regresses heavily on JSON & grounded QA
        if task.category in {"structured_json", "grounded_qa"}:
            return bucket < 45
        return bucket < 32

    def _sim_latency_ms(self, task: EvalTask) -> float:
        bucket = self._task_bucket(task.task_id)
        if self.profile == "baseline_v1":
            return 280.0 + (bucket * 4.5)
        if self.profile == "candidate_v2":
            return 240.0 + (bucket * 3.8)
        # Regressed model breaches 1500ms p95 SLA
        return 750.0 + (bucket * 11.5)

    def generate(self, task: EvalTask) -> AgentResponse:
        fail = self._should_fail(task)
        latency = self._sim_latency_ms(task)
        prompt_tok = max(60, len(task.prompt) // 4 + (len(task.context or "") // 4))

        if task.category == "code_synthesis":
            if not fail:
                code = f"```python\n{task.reference['reference_solution']}\n```"
            else:
                # Realistic off-by-one / wrong return failure
                fn_name = task.reference.get("function_name", "solve")
                code = f"```python\ndef {fn_name}(*args, **kwargs):\n    return None\n```"
            return AgentResponse(
                task_id=task.task_id,
                output_text=code,
                latency_ms=latency,
                prompt_tokens=prompt_tok,
                completion_tokens=len(code) // 4,
            )

        if task.category == "tool_trajectory":
            exp_tools = task.reference.get("expected_tools", [])
            if not fail:
                calls = [ToolCall(name=t["name"], arguments=dict(t["arguments"])) for t in exp_tools]
            else:
                # Realistic failure: dropped final step or wrong argument
                calls = []
                for idx, t in enumerate(exp_tools):
                    bad_args = dict(t["arguments"])
                    if idx == 0 and bad_args:
                        first_key = next(iter(bad_args))
                        bad_args[first_key] = "INVALID_ARG"
                    calls.append(ToolCall(name=t["name"], arguments=bad_args))
                if len(calls) > 1:
                    calls = calls[:-1]
            return AgentResponse(
                task_id=task.task_id,
                output_text="Executed tool sequence.",
                tool_calls=calls,
                latency_ms=latency,
                prompt_tokens=prompt_tok,
                completion_tokens=85,
            )

        if task.category == "grounded_qa":
            facts = task.reference.get("required_facts", [])
            if not fail:
                ans = "Based on the provided context: " + "; ".join(facts) + "."
            else:
                # Realistic hallucination: invents an ungrounded figure 99.7%
                ans = f"Based on the context: {facts[0] if facts else 'N/A'}, with an estimated 99.7% uplift."
            return AgentResponse(
                task_id=task.task_id,
                output_text=ans,
                latency_ms=latency,
                prompt_tokens=prompt_tok,
                completion_tokens=len(ans) // 4,
            )

        if task.category == "structured_json":
            expected_vals = dict(task.reference.get("expected_values", {}))
            if not fail:
                payload = json.dumps(expected_vals)
            else:
                # Realistic schema violation: wrong field type (string instead of int/float)
                bad_payload = {k: str(v) if isinstance(v, (int, float, bool)) else 12345 for k, v in expected_vals.items()}
                payload = json.dumps(bad_payload)
            return AgentResponse(
                task_id=task.task_id,
                output_text=payload,
                latency_ms=latency,
                prompt_tokens=prompt_tok,
                completion_tokens=len(payload) // 4,
            )

        raise ValueError(f"Unsupported task category: {task.category}")


class OpenAICompatibleAdapter(BaseAgentAdapter):
    """Live HTTP adapter for OpenAI, vLLM, or local Ollama OpenAI-compatible endpoints."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.name = model
        self.model = model
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature

    def generate(self, task: EvalTask) -> AgentResponse:
        messages = []
        if task.context:
            messages.append({"role": "system", "content": f"Use only this context:\n{task.context}"})
        messages.append({"role": "user", "content": task.prompt})

        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body, headers=headers)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            latency_ms = (time.perf_counter() - t0) * 1000.0
            msg = data["choices"][0]["message"]
            usage = data.get("usage", {})
            raw_tools = msg.get("tool_calls") or []
            parsed_tools = [
                ToolCall(
                    name=tc["function"]["name"],
                    arguments=json.loads(tc["function"].get("arguments", "{}")),
                )
                for tc in raw_tools
            ]
            return AgentResponse(
                task_id=task.task_id,
                output_text=msg.get("content") or "",
                tool_calls=parsed_tools,
                latency_ms=latency_ms,
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            )
        except Exception as exc:
            return AgentResponse(
                task_id=task.task_id,
                output_text="",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                error=str(exc),
            )
