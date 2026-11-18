"""Objective programmatic evaluation scorers for frontier LLMs and AI agents.

Implements:
- Unbiased combinatorial Pass@k estimator (Chen et al., Codex)
- Subprocess-isolated Python code execution scorer with strict wall-clock timeout
- Multi-step tool-call trajectory evaluator (F1, argument match, causal ordering)
- Numerical & factual groundedness (anti-hallucination) scorer
- Strict JSON schema & type-contract validator
"""

import json
import math
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from src.schemas import AgentResponse, EvalTask, TaskEvalResult, ToolCall


TYPE_MAP = {
    "str": str,
    "int": int,
    "float": (int, float),
    "bool": bool,
    "list": list,
    "dict": dict,
}


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased Pass@k estimator: 1 - C(n - c, k) / C(n, k)."""
    if k <= 0 or n <= 0 or k > n:
        raise ValueError(f"Invalid parameters: n={n}, c={c}, k={k}")
    if c < 0 or c > n:
        raise ValueError(f"Correct count c={c} must lie in [0, n={n}]")
    if n - c < k:
        return 1.0
    return float(1.0 - (math.comb(n - c, k) / math.comb(n, k)))


def _extract_code_block(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def score_code_synthesis(
    response: AgentResponse,
    reference: Dict[str, Any],
    timeout_sec: float = 3.0,
) -> Tuple[bool, float, Dict[str, Any]]:
    """Execute generated Python code against hidden assertions in an isolated subprocess."""
    if response.error:
        return False, 0.0, {"failure_reason": f"adapter_error: {response.error}"}

    code = _extract_code_block(response.output_text)
    test_code = reference.get("test_code", "")
    full_script = f"{code}\n\n{test_code}\n"

    try:
        proc = subprocess.run(
            [sys.executable, "-c", full_script],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if proc.returncode == 0:
            return True, 1.0, {"exit_code": 0, "status": "all_assertions_passed"}
        stderr_tail = (proc.stderr or "").strip().splitlines()[-3:]
        return False, 0.0, {
            "exit_code": proc.returncode,
            "failure_reason": " | ".join(stderr_tail) if stderr_tail else "runtime_error",
        }
    except subprocess.TimeoutExpired:
        return False, 0.0, {"failure_reason": f"timeout_exceeded_{timeout_sec}s"}


def score_tool_trajectory(
    response: AgentResponse,
    reference: Dict[str, Any],
) -> Tuple[bool, float, Dict[str, Any]]:
    """Score multi-step agent tool-calling trajectories across selection, arguments, and order."""
    expected: List[Dict[str, Any]] = reference.get("expected_tools", [])
    actual: List[ToolCall] = response.tool_calls

    if not expected:
        passed = len(actual) == 0
        return passed, (1.0 if passed else 0.0), {"expected_count": 0, "actual_count": len(actual)}

    exp_names = [t["name"] for t in expected]
    act_names = [t.name for t in actual]

    # Multiset tool selection precision & recall
    exp_remaining = list(exp_names)
    matched_names = 0
    for name in act_names:
        if name in exp_remaining:
            matched_names += 1
            exp_remaining.remove(name)

    precision = matched_names / len(act_names) if act_names else 0.0
    recall = matched_names / len(exp_names) if exp_names else 0.0
    tool_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Argument exactness for matched tool steps
    arg_matches = 0
    for idx, exp_step in enumerate(expected):
        exp_args = exp_step.get("arguments", {})
        if idx < len(actual) and actual[idx].name == exp_step["name"]:
            act_args = actual[idx].arguments
            if all(act_args.get(k) == v for k, v in exp_args.items()):
                arg_matches += 1
    arg_accuracy = arg_matches / len(expected)

    # Causal ordering preservation (exact subsequence / prefix alignment)
    order_hits = sum(
        1 for i, name in enumerate(exp_names) if i < len(act_names) and act_names[i] == name
    )
    order_score = order_hits / len(exp_names)

    composite = round(0.40 * tool_f1 + 0.40 * arg_accuracy + 0.20 * order_score, 4)
    passed = composite >= 0.90

    return passed, composite, {
        "tool_f1": round(tool_f1, 4),
        "arg_accuracy": round(arg_accuracy, 4),
        "order_score": round(order_score, 4),
        "expected_tools": exp_names,
        "actual_tools": act_names,
    }


def _extract_numbers(text: str) -> List[str]:
    raw = re.findall(r"(?<![A-Za-z0-9_])\$?(-?\d+(?:,\d{3})*(?:\.\d+)?%?)", text)
    return [num.replace(",", "").rstrip("%") for num in raw]


def score_grounded_qa(
    response: AgentResponse,
    reference: Dict[str, Any],
    context: Optional[str],
) -> Tuple[bool, float, Dict[str, Any]]:
    """Verify answer correctness and strict numerical groundedness against retrieved context."""
    answer = response.output_text or ""
    ctx = context or ""

    required_facts: List[str] = reference.get("required_facts", [])
    fact_hits = sum(1 for fact in required_facts if fact.lower() in answer.lower())
    fact_recall = (fact_hits / len(required_facts)) if required_facts else 1.0

    # Groundedness check: every numeric claim in the answer must exist in context
    ctx_numbers = set(_extract_numbers(ctx))
    ans_numbers = _extract_numbers(answer)
    ungrounded = [n for n in ans_numbers if n not in ctx_numbers]
    hallucination_free = len(ungrounded) == 0

    faithfulness = 1.0 if not ans_numbers else (len(ans_numbers) - len(ungrounded)) / len(ans_numbers)
    composite = round(0.55 * fact_recall + 0.45 * faithfulness, 4)
    passed = (fact_recall >= 1.0) and hallucination_free

    return passed, composite, {
        "fact_recall": round(fact_recall, 4),
        "faithfulness": round(faithfulness, 4),
        "ungrounded_numbers": ungrounded,
    }


def score_structured_json(
    response: AgentResponse,
    reference: Dict[str, Any],
) -> Tuple[bool, float, Dict[str, Any]]:
    """Validate strict JSON syntax, required schema types, and exact reference values."""
    text = response.output_text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    payload_str = fence.group(1).strip() if fence else text

    try:
        parsed = json.loads(payload_str)
    except json.JSONDecodeError as exc:
        return False, 0.0, {"schema_valid": False, "error": f"json_decode_error: {exc.msg}"}

    if not isinstance(parsed, dict):
        return False, 0.0, {"schema_valid": False, "error": "root_must_be_json_object"}

    required_schema: Dict[str, str] = reference.get("schema", {})
    expected_values: Dict[str, Any] = reference.get("expected_values", {})

    type_checks = 0
    for key, type_name in required_schema.items():
        if key in parsed:
            expected_cls = TYPE_MAP.get(type_name, object)
            val = parsed[key]
            # Note: bool is subclass of int in Python; guard explicitly
            if type_name == "int" and isinstance(val, bool):
                continue
            if isinstance(val, expected_cls):
                type_checks += 1

    schema_score = (type_checks / len(required_schema)) if required_schema else 1.0

    val_checks = 0
    for key, exp_val in expected_values.items():
        if key in parsed and parsed[key] == exp_val:
            val_checks += 1
    value_score = (val_checks / len(expected_values)) if expected_values else 1.0

    composite = round(0.50 * schema_score + 0.50 * value_score, 4)
    passed = (schema_score == 1.0) and (value_score == 1.0)

    return passed, composite, {
        "schema_valid": schema_score == 1.0,
        "schema_score": round(schema_score, 4),
        "value_score": round(value_score, 4),
    }


def evaluate_task(task: EvalTask, response: AgentResponse, timeout_sec: float = 3.0) -> TaskEvalResult:
    """Dispatch task to its category-specific objective scorer."""
    if task.category == "code_synthesis":
        passed, score, details = score_code_synthesis(response, task.reference, timeout_sec=timeout_sec)
    elif task.category == "tool_trajectory":
        passed, score, details = score_tool_trajectory(response, task.reference)
    elif task.category == "grounded_qa":
        passed, score, details = score_grounded_qa(response, task.reference, task.context)
    elif task.category == "structured_json":
        passed, score, details = score_structured_json(response, task.reference)
    else:
        raise ValueError(f"Unsupported task category: {task.category}")

    return TaskEvalResult(
        task_id=task.task_id,
        category=task.category,
        passed=passed,
        score=score,
        details=details,
        latency_ms=round(response.latency_ms, 2),
        total_tokens=response.prompt_tokens + response.completion_tokens,
    )
