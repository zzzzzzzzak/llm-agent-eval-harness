import pytest

from src.schemas import AgentResponse, ToolCall
from src.scorers import (
    estimate_pass_at_k,
    score_code_synthesis,
    score_grounded_qa,
    score_structured_json,
    score_tool_trajectory,
)


def test_pass_at_k_combinatorial_exactness():
    assert estimate_pass_at_k(n=10, c=0, k=1) == 0.0
    assert estimate_pass_at_k(n=10, c=10, k=1) == 1.0
    assert abs(estimate_pass_at_k(n=10, c=5, k=1) - 0.5) < 1e-9
    # C(5, 2) / C(10, 2) = 10 / 45 => 1 - 10/45 = 35/45
    assert abs(estimate_pass_at_k(n=10, c=5, k=2) - (35.0 / 45.0)) < 1e-9
    assert estimate_pass_at_k(n=10, c=8, k=5) == 1.0


def test_code_synthesis_pass_fail_and_timeout_isolation():
    ref = {"test_code": "assert add(2, 3) == 5\nassert add(-1, 1) == 0"}

    good = AgentResponse(task_id="c1", output_text="```python\ndef add(a, b): return a + b\n```")
    passed, score, _ = score_code_synthesis(good, ref, timeout_sec=2.0)
    assert passed is True and score == 1.0

    bad = AgentResponse(task_id="c2", output_text="def add(a, b): return a - b")
    passed, score, details = score_code_synthesis(bad, ref, timeout_sec=2.0)
    assert passed is False and score == 0.0 and details["exit_code"] != 0

    hang = AgentResponse(task_id="c3", output_text="def add(a, b):\n    while True: pass")
    passed, score, details = score_code_synthesis(hang, ref, timeout_sec=0.5)
    assert passed is False and score == 0.0 and "timeout" in details["failure_reason"]


def test_tool_trajectory_scorer():
    ref = {
        "expected_tools": [
            {"name": "fetch_metrics", "arguments": {"service": "auth"}},
            {"name": "create_incident", "arguments": {"severity": "SEV2"}},
        ]
    }
    perfect = AgentResponse(
        task_id="t1",
        output_text="",
        tool_calls=[
            ToolCall("fetch_metrics", {"service": "auth"}),
            ToolCall("create_incident", {"severity": "SEV2"}),
        ],
    )
    passed, score, _ = score_tool_trajectory(perfect, ref)
    assert passed is True and score == 1.0

    wrong_order = AgentResponse(
        task_id="t2",
        output_text="",
        tool_calls=[
            ToolCall("create_incident", {"severity": "SEV2"}),
            ToolCall("fetch_metrics", {"service": "auth"}),
        ],
    )
    passed, score, details = score_tool_trajectory(wrong_order, ref)
    assert passed is False and details["order_score"] == 0.0


def test_grounded_qa_catches_hallucinated_numbers():
    ctx = "Model compression reduced memory from 49.06 MB down to 6.56 MB (7.47x reduction)."
    ref = {"required_facts": ["6.56 MB", "7.47x"]}

    faithful = AgentResponse(task_id="q1", output_text="Memory dropped to 6.56 MB, a 7.47x reduction.")
    passed, score, details = score_grounded_qa(faithful, ref, ctx)
    assert passed is True and score == 1.0 and details["ungrounded_numbers"] == []

    hallucinated = AgentResponse(
        task_id="q2",
        output_text="Memory dropped to 6.56 MB (7.47x) and boosted throughput by 412%.",
    )
    passed, score, details = score_grounded_qa(hallucinated, ref, ctx)
    assert passed is False and "412" in details["ungrounded_numbers"]


def test_structured_json_schema_and_bool_int_guard():
    ref = {
        "schema": {"service": "str", "severity": "int", "resolved": "bool"},
        "expected_values": {"service": "api", "severity": 1, "resolved": True},
    }
    valid = AgentResponse(task_id="j1", output_text='{"service": "api", "severity": 1, "resolved": true}')
    passed, score, _ = score_structured_json(valid, ref)
    assert passed is True and score == 1.0

    # Ensure Python bool is rejected where int is required
    bool_as_int = AgentResponse(task_id="j2", output_text='{"service": "api", "severity": true, "resolved": true}')
    passed, score, details = score_structured_json(bool_as_int, ref)
    assert passed is False and details["schema_valid"] is False
