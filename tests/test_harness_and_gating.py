import os

from src.adapters import DeterministicBenchmarkAdapter
from src.harness import (
    evaluate_regression_gate,
    load_eval_config,
    load_tasks_jsonl,
    run_single_adapter,
)
from src.schemas import compute_dataset_fingerprint

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "configs", "eval_suite.yaml")


def test_dataset_loading_and_deterministic_fingerprint():
    dataset_rel, _, _ = load_eval_config(CONFIG_PATH)
    tasks = load_tasks_jsonl(os.path.join(ROOT_DIR, dataset_rel))
    assert len(tasks) == 60
    categories = {t.category for t in tasks}
    assert categories == {"code_synthesis", "tool_trajectory", "grounded_qa", "structured_json"}
    assert compute_dataset_fingerprint(tasks) == compute_dataset_fingerprint(tasks)


def test_regression_gate_approves_improved_candidate_and_blocks_regressed_candidate():
    dataset_rel, gate_spec, timeout_sec = load_eval_config(CONFIG_PATH)
    tasks = load_tasks_jsonl(os.path.join(ROOT_DIR, dataset_rel))

    base_run = run_single_adapter(DeterministicBenchmarkAdapter("baseline_v1"), tasks, timeout_sec)
    good_run = run_single_adapter(DeterministicBenchmarkAdapter("candidate_v2"), tasks, timeout_sec)
    bad_run = run_single_adapter(DeterministicBenchmarkAdapter("candidate_v2_regressed"), tasks, timeout_sec)

    gate_good = evaluate_regression_gate(base_run, good_run, gate_spec)
    assert gate_good["gate_passed"] is True
    assert gate_good["paired_permutation_test"]["mean_delta"] > 0.05
    assert gate_good["paired_permutation_test"]["p_value"] < 0.05

    gate_bad = evaluate_regression_gate(base_run, bad_run, gate_spec)
    assert gate_bad["gate_passed"] is False
    assert len(gate_bad["failures"]) >= 2
    assert any("regression" in msg.lower() for msg in gate_bad["failures"])
    assert any("latency" in msg.lower() for msg in gate_bad["failures"])
