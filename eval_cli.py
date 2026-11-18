"""CLI entrypoint for reproducible agent evaluations and CI regression gating."""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.adapters import DeterministicBenchmarkAdapter, OpenAICompatibleAdapter
from src.harness import (
    evaluate_regression_gate,
    export_reports,
    load_eval_config,
    load_tasks_jsonl,
    run_single_adapter,
)


def _resolve_adapter(spec: str):
    if spec in {"baseline_v1", "candidate_v2", "candidate_v2_regressed"}:
        return DeterministicBenchmarkAdapter(profile=spec)
    if spec.startswith("openai:"):
        return OpenAICompatibleAdapter(model=spec.split(":", 1)[1])
    raise ValueError(
        f"Unsupported adapter '{spec}'. Use baseline_v1, candidate_v2, "
        "candidate_v2_regressed, or openai:<model_name>."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic LLM & AI Agent Evaluation Harness with CI Regression Gating"
    )
    parser.add_argument("--config", default="configs/eval_suite.yaml", help="Path to YAML eval suite config")
    parser.add_argument("--baseline", default="baseline_v1", help="Baseline adapter identifier")
    parser.add_argument("--candidate", default="candidate_v2", help="Candidate adapter identifier")
    parser.add_argument("--output-dir", default="reports", help="Directory for JSON/Markdown scorecards")
    args = parser.parse_args()

    config_path = os.path.join(ROOT_DIR, args.config) if not os.path.isabs(args.config) else args.config
    dataset_rel, gate_spec, timeout_sec = load_eval_config(config_path)
    dataset_path = os.path.join(ROOT_DIR, dataset_rel) if not os.path.isabs(dataset_rel) else dataset_rel
    output_dir = os.path.join(ROOT_DIR, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir

    tasks = load_tasks_jsonl(dataset_path)
    baseline_adapter = _resolve_adapter(args.baseline)
    candidate_adapter = _resolve_adapter(args.candidate)

    print("=" * 76)
    print(f"LLM AGENT EVAL HARNESS | Tasks: {len(tasks)} | {args.candidate} vs. {args.baseline}")
    print("=" * 76)

    baseline_run = run_single_adapter(baseline_adapter, tasks, timeout_sec=timeout_sec)
    candidate_run = run_single_adapter(candidate_adapter, tasks, timeout_sec=timeout_sec)
    gate_result = evaluate_regression_gate(baseline_run, candidate_run, gate_spec)

    json_path, md_path = export_reports(tasks, baseline_run, candidate_run, gate_result, output_dir)

    b_mean = baseline_run["overall_score"]["mean"]
    c_mean = candidate_run["overall_score"]["mean"]
    perm = gate_result["paired_permutation_test"]
    mcn = gate_result["mcnemar_test"]

    print(f"Baseline  ({args.baseline}): score={b_mean:.4f} | pass={baseline_run['overall_pass_rate']['mean']*100:.1f}% | p95={baseline_run['operational_metrics']['p95_latency_ms']:.1f} ms")
    print(f"Candidate ({args.candidate}): score={c_mean:.4f} | pass={candidate_run['overall_pass_rate']['mean']*100:.1f}% | p95={candidate_run['operational_metrics']['p95_latency_ms']:.1f} ms")
    print(f"Delta: {perm['mean_delta']:+.4f} | Permutation p={perm['p_value']:.5f} | McNemar p={mcn['p_value']:.5f} (wins={mcn['improvements_c']}, losses={mcn['regressions_b']})")
    print(f"Reports written: {json_path} & {md_path}")

    if gate_result["gate_passed"]:
        print("\n[GATE: PASS] Candidate meets all quality, significance, and latency SLAs.")
        return 0

    print("\n[GATE: BLOCKED] Candidate failed regression gate:")
    for err in gate_result["failures"]:
        print(f"  - {err}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
