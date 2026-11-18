"""Reproducible evaluation runner, statistical comparator, and CI regression gatekeeper."""

from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional, Tuple
import yaml

from src.adapters import BaseAgentAdapter
from src.schemas import (
    EvalTask,
    RegressionGateSpec,
    TaskEvalResult,
    compute_dataset_fingerprint,
)
from src.scorers import evaluate_task
from src.statistics import (
    bootstrap_ci,
    mcnemar_test,
    paired_permutation_test,
    summarize_operational_metrics,
)


def load_tasks_jsonl(filepath: str) -> List[EvalTask]:
    tasks: List[EvalTask] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                tasks.append(EvalTask.from_dict(json.loads(stripped)))
    return tasks


def load_eval_config(config_path: str) -> Tuple[str, RegressionGateSpec, float]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dataset_rel = cfg["dataset"]["path"]
    gate_cfg = cfg.get("regression_gate", {})
    gate = RegressionGateSpec(
        min_overall_score=float(gate_cfg.get("min_overall_score", 0.75)),
        max_score_regression=float(gate_cfg.get("max_score_regression", 0.02)),
        significance_alpha=float(gate_cfg.get("significance_alpha", 0.05)),
        max_p95_latency_ms=float(gate_cfg.get("max_p95_latency_ms", 1500.0)),
        min_category_scores=dict(gate_cfg.get("min_category_scores", {})),
    )
    timeout_sec = float(cfg.get("execution", {}).get("code_timeout_sec", 3.0))
    return dataset_rel, gate, timeout_sec


def run_single_adapter(
    adapter: BaseAgentAdapter,
    tasks: List[EvalTask],
    timeout_sec: float = 3.0,
) -> Dict[str, Any]:
    """Execute an adapter deterministically over all tasks and aggregate stats."""
    results: List[TaskEvalResult] = []
    for task in tasks:
        resp = adapter.generate(task)
        res = evaluate_task(task, resp, timeout_sec=timeout_sec)
        results.append(res)

    scores = [r.score for r in results]
    passes = [1.0 if r.passed else 0.0 for r in results]

    overall_score_ci = bootstrap_ci(scores, confidence=0.95)
    overall_pass_ci = bootstrap_ci(passes, confidence=0.95)
    ops = summarize_operational_metrics(results)

    by_cat: Dict[str, List[TaskEvalResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    category_breakdown: Dict[str, Dict[str, Any]] = {}
    for cat, cat_res in sorted(by_cat.items()):
        cat_scores = [r.score for r in cat_res]
        cat_passes = [1.0 if r.passed else 0.0 for r in cat_res]
        category_breakdown[cat] = {
            "count": len(cat_res),
            "mean_score": bootstrap_ci(cat_scores)["mean"],
            "pass_rate": bootstrap_ci(cat_passes)["mean"],
        }

    return {
        "adapter_name": adapter.name,
        "task_count": len(results),
        "overall_score": overall_score_ci,
        "overall_pass_rate": overall_pass_ci,
        "operational_metrics": ops,
        "category_breakdown": category_breakdown,
        "task_results": results,
    }


def evaluate_regression_gate(
    baseline_run: Dict[str, Any],
    candidate_run: Dict[str, Any],
    gate: RegressionGateSpec,
) -> Dict[str, Any]:
    """Compare candidate vs. baseline statistically and enforce CI regression gating."""
    base_res: List[TaskEvalResult] = baseline_run["task_results"]
    cand_res: List[TaskEvalResult] = candidate_run["task_results"]

    base_scores = [r.score for r in base_res]
    cand_scores = [r.score for r in cand_res]
    base_passed = [r.passed for r in base_res]
    cand_passed = [r.passed for r in cand_res]

    perm = paired_permutation_test(base_scores, cand_scores)
    mcnemar = mcnemar_test(base_passed, cand_passed)

    failures: List[str] = []

    cand_mean = candidate_run["overall_score"]["mean"]
    if cand_mean < gate.min_overall_score:
        failures.append(
            f"Overall score {cand_mean:.4f} dropped below minimum floor {gate.min_overall_score:.4f}"
        )

    if perm["mean_delta"] < -gate.max_score_regression and perm["p_value"] < gate.significance_alpha:
        failures.append(
            f"Statistically significant regression detected: delta={perm['mean_delta']:+.4f} "
            f"(p={perm['p_value']:.5f} < alpha={gate.significance_alpha})"
        )

    cand_p95 = candidate_run["operational_metrics"]["p95_latency_ms"]
    if cand_p95 > gate.max_p95_latency_ms:
        failures.append(
            f"p95 latency {cand_p95:.1f} ms exceeded SLA ceiling {gate.max_p95_latency_ms:.1f} ms"
        )

    for cat, min_cat_score in gate.min_category_scores.items():
        actual_cat = candidate_run["category_breakdown"].get(cat, {}).get("mean_score", 0.0)
        if actual_cat < min_cat_score:
            failures.append(
                f"Category '{cat}' score {actual_cat:.4f} below minimum floor {min_cat_score:.4f}"
            )

    return {
        "gate_passed": len(failures) == 0,
        "failures": failures,
        "paired_permutation_test": perm,
        "mcnemar_test": mcnemar,
    }


def export_reports(
    tasks: List[EvalTask],
    baseline_run: Dict[str, Any],
    candidate_run: Dict[str, Any],
    gate_result: Dict[str, Any],
    output_dir: str,
) -> Tuple[str, str]:
    """Write machine-readable JSON report and executive Markdown scorecard."""
    os.makedirs(output_dir, exist_ok=True)
    fingerprint = compute_dataset_fingerprint(tasks)

    def _strip_raw(run: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in run.items() if k != "task_results"}

    report_payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "dataset_fingerprint_sha256": fingerprint,
        "total_tasks": len(tasks),
        "baseline": _strip_raw(baseline_run),
        "candidate": _strip_raw(candidate_run),
        "comparison_and_gate": gate_result,
    }

    json_path = os.path.join(output_dir, "eval_comparison_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)

    perm = gate_result["paired_permutation_test"]
    mcn = gate_result["mcnemar_test"]
    b_sc = baseline_run["overall_score"]
    c_sc = candidate_run["overall_score"]
    b_op = baseline_run["operational_metrics"]
    c_op = candidate_run["operational_metrics"]
    status_str = "PASS (APPROVED FOR RELEASE)" if gate_result["gate_passed"] else "FAIL (REGRESSION BLOCKED)"

    md_lines = [
        f"# Evaluation Scorecard: `{candidate_run['adapter_name']}` vs. `{baseline_run['adapter_name']}`",
        "",
        f"**Gate Decision:** `{status_str}`  ",
        f"**Dataset Fingerprint (SHA-256):** `{fingerprint}` | **Tasks:** `{len(tasks)}`",
        "",
        "## Headline Comparison (95% Bootstrap CI)",
        "",
        "| Metric | Baseline | Candidate | Delta | Statistical Test |",
        "|---|:---:|:---:|:---:|:---|",
        f"| **Mean Score** | {b_sc['mean']:.4f} `[{b_sc['ci_low']:.3f}, {b_sc['ci_high']:.3f}]` | "
        f"**{c_sc['mean']:.4f}** `[{c_sc['ci_low']:.3f}, {c_sc['ci_high']:.3f}]` | "
        f"**{perm['mean_delta']:+.4f}** | Permutation `p = {perm['p_value']:.5f}` |",
        f"| **Pass Rate** | {baseline_run['overall_pass_rate']['mean'] * 100:.1f}% | "
        f"**{candidate_run['overall_pass_rate']['mean'] * 100:.1f}%** | "
        f"**{(candidate_run['overall_pass_rate']['mean'] - baseline_run['overall_pass_rate']['mean']) * 100:+.1f}%** | "
        f"McNemar `p = {mcn['p_value']:.5f}` (win={mcn['improvements_c']}, loss={mcn['regressions_b']}) |",
        f"| **p95 Latency** | {b_op['p95_latency_ms']:.1f} ms | **{c_op['p95_latency_ms']:.1f} ms** | "
        f"{c_op['p95_latency_ms'] - b_op['p95_latency_ms']:+.1f} ms | SLA Check |",
        "",
        "## Category Breakdown",
        "",
        "| Category | Tasks | Baseline Score | Candidate Score | Delta |",
        "|---|:---:|:---:|:---:|:---:|",
    ]
    for cat in sorted(baseline_run["category_breakdown"].keys()):
        bs = baseline_run["category_breakdown"][cat]["mean_score"]
        cs = candidate_run["category_breakdown"][cat]["mean_score"]
        cnt = baseline_run["category_breakdown"][cat]["count"]
        md_lines.append(f"| `{cat}` | {cnt} | {bs:.4f} | **{cs:.4f}** | {cs - bs:+.4f} |")

    if gate_result["failures"]:
        md_lines.append("\n## Regression Gate Violations\n")
        for fail in gate_result["failures"]:
            md_lines.append(f"- ❌ {fail}")

    md_path = os.path.join(output_dir, "eval_scorecard.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    return json_path, md_path
