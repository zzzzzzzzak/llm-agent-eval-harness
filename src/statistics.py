"""Statistical inference engine for rigorous model evaluation and regression gating.

Provides:
- Deterministic bootstrap 95% confidence intervals
- Non-parametric paired permutation test for continuous task scores
- Exact binomial / continuity-corrected McNemar's test for paired pass/fail rates
- Operational latency percentiles (p50, p95, p99) and token cost profiling
"""

from typing import Any, Dict, List, Sequence
import numpy as np
from scipy.stats import binomtest, chi2

from src.schemas import TaskEvalResult


def bootstrap_ci(
    scores: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute deterministic percentile bootstrap confidence interval for mean score."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    sample_means = arr[idx].mean(axis=1)

    alpha = 1.0 - confidence
    ci_low = float(np.percentile(sample_means, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(sample_means, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "mean": round(float(arr.mean()), 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
    }


def paired_permutation_test(
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    n_permutations: int = 5000,
    seed: int = 42,
) -> Dict[str, float]:
    """Two-sided paired permutation test comparing candidate vs. baseline on identical tasks."""
    base = np.asarray(baseline_scores, dtype=np.float64)
    cand = np.asarray(candidate_scores, dtype=np.float64)
    if base.shape != cand.shape:
        raise ValueError(f"Mismatched score lengths: {base.shape} vs {cand.shape}")

    diffs = cand - base
    obs_delta = float(diffs.mean())

    if np.allclose(diffs, 0.0):
        return {"mean_delta": 0.0, "p_value": 1.0}

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, diffs.size))
    perm_means = (signs * diffs).mean(axis=1)

    # +1 smoothing avoids reporting exact 0.0 p-value under finite Monte Carlo sampling
    extreme_count = int(np.sum(np.abs(perm_means) >= abs(obs_delta) - 1e-12))
    p_value = float((extreme_count + 1) / (n_permutations + 1))

    return {
        "mean_delta": round(obs_delta, 4),
        "p_value": round(p_value, 6),
    }


def mcnemar_test(
    baseline_passed: Sequence[bool],
    candidate_passed: Sequence[bool],
) -> Dict[str, Any]:
    """Paired McNemar's test on binary pass/fail outcomes.

    Uses exact two-sided binomial test when discordant pairs (b + c) < 25,
    and continuity-corrected chi-square otherwise.
    """
    if len(baseline_passed) != len(candidate_passed):
        raise ValueError("Baseline and candidate pass arrays must have equal length.")

    b = sum(1 for bp, cp in zip(baseline_passed, candidate_passed) if bp and not cp)  # Regressions
    c = sum(1 for bp, cp in zip(baseline_passed, candidate_passed) if not bp and cp)  # Improvements
    discordant = b + c

    if discordant == 0:
        return {
            "regressions_b": 0,
            "improvements_c": 0,
            "method": "exact_binomial",
            "p_value": 1.0,
        }

    if discordant < 25:
        res = binomtest(k=min(b, c), n=discordant, p=0.5, alternative="two-sided")
        return {
            "regressions_b": b,
            "improvements_c": c,
            "method": "exact_binomial",
            "p_value": round(float(res.pvalue), 6),
        }

    chi2_stat = ((abs(b - c) - 1.0) ** 2) / discordant
    p_value = float(chi2.sf(chi2_stat, df=1))
    return {
        "regressions_b": b,
        "improvements_c": c,
        "method": "chi2_yates",
        "chi2_statistic": round(float(chi2_stat), 4),
        "p_value": round(p_value, 6),
    }


def summarize_operational_metrics(results: List[TaskEvalResult]) -> Dict[str, float]:
    """Compute latency percentiles and token usage summary for a run."""
    if not results:
        return {"p50_latency_ms": 0.0, "p95_latency_ms": 0.0, "p99_latency_ms": 0.0, "mean_tokens": 0.0}

    latencies = np.array([r.latency_ms for r in results], dtype=np.float64)
    tokens = np.array([r.total_tokens for r in results], dtype=np.float64)

    return {
        "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 2),
        "mean_tokens": round(float(tokens.mean()), 1),
        "total_tokens": int(tokens.sum()),
    }
