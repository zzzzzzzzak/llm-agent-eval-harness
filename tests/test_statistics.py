from src.statistics import bootstrap_ci, mcnemar_test, paired_permutation_test


def test_bootstrap_ci_determinism_and_bounds():
    scores = [0.0, 1.0, 1.0, 0.5, 1.0, 0.8, 0.9, 1.0]
    ci1 = bootstrap_ci(scores, confidence=0.95, n_resamples=1000, seed=42)
    ci2 = bootstrap_ci(scores, confidence=0.95, n_resamples=1000, seed=42)

    assert ci1 == ci2
    assert ci1["ci_low"] <= ci1["mean"] <= ci1["ci_high"]


def test_paired_permutation_test_identity_and_signal():
    base = [0.5, 0.6, 0.4, 0.7, 0.5, 0.6, 0.5, 0.4, 0.6, 0.5]
    same = paired_permutation_test(base, base)
    assert same["mean_delta"] == 0.0
    assert same["p_value"] == 1.0

    better = [x + 0.3 for x in base]
    sig = paired_permutation_test(base, better, n_permutations=2000, seed=42)
    assert sig["mean_delta"] == 0.3
    assert sig["p_value"] < 0.01


def test_mcnemar_exact_and_yates_branches():
    # Exact binomial branch (discordant < 25)
    base = [False] * 12 + [True] * 8
    cand = [True] * 12 + [True] * 8  # 12 improvements, 0 regressions
    res_small = mcnemar_test(base, cand)
    assert res_small["method"] == "exact_binomial"
    assert res_small["improvements_c"] == 12
    assert res_small["regressions_b"] == 0
    assert res_small["p_value"] < 0.001

    # Chi-square Yates branch (discordant >= 25)
    base_lg = [False] * 30 + [True] * 10
    cand_lg = [True] * 30 + [True] * 10
    res_lg = mcnemar_test(base_lg, cand_lg)
    assert res_lg["method"] == "chi2_yates"
    assert res_lg["p_value"] < 0.001
