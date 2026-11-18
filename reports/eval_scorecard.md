# Evaluation Scorecard: `candidate_v2` vs. `baseline_v1`

**Gate Decision:** `PASS (APPROVED FOR RELEASE)`  
**Dataset Fingerprint (SHA-256):** `fda65077f0588ca4` | **Tasks:** `60`

## Headline Comparison (95% Bootstrap CI)

| Metric | Baseline | Candidate | Delta | Statistical Test |
|---|:---:|:---:|:---:|:---|
| **Mean Score** | 0.8668 `[0.788, 0.936]` | **0.9728** `[0.939, 1.000]` | **+0.1060** | Permutation `p = 0.00260` |
| **Pass Rate** | 80.0% | **95.0%** | **+15.0%** | McNemar `p = 0.00391` (win=9, loss=0) |
| **p95 Latency** | 680.5 ms | **578.2 ms** | -102.3 ms | SLA Check |

## Category Breakdown

| Category | Tasks | Baseline Score | Candidate Score | Delta |
|---|:---:|:---:|:---:|:---:|
| `code_synthesis` | 15 | 0.8667 | **1.0000** | +0.1333 |
| `grounded_qa` | 15 | 0.8183 | **0.9333** | +0.1150 |
| `structured_json` | 15 | 0.8667 | **1.0000** | +0.1333 |
| `tool_trajectory` | 15 | 0.9156 | **0.9578** | +0.0422 |
