# Sum10 solver optimization: measured results

Date: 2026-10-02. Baseline: `8cb00a3f8161359d4c2310de592822b39d125496`.

## Changes

- Batch scanning, stable count sorting, move application and evaluation inside Numba
- Materialize Python state dictionaries and paths only for surviving beam candidates
- Copy surviving map rows out of batch allocations; preserve original move and random draw order
- Cache compiled kernels for reuse; explicitly seed the separate Numba RNG in each worker
- Add a headless benchmark, a preserved original-loop oracle, and regression tests

## Warmed fixed-work performance

Beam 200, depth cap 160 per phase, weights 63.0 / 1.0. Three RNG seeds (17, 42, 97), three repeats per seed, and alternating timing order across trials. Times below are medians in seconds; each row includes nine paired trials.

| Fixture | Mode | Before | After | Speedup | Scores by seed |
|---|---|---:|---:|---:|---|
| repository_16x10 | classic | 1.8323 | 0.8354 | 2.19× | [102, 106, 106] |
| repository_16x10 | omni | 1.5835 | 0.6345 | 2.50× | [105, 107, 105] |
| repository_16x10 | god | 1.9743 | 0.9316 | 2.12× | [143, 141, 141] |
| random_16x10 | classic | 1.0191 | 0.5926 | 1.72× | [68, 72, 68] |
| random_16x10 | omni | 1.0294 | 0.5318 | 1.94× | [97, 97, 97] |
| random_16x10 | god | 1.2084 | 0.6915 | 1.75× | [107, 115, 119] |
| sparse_16x10 | classic | 0.0282 | 0.0210 | 1.35× | [20, 20, 20] |
| sparse_16x10 | omni | 0.0642 | 0.0314 | 2.05× | [31, 31, 31] |
| sparse_16x10 | god | 0.0349 | 0.0289 | 1.21× | [38, 38, 38] |

Across all 81 paired trials: **79.07s → 39.12s**, or **2.02×** faster (50.5% less warmed compute time).

Every trial matched the baseline exactly: score, complete move path, final map, heuristic value and subsequent RNG draw. Each path was independently replayed for rectangle bounds, sum 10, removed-cell count and score. These results are for fixed work, not a claim of better solutions or strict wall-clock deadline compliance.

First-ever warm-up/compilation in this harness: baseline **1.29s**, optimized **4.73s**. This one-time cost is higher; cached workers can reuse compiled code. Do not apply warmed speedups directly to very short first-run tasks.

## Validation

- 284 tests passed with Numba enabled (6.25s)
- 284 tests passed with `NUMBA_DISABLE_JIT=1` (9.80s)
- Two real `spawn` ProcessPoolExecutor workers returned the same valid full-clear solution (12/12 cells) for the same seed
- `compileall` passed for the modified solver, benchmark helpers and tests
- `git diff --check` passed

Coverage includes classic/omni/two-phase god, multiple beams/depths/personalities, dense/sparse/empty/no-move boards, active zero values, duplicate moves, stable ties, survivor ownership, worker seed normalization, and the score 80/81/119/120/121 beam/window thresholds.

## Reproduce and scope

```sh
python -m pip install numpy numba pytest
python -m pytest -q tests
NUMBA_DISABLE_JIT=1 python -m pytest -q tests
python benchmarks/benchmark_solver.py --baseline-ref 8cb00a3 --output results.json
```

Environment: Python 3.12.14, NumPy 2.3.5, Numba 0.68.0; Linux-6.18.44-x86_64-with-glibc2.41. Shared-cloud wall-clock timings vary with hardware and load. Raw measurements are in `results-2026-10-02.json`.

See [README.md](README.md) for limitations deliberately preserved from the baseline, including incomplete omni move enumeration, duplicate states, classic rollback using omni, and non-strict time limits. OCR, WebSocket behavior, Windows input, and game integration were not validated. No push, PR, deployment or external repository change was made.
