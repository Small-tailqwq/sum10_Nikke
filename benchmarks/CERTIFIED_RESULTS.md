# Sum10: certified full-clear optimization, 2026-10-02

## Publication follow-up

Publication checks: **424/424 with Numba, 424/424 with JIT disabled**, 32 focused eye/hand/import tests within that suite, and six Node VM front-end scenarios. All 64 default numerical outputs still match the frozen candidate. Actual Windows hardware/gameplay remains untested.

The frozen measurements below are preserved. The publication version additionally fixes OCR-zero handling, optional-Numba fallback, package/script imports and failure/proof messages without changing the default numerical search. See [eye/hand compatibility](../Head/COMPATIBILITY.md) and [reproducible data](certified/README.md). Hardware details and unavailable CPU quota are recorded in [platform metadata](certified/results/platform_publication.json).

## Result

On **32 previously unseen 16×10 boards, tested with two solver seeds each**, the new candidate returned **64/64 independently verified full-clear paths**. Original and previous optimized God search each cleared **22/64**. Every board has a separate legal full-clear certificate, so a verified score of 160 is globally optimal.

All three versions used beam 200, one base search, no restarts/repairs in this primary comparison. Candidate was frozen before heldout board-only inputs were released. Timing began immediately before target JSON ingress to a ready worker and ended after the independently validated complete result JSON was received. Warmup used a different 20-cell board. This includes target decoding, conversion, search, reconstruction, validation, output serialization and process-pipe dispatch.

| Version | Full clears | Median complete-output time | Interpretation |
|---|---:|---:|---|
| Original 8cb00a3 God |22/64|2.225s|42 best-found failures; not optimum timing samples|
| Previous optimized God |22/64|0.934s|Same scores and paths as original under matching seeds|
| New complete-move solver |64/64|0.140s|All160/160, range0.121–0.161s|

Among the **22 paired runs where both old and new versions actually solved the board**, median per-pair time-to-optimum speedup was **6.71× over previous optimized** and **16.90× over original**. Failed old runs were excluded from this speedup statistic. Comparing the median durations of all 64 calls alone would conflate solved and unsolved output, so it is not our time-to-optimum claim.

## Coverage and limitations

There are 16 development boards, 32 heldout boards, and 3 separate warmup boards across four construction families. All 51 full-clear witnesses are independently replayed with a pure-Python direct-rectangle checker. The optimizer receives only board digits, dimensions, seed and search settings; it never receives a witness. Board/certificate/seed hashes were committed before evaluation; the freeze record is `certified/evaluation/candidate_freeze.json` (2026-10-02T06:58:44Z).

| Family | Heldout board/seed runs | Old/previous full clears | New full clears |
|---|---:|---:|---:|
| Rectangle carving |16|2|16|
| Orthogonal weave |16|3|16|
| Layered anchors |16|14|16|
| Higher-digit pair weave |16|3|16|

The first three distributions average about 3.65 per digit; the higher-digit holdout averages 4.60, closer to the public arena's 4.82. Witnesses include dependent and long-range overlapping rectangles. Three families have certificates fully supported by the original live-opposite-corner scanner; layered anchors also covers moves that scanner can miss.

**These are constructively correlated guaranteed-clearable distributions, not uniform random boards and not proof of equal difficulty to every real board.** Largest dimensions match the original16×10 scale; the source never provides a formal “hardest” difficulty designation. Heldout success establishes this corpus result, not a universal solve guarantee. The 32 boards, not 64 repeated seed runs, are the distinct test instances. The holdout embargo is operational discipline, not a security sandbox.

## Cold starts and cooperative budgets

Fresh processes with empty Numba caches were timed from before launch through validated output, using three predeclared heldout boards. No target or other board was warmed in cold mode.

| Version | Cold output times | Full clears |
|---|---|---:|
| Original |3.64–4.67s|1/3|
| Previous optimized |6.24–6.72s|1/3|
| Candidate |5.10–5.45s|3/3|

Cold compilation remains material: the candidate is slower than original on first output, although it solves more boards. Warmed speedups must not be presented as cold speedups.

On eight separately predeclared heldout cases with nominal 3-second budgets, original/previous each solved 5/8 and candidate 8/8. Candidate median 0.141s; original median 4.231s, maximum 6.112s; previous median 2.076s, maximum 3.626s. The legacy comparison wrapper restarts complete base searches between budget checks, so it can overshoot substantially. Candidate checks between compiled layers; it too is cooperative, not hard real-time. Actual elapsed time, not just the nominal budget, is retained in every raw trial.

A real two-process `spawn` adapter check, warmed on another board, returned the first independently verified full clear in 0.124s and both in 0.127s. This includes submission, conversion, search, reconstruction, validation and return IPC. It is a correctness/dispatch check on one board, not an independent performance distribution. Full browser/WebSocket/OCR/gameplay latency has not been benchmarked.

## Real public test arena

Source: [Small-tailqwq/sum10_Nikke issue1](https://github.com/Small-tailqwq/sum10_Nikke/issues/1), including all three comments. No issue image or solution witness was present. The exact board sums to 771. Every legal move removes value 10, so a fullclear is impossible. A residual digit 1 gives an optimistic cell-count upper bound of 159; attainability of 159 was not established.

The author's actual target is≥144/160, with V4 under 30s on a 4800H and beam 500–1000. A second setting repeats “BEAM WIDTH: 16”; it may mean threads but that typo is unresolved. The following is our same-machine, single-worker comparison at beam 500, seeds 17/42/97; it does not reproduce an unspecified 16-worker 4800H setup.

| Algorithm | First complete outputs reaching 144 | Times for successful outputs | Nominal 3s best scores, seeds 17/42/97 |
|---|---:|---|---|
|Original God|1/3|5.060s|144/142/141|
|Previous optimized God|2/3|2.347s, 4.518s|144/142/144|
|Upstream V4 God|1/3|3.693s|145/139/143|
|New candidate|3/3|0.337s, 0.317s, 0.325s|157/158/156|

“Time to 144” means the first complete independently validated result exposed by each adapter; it is not an instrumented internal first-hit timestamp. Old runs that missed 144 are failures and are never treated as successful target times. Old base calls can exceed 3s before returning; see raw actual times.

The frozen candidate produced scores 157/158/156 at approximately 3.002s. At an additional 30-second budget, with no algorithm changes, all three seeds produced **158/160 (98.75%)**. These are best-found results, **not proved optimal**; the gap to the 159 upper bound remains 1 cell. Timed attempts vary by hardware/load. This 30-second diagnostic was declared after primary holdout evaluation; it did not drive any tuning.

The original production Hydra rollback wrapper was also tested unchanged with nominal 3s. Scores were 144/142/141 (original) and 144/143/141 (previous optimized); actual runtime ranged 3.57–7.85s, reflecting its existing base/repair overshoot and reward-extension behavior. This confirms the strong arena result is not solely an artifact of comparing only legacy base restarts. See `certified/results/arena_legacy_hydra.json`.

## What changed

- Enumerate every distinct legal rectangle removal using row bands and positive-sum column windows, including legal shapes with empty opposite corners
- Represent live cells in three uint64 masks, comparing full masks for exact deduplication; hashes never establish equality alone
- Keep only a unique top beam in compiled code and reconstruct final paths with parent pointers
- Prefer preserving useful small digits at equal removed value, rather than immediately maximizing removed-cell count
- Add seven necessary digit-multiset inequalities as soft ranking signals
- Stop immediately when a fullclear or proven bound is reached; otherwise support seeded restarts and suffix repair within a cooperative budget
- Add an opt-in CLEAR SEARCH UI mode (`complete`) and headless JSON CLI; existing classic/omni/god defaults are preserved

Development ablation: the initial positive-count prototype cleared 0/12; negative-count ranking alone and negative ranking plus cuts both cleared 12/12. Consequently these experiments do not attribute the improvement to the multiset penalties alone. Complete enumeration, unique states, ranking and compiled data structures change together versus the old solver; this is an algorithm-quality change, not a behavior-preserving 2× micro-optimization.

## Verification

- 392 repository tests passed with Numba
- 392 passed with `NUMBA_DISABLE_JIT=1`; expected uint64 wraparound produced 304 overflow warnings
- 372 independent differential scanner/path/tiny-exhaustive tests passed on each candidate stage, including the final frozen candidate
- 17 corpus checker/corruption/release tests passed; generator outputs reproduced byte-for-byte under different Python hash seeds
- All 51 known-clearable certificates replayed to 160
- Real spawned worker integration and path validation passed
- Source review checked collision handling, heap/tombstone updates, move completeness, parent links, repair prefixes, residue bounds and all 41 possible sum10 digit multisets
- Delivered source adds only a pathological custom-weight magnitude guard and comment corrections after freeze. All 64 heldout default outputs match the frozen candidate exactly (paths, scores, remaining counts, bounds and statistics); no parameter tuning followed holdout release

The frozen algorithm hash is 0734651940e56bf5bbb7c8c4ea6a1af5492a91a779c1cd60676dbce5221bf6a6. The pre-publication integration source hash was 4459ba693b02640c052725a3f7f6ed1583faeb54b3afb7dfc986ff9e87198701; compatibility changes above were made afterward and are tested separately. The numerical input guard rejects pathological parameter magnitudes; publication compatibility changes are documented separately.

The cloud browser blocked the local preview URL, so the new button's visual layout is unverified. OCR, live WebSocket interactions, Windows input and actual gameplay were not exercised. During the original benchmark phase no external repository write or game action occurred. This report is now included in the authorized draft PR.

## Reproduce

Environment: Python 3.12.14, NumPy 2.3.5, Numba 0.68.0, Linux, AMD EPYC 9V74; shared cloud reports 9 logical CPUs and affinity CPUs 0–8. CPU quota files are absent, so dedicated physical core allocation is unknown. Timed runs use one solver process and 1 BLAS/Numba thread. Wall-clock results can vary with load and hardware.

Use this PR branch, based on 8cb00a3f8161359d4c2310de592822b39d125496. It includes the previous behavior-preserving optimization and the opt-in new solver.

```sh
python -m pip install numpy numba pytest
python -m pytest -q tests
NUMBA_DISABLE_JIT=1 python -m pytest -q tests
python benchmarks/solve_certified.py board.json --beam 200 --seconds 3 --seed 17
```

Input is a 2-D integer JSON board or an object with `board`. Zeros denote empty cells. The new solver supports up to 192 cells and 63 columns and requires Numba for the accelerated path.

For benchmark reproduction, use the committed harness and fixed data:

```sh
python benchmarks/certified/benchmark_certified.py --boards benchmarks/certified/evaluation/all_boards.json --variants original,previous,candidate --candidate fast_solver_finalist.py --mode god --beam 200 --single-pass --seeds 17 97 --output reproduced.json
python benchmarks/certified/corpus/check_corpus.py
python benchmarks/certified/corpus/check_corpus.py --root benchmarks/certified/corpus/high_digit_extension
python benchmarks/certified/summarize_results.py
```

`certified/results/summary.json` is mechanically derived from raw trial files. Every completed trial contains its full output path for replay. An initial harness-only warmup target-setting error was fixed before the committed complete measurements; partial runs are excluded. Frozen source snapshots, seed commitments, generation code, certificates and independent checker are included for auditability.
