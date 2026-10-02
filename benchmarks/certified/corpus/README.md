# Certified full-clear Sum10 corpus

## What is certified

Every board starts with all **160 cells active on a 16-row × 10-column board**, with integer values 1–9. A move selects inclusive coordinates `[r1,c1,r2,c2]`, sums every currently active cell in that axis-aligned rectangle, and is legal exactly when that sum is 10. It removes all those cells, with no gravity or refill. The objective is removed-cell count. Every board has a directly constructed, independently replayed full-clear witness, so its optimal score is **exactly 160**, the trivial global cell-count upper bound.

The supplied source was inspected at `Head/god_brain_v62.py` (`_fast_scan_rects_v6`, `_apply_move_fast`, `_run_core_search_logic`) and `tests/reference_search.py`. The source’s classic mode additionally requires exactly two removed cells; omni permits two or more. These corpus optimum certificates concern the stated unrestricted rectangle game / omni mode. They do not certify a classic-only full clear.

### Important original-scanner limitation

The original scanner enumerates rectangles from **two active opposite corners**. The game rule itself has no such restriction. A legal rectangle can have no occupied opposite-corner pair. Every certificate move is annotated independently for this distinction in the split’s `move_annotations.json`.

- `rectangle_carving`, `orthogonal_weave`, and `high_digit_pair_weave`: **all certificate moves** are supported by the original pair-anchored scanner
- `layered_anchors`: some late moves can be unanchored; this is deliberately included as coverage of the full game rule
- An unsupported witness move does **not** prove the board has no alternative full-clear path supported by the original scanner

## Inventory and separation

The original corpus is fixed at 39 boards: **3 warmup, 12 development, 24 holdout**, with one, four, and eight boards respectively per original family. The separate high-digit extension adds **4 development and 8 holdout** boards. Combined evaluation therefore has **16 development / 32 holdout**, balanced four/eight per family, plus three separate warmup boards.

| Root | Split | Boards | Average digit | Average witness moves | Average dependent moves |
|---|---|---:|---:|---:|---:|
| Original | Warmup | 3 | 3.562500 | 57.00 | 21.33 |
| Original | Development | 12 | 3.671875 | 58.75 | 19.50 |
| Original | Holdout | 24 | 3.645833 | 58.33 | 20.08 |
| High-digit extension | Development | 4 | 4.640625 | 74.25 | 32.25 |
| High-digit extension | Holdout | 8 | 4.601562 | 73.63 | 30.50 |

A dependent move’s rectangle contains positive-valued cells that were removed by previous moves, so that exact rectangle was **not initially legal**. Every board has such dependencies and long-range moves. Across the original heldout corpus, witness rectangles span up to the entire 16×10 board, dependency chains reach depth 10, and rectangles can cover up to 153 previously removed cells. The high-digit development boards have 30–33 dependent moves and 5–8 long-range moves each.

`manifest.json` commits canonical JSON SHA-256 hashes of boards, certificates, and seed lists. `high_digit_extension/manifest.json` independently commits the extension; the original boards and commitments were **not changed** when the extension was added. All 51 boards across the two corpora are distinct. Seeds use domain-separated SHA-256 of corpus version, split, family, and index; they were fixed without consulting solver performance.

## Generator families

All generators construct geometry first, partitioning cells into the groups removed by an explicit sequence. **Only afterward** do they assign each group a randomized ordered positive composition of 10. The construction never calls a solver, examines a solver score, or rejects a board based on solver difficulty.

1. **rectangle_carving.** Repeatedly choose bounding rectangles of two active cells, with even occupancy 2, 4, 6, 8, or 10. Selection is stochastic, biased toward desired small occupancy and wider empty-spanning rectangles once cavities develop. Even group sizes guarantee a legal pair can finish. This supplies interleaved 2-D overlap and long-range dependencies.
2. **orthogonal_weave.** Alternate horizontal and vertical active-cell segments, preferring even groups of 2/4/6/8. When a direction cannot remove a pair, use a 2-D pair-anchored fallback. This gives a different row/column interaction pattern, plus occasional 2-D terminal stitching.
3. **layered_anchors.** Independently choose a two-cell survivor block in each row, clear the even-length outer runs in shuffled order, then use general arbitrary 2-D rectangles to clear the 32 survivor cells. Late rectangles intersect many previously cleared cells, including some with no live opposite-corner pair. This family intentionally has more visibly separable early row segments.
4. **high_digit_pair_weave** (separate extension). Mix orthogonal and 2-D pair-anchored carving, with at most eight 4-cell groups and all remaining groups pairs. If there are `q` 4-cell groups, the exact mean digit is `5 - q/16`, guaranteeing 4.5–5.0. Actual generated boards have mean 4.5–4.75. Every move remains supported by the original scanner, and roughly 30 moves per board depend on earlier removals. This is a high-digit stress variant of the carving/weaving design, not an entirely unrelated sampling distribution.

For a fixed group size, digit assignment samples uniformly among ordered positive compositions of 10. It does not force globally uniform 1–9 frequencies.

## Files and schema

Each split has:

- `boards.json`: array of records with `id`, `family`, `rows`, `cols`, flat row-major `values`, string-row `grid`, `initial_active`, and `certified_optimum`
- `certificates.json`: corresponding board hash, seed, rectangle `path`, and `removed_counts`
- `seeds.json`: reproducibility seeds, separate from board inputs
- `validation.json`: independent per-board summary statistics
- `move_annotations.json`: independent per-move rectangle, removed count, geometry, live-opposite-corner flag, and dependency predecessors

The optimizer should consume **only `boards.json`**, never certificates, seeds, or per-move annotations. The independent checker may consume the witness for correctness certification. Do not provide witness paths to a solver, initialize searches from them, derive per-board move schedules from them, or reward reconstruction of generator-specific witnesses.

Public `validation_summary.json` contains aggregate structural statistics without board values, witness paths, or seeds. Holdout certificates/values/seeds/per-board annotations stay under `sealed_holdout/` until the agreed release boundary.

## Holdout protocol

This is an **operational embargo**, not encryption or filesystem access control. The construction/checking component necessarily reads holdout inputs, but candidate optimization must not read them. Public generator code also makes the seeds reproducible; deliberately regenerating holdout during tuning would violate the same embargo.

1. Tune only on development. Use warmup solely to warm compilation/execution, and do not include it in success statistics.
2. Freeze every candidate implementation/dependency file, all search budgets, modes, random seeds, weights, selection rules, and comparison settings before opening holdout.
3. Run `release_holdout.py` once for the predeclared evaluation. It records candidate byte hashes, full configuration, and UTC time before copying board JSON. It copies no certificate, seed, or annotation file.
4. Evaluate the frozen candidate and baseline on all released boards under the same predeclared protocol; independently replay returned paths.
5. Report full-clear success, residual cells/score, wall time, and paired per-board differences, both overall and per family. With only 32 heldout boards (8 per family), uncertainty remains substantial.
6. If results are used for another optimization round, these boards are development data thereafter. A fresh, previously unseen seed set is required for a new final holdout claim.

Example release (the destination must be new):

```sh
python release_holdout.py \
  --candidate /absolute/path/to/candidate.py \
  --candidate /absolute/path/to/other_runtime_dependency.py \
  --configuration /absolute/path/to/frozen_configuration.json \
  --destination /absolute/path/to/new_evaluation_directory
```

The default release includes both original 24-board holdout and 8-board extension. The helper writes `candidate_freeze.json`, `primary_holdout_boards.json`, and `high_digit_extension_holdout_boards.json`. It refuses an existing destination and rejects a holdout hash mismatch. It cannot enforce experimental discipline after export, identify omitted runtime dependencies, or physically prevent later candidate edits; the evaluator should recheck the recorded candidate hashes before and after the run.

## Independent checker and verification

`check_corpus.py` uses only standard-library Python. It imports neither generators nor solver code. For every move it directly loops over rows and columns and adds currently active values, checks the sum is exactly 10 and at least two cells are removed, then marks those cells inactive. It checks all 160 cells are gone, coordinates/digits/schema are valid, claimed removal counts match, hashes match, and split board/seed duplicates do not occur. It also records dependency and anchoring statistics. The global 51-board cross-corpus duplicate check was run separately.

```sh
python check_corpus.py
python check_corpus.py --root high_digit_extension
python -m unittest -v test_checker.py test_release.py
```

For independent replay of a solver result:

```python
from check_corpus import replay
checked = replay(board_record, returned_path, require_full=False)
assert returned_score == checked['score']
# Full-clear success is checked['score'] == 160.
```

Verification completed:

- All **51/51** certificates independently replay to 160
- **17/17** checker/release tests pass, covering illegal sums, duplicate moves, partial paths, bounds, integer/digit validation, false counts, nondense maps, incompatible grids, freeze/config recording, board-only export, repeat destination refusal, and holdout tampering
- All 15 generated payload files and both manifests reproduce **byte for byte**, including under different `PYTHONHASHSEED` values, on Python 3.12
- Release tests use dummy data only; tests do not release the real holdout
- Every original development board contains all digits 1–9 and nontrivial long-range dependency moves

Reproduce into a separate directory, without replacing fixed artifacts:

```sh
python generate_corpus.py --output /tmp/reproduced_sum10
python generate_high_digit_extension.py --output /tmp/reproduced_sum10/high_digit_extension
```

## Limitations and claims to avoid

- These are **constructed clearable boards**, not an unbiased sample of arbitrary game boards, uniformly random clearable boards, or boards captured from real play
- Conditioning independent uniform digits on full clear is not what this generator does; even matching a mean digit does not eliminate witness-induced correlations
- Original families are low-digit-heavy (heldout average 3.646), potentially much easier than a high-digit real/random board. The extension raises the mean (4.602) but does not establish representative real-world difficulty
- All boards’ total value is a multiple of 10, which is necessary for full clear. Pair-heavy families also have strong complement-pair structure. Group sizes and geometric patterns are deliberately biased
- Many early witness moves are local and initially legal; the corpus is not an adversarial hardness proof. Long-range overlap/dependency metrics describe supplied witnesses, not unavoidable dependencies in every solution
- The third family contains original-scanner-inaccessible witness moves. Separate family/anchoring reporting is important when attributing gains to search quality versus correcting enumeration completeness
- A known full-clear witness certifies optimum, but not witness uniqueness or minimal number of moves. Higher witness depth does not automatically mean greater solver difficulty
- Some family geometry and scoring choices are related; the fourth family is a stress extension rather than independent evidence from an unrelated generator
- No solver performance was used to filter instances. Consequently some may be easy for the baseline, which should be reported rather than silently replaced
- Small family sample sizes limit statistical conclusions. Success on this corpus supports regression testing on these explicit distributions, not a universal full-clear guarantee
