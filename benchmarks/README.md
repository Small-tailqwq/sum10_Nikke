# Solver performance regression harness

The V7.1 search policy is unchanged: move enumeration (including duplicate moves),
count-descending truncation, heuristic formulas, random draw order, beam widths,
and rollback policy are retained. Candidate expansion now stays inside Numba;
only surviving candidates get Python dictionaries and copied paths. Both sorts
are stable, and survivor maps are copied out of the temporary batch buffers.

The worker now seeds Numba's RNG explicitly. Previously `np.random.seed()` was
called only from Python, which does not seed Numba's independent RNG. Specifying
a worker seed now makes its fixed-work base search reproducible. Wall-clock
rollback runs are still timing-dependent.

## Dependencies and verification

From the repository root, using Python 3.8+:

```sh
python -m pip install numpy numba pytest
python -m pytest -q tests
NUMBA_DISABLE_JIT=1 python -m pytest -q tests
python benchmarks/benchmark_solver.py --baseline-ref 8cb00a3 --output results.json
```

On Windows, set `NUMBA_DISABLE_JIT=1` using the shell's environment-variable
syntax before running pytest. You can also run the harness without Numba, but
that is a different performance profile.

The harness extracts the actual solver functions without modifying them into
temporary modules. It does not import FastAPI, OCR, Torch, screen capture or mouse
control, and requires no display, game window or GPU. The baseline source comes
from `git show <ref>:Head/god_brain.py`; that revision must exist in your checkout.

The default benchmark uses:

- The 16×10 example board in `Head/deep_dive.py`
- A synthetic 16×10 board and a 48-active-cell sparse board, generated from seed
  20261002
- Worker RNG seeds 17, 42 and 97; three repeats each
- Beam width 200, depth cap 160 per phase, weights 63.0 / 1.0
- Classic, omni and the two-phase god base search
- The same compiled RNG seed for both versions; alternating timing order

Every measured run replays each returned rectangle to verify sum 10, move count,
score and final board. It also requires exact before/after equality of the
complete path, score, board, heuristic value and subsequent RNG draw. Timing
excludes validation and initial JIT warm-up. Warm-up durations are reported
separately; a new compiled batch kernel increases first-ever compilation work,
while cached kernels can be reused by subsequent workers.

Tests use the original beam loop preserved as an oracle, plus candidate-level
and move-replay checks. They cover sparse/dense boards, three modes, several beam
widths/depths/personalities, empty/no-move boards, zeros, duplicate candidates,
stable heuristic ties, map ownership and worker seed reproducibility.

## Scope and existing limitations

This is a performance-preserving change, not a claim of optimal search or full
board-clearing ability. The following existing behaviors are deliberately
unchanged and need a separate search-quality/correctness change:

- The scanner only considers rectangles spanned by two active opposite corners.
  In omni mode it can miss a legal rectangle, such as a 3×3 cross of five 2s with
  inactive corners
- Duplicate rectangles/states consume beam slots
- Best-state tracking follows the existing heuristic-ranked beam behavior;
  it does not retain every highest-raw-score generated child
- Rollback repairs use omni mode even when the initial mode is classic
- `time_limit` is checked between entire searches, can overrun, and is extended
  after improvements; the benchmark therefore compares fixed work, not a strict
  wall-clock SLA

OCR, WebSocket operation, Windows mouse execution, and real-game integration are
outside these headless tests. No changes are made to those modules or interfaces.
