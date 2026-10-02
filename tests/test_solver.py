"""Behavior-preserving solver regression tests; no web/OCR dependencies."""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from solver_support import load_core, solve_base, validate_path
import reference_search as reference


@pytest.fixture(scope='session')
def core():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='sum10-tests-') as directory:
        module = load_core((root / 'Head/god_brain.py').read_text(encoding='utf-8'),
                           '_sum10_test_core', directory)
        for name in ['_evaluate_state', '_fast_scan_rects_v6', '_apply_move_fast']:
            setattr(reference, name, getattr(module, name))
        yield module


def board(seed, rows, cols, sparse):
    rng = np.random.default_rng(seed)
    vals = rng.integers(1, 10, rows * cols, dtype=np.int8)
    mask = np.ones(rows * cols, np.int8)
    if sparse:
        mask[rng.random(rows * cols) < 0.45] = 0
    return mask, vals


WEIGHTS = [
    {'w_island': 0.0, 'w_fragment': 0.0},
    {'w_island': 63.0, 'w_fragment': 1.0},
    {'w_island': 250.0, 'w_fragment': 2.0},
]


@pytest.mark.parametrize('rows,cols,sparse', [(2, 3, False), (3, 4, True), (4, 5, False)])
@pytest.mark.parametrize('mode', ['classic', 'omni', 'god'])
@pytest.mark.parametrize('beam', [1, 2, 5])
@pytest.mark.parametrize('depth', [0, 1, 8])
@pytest.mark.parametrize('weights', WEIGHTS)
def test_search_matches_original_loop(core, rows, cols, sparse, mode, beam, depth, weights):
    mask, vals = board(42, rows, cols, sparse)
    before_mask, before_vals = mask.copy(), vals.copy()
    old = SimpleNamespace(_run_core_search_logic=reference.reference_search)
    outputs = []
    for module in (old, core):
        core._seed_search_rng(97)
        result = solve_base(module, mask, vals, rows, cols, beam, mode, weights, depth)
        probe = core._evaluate_state(0, np.zeros_like(mask), rows, cols, 0., 0.)
        outputs.append((result, probe))
        validate_path(result, mask, vals, rows, cols, mode)
    expected, actual = outputs[0][0], outputs[1][0]
    assert actual['path'] == expected['path']
    assert actual['score'] == expected['score']
    assert actual['h_score'] == expected['h_score']
    assert outputs[0][1] == outputs[1][1]
    np.testing.assert_array_equal(actual['map'], expected['map'])
    np.testing.assert_array_equal(mask, before_mask)
    np.testing.assert_array_equal(vals, before_vals)
    assert isinstance(actual['score'], int)


@pytest.mark.parametrize('classic', [True, False])
@pytest.mark.parametrize('window', [1, 3, 60, 100])
@pytest.mark.parametrize('sparse', [True, False])
def test_batch_matches_ordered_candidates(core, classic, window, sparse):
    mask, vals = board(17, 6, 7, sparse)
    active = np.where(mask == 1)[0].astype(np.int32)
    moves = core._fast_scan_rects_v6(mask, vals, 6, 7, active)
    valid = [m for m in moves if m[4] == 2 or (not classic and m[4] > 2)]
    valid.sort(key=lambda move: move[4], reverse=True)
    valid = valid[:window]
    core._seed_search_rng(42)
    expected = []
    for move in valid:
        child = core._apply_move_fast(mask, move[:4], 7)
        score = 13 + move[4]
        h = core._evaluate_state(score, child, 6, 7, 63., 1.)
        expected.append((child, move[:4], score, h))
    probe = core._evaluate_state(0, np.zeros_like(mask), 6, 7, 0., 0.)
    core._seed_search_rng(42)
    maps, rects, scores, h_scores = core._expand_state_fast(
        mask, vals, 6, 7, classic, 13, 63., 1., window)
    assert len(scores) == len(expected)
    for i, (child, rect, score, h) in enumerate(expected):
        np.testing.assert_array_equal(maps[i], child)
        np.testing.assert_array_equal(rects[i], rect)
        assert scores[i] == score
        assert h_scores[i] == h
    assert probe == core._evaluate_state(0, np.zeros_like(mask), 6, 7, 0., 0.)


@pytest.mark.parametrize('mask,vals', [
    ([0, 0, 0, 0], [1, 9, 2, 8]),
    ([1, 0, 0, 0], [5, 9, 2, 8]),
    ([1, 1, 1, 1], [1, 1, 1, 1]),
    ([1, 1, 1, 1], [0, 0, 5, 5]),
    ([1, 1, 1, 1], [1, 2, 3, 4]),
])
def test_empty_no_move_zero_and_duplicate_cases(core, mask, vals):
    mask, vals = np.array(mask, np.int8), np.array(vals, np.int8)
    old = SimpleNamespace(_run_core_search_logic=reference.reference_search)
    core._seed_search_rng(1)
    expected = solve_base(old, mask, vals, 2, 2, 5, 'omni', WEIGHTS[0])
    core._seed_search_rng(1)
    actual = solve_base(core, mask, vals, 2, 2, 5, 'omni', WEIGHTS[0])
    assert actual['path'] == expected['path']
    assert actual['score'] == expected['score']
    validate_path(actual, mask, vals, 2, 2, 'omni')


def test_stable_heuristic_ties_and_survivor_ownership(core, monkeypatch):
    # All scores tie. Truncation must retain scanner order and selected maps
    # must not be views retaining the entire batch.
    mask, vals = np.ones(6, np.int8), np.array([1, 9, 2, 8, 5, 5], np.int8)
    maps = np.array([[0, 0, 1, 1, 1, 1], [1, 1, 0, 0, 1, 1]], np.int8)
    rects = np.array([[0, 0, 0, 1], [1, 0, 1, 1]], np.int32)
    monkeypatch.setattr(core, '_expand_state_fast', lambda *args: (
        maps, rects, np.array([2, 2]), np.array([100., 100.])))
    state = core._run_core_search_logic(mask, vals, 2, 3, 1, 'omni', 0, [], WEIGHTS[0], max_depth=1)
    assert state['path'] == [[0, 0, 0, 1]]
    assert not np.shares_memory(state['map'], maps)
    maps[:] = 1
    np.testing.assert_array_equal(state['map'], [0, 0, 1, 1, 1, 1])


@pytest.mark.parametrize('mode', ['classic', 'omni', 'god'])
@pytest.mark.parametrize('seed', [42, -1, 2**32 + 7])
def test_worker_seed_is_repeatable(core, mode, seed):
    mask, vals = board(42, 4, 5, False)
    args = (mask.tolist(), vals.tolist(), 4, 5, 3, mode, seed, 0., WEIGHTS[1])
    first = core._solve_process_hydra(args)
    second = core._solve_process_hydra(args)
    assert first == second


@pytest.mark.parametrize('start_score', [80, 81, 119, 120, 121])
@pytest.mark.parametrize('mode', ['classic', 'omni'])
def test_continuation_crosses_beam_and_window_thresholds(core, start_score, mode):
    # A dense low-valued board supplies >100 possible candidates, covering
    # both dynamic beam thresholds and the 60/100 candidate-window boundary.
    rng = np.random.default_rng(17)
    mask = np.ones(144, np.int8)
    vals = (np.full(144, 5, np.int8) if mode == 'classic'
            else rng.integers(1, 4, 144, dtype=np.int8))
    outputs = []
    for search in (reference.reference_search, core._run_core_search_logic):
        core._seed_search_rng(42)
        result = search(mask, vals, 12, 12, 5, mode, start_score, [], WEIGHTS[1], max_depth=3)
        outputs.append((result, core._evaluate_state(0, np.zeros_like(mask), 12, 12, 0., 0.)))
        validate_path(dict(result, score=result['score'] - start_score),
                      mask, vals, 12, 12, mode)
    expected, actual = outputs[0][0], outputs[1][0]
    assert actual['path'] == expected['path']
    assert actual['score'] == expected['score']
    assert actual['h_score'] == expected['h_score']
    assert outputs[0][1] == outputs[1][1]
    np.testing.assert_array_equal(actual['map'], expected['map'])
