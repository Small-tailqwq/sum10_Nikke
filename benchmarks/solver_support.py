"""Load only the real solver functions, without OCR, GUI or web-server imports."""
import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np

CORE_NAMES = {
    '_calc_prefix_sum', '_get_rect_sum', '_get_rect_count', '_count_islands',
    '_evaluate_state', '_fast_scan_rects_v6', '_apply_move_fast',
    '_seed_search_rng', '_expand_state_fast', '_run_core_search_logic',
    '_solve_process_hydra',
}
HEADER = '''import numpy as np
import random
import time
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorate(fn): return fn
        return decorate
'''


def load_core(source, name, directory):
    """Materialize unchanged function source so Numba's disk cache also works."""
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in CORE_NAMES]
    lines = source.splitlines(keepends=True)
    chunks = []
    for node in nodes:
        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
        chunks.append(''.join(lines[first - 1:node.end_lineno]))
    path = Path(directory) / (name + '.py')
    path.write_text(HEADER + '\n\n'.join(chunks), encoding='utf-8')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REPOSITORY_BOARD = '''3174268574
6982841133
1744247217
6675567919
8981272644
9923228683
3699186393
1557354841
6793751936
4242945534
3758137661
9737251917
1938446324
1722548335
2365168672
5166428486'''


def fixtures():
    """One repository example and deterministic dense/sparse synthetic boards."""
    vals = np.array([int(c) for c in REPOSITORY_BOARD if c.isdigit()], np.int8)
    rng = np.random.default_rng(20261002)
    random_vals = rng.integers(1, 10, 160, dtype=np.int8)
    sparse_map = np.zeros(160, np.int8)
    sparse_map[rng.choice(160, 48, replace=False)] = 1
    return [
        ('repository_16x10', np.ones(160, np.int8), vals, 16, 10),
        ('random_16x10', np.ones(160, np.int8), random_vals, 16, 10),
        ('sparse_16x10', sparse_map, random_vals, 16, 10),
    ]


def solve_base(module, mask, vals, rows, cols, beam, mode, weights, depth=160):
    if mode == 'god':
        phase_one_weights = weights.copy()
        if phase_one_weights['w_island'] > 0:
            phase_one_weights['w_island'] *= 0.5
        first = module._run_core_search_logic(
            mask, vals, rows, cols, beam, 'classic', 0, [],
            phase_one_weights, max_depth=depth)
        return module._run_core_search_logic(
            first['map'], vals, rows, cols, beam, 'omni', first['score'],
            first['path'], weights, max_depth=depth)
    return module._run_core_search_logic(
        mask, vals, rows, cols, beam, mode, 0, [], weights, max_depth=depth)


def validate_path(result, mask, vals, rows, cols, mode):
    active = mask.copy()
    score = 0
    for rect in result['path']:
        r1, c1, r2, c2 = rect
        assert 0 <= r1 <= r2 < rows and 0 <= c1 <= c2 < cols
        indices = [r * cols + c for r in range(r1, r2 + 1)
                   for c in range(c1, c2 + 1) if active[r * cols + c] == 1]
        assert sum(int(vals[i]) for i in indices) == 10
        assert len(indices) == 2 if mode == 'classic' else len(indices) >= 2
        active[indices] = 0
        score += len(indices)
    assert score == result['score']
    np.testing.assert_array_equal(active, result['map'])
