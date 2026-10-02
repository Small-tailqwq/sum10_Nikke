"""Reproducible, fixed-work comparison against a specific Git revision.

Run from the checkout: python benchmarks/benchmark_solver.py --output results.json
Only NumPy and optional Numba are needed; OCR/FastAPI/Windows are never imported.
"""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from solver_support import fixtures, load_core, solve_base, validate_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default='8cb00a3')
    parser.add_argument('--beam', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seeds', type=int, nargs='+', default=[17, 42, 97])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.beam < 1 or args.repeats < 1:
        parser.error('beam and repeats must be positive')
    root = Path(__file__).resolve().parents[1]
    old_source = subprocess.check_output(
        ['git', 'show', f'{args.baseline_ref}:Head/god_brain.py'], cwd=root, text=True)
    new_source = (root / 'Head/god_brain.py').read_text(encoding='utf-8')
    weights = {'w_island': 63.0, 'w_fragment': 1.0}
    result = {
        'baseline_ref': subprocess.check_output(
            ['git', 'rev-parse', args.baseline_ref], cwd=root, text=True).strip(),
        'current_source_sha256': hashlib.sha256(new_source.encode()).hexdigest(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'numpy': np.__version__, 'beam': args.beam, 'max_depth_per_phase': 160,
        'weights': weights, 'repeats': args.repeats, 'seeds': args.seeds,
        'budget': 'fixed beam/depth; excludes timed rollback and JIT warm-up',
        'cases': [],
    }
    try:
        import numba
        result['numba'] = numba.__version__
    except ImportError:
        result['numba'] = None
    with tempfile.TemporaryDirectory(prefix='sum10-benchmark-') as directory:
        old = load_core(old_source, '_sum10_baseline', directory)
        new = load_core(new_source, '_sum10_optimized', directory)
        for label, module in [('baseline', old), ('optimized', new)]:
            new._seed_search_rng(0)
            start = time.perf_counter()
            module._run_core_search_logic(
                np.ones(6, np.int8), np.array([1, 9, 2, 8, 5, 5], np.int8),
                2, 3, 2, 'omni', 0, [], weights, max_depth=2)
            result[label + '_warmup_seconds'] = time.perf_counter() - start
        trial_index = 0
        for name, mask, vals, rows, cols in fixtures():
            for mode in ['classic', 'omni', 'god']:
                baseline_times, optimized_times, scores = [], [], []
                for seed in args.seeds:
                    for repeat in range(args.repeats):
                        # Alternate timing order to reduce warm CPU/order bias.
                        modules = [('baseline', old), ('optimized', new)]
                        if trial_index % 2:
                            modules.reverse()
                        trial_index += 1
                        outputs = {}
                        for label, module in modules:
                            # Seed the compiled RNG for BOTH implementations.
                            new._seed_search_rng(seed)
                            start = time.perf_counter()
                            state = solve_base(module, mask, vals, rows, cols,
                                               args.beam, mode, weights)
                            elapsed = time.perf_counter() - start
                            validate_path(state, mask, vals, rows, cols, mode)
                            probe = module._evaluate_state(
                                0, np.zeros(rows * cols, np.int8), rows, cols, 0., 0.)
                            outputs[label] = (state, probe)
                            (baseline_times if label == 'baseline' else optimized_times).append(elapsed)
                        before, after = outputs['baseline'][0], outputs['optimized'][0]
                        assert before['score'] == after['score']
                        assert before['path'] == after['path']
                        assert before['h_score'] == after['h_score']
                        assert outputs['baseline'][1] == outputs['optimized'][1]
                        np.testing.assert_array_equal(before['map'], after['map'])
                        if repeat == 0:
                            scores.append(before['score'])
                old_median = statistics.median(baseline_times)
                new_median = statistics.median(optimized_times)
                case = {'fixture': name, 'mode': mode, 'scores_by_seed': scores,
                        'baseline_seconds': baseline_times,
                        'optimized_seconds': optimized_times,
                        'baseline_median': old_median, 'optimized_median': new_median,
                        'speedup': old_median / new_median,
                        'exact_result_and_rng_match': True}
                result['cases'].append(case)
                print(f'{name:20s} {mode:7s}: {old_median:.4f}s -> {new_median:.4f}s '
                      f'({old_median / new_median:.2f}x), scores={scores}, exact match', flush=True)
    result['total_baseline_seconds'] = sum(sum(c['baseline_seconds']) for c in result['cases'])
    result['total_optimized_seconds'] = sum(sum(c['optimized_seconds']) for c in result['cases'])
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    else:
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
