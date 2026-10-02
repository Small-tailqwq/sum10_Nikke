"""Original beam loop from commit 8cb00a3, retained as a regression oracle."""
import numpy as np

def reference_search(start_map, vals_arr, rows, cols, beam_width, search_mode, start_score, start_path, weights, max_depth=160):
    w_island = weights.get('w_island', 0)
    w_fragment = weights.get('w_fragment', 0)

    initial_h = _evaluate_state(start_score, start_map, rows, cols, w_island, w_fragment)
    current_beam = [{'map': start_map, 'path': list(start_path), 'score': start_score, 'h_score': initial_h}]
    best_state_in_run = current_beam[0]

    for _ in range(max_depth):
        # [V7 特性] 动态算力漏斗
        current_max_score = best_state_in_run['score']
        effective_beam_width = beam_width

        # 残局算力爆发
        if current_max_score > 120:
            effective_beam_width = int(beam_width * 3.0)
        elif current_max_score > 80:
            effective_beam_width = int(beam_width * 1.5)

        next_candidates = []
        found_any_move = False

        for state in current_beam:
            active_indices = np.where(state['map'] == 1)[0].astype(np.int32)
            if len(active_indices) < 2:
                if state['score'] > best_state_in_run['score']: best_state_in_run = state
                continue

            raw_moves = _fast_scan_rects_v6(state['map'], vals_arr, rows, cols, active_indices)
            if not raw_moves:
                if state['score'] > best_state_in_run['score']: best_state_in_run = state
                continue

            valid_moves_for_state = []
            for m in raw_moves:
                count = m[4]
                rule_pass = False
                if search_mode == 'classic':
                    if count == 2: rule_pass = True
                else:
                    if count >= 2: rule_pass = True
                if rule_pass: valid_moves_for_state.append(m)

            if not valid_moves_for_state:
                if state['score'] > best_state_in_run['score']: best_state_in_run = state
                continue

            found_any_move = True

            # 排序 + 截断
            valid_moves_for_state.sort(key=lambda x: x[4], reverse=True)
            window_size = 60 if current_max_score < 120 else 100
            top_moves = valid_moves_for_state[:window_size]

            for move in top_moves:
                r1, c1, r2, c2, count = move
                new_map = _apply_move_fast(state['map'], (r1, c1, r2, c2), cols)
                new_score = state['score'] + count
                h = _evaluate_state(new_score, new_map, rows, cols, w_island, w_fragment)
                new_path = list(state['path'])
                new_path.append([int(r1), int(c1), int(r2), int(c2)])
                next_candidates.append({'map': new_map, 'path': new_path, 'score': new_score, 'h_score': h})

        if not found_any_move or not next_candidates: break

        next_candidates.sort(key=lambda x: x['h_score'], reverse=True)
        current_beam = next_candidates[:effective_beam_width]

        if current_beam[0]['score'] > best_state_in_run['score']:
            best_state_in_run = current_beam[0]

    return best_state_in_run
