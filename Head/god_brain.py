"""
Sum10 God Engine V6.2 - "God Is Dead" Edition
=============================================
Philosophy: "Chaos is the only truth."

核心变革：
1. [Stochastic Beam Search]: 引入"幸存者偏差"机制，强制保留低分变异体。
2. [Entropy Heuristic]: 替代孤岛检测，计算盘面自由熵（基于Numba）。
3. [Hyper-Shuffle]: 彻底消除位置偏见，让狂战士的攻击覆盖全图。

此版本不再寻求"正确"的走法，而是穷举所有"可能"的奇迹。
"""

import asyncio
import json
import random
import time
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from concurrent.futures import ProcessPoolExecutor
import uvicorn
import ctypes

# --- 系统配置 ---
try:
    ctypes.windll.user32.SetProcessDPIAware()
except: pass

# ... (输入法配置保持不变，省略) ...

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator

app = FastAPI()

# --- 极速数学内核 (Numba Powered) ---

@njit(fastmath=True, nogil=True)
def _calc_prefix_sum(vals, rows, cols):
    P = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    for r in range(rows):
        row_sum = 0
        for c in range(cols):
            row_sum += vals[r * cols + c]
            P[r + 1][c + 1] = P[r][c + 1] + row_sum
    return P

@njit(fastmath=True, nogil=True)
def _get_rect_sum(P, r1, c1, r2, c2):
    return P[r2+1][c2+1] - P[r1][c2+1] - P[r2+1][c1] + P[r1][c1]

@njit(fastmath=True, nogil=True)
def _get_rect_count(P_count, r1, c1, r2, c2):
    return P_count[r2+1][c2+1] - P_count[r1][c2+1] - P_count[r2+1][c1] + P_count[r1][c1]

@njit(fastmath=True, nogil=True)
def _apply_move_fast(map_data, rect, cols):
    new_map = map_data.copy()
    r1, c1, r2, c2 = rect
    for r in range(r1, r2 + 1):
        base = r * cols
        for c in range(c1, c2 + 1):
            new_map[base + c] = 0
    return new_map

# --- 新武器：自由熵评估 (Entropy Heuristic) ---
# 比孤岛检测快 50 倍，但能提供足够的空间感
@njit(fastmath=True, nogil=True)
def _calc_local_entropy(map_data, rows, cols, r1, c1, r2, c2):
    """
    计算消除区域周围的"自由空间" (0的数量)。
    狂战士不仅要杀敌，还要把战场炸开阔。
    """
    free_space = 0
    # 扫描消除矩形的外围一圈
    r_start = max(0, r1 - 1)
    r_end = min(rows - 1, r2 + 1)
    c_start = max(0, c1 - 1)
    c_end = min(cols - 1, c2 + 1)
    
    for r in range(r_start, r_end + 1):
        for c in range(c_start, c_end + 1):
            # 如果是 0，说明是空地，加分
            if map_data[r * cols + c] == 0:
                free_space += 1
    return free_space

@njit(fastmath=True, nogil=True)
def _evaluate_chaos(score, map_data, rows, cols, rect, w_entropy, noise_level):
    """
    混沌评估函数：
    H = (分数 * 1000) + (自由熵 * w_entropy) + (巨大随机噪音)
    """
    h = float(score * 1000)
    
    # 自由熵加分：鼓励打通关节
    if w_entropy > 0:
        entropy = _calc_local_entropy(map_data, rows, cols, rect[0], rect[1], rect[2], rect[3])
        h += entropy * w_entropy
        
    # 注入混沌：这是狂战士的灵魂
    # noise_level 越高，越容易选中"非最优解"，从而跳出 148 的陷阱
    if noise_level > 0:
        h += np.random.random() * noise_level
        
    return h

@njit(fastmath=True, nogil=True)
def _fast_scan_rects_v7(map_data, vals, rows, cols, active_indices):
    # 标准 Numba 扫描，保持原样
    moves = []
    n_active = len(active_indices)
    current_vals = np.zeros(rows * cols, dtype=np.int32)
    current_counts = np.zeros(rows * cols, dtype=np.int32)
    for i in range(rows * cols):
        if map_data[i] == 1:
            current_vals[i] = vals[i]
            current_counts[i] = 1
    P_val = _calc_prefix_sum(current_vals, rows, cols)
    P_cnt = _calc_prefix_sum(current_counts, rows, cols)
    for i in range(n_active):
        for j in range(i, n_active):
            idx1 = active_indices[i]; idx2 = active_indices[j]
            r1_raw = idx1 // cols; c1_raw = idx1 % cols
            r2_raw = idx2 // cols; c2_raw = idx2 % cols
            min_r = min(r1_raw, r2_raw); max_r = max(r1_raw, r2_raw)
            min_c = min(c1_raw, c2_raw); max_c = max(c1_raw, c2_raw)
            if _get_rect_sum(P_val, min_r, min_c, max_r, max_c) != 10: continue
            count = _get_rect_count(P_cnt, min_r, min_c, max_r, max_c)
            moves.append((min_r, min_c, max_r, max_c, count))
    return moves

# --- 核心逻辑：随机波束搜索 (Stochastic Beam Search) ---

def _run_chaos_core(start_map, vals_arr, rows, cols, beam_width, start_score, start_path, strategy, max_depth=160):
    w_entropy = strategy['w_entropy']
    noise = strategy['noise']
    stochastic_ratio = strategy['stochastic_ratio'] # 混沌因子：保留多少"垃圾"
    
    current_beam = [{
        'map': start_map,
        'path': list(start_path), 
        'score': start_score,
        'h_score': 0
    }]
    
    best_state_in_run = current_beam[0]
    
    for _ in range(max_depth):
        next_candidates = []
        found_any_move = False
        
        for state in current_beam:
            active_indices = np.where(state['map'] == 1)[0].astype(np.int32)
            if len(active_indices) < 2:
                if state['score'] > best_state_in_run['score']: best_state_in_run = state
                continue

            raw_moves = _fast_scan_rects_v7(state['map'], vals_arr, rows, cols, active_indices)
            if not raw_moves:
                if state['score'] > best_state_in_run['score']: best_state_in_run = state
                continue
            
            # --- 彻底解除封印 ---
            # 1. 只要 >= 2 都要 (Omni only)
            valid_moves = [m for m in raw_moves if m[4] >= 2]
            
            if not valid_moves: continue
            found_any_move = True
            
            # 2. 【关键】必须打乱！打破左上角的诅咒
            random.shuffle(valid_moves)
            
            # 3. 性能截断 (仅为了防爆内存，设得很大)
            top_moves = valid_moves[:300]
            
            for move in top_moves:
                r1, c1, r2, c2, count = move
                rect_tuple = (r1, c1, r2, c2)
                new_map = _apply_move_fast(state['map'], rect_tuple, cols)
                new_score = state['score'] + count
                
                # 评估：分数 + 自由熵 + 混沌噪音
                h = _evaluate_chaos(new_score, new_map, rows, cols, rect_tuple, w_entropy, noise)
                
                new_path = list(state['path'])
                new_path.append([int(r1), int(c1), int(r2), int(c2)])
                
                next_candidates.append({
                    'map': new_map, 'path': new_path,
                    'score': new_score, 'h_score': h
                })

        if not found_any_move: break
        if not next_candidates: break
        
        # --- 弑神之选：Stochastic Selection ---
        # 不再只选分数最高的，而是强制混入一些"坏"的，保持基因多样性
        
        # 1. 先按 H-score 排序
        next_candidates.sort(key=lambda x: x['h_score'], reverse=True)
        
        # 2. 决定保留多少精英 (比如 80%)
        elite_count = int(beam_width * (1.0 - stochastic_ratio))
        elites = next_candidates[:elite_count]
        
        # 3. 从剩下的垃圾堆里，随机捡回一些 (20%)
        # 这就是让疯子拿到"拐棍"的关键：不要让所有疯子都去挤独木桥
        remaining = next_candidates[elite_count:]
        if remaining and stochastic_ratio > 0:
            random_picks_count = min(len(remaining), beam_width - elite_count)
            # 使用 random.sample 可能会慢，这里简单切片再 shuffle 即可
            # 或者直接取 random pointers
            lucky_dogs = random.sample(remaining, random_picks_count)
            current_beam = elites + lucky_dogs
        else:
            current_beam = elites
            
        # 更新最佳 (依然以纯分数为标准)
        curr_best = max(current_beam, key=lambda x: x['score'])
        if curr_best['score'] > best_state_in_run['score']:
            best_state_in_run = curr_best
            
    return best_state_in_run

def _solve_process_god_is_dead(args):
    map_list, val_list, rows, cols, beam_width, seed, time_limit, role_config = args
    
    safe_seed = seed % (2**32 - 1)
    np.random.seed(safe_seed)
    random.seed(safe_seed)
    
    initial_map_arr = np.array(map_list, dtype=np.int8)
    vals_arr = np.array(val_list, dtype=np.int8)
    
    final_state = _run_chaos_core(
        initial_map_arr, vals_arr, rows, cols, 
        beam_width, 0, [], 
        role_config
    )
    
    return {
        'worker_id': seed,
        'score': final_state['score'],
        'path': final_state['path'],
        'role': role_config['name']
    }

# --- 角色配置与分发 ---
# 在这里，我们塑造不同的角色，给他们不同的武器

ROLE_DEFINITIONS = [
    # 角色1: 毁灭者 (The Destroyer)
    # 纯粹的狂战士，极高的随机噪音，完全不看盘面，只看运气。
    # 武器：加特林 (High Noise)
    {
        'name': 'Destroyer (High Noise)',
        'w_entropy': 0,
        'noise': 3000.0,      # 极高噪音，甚至敢走负分的路
        'stochastic_ratio': 0.3 # 30% 的概率保留垃圾路径
    },
    # 角色2: 拓荒者 (The Pioneer)
    # 关注"自由熵"，喜欢把盘面炸开，不一定是得分最高的，但一定是剩余空间最大的。
    # 武器：工兵铲 (Entropy)
    {
        'name': 'Pioneer (Entropy)',
        'w_entropy': 50.0,    # 重视空间释放
        'noise': 500.0,       # 中等噪音
        'stochastic_ratio': 0.1
    },
    # 角色3: 赌徒 (The Gambler)
    # 极端随机保留机制，专门走没人走的路。
    # 武器：骰子 (High Stochastic)
    {
        'name': 'Gambler (Stochastic)',
        'w_entropy': 10.0,
        'noise': 1000.0,
        'stochastic_ratio': 0.5 # 50% 的路径都是随机捡来的垃圾，极度不可控
    },
     # 角色4: 纯粹V4 (Legacy)
    # 什么都不加，就是快。
    {
        'name': 'Legacy V4 (Pure)',
        'w_entropy': 0,
        'noise': 500.0,
        'stochastic_ratio': 0
    }
]

# --- WebSocket 适配 ---
# 修改 START 指令中的分发逻辑

# for i in range(threads):
#     # 轮询分配角色
#     role = ROLE_DEFINITIONS[i % len(ROLE_DEFINITIONS)]
#     
#     # 这里的 Beam Width 可以开得很大，因为计算量很小
#     actual_beam = 3000 
#     
#     args = (map_data, vals, rows, cols, actual_beam, base_seed + i, TIME_LIMIT, role)
#     task = loop.run_in_executor(executor, _solve_process_god_is_dead, args)
# ...