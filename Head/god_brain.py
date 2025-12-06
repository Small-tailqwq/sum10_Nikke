import asyncio
import json
import random
import time
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ProcessPoolExecutor
import uvicorn
import os
import ctypes
from datetime import datetime

# --- 系统配置 ---
try:
    ctypes.windll.user32.SetProcessDPIAware()
    print(">> [系统] Windows DPI 感知模式已激活。")
except: pass

INPUT_METHOD = "NONE"
try:
    import pydirectinput
    pydirectinput.FAILSAFE = False
    pydirectinput.PAUSE = 0.001 
    INPUT_METHOD = "DIRECT_INPUT"
    print(">> [系统] 硬件模拟层 (pydirectinput) 已加载。")
except ImportError:
    try:
        import pyautogui
        pyautogui.FAILSAFE = True 
        pyautogui.PAUSE = 0.01 
        INPUT_METHOD = "WIN32_API"
        print(">> [系统] 软件模拟层 (pyautogui) 已加载。")
    except: pass

try:
    from numba import njit, int8, int32, float32
    print(">> [系统] Numba 加速引擎已装载 (积分图模式)。")
    HAS_NUMBA = True
except ImportError:
    print(">> [警告] 未检测到 Numba！性能将受限。")
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- 数据收集器 ---
class DataCollector:
    def __init__(self, filename="sum10_training_data.jsonl"):
        self.filename = filename
        # 消除率阈值 (例如 0.85 表示 85%)，只有超过这个质量的局才会被记录
        # 注意：这里我们用分数来估算，或者需要在后端计算消除率
        # 简单起见，我们记录所有结果，后续在训练时再清洗
        self.min_score_threshold = 50000 

    def save_record(self, initial_map, initial_vals, rows, cols, path, score, mode):
        # 构造训练样本
        record = {
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "rows": rows,
            "cols": cols,
            "map": initial_map, # 原始 0/1 掩码
            "vals": initial_vals, # 原始数值
            "score": score,
            "path": path # 关键标签：专家的操作序列
        }
        
        try:
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(f">> [数据收集] 已保存高分局 (Score: {score}) 到 {self.filename}")
            return True
        except Exception as e:
            print(f">> [数据收集] 保存失败: {e}")
            return False

data_collector = DataCollector()

# --- 神之手 (保持 V3.3 的偏移微调版) ---
class GodHand:
    def __init__(self):
        self.tl = None; self.tr = None; self.bl = None; self.br = None
        self.rows = 0; self.cols = 0
        self.is_calibrated = False
        self.offset_x = 0; self.offset_y = 0

    def calibrate(self, tl, tr, bl, br, rows, cols):
        self.tl = tl; self.tr = tr; self.bl = bl; self.br = br
        self.rows = rows; self.cols = cols
        self.is_calibrated = True
        return f"校准完成 (Mode: {INPUT_METHOD})"

    def set_offset(self, x, y):
        self.offset_x = x; self.offset_y = y
        return f"偏移: X{x:+d}, Y{y:+d}"

    def get_screen_pos(self, r, c):
        if not self.is_calibrated: return (0, 0)
        u = c / (self.cols - 1) if self.cols > 1 else 0
        v = r / (self.rows - 1) if self.rows > 1 else 0
        top_x = self.tl[0] + (self.tr[0] - self.tl[0]) * u
        top_y = self.tl[1] + (self.tr[1] - self.tl[1]) * u
        bot_x = self.bl[0] + (self.br[0] - self.bl[0]) * u
        bot_y = self.bl[1] + (self.br[1] - self.bl[1]) * u
        final_x = top_x + (bot_x - top_x) * v
        final_y = top_y + (bot_y - top_y) * v
        return (int(final_x + self.offset_x), int(final_y + self.offset_y))

    def get_mouse_pos(self):
        import pyautogui
        return pyautogui.position()

    def move_to(self, x, y):
        if INPUT_METHOD == "DIRECT_INPUT": pydirectinput.moveTo(x, y)
        elif INPUT_METHOD == "WIN32_API": import pyautogui; pyautogui.moveTo(x, y)

    def execute_move(self, rect):
        if INPUT_METHOD == "NONE" or not self.is_calibrated: return
        r1, c1, r2, c2 = rect
        start_x, start_y = self.get_screen_pos(r1, c1)
        end_x, end_y = self.get_screen_pos(r2, c2)
        
        if INPUT_METHOD == "DIRECT_INPUT":
            pydirectinput.moveTo(start_x, start_y); time.sleep(0.015)
            pydirectinput.mouseDown(); time.sleep(0.02)
            steps = 4 
            dx = (end_x - start_x) / steps; dy = (end_y - start_y) / steps
            for i in range(1, steps + 1):
                pydirectinput.moveTo(int(start_x + dx * i), int(start_y + dy * i))
            pydirectinput.moveTo(end_x, end_y); time.sleep(0.015)
            pydirectinput.mouseUp(); time.sleep(0.02)
        elif INPUT_METHOD == "WIN32_API":
            import pyautogui
            pyautogui.moveTo(start_x, start_y); time.sleep(0.01)
            pyautogui.mouseDown(x=start_x, y=start_y); time.sleep(0.02)
            pyautogui.moveTo(end_x, end_y, duration=0.1)
            pyautogui.mouseUp(x=end_x, y=end_y)
        time.sleep(0.03)

god_hand = GodHand()

# --- 核心算法升级：二维前缀和 (2D Prefix Sum) ---

@njit(fastmath=True, nogil=True, cache=True)
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
def _fast_scan_rects_v4(map_data, vals, rows, cols, active_indices):
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

@njit(fastmath=True, nogil=True)
def _apply_move_fast(map_data, rect, cols):
    new_map = map_data.copy()
    r1, c1, r2, c2 = rect
    for r in range(r1, r2 + 1):
        base = r * cols
        for c in range(c1, c2 + 1):
            new_map[base + c] = 0
    return new_map

# --- 核心搜索逻辑 (独立出来以便复用) ---
def _run_core_search_logic(start_map, vals_arr, rows, cols, beam_width, search_mode, start_score, start_path, max_depth=160):
    """
    通用 Beam Search 内核
    """
    current_beam = [{
        'map': start_map,
        'path': list(start_path), 
        'score': start_score,
        'h_score': float(start_score * 1000)
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

            raw_moves = _fast_scan_rects_v4(state['map'], vals_arr, rows, cols, active_indices)
            
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
            
            # 扩展策略
            valid_moves_for_state.sort(key=lambda x: x[4], reverse=True)
            top_moves = valid_moves_for_state[:20]
            
            for move in top_moves:
                r1, c1, r2, c2, count = move
                rect_tuple = (r1, c1, r2, c2)
                new_map = _apply_move_fast(state['map'], rect_tuple, cols)
                new_score = state['score'] + count
                h = new_score * 1000 + random.random() * 500
                
                new_path = list(state['path'])
                new_path.append([int(r1), int(c1), int(r2), int(c2)])
                
                next_candidates.append({
                    'map': new_map, 'path': new_path,
                    'score': new_score, 'h_score': h
                })

        if not found_any_move: break
        if not next_candidates: break
        
        next_candidates.sort(key=lambda x: x['h_score'], reverse=True)
        current_beam = next_candidates[:beam_width]
        
        if current_beam[0]['score'] > best_state_in_run['score']:
            best_state_in_run = current_beam[0]
            
    return best_state_in_run

# --- V5 逆脉冲引擎 ---

def _solve_process_reverse_pulse(args):
    """
    V5 核心：带逆脉冲优化的求解器
    """
    map_list, val_list, rows, cols, beam_width, mode, seed, time_limit = args
    safe_seed = seed % (2**32 - 1)
    np.random.seed(safe_seed)
    random.seed(safe_seed)
    
    initial_map_arr = np.array(map_list, dtype=np.int8)
    vals_arr = np.array(val_list, dtype=np.int8)
    
    start_time = time.time()
    
    # 1. 初始冲刺 (Base Run)
    base_state = None
    if mode == 'god':
        p1 = _run_core_search_logic(initial_map_arr, vals_arr, rows, cols, beam_width, 'classic', 0, [])
        p2 = _run_core_search_logic(p1['map'], vals_arr, rows, cols, beam_width, 'omni', p1['score'], p1['path'])
        base_state = p2
    else:
        base_state = _run_core_search_logic(initial_map_arr, vals_arr, rows, cols, beam_width, mode, 0, [])
        
    best_final_state = base_state
    
    # 2. 逆脉冲优化循环 (Reverse Pulse Loop)
    iteration = 0
    while (time.time() - start_time) < time_limit:
        iteration += 1
        current_best_path = best_final_state['path']
        
        if len(current_best_path) < 5: 
            break
            
        # --- 破坏阶段 (Destroy) ---
        cut_len = random.randint(3, min(10, len(current_best_path) // 2))
        cut_start = random.randint(0, len(current_best_path) - cut_len)
        prefix_path = current_best_path[:cut_start]
        
        temp_map = initial_map_arr.copy()
        prefix_score = 0
        
        for rect in prefix_path:
            r1, c1, r2, c2 = rect
            step_count = 0
            for r in range(r1, r2+1):
                for c in range(c1, c2+1):
                    idx = r * cols + c
                    if temp_map[idx] == 1:
                        step_count += 1
                        temp_map[idx] = 0
            prefix_score += step_count
            
        # --- 重建阶段 (Repair) ---
        repaired_state = _run_core_search_logic(
            temp_map, vals_arr, rows, cols, 
            beam_width,
            'omni', 
            prefix_score, 
            prefix_path
        )
        
        # --- 评估阶段 (Evaluate) ---
        if repaired_state['score'] > best_final_state['score']:
            best_final_state = repaired_state

    return {
        'worker_id': seed,
        'score': best_final_state['score'],
        'path': best_final_state['path'],
        'iterations': iteration
    }

# --- WebSocket 服务端 ---
@app.websocket("/ws/optimize")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    executor = ProcessPoolExecutor()
    calibration_points = {'tl': None, 'tr': None, 'bl': None, 'br': None}
    
    # 全局变量记录当前会话的初始信息，用于数据收集
    current_session_data = {
        'initial_map': None,
        'initial_vals': None,
        'rows': 0, 'cols': 0, 'mode': ''
    }
    
    try:
        while True:
            data = await websocket.receive_text()
            req = json.loads(data)
            cmd = req.get('cmd')
            
            if cmd == 'CAPTURE_POS':
                target = req.get('target')
                delay = 2
                await websocket.send_json({"type": "LOG", "msg": f"请在 {delay} 秒内移动到目标..."})
                for i in range(delay, 0, -1):
                    await websocket.send_json({"type": "COUNTDOWN", "val": i, "target": target})
                    await asyncio.sleep(1)
                x, y = god_hand.get_mouse_pos(); calibration_points[target] = (x, y)
                await websocket.send_json({"type": "POS_CAPTURED", "target": target, "pos": [x, y], "msg": "OK"})

            elif cmd == 'APPLY_CALIBRATION':
                rows = req['rows']; cols = req['cols']
                tl = calibration_points['tl']; tr = calibration_points['tr']
                bl = calibration_points['bl']; br = calibration_points['br']
                if all([tl, tr, bl, br]):
                    res = god_hand.calibrate(tl, tr, bl, br, rows, cols)
                    await websocket.send_json({"type": "LOG", "msg": res})
                    await websocket.send_json({"type": "CALIBRATION_DONE", "status": True})
                else: await websocket.send_json({"type": "LOG", "msg": "校准失败：点位缺失"})

            elif cmd == 'SET_OFFSET':
                ox = int(req.get('x', 0)); oy = int(req.get('y', 0))
                await websocket.send_json({"type": "LOG", "msg": god_hand.set_offset(ox, oy)})

            elif cmd == 'TEST_ALIGNMENT':
                if not god_hand.is_calibrated: await websocket.send_json({"type": "LOG", "msg": "未校准"})
                else:
                    cx, cy = god_hand.get_screen_pos(god_hand.rows // 2, god_hand.cols // 2)
                    god_hand.move_to(cx, cy)
                    await websocket.send_json({"type": "LOG", "msg": f"准心测试: ({cx}, {cy})"})

            elif cmd == 'EXECUTE_PATH':
                path = req['path']
                if not god_hand.is_calibrated: await websocket.send_json({"type": "LOG", "msg": "未校准"})
                else:
                    await websocket.send_json({"type": "LOG", "msg": f"⚠️ 2秒后接管..."})
                    for i in range(2, 0, -1): await asyncio.sleep(1)
                    await websocket.send_json({"type": "LOG", "msg": ">>> 极速执行中 <<<"})
                    for i, rect in enumerate(path):
                        god_hand.execute_move(rect)
                        if i % 10 == 0: await websocket.send_json({"type": "EXEC_PROGRESS", "val": i, "total": len(path)}); await asyncio.sleep(0.001)
                    await websocket.send_json({"type": "LOG", "msg": "执行完毕"})

            elif cmd == 'START':
                rows = req['rows']; cols = req['cols']
                map_data = req['map']; vals = req['vals']
                beam_width = req['beamWidth']; mode = req['mode']; threads = req['threads']
                
                # 记录本局初始状态
                current_session_data['initial_map'] = map_data
                current_session_data['initial_vals'] = vals
                current_session_data['rows'] = rows
                current_session_data['cols'] = cols
                current_session_data['mode'] = mode
                
                TIME_LIMIT = 25.0
                
                msg = f"GOD ENGINE V5.1 (Data Collector) | {INPUT_METHOD} | Limit:{TIME_LIMIT}s"
                await websocket.send_json({"type": "LOG", "msg": msg})
                
                loop = asyncio.get_running_loop()
                tasks = []
                max_seed = 2**32 - 1 - threads
                base_seed = random.randint(0, max_seed)
                
                for i in range(threads):
                    args = (map_data, vals, rows, cols, beam_width, mode, base_seed + i, TIME_LIMIT)
                    task = loop.run_in_executor(executor, _solve_process_reverse_pulse, args)
                    tasks.append(task)
                
                best_score = -1; done_count = 0
                best_path_found = []
                
                for coro in asyncio.as_completed(tasks):
                    try:
                        result = await coro
                        done_count += 1
                        
                        await websocket.send_json({"type": "PROGRESS", "val": int((done_count / threads) * 100)})
                        
                        if result['score'] > best_score:
                            best_score = result['score']
                            best_path_found = result['path']
                            iters = result.get('iterations', 0)
                            await websocket.send_json({
                                "type": "BETTER_SOLUTION", 
                                "score": result['score'], 
                                "path": result['path'], 
                                "worker": result['worker_id']
                            })
                    except Exception as e: print(f"Task Error: {e}")
                
                # --- 自动保存高分局 ---
                if best_score > 0:
                    saved = data_collector.save_record(
                        current_session_data['initial_map'],
                        current_session_data['initial_vals'],
                        current_session_data['rows'],
                        current_session_data['cols'],
                        best_path_found,
                        best_score,
                        current_session_data['mode']
                    )
                    if saved:
                        await websocket.send_json({"type": "LOG", "msg": "【系统】高分数据已归档 💾"})
                
                await websocket.send_json({"type": "DONE", "msg": "逆脉冲优化完毕"})

    except WebSocketDisconnect:
        executor.shutdown(wait=False)

if __name__ == "__main__":
    print(">> Sum10 外置大脑 V5.1 (Data Collector) 启动中...")
    print(">> 启动逆向脉冲引擎，压榨每一秒算力...")
    print(">> 训练数据将自动保存至 sum10_training_data.jsonl")
    uvicorn.run(app, host="0.0.0.0", port=8000)