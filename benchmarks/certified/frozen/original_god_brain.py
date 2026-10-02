"""
Sum10 God Brain V7.1 (Time Traveler + OCR)
==============================================
主脑程序 "时空行者" 融合版

更新日志 (2025-12-06):
[版本融合] 将 V7 的核心算法植入 V6.2 的 OCR 架构中

核心机制 (V7 内核):
1. **S/L 大法 (Smart Rollback)**: 死磕最后 20 步，直到从 148 变 150。
2. **动态算力 (Dynamic Beam)**: 残局算力翻 3 倍，不做选择题，全都要。
3. **残局恐惧 (Endgame Panic)**: 剩 <20 个子时，孤岛惩罚 x5，强迫清盘。

保留功能:
- OCR 屏幕识别模块 (RUN_OCR)
- 硬件/软件模拟输入兼容
- 优雅的进程退出机制
"""

import asyncio
import json
import random
import time
import numpy as np
import signal
import sys
import atexit
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
    print(">> [系统] Numba 加速引擎已装载 (积分图 + 孤岛检测)。")
    HAS_NUMBA = True
except ImportError:
    print(">> [警告] 未检测到 Numba！性能将受限。")
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func): return func
        return decorator

# --- OCR 模块初始化 (保留) ---
OCR_AVAILABLE = False
recognizer = None
try:
    eyes_path = os.path.join(os.path.dirname(__file__), '..', 'eyes')
    labeling_tool_path = os.path.join(eyes_path, 'Sum10_Labeling_Tool')
    sys.path.insert(0, eyes_path)
    sys.path.insert(0, labeling_tool_path)

    from auto_capture import auto_capture_and_unwarp
    from predict import Sum10Recognizer

    model_path = os.path.join(labeling_tool_path, 'sum10_model.pth')
    if os.path.exists(model_path):
        recognizer = Sum10Recognizer(model_path)
        OCR_AVAILABLE = True
        print("✅ OCR模块已加载")
    else:
        print(f"⚠️ 模型文件未找到: {model_path}")
except Exception as e:
    print(f"⚠️ OCR模块加载失败: {e}")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- 全局执行器注册 ---
EXECUTORS = set()

def _shutdown_all_executors():
    for ex in list(EXECUTORS):
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
    EXECUTORS.clear()

atexit.register(_shutdown_all_executors)

def _sigint_handler(signum, frame):
    print("\n>> [系统] 捕获 Ctrl+C，正在优雅关闭进程池...")
    _shutdown_all_executors()
    sys.exit(0)

signal.signal(signal.SIGINT, _sigint_handler)

# --- 数据收集器 ---
class DataCollector:
    def __init__(self, filename="sum10_elite_data.jsonl"):
        self.filename = filename
    def save_record(self, record):
        try:
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(f">> [时空行者] 传奇数据已归档 (Score: {record['score']})")
        except: pass

data_collector = DataCollector()

# --- 神之手 ---
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

# --- V7 内核开始 ---

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
def _count_islands(map_data, rows, cols):
    islands = 0
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if map_data[idx] == 1:
                if r > 0 and map_data[(r-1)*cols + c] == 1: continue
                if r < rows - 1 and map_data[(r+1)*cols + c] == 1: continue
                if c > 0 and map_data[r*cols + (c-1)] == 1: continue
                if c < cols - 1 and map_data[r*cols + (c+1)] == 1: continue
                islands += 1
    return islands

# --- V7 神之眼 (残局恐惧版) ---
@njit(fastmath=True, nogil=True)
def _evaluate_state(score, map_data, rows, cols, w_island, w_fragment):
    # 1. 基础攻击性
    h = float(score * 2000)
    
    # 2. 剩余统计
    remaining_count = 0
    for i in range(rows * cols):
        if map_data[i] == 1: remaining_count += 1
    
    # 3. 孤岛惩罚 (动态加权)
    if w_island > 0:
        islands = _count_islands(map_data, rows, cols)
        
        # [V7 特性] 残局恐惧症 (Endgame Panic)
        # 剩 < 20 个子时，孤岛惩罚 x5.0，强迫清盘
        panic_multiplier = 1.0
        if remaining_count < 20 and remaining_count > 0:
            panic_multiplier = 5.0
            
        h -= islands * w_island * panic_multiplier
        
        if remaining_count < 10:
            h -= remaining_count * w_island * 2.0
    
    # 4. 中心引力
    if w_fragment > 0 and remaining_count > 30:
        center_mass = 0
        center_r, center_c = rows // 2, cols // 2
        for r in range(rows):
            for c in range(cols):
                if map_data[r * cols + c] == 1:
                    dist = abs(r - center_r) + abs(c - center_c)
                    center_mass += (20 - dist)
        h -= center_mass * w_fragment
    
    # 5. 随机噪音 (S/L 专用优化)
    # 赌徒模式的 2000 分噪音在残局时会坏事，V7 在残局时会收敛噪音
    noise_level = 50.0
    if w_island < 20 and w_fragment < 1:
        noise_level = 2000.0 # 开局敢赌
        
    if remaining_count < 30:
        noise_level *= 0.2 # 残局求稳
        
    h += np.random.random() * noise_level
    return h

@njit(fastmath=True, nogil=True)
def _fast_scan_rects_v6(map_data, vals, rows, cols, active_indices):
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

# --- 核心搜索逻辑 (V7 动态算力版) ---
def _run_core_search_logic(start_map, vals_arr, rows, cols, beam_width, search_mode, start_score, start_path, weights, max_depth=160):
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

# --- V7 九头蛇 (S/L 大法版) ---
def _solve_process_hydra(args):
    map_list, val_list, rows, cols, beam_width, mode, seed, time_limit, personality = args
    safe_seed = seed % (2**32 - 1)
    np.random.seed(safe_seed)
    random.seed(safe_seed)
    
    initial_map_arr = np.array(map_list, dtype=np.int8)
    vals_arr = np.array(val_list, dtype=np.int8)
    
    weights = {'w_island': personality.get('w_island', 0), 'w_fragment': personality.get('w_fragment', 0)}
    start_time = time.time()
    
    # 1. Base Run
    base_state = None
    if mode == 'god':
        p1_weights = weights.copy()
        if p1_weights['w_island'] > 0: p1_weights['w_island'] *= 0.5 
        p1 = _run_core_search_logic(initial_map_arr, vals_arr, rows, cols, beam_width, 'classic', 0, [], p1_weights)
        p2 = _run_core_search_logic(p1['map'], vals_arr, rows, cols, beam_width, 'omni', p1['score'], p1['path'], weights)
        base_state = p2
    else:
        base_state = _run_core_search_logic(initial_map_arr, vals_arr, rows, cols, beam_width, mode, 0, [], weights)
        
    best_final_state = base_state
    
    # 2. 时光倒流循环 (S/L Loop)
    iteration = 0
    max_score_history = best_final_state['score']
    
    while (time.time() - start_time) < time_limit:
        iteration += 1
        path = best_final_state['path']
        path_len = len(path)
        if path_len < 10: break
            
        # [V7 策略] 80% 概率只回滚最后 15-30 步 (死磕残局)
        if random.random() < 0.8:
            rollback_steps = random.randint(10, 30)
        else:
            rollback_steps = random.randint(30, max(31, int(path_len * 0.6)))
            
        if rollback_steps >= path_len: rollback_steps = path_len - 2
        cut_start = path_len - rollback_steps
        prefix_path = path[:cut_start]
        
        # 重建状态
        temp_map = initial_map_arr.copy()
        prefix_score = 0
        for rect in prefix_path:
            r1, c1, r2, c2 = rect
            s = 0
            for r in range(r1, r2+1):
                for c in range(c1, c2+1):
                    if temp_map[r*cols+c] == 1:
                        s += 1
                        temp_map[r*cols+c] = 0
            prefix_score += s
            
        # [V7] 极端性格修补 (Extreme Repair)
        repair_weights = weights.copy()
        dice = random.random()
        if dice < 0.4:
            repair_weights['w_island'] = random.randint(150, 300) # 极度恐慌
        elif dice < 0.7:
            repair_weights['w_island'] = random.randint(-50, -10) # 混乱邪恶
        else:
            repair_weights['w_island'] += random.randint(-20, 20)
            
        huge_beam = int(beam_width * 5.0) # 局部显微镜
        
        repaired_state = _run_core_search_logic(
            temp_map, vals_arr, rows, cols, huge_beam, 'omni', 
            prefix_score, prefix_path, repair_weights
        )
        
        if repaired_state['score'] > best_final_state['score']:
            best_final_state = repaired_state
            if best_final_state['score'] > max_score_history:
                max_score_history = best_final_state['score']
                time_limit += 2.0 # 奖励时间
        elif repaired_state['score'] == best_final_state['score']:
            if random.random() < 0.3:
                best_final_state = repaired_state

    return {
        'worker_id': seed,
        'score': best_final_state['score'],
        'path': best_final_state['path'],
        'iterations': iteration,
        'personality': personality
    }

# --- WebSocket ---
@app.websocket("/ws/optimize")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    executor = ProcessPoolExecutor()
    EXECUTORS.add(executor)
    tasks = []
    calibration_points = {'tl': None, 'tr': None, 'bl': None, 'br': None}
    current_session_data = {'initial_map': None, 'initial_vals': None, 'rows': 0, 'cols': 0, 'mode': ''}
    
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
                else: await websocket.send_json({"type": "LOG", "msg": "校准失败"})

            elif cmd == 'SET_OFFSET':
                ox = int(req.get('x', 0)); oy = int(req.get('y', 0))
                await websocket.send_json({"type": "LOG", "msg": god_hand.set_offset(ox, oy)})

            elif cmd == 'TEST_ALIGNMENT':
                if not god_hand.is_calibrated: await websocket.send_json({"type": "LOG", "msg": "未校准"})
                else:
                    cx, cy = god_hand.get_screen_pos(god_hand.rows // 2, god_hand.cols // 2)
                    god_hand.move_to(cx, cy)
                    await websocket.send_json({"type": "LOG", "msg": f"准心测试: ({cx}, {cy})"})
            
            # --- OCR 指令 (保留) ---
            elif cmd == 'RUN_OCR':
                if not OCR_AVAILABLE:
                    await websocket.send_json({"type": "OCR_ERROR", "msg": "OCR模块未加载"})
                else:
                    try:
                        await websocket.send_json({"type": "LOG", "msg": "📸 正在截取屏幕..."})
                        await asyncio.sleep(0.1)
                        loop = asyncio.get_event_loop()
                        def capture_screenshot():
                            warped, save_path = auto_capture_and_unwarp(coords=None, use_timestamp=True, silent=True)
                            return warped, save_path
                        warped, save_path = await loop.run_in_executor(None, capture_screenshot)
                        if warped is None or save_path is None:
                            await websocket.send_json({"type": "OCR_ERROR", "msg": "截图失败,请先运行coordinate_picker.py设置坐标"})
                        else:
                            await websocket.send_json({"type": "LOG", "msg": f"✅ 截图已保存: {os.path.basename(save_path)}"})
                            await websocket.send_json({"type": "LOG", "msg": "🔍 正在识别棋盘..."})
                            def run_ocr(): return recognizer.recognize_board(save_path)
                            matrix = await loop.run_in_executor(None, run_ocr)
                            raw_data = ''.join(str(cell) for row in matrix for cell in row)
                            await websocket.send_json({"type": "OCR_RESULT", "raw_data": raw_data, "matrix": matrix})
                            await websocket.send_json({"type": "LOG", "msg": f"✅ OCR识别完成 ({len(raw_data)}位数字)"})
                    except Exception as e:
                        await websocket.send_json({"type": "OCR_ERROR", "msg": f"OCR处理失败: {str(e)}"})

            elif cmd == 'EXECUTE_PATH':
                path = req['path']
                if not god_hand.is_calibrated: await websocket.send_json({"type": "LOG", "msg": "未校准"})
                else:
                    await websocket.send_json({"type": "LOG", "msg": f"⚠️ 2秒后接管..."})
                    for i in range(2, 0, -1): await asyncio.sleep(1)
                    await websocket.send_json({"type": "LOG", "msg": ">>> 神之手启动 <<<"})
                    for i, rect in enumerate(path):
                        god_hand.execute_move(rect)
                        if i % 10 == 0: await websocket.send_json({"type": "EXEC_PROGRESS", "val": i, "total": len(path)}); await asyncio.sleep(0.001)
                    await websocket.send_json({"type": "LOG", "msg": "执行完毕"})

            elif cmd == 'START':
                if tasks:
                    executor.shutdown(wait=False, cancel_futures=True)
                    executor = ProcessPoolExecutor()
                    tasks = []

                rows = req['rows']; cols = req['cols']
                map_data = req['map']; vals = req['vals']
                beam_width = req['beamWidth']; mode = req['mode']; threads = req['threads']
                current_session_data.update({'initial_map': map_data, 'initial_vals': vals, 'rows': rows, 'cols': cols, 'mode': mode})
                
                # V7 增加了计算量，默认给 30 秒，如果突破高分会自动延时
                TIME_LIMIT = 30.0 
                msg = f"GOD ENGINE V7.1 (Time Traveler + OCR) | {INPUT_METHOD} | S/L Mode Active"
                await websocket.send_json({"type": "LOG", "msg": msg})
                
                loop = asyncio.get_running_loop()
                max_seed = 2**32 - 1 - threads
                base_seed = random.randint(0, max_seed)
                
                for i in range(threads):
                    personality = {'name': f"Core-{i}"}
                    # 0-1. 稳健派 (保底)
                    if i < 2:
                        personality['w_island'] = 50; personality['w_fragment'] = 2; personality['role'] = 'Balancer (稳健)'
                    # 2-5. 主攻手
                    elif i < 6:
                        personality['w_island'] = 24; personality['w_fragment'] = 0.5; personality['role'] = 'Striker (主攻)'
                    # 6-9. 极端派 (S/L 变异源)
                    else:
                        personality['w_island'] = 80; personality['w_fragment'] = 1.0; personality['role'] = 'Heavy (重装)'
                    
                    args = (map_data, vals, rows, cols, beam_width, mode, base_seed + i, TIME_LIMIT, personality)
                    task = loop.run_in_executor(executor, _solve_process_hydra, args)
                    tasks.append(task)
                
                best_score = -1; done_count = 0; best_record = None
                
                for coro in asyncio.as_completed(tasks):
                    try:
                        result = await coro
                        done_count += 1
                        await websocket.send_json({"type": "PROGRESS", "val": int((done_count / threads) * 100)})
                        if result['score'] > best_score:
                            best_score = result['score']
                            best_record = result
                            await websocket.send_json({"type": "BETTER_SOLUTION", "score": result['score'], "path": result['path'], "worker": result['worker_id']})
                    except Exception as e: print(f"Task Error: {e}")
                
                if best_record:
                    full_record = {
                        "timestamp": datetime.now().isoformat(), "mode": mode,
                        "rows": rows, "cols": cols, "score": best_score, "path": best_record['path'],
                        "winning_personality": best_record['personality']
                    }
                    data_collector.save_record(full_record)
                    
                await websocket.send_json({"type": "DONE", "msg": "时空演算完毕"})

            elif cmd == 'EMERGENCY_EXECUTE':
                if tasks:
                    executor.shutdown(wait=False, cancel_futures=True)
                    executor = ProcessPoolExecutor()
                    tasks = []
                path = req.get('path', [])
                if not path:
                    await websocket.send_json({"type": "LOG", "msg": "无可用解"})
                    continue
                if not god_hand.is_calibrated:
                    await websocket.send_json({"type": "LOG", "msg": "未校准"})
                    continue
                await websocket.send_json({"type": "LOG", "msg": "⚠️ 1秒后接管"})
                await asyncio.sleep(1)
                for i, rect in enumerate(path):
                    god_hand.execute_move(rect)
                    if i % 10 == 0: await websocket.send_json({"type": "EXEC_PROGRESS", "val": i, "total": len(path)}); await asyncio.sleep(0.001)
                await websocket.send_json({"type": "LOG", "msg": "紧急执行完毕"})

    except WebSocketDisconnect: pass
    finally:
        try:
            EXECUTORS.discard(executor)
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception: pass

if __name__ == "__main__":
    print(">> Sum10 外置大脑 V7.1 (Time Traveler + OCR) 启动中...")
    print(">> 集成: S/L回滚 | 残局核爆搜索 | 极端性格修补 | OCR视觉")
    uvicorn.run(app, host="0.0.0.0", port=8000)