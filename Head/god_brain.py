"""
Sum10 God Brain V6.2 (Genesis - Gambler Mode)
==============================================
主脑程序"赌徒模式"版本

更新日志 (2025-12-06 赌徒模式):
[教授处方] 实施"赌徒模式"优化，重现V4荣光

问题诊断：
- V6.1太"懂事"了，过度平滑导致不敢冒险
- 惩罚函数封死了"险中求胜"的路径
- 那93.8%的神局，往往是在险中求胜的

核心修复：
1. **随机噪音放大** (50 → 2000)
   - 之前50分对几万分score无影响
   - 现在2000分能改变Beam选择
   - 算法会偶尔选择"看着顺眼"的路径

2. **性格重新配置** (复刻V4美学)
   - 稳健派 (2核): 保底146分
   - 狂战士 (4核): V4复刻版，纯贪婪
   - 微醺赌徒 (6核): 极低惩罚，允许冒险
   - 疯子 (4核): 高随机扰动，创造奇迹

目标：复刻V4的暴力美学 + V6的Numba速度 = 重现93.8%

历史更新 (V6.1.1):
- [致命缺陷修复] 参数虚设、惩罚过重、性格参数失调

历史更新 (V6.1):
- [核心修复] 预筛选窗口从 25 扩大到 60
- [性能优化] 评估函数注入"中心引力"惩罚机制
"""

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
import sys
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

# --- OCR 模块初始化 ---
OCR_AVAILABLE = False
recognizer = None
try:
    # 添加 eyes 目录到路径
    eyes_path = os.path.join(os.path.dirname(__file__), '..', 'eyes')
    labeling_tool_path = os.path.join(eyes_path, 'Sum10_Labeling_Tool')
    sys.path.insert(0, eyes_path)
    sys.path.insert(0, labeling_tool_path)
    
    from auto_capture import auto_capture_and_unwarp, load_coordinates
    from predict import Sum10Recognizer
    
    # 初始化识别器
    model_path = os.path.join(labeling_tool_path, 'sum10_model.pth')
    if os.path.exists(model_path):
        recognizer = Sum10Recognizer(model_path)
        OCR_AVAILABLE = True
        print("✅ OCR模块已加载")
    else:
        print(f"⚠️  模型文件未找到: {model_path}")
except Exception as e:
    print(f"⚠️  OCR模块加载失败: {e}")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- 数据收集器 ---
class DataCollector:
    def __init__(self, filename="sum10_elite_data.jsonl"):
        self.filename = filename
    def save_record(self, record):
        try:
            with open(self.filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(f">> [创世纪] 精英数据已归档 (Score: {record['score']})")
        except: pass

data_collector = DataCollector()

# --- 神之手 (保持稳定版) ---
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

# --- 核心算法 V6：神格化内核 ---

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

# --- 新增：孤岛检测内核 (Island Detection) ---
@njit(fastmath=True, nogil=True)
def _count_islands(map_data, rows, cols):
    """
    计算盘面上的孤岛数量。
    孤岛定义：一个非0数字，其上下左右4个邻居都是0（或边界）。
    """
    islands = 0
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if map_data[idx] == 1:
                # 检查四周
                # 上
                if r > 0 and map_data[(r-1)*cols + c] == 1: continue
                # 下
                if r < rows - 1 and map_data[(r+1)*cols + c] == 1: continue
                # 左
                if c > 0 and map_data[r*cols + (c-1)] == 1: continue
                # 右
                if c < cols - 1 and map_data[r*cols + (c+1)] == 1: continue
                
                islands += 1
    return islands

# --- 新增：神之眼评估函数 (Heuristic Evaluation) ---
# V6.1 升级：注入"中心引力"机制 (教授补丁)
# V6.1.1 紧急修复：修复参数虚设 + 惩罚过重问题
# V6.2 赌徒模式：大幅提高随机噪音，重现V4荣光
@njit(fastmath=True, nogil=True)
def _evaluate_state(score, map_data, rows, cols, w_island, w_fragment):
    """
    计算状态评分 H = Score - Penalty
    w_island: 孤岛惩罚权重
    w_fragment: 中心引力惩罚系数 (建议 0-5 范围)
    
    V6.1 新增：中心引力惩罚 (Center Gravity Penalty)
    V6.1.1 修复：
    - 提高基础分权重 (1000 -> 2000)，让算法更攻击性
    - 修复 w_fragment 参数虚设问题，真正启用性格差异
    - 降低中心惩罚基础值，防止矫枉过正
    V6.2 赌徒模式 (教授处方):
    - 为低惩罚性格注入巨大随机噪音 (50 -> 2000)
    - 原理：50分随机对几万分score无影响，2000分才能改变Beam选择
    - 效果：算法会偶尔选择"这步虽然少得1000分，但我看着顺眼，走你！"
    """
    # 1. 提高基础分权重，让算法更想拿分 (1000 -> 2000)
    h = float(score * 2000)
    
    # 2. 孤岛惩罚 (保留原有逻辑)
    if w_island > 0:
        islands = _count_islands(map_data, rows, cols)
        h -= islands * w_island
    
    # 3. 修复参数断连：使用 w_fragment 控制中心引力
    # 如果 w_fragment 为 0 (狂战士)，则完全不考虑中心引力
    if w_fragment > 0:
        center_mass = 0
        center_r, center_c = rows // 2, cols // 2
        for r in range(rows):
            for c in range(cols):
                if map_data[r * cols + c] == 1:
                    # 距离中心的曼哈顿距离越小，惩罚越大
                    dist = abs(r - center_r) + abs(c - center_c)
                    # 降低基础惩罚力度，并乘以 w_fragment 系数
                    center_mass += (20 - dist)
        
        # 减去中心堆积的惩罚 (现在是可控的了)
        h -= center_mass * w_fragment
    
    # 4. [V6.2 关键修改] 随机扰动 - 根据性格动态调整噪音级别
    # 原理：之前的50分对于几万分的score来说太小，根本影响不了排序
    # 对赌徒/狂战士给予500-2000的随机加分，让它有机会选择得分略低但运气可能更好的路径
    noise_level = 50.0
    if w_island < 20 and w_fragment < 1:
        # 赌徒/狂战士模式：注入巨大随机性 (这是创造奇迹的源头)
        noise_level = 2000.0
    
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
            
            # 前缀和快速校验
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

# --- 核心搜索逻辑 (进化版) ---
def _run_core_search_logic(start_map, vals_arr, rows, cols, beam_width, search_mode, start_score, start_path, weights, max_depth=160):
    """
    进化版 Beam Search (V6.3 Fix)
    修复：移除了预筛选阶段的致命截断，防止因位置偏见导致丢失关键路径。
    """
    w_island = weights.get('w_island', 0)
    w_fragment = weights.get('w_fragment', 0)
    
    # 初始评估
    initial_h = _evaluate_state(start_score, start_map, rows, cols, w_island, w_fragment)
    
    current_beam = [{
        'map': start_map,
        'path': list(start_path), 
        'score': start_score,
        'h_score': initial_h
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
            
            # --- [V6.3 核心修复] ---
            # 1. 彻底移除 [:60] 截断。
            #    Beam Width (1000) 本身就是过滤器，不需要在这里提前过滤。
            #    提前过滤会导致位于盘面后方的关键步被无视。
            
            # 2. 引入随机性打乱 (Shuffle)
            #    如果所有步数得分相同(Classic模式)，我们要随机选，而不是只选左上角的。
            #    这让"狂战士"和"赌徒"能真正随机地探索全图。
            
            # 这里的 random.shuffle 是 Python 原生的，对 Numba 列表不起作用
            # 但 valid_moves_for_state 是普通 list (因为 append 操作)，所以可以用
            random.shuffle(valid_moves_for_state)
            
            # 如果实在太多(超过200)，为了性能可以截断，但必须是在 Shuffle 之后！
            # 这样保证了全图都有机会被选中
            if len(valid_moves_for_state) > 200:
                top_moves = valid_moves_for_state[:200]
            else:
                top_moves = valid_moves_for_state
            
            for move in top_moves:
                r1, c1, r2, c2, count = move
                rect_tuple = (r1, c1, r2, c2)
                new_map = _apply_move_fast(state['map'], rect_tuple, cols)
                new_score = state['score'] + count
                
                # 评估
                h = _evaluate_state(new_score, new_map, rows, cols, w_island, w_fragment)
                
                new_path = list(state['path'])
                new_path.append([int(r1), int(c1), int(r2), int(c2)])
                
                next_candidates.append({
                    'map': new_map, 'path': new_path,
                    'score': new_score, 'h_score': h
                })

        if not found_any_move: break
        if not next_candidates: break
        
        # Beam Selection
        next_candidates.sort(key=lambda x: x['h_score'], reverse=True)
        current_beam = next_candidates[:beam_width]
        
        if current_beam[0]['score'] > best_state_in_run['score']:
            best_state_in_run = current_beam[0]
            
    return best_state_in_run

# --- V6 九头蛇引擎 (Hydra Engine) ---

def _solve_process_hydra(args):
    """
    V6 核心：多性格 + 定向爆破
    """
    map_list, val_list, rows, cols, beam_width, mode, seed, time_limit, personality = args
    safe_seed = seed % (2**32 - 1)
    np.random.seed(safe_seed)
    random.seed(safe_seed)
    
    initial_map_arr = np.array(map_list, dtype=np.int8)
    vals_arr = np.array(val_list, dtype=np.int8)
    
    # 解析性格参数
    weights = {
        'w_island': personality.get('w_island', 0),
        'w_fragment': personality.get('w_fragment', 0)
    }
    
    start_time = time.time()
    
    # 1. 初始冲刺 (Base Run)
    base_state = None
    if mode == 'god':
        # P1 阶段通常比较简单，稍微加一点点孤岛惩罚防止把自己堵死
        p1_weights = weights.copy()
        if p1_weights['w_island'] > 0: p1_weights['w_island'] *= 0.5 
        
        p1 = _run_core_search_logic(initial_map_arr, vals_arr, rows, cols, beam_width, 'classic', 0, [], p1_weights)
        p2 = _run_core_search_logic(p1['map'], vals_arr, rows, cols, beam_width, 'omni', p1['score'], p1['path'], weights)
        base_state = p2
    else:
        base_state = _run_core_search_logic(initial_map_arr, vals_arr, rows, cols, beam_width, mode, 0, [], weights)
        
    best_final_state = base_state
    
    # 2. 定向爆破循环 (Directed Destruction Loop)
    iteration = 0
    while (time.time() - start_time) < time_limit:
        iteration += 1
        path = best_final_state['path']
        if len(path) < 5: break
            
        # --- 定向选择切点 ---
        # 策略：分析路径，找到“低效区”。
        # 简单 heuristic: 切掉中间段，或者切掉消除数量较小的段。
        # 这里使用随机+启发式混合：
        # 70% 概率随机切，30% 概率切后半段（残局往往最难）
        
        if random.random() < 0.3:
            cut_start = random.randint(len(path)//2, len(path)-3)
        else:
            cut_start = random.randint(0, len(path)-3)
            
        cut_len = random.randint(3, min(12, len(path) - cut_start))
        
        prefix_path = path[:cut_start]
        
        # 快速重建状态
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
            
        # --- 重建 (Repair) ---
        # 在修补阶段，我们可以尝试稍微改变一下性格（微调权重），增加变异性
        repair_weights = weights.copy()
        repair_weights['w_island'] += random.randint(-50, 50) # 基因突变
        
        repaired_state = _run_core_search_logic(
            temp_map, vals_arr, rows, cols, 
            int(beam_width * 1.2), # 局部修补时算力加码
            'omni', 
            prefix_score, 
            prefix_path,
            repair_weights
        )
        
        if repaired_state['score'] > best_final_state['score']:
            best_final_state = repaired_state

    return {
        'worker_id': seed,
        'score': best_final_state['score'],
        'path': best_final_state['path'],
        'iterations': iteration,
        'personality': personality # 记录是谁立了大功
    }

# --- WebSocket 服务端 ---
@app.websocket("/ws/optimize")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    executor = ProcessPoolExecutor()
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

            elif cmd == 'RUN_OCR':
                # OCR 自动截图识别流程
                if not OCR_AVAILABLE:
                    await websocket.send_json({"type": "OCR_ERROR", "msg": "OCR模块未加载"})
                else:
                    try:
                        await websocket.send_json({"type": "LOG", "msg": "📸 正在截取屏幕..."})
                        await asyncio.sleep(0.1)  # 让UI更新
                        
                        # 在线程池中运行OCR (避免阻塞事件循环)
                        loop = asyncio.get_event_loop()
                        
                        # 截图并保存
                        def capture_screenshot():
                            warped, save_path = auto_capture_and_unwarp(
                                coords=None, 
                                use_timestamp=True, 
                                silent=True
                            )
                            return warped, save_path
                        
                        warped, save_path = await loop.run_in_executor(None, capture_screenshot)
                        
                        if warped is None or save_path is None:
                            await websocket.send_json({"type": "OCR_ERROR", "msg": "截图失败,请先运行coordinate_picker.py设置坐标"})
                        else:
                            await websocket.send_json({"type": "LOG", "msg": f"✅ 截图已保存: {os.path.basename(save_path)}"})
                            await websocket.send_json({"type": "LOG", "msg": "🔍 正在识别棋盘..."})
                            
                            # OCR识别
                            def run_ocr():
                                return recognizer.recognize_board(save_path)
                            
                            matrix = await loop.run_in_executor(None, run_ocr)
                            
                            # 转换为Web UI需要的格式
                            raw_data = ''.join(str(cell) for row in matrix for cell in row)
                            
                            await websocket.send_json({
                                "type": "OCR_RESULT",
                                "raw_data": raw_data,
                                "matrix": matrix
                            })
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
                rows = req['rows']; cols = req['cols']
                map_data = req['map']; vals = req['vals']
                beam_width = req['beamWidth']; mode = req['mode']; threads = req['threads']
                
                current_session_data.update({'initial_map': map_data, 'initial_vals': vals, 'rows': rows, 'cols': cols, 'mode': mode})
                
                TIME_LIMIT = 25.0
                msg = f"GOD ENGINE V6.2.2 (Dual-Path Strategy) | {INPUT_METHOD} | Hydra Cores:{threads}"
                await websocket.send_json({"type": "LOG", "msg": msg})
                
                loop = asyncio.get_running_loop()
                tasks = []
                max_seed = 2**32 - 1 - threads
                base_seed = random.randint(0, max_seed)
                
                # --- 九头蛇分发策略 (Hydra Dispatch) ---
                # V6.2 赌徒模式：根据教授处方，重新配置性格参数
                # V6.2.1 Optuna优化：应用训练最优参数
                # V6.2.2 混合派：双峰策略 - 同时利用两个局部最优
                # 目标：路径A(24,0.5微醺) + 路径B(63,1.0理性) → 突破150分
                for i in range(threads):
                    personality = {'name': f"Core-{i}"}
                    
                    # 0. 稳健派 (保底 - 用来稳定146分)
                    if i < 2:
                        personality['w_island'] = 50
                        personality['w_fragment'] = 2
                        personality['role'] = 'Balancer (稳健派)'
                    
                    # 1. 狂战士 - V4复刻版 (完全无视惩罚，纯贪婪)
                    elif i < 4:
                        personality['w_island'] = 0
                        personality['w_fragment'] = 0
                        personality['role'] = 'Berserker (V4复刻版)'
                    
                    # 2. 微醺赌徒 - 路径A (Optuna训练1: 24, 0.5 → 92.50%)
                    # 策略: 极低约束 + 巨大随机噪音(2000) + 靠运气打破常规
                    elif i < 9:
                        personality['w_island'] = 24
                        personality['w_fragment'] = 0.5
                        personality['role'] = 'Gambler-Elite (路径A-92.50%)'
                    
                    # 3. 理性冒险 - 路径B (Optuna训练2: 63, 1.0 → 92.50%)
                    # 策略: 中等约束 + 常规噪音(50) + 稳扎稳打
                    elif i < 14:
                        personality['w_island'] = 63
                        personality['w_fragment'] = 1.0  # 临界点
                        personality['role'] = 'Tactician-Elite (路径B-92.50%)'
                    
                    # 4. 临界探索 (基于前5名的混合组合)
                    else:
                        personality['w_island'] = 66
                        personality['w_fragment'] = 0.5
                        personality['role'] = 'Boundary-Explorer'
                    
                    args = (map_data, vals, rows, cols, beam_width, mode, base_seed + i, TIME_LIMIT, personality)
                    task = loop.run_in_executor(executor, _solve_process_hydra, args)
                    tasks.append(task)
                
                best_score = -1; done_count = 0
                best_record = None
                
                for coro in asyncio.as_completed(tasks):
                    try:
                        result = await coro
                        done_count += 1
                        await websocket.send_json({"type": "PROGRESS", "val": int((done_count / threads) * 100)})
                        
                        if result['score'] > best_score:
                            best_score = result['score']
                            best_record = result
                            p = result['personality']
                            # 可选：显示是谁立功了
                            # log_msg = f"[{p['role']}] 发现新解: {best_score} (孤岛权重:{p['w_island']})"
                            # await websocket.send_json({"type": "LOG", "msg": log_msg})
                            
                            await websocket.send_json({
                                "type": "BETTER_SOLUTION", 
                                "score": result['score'], 
                                "path": result['path'], 
                                "worker": result['worker_id']
                            })
                    except Exception as e: print(f"Task Error: {e}")
                
                # 自动归档
                if best_record:
                    # 记录获胜的性格参数，方便你分析哪种性格最强
                    full_record = {
                        "timestamp": datetime.now().isoformat(),
                        "mode": mode,
                        "rows": rows, "cols": cols,
                        "map": current_session_data['initial_map'],
                        "vals": current_session_data['initial_vals'],
                        "score": best_score,
                        "path": best_record['path'],
                        "winning_personality": best_record['personality']
                    }
                    data_collector.save_record(full_record)
                    
                await websocket.send_json({"type": "DONE", "msg": "创世纪演算完毕"})

    except WebSocketDisconnect:
        executor.shutdown(wait=False)

if __name__ == "__main__":
    print(">> Sum10 外置大脑 V6.2 (Genesis - Gambler Mode) 启动中...")
    print(">> 集成: 积分图加速 | 孤岛惩罚评估 | 多性格进化搜索 | 定向爆破")
    uvicorn.run(app, host="0.0.0.0", port=8000)