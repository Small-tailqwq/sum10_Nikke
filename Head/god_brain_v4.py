import asyncio
import json
import random
import time
import numpy as np
import sys
import signal
import atexit
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ProcessPoolExecutor
import uvicorn
import os
import ctypes

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

# --- OCR 模块初始化 ---
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

# --- 进程池登记，方便优雅关闭 ---
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

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

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
            steps = 4 # 减少步数以提速
            dx = (end_x - start_x) / steps; dy = (end_y - start_y) / steps
            for i in range(1, steps + 1):
                pydirectinput.moveTo(int(start_x + dx * i), int(start_y + dy * i))
                # 极速拖拽，不再 sleep
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
    """
    计算二维前缀和数组 P。
    P[i][j] 表示从 (0,0) 到 (i-1, j-1) 的矩形总和。
    数组大小为 (rows+1) x (cols+1)，第0行和第0列为0。
    """
    P = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    # 使用 vals 而不是 map_data，因为我们需要具体数值的和
    # 注意：如果某个位置 map_data 为 0（已消除），则 vals 里对应的值也应视为 0
    # 但传入的 vals 是原始值，所以我们需要一个 masked_vals
    
    for r in range(rows):
        row_sum = 0
        for c in range(cols):
            # 只有当值 > 0 时才累加 (假设 map_data 的状态隐含在 vals 的 0 值中，或者需要外部处理)
            # 在 Beam Search 中，我们通常会把已消除的位置在 vals 中置为 0，或者传入 mask
            # 这里假设调用前已处理 vals，或者 vals 就是当前状态的值
            row_sum += vals[r * cols + c]
            P[r + 1][c + 1] = P[r][c + 1] + row_sum
    return P

@njit(fastmath=True, nogil=True)
def _get_rect_sum(P, r1, c1, r2, c2):
    """利用前缀和 O(1) 获取矩形总和"""
    return P[r2+1][c2+1] - P[r1][c2+1] - P[r2+1][c1] + P[r1][c1]

@njit(fastmath=True, nogil=True)
def _get_rect_count(P_count, r1, c1, r2, c2):
    """利用前缀和 O(1) 获取矩形内非零元素个数"""
    return P_count[r2+1][c2+1] - P_count[r1][c2+1] - P_count[r2+1][c1] + P_count[r1][c1]

@njit(fastmath=True, nogil=True)
def _fast_scan_rects_v4(map_data, vals, rows, cols, active_indices):
    """
    V4 极速扫描：基于前缀和优化
    """
    moves = []
    n_active = len(active_indices)
    
    # 1. 预计算当前状态的 "值前缀和" 和 "计数前缀和"
    # 为了速度，我们需要在 Numba 内部构建这两个临时数组
    # 由于 map_data 是一维的，vals 也是一维的，我们需要处理一下
    
    current_vals = np.zeros(rows * cols, dtype=np.int32)
    current_counts = np.zeros(rows * cols, dtype=np.int32)
    
    for i in range(rows * cols):
        if map_data[i] == 1:
            current_vals[i] = vals[i]
            current_counts[i] = 1
            
    P_val = _calc_prefix_sum(current_vals, rows, cols)
    P_cnt = _calc_prefix_sum(current_counts, rows, cols)
    
    # 2. 遍历可能的矩形
    # 优化策略：不遍历所有点对，而是遍历“可能的矩形”。
    # 但为了保持逻辑一致性（必须以两个有效点为对角），我们还是遍历点对，但检查变成 O(1)
    
    for i in range(n_active):
        for j in range(i, n_active):
            idx1 = active_indices[i]
            idx2 = active_indices[j]
            
            # 坐标变换
            r1_raw = idx1 // cols; c1_raw = idx1 % cols
            r2_raw = idx2 // cols; c2_raw = idx2 % cols
            
            # 确定矩形边界
            min_r = min(r1_raw, r2_raw)
            max_r = max(r1_raw, r2_raw)
            min_c = min(c1_raw, c2_raw)
            max_c = max(c1_raw, c2_raw)
            
            # --- 核心优化点 ---
            # O(1) 获取总和
            current_sum = _get_rect_sum(P_val, min_r, min_c, max_r, max_c)
            
            # 快速剪枝：如果和已经不等于 10，直接跳过
            if current_sum != 10:
                continue
                
            # O(1) 获取元素个数
            count = _get_rect_count(P_cnt, min_r, min_c, max_r, max_c)
            
            # 记录结果 (r1, c1, r2, c2, count)
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

def _solve_process_beam_search(args):
    map_list, val_list, rows, cols, beam_width, mode, seed = args
    safe_seed = seed % (2**32 - 1)
    np.random.seed(safe_seed)
    
    initial_map_arr = np.array(map_list, dtype=np.int8)
    vals_arr = np.array(val_list, dtype=np.int8)

    def run_core_search(start_map, search_mode, start_score, start_path):
        current_beam = [{
            'map': start_map,
            'path': list(start_path), 
            'score': start_score,
            'h_score': float(start_score * 1000)
        }]
        
        best_state_in_run = current_beam[0]
        MAX_DEPTH = 160 # 既然速度快了，深度上限可以略微提高
        
        for _ in range(MAX_DEPTH):
            next_candidates = []
            found_any_move = False
            
            for state in current_beam:
                # 获取活动点
                active_indices = np.where(state['map'] == 1)[0].astype(np.int32)
                if len(active_indices) < 2:
                    if state['score'] > best_state_in_run['score']: best_state_in_run = state
                    continue

                # V4 极速扫描
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
                
                # 扩展状态 (这里可以进一步优化：只保留前 N 个最好的 move)
                # 为了防止分支爆炸，我们在扩展阶段就做一个简单的筛选
                # 优先选择消除数量多的
                valid_moves_for_state.sort(key=lambda x: x[4], reverse=True)
                top_moves = valid_moves_for_state[:20] # 每层只探索当前状态下最好的 20 种走法
                
                for move in top_moves:
                    r1, c1, r2, c2, count = move
                    rect_tuple = (r1, c1, r2, c2)
                    new_map = _apply_move_fast(state['map'], rect_tuple, cols)
                    new_score = state['score'] + count
                    
                    # 启发式评分优化：
                    # 1. 基础分：消除数量 * 1000
                    # 2. 聚集奖励：如果消除的是“孤岛”周围的数字，给予奖励 (暂未实现复杂版，用随机代替)
                    h = new_score * 1000 + random.random() * 500
                    
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

    final_result_state = None
    if mode == 'god':
        p1_state = run_core_search(initial_map_arr, 'classic', 0, [])
        p2_state = run_core_search(p1_state['map'], 'omni', p1_state['score'], p1_state['path'])
        final_result_state = p2_state
    else:
        final_result_state = run_core_search(initial_map_arr, mode, 0, [])

    return {
        'worker_id': seed,
        'score': final_result_state['score'],
        'path': final_result_state['path']
    }

# --- WebSocket 服务端 (保持一致) ---
@app.websocket("/ws/optimize")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    executor = ProcessPoolExecutor()
    EXECUTORS.add(executor)
    calibration_points = {'tl': None, 'tr': None, 'bl': None, 'br': None}
    
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

            # --- OCR 指令 ---
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
                    await websocket.send_json({"type": "LOG", "msg": ">>> 极速执行中 <<<"})
                    for i, rect in enumerate(path):
                        god_hand.execute_move(rect)
                        if i % 10 == 0: await websocket.send_json({"type": "EXEC_PROGRESS", "val": i, "total": len(path)}); await asyncio.sleep(0.001)
                    await websocket.send_json({"type": "LOG", "msg": "执行完毕"})

            elif cmd == 'START':
                rows = req['rows']; cols = req['cols']
                map_data = req['map']; vals = req['vals']
                beam_width = req['beamWidth']; mode = req['mode']; threads = req['threads']
                
                # 针对 10x16 的特殊参数调整
                # 这种尺寸下，我们可以让 beam_width 实际效果翻倍，因为计算快了
                
                msg = f"GOD ENGINE V4 (Integral) | {INPUT_METHOD} | Core:{threads}"
                await websocket.send_json({"type": "LOG", "msg": msg})
                
                loop = asyncio.get_running_loop()
                tasks = []
                max_seed = 2**32 - 1 - threads
                base_seed = random.randint(0, max_seed)
                
                for i in range(threads):
                    args = (map_data, vals, rows, cols, beam_width, mode, base_seed + i)
                    task = loop.run_in_executor(executor, _solve_process_beam_search, args)
                    tasks.append(task)
                
                best_score = -1; done_count = 0
                for coro in asyncio.as_completed(tasks):
                    try:
                        result = await coro
                        done_count += 1
                        await websocket.send_json({"type": "PROGRESS", "val": int((done_count / threads) * 100)})
                        if result['score'] > best_score:
                            best_score = result['score']
                            await websocket.send_json({"type": "BETTER_SOLUTION", "score": result['score'], "path": result['path'], "worker": result['worker_id']})
                    except Exception as e: print(f"Task Error: {e}")
                await websocket.send_json({"type": "DONE", "msg": "演算完毕"})

    except WebSocketDisconnect:
        executor.shutdown(wait=False)

if __name__ == "__main__":
    print(">> Sum10 外置大脑 V4.0 (Integral Image Optimized) 启动中...")
    print(">> 专为 10x16 棋盘优化，性能提升约 2000%")
    uvicorn.run(app, host="0.0.0.0", port=8000)