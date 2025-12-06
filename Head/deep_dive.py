"""
Sum10 Deep Dive Miner (破壁者)
文件名: deep_dive.py
描述: 锁定黄金参数，利用高 Beam Width 和多随机种子，暴力挖掘 148 分以上的路径

教授的核心洞察:
1. 参数调优已完成使命 (找到了 w_island=63, w_fragment=1.0)
2. 148分瓶颈 = 搜索视野不够宽 (神之一手被剪枝掉了)
3. 突破策略: 锁定参数 + 超大Beam Width + 多随机种子
"""

import asyncio
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from god_brain import _solve_process_hydra  # 回退到V6.2 Hydra引擎
import json
from datetime import datetime

# --- 1. 经典题目录入 ---
raw_map_str = """
3174268574
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
5166428486
"""

def parse_map(text):
    lines = text.strip().split('\n')
    rows = len(lines)
    cols = len(lines[0])
    vals = []
    for line in lines:
        for char in line:
            vals.append(int(char))
    map_data = [1] * (rows * cols)
    return map_data, vals, rows, cols

TARGET_MAP, TARGET_VALS, ROWS, COLS = parse_map(raw_map_str)

# --- 2. 配置你的"重型火炮" ---
# 基于第二次训练的分析，我们锁定这组"战术家"参数
GOLDEN_CONFIG = {
    'w_island': 63,      # Optuna训练最佳实践
    'w_fragment': 1.0,   # 临界点发现 (前5名中出现3次)
    'role': 'DeepMiner-壁破者'
}

# 关键：大幅提升搜索宽度！
# 教授的核心论点: 从148到160的距离 = beam_width从200到1000的距离
BEAM_WIDTH = 1000  # 重型火炮! (训练时可能只有100-200)

TIME_LIMIT = 30.0  # 给每局充足时间思考 (训练时只有20秒)

def run_miner():
    # 你的电脑有多少核就开多少，尽量压榨
    THREADS = 16 
    # Beam Width 保持 1000，这是我们唯一的优势
    BEAM_WIDTH = 1000
    # 时间给足，让疯子们多想一会儿
    TIME_LIMIT = 25.0 
    
    TOTAL_ROUNDS = 200 
    
    print(f"==================================================")
    print(f"🧨 SUM10 越狱行动启动 (Jailbreak Mode)")
    print(f"🚫 抛弃固定参数，启用全频谱混沌攻击")
    print(f"🔦 Beam Width: {BEAM_WIDTH}")
    print(f"==================================================")

    executor = ProcessPoolExecutor(max_workers=THREADS)
    loop = asyncio.get_event_loop()
    
    best_score_global = 0

    for round_idx in range(TOTAL_ROUNDS):
        print(f"\n>> Round {round_idx + 1}/{TOTAL_ROUNDS} dispatching chaos...")
        tasks = []
        
        base_seed = int(time.time() * 1000) + (round_idx * 99999)
        
        for i in range(THREADS):
            current_seed = base_seed + i
            
            # --- 越狱行动：每条线程分配完全不同的战术 ---
            personality = {'name': f"Worker-{i}"}
            
            if i < 4:
                # 【A组：V4 复刻版】(纯贪婪，0惩罚)
                personality['w_island'] = 0
                personality['w_fragment'] = 0
                personality['role'] = 'Berserker (V4 Legacy)'
                
            elif i < 8:
                # 【B组：微量约束】
                personality['w_island'] = 10
                personality['w_fragment'] = 0.1
                personality['role'] = 'Light Walker'
                
            elif i < 12:
                # 【C组：极端随机】(触发2000噪音)
                personality['w_island'] = 5
                personality['w_fragment'] = 0.01 
                personality['role'] = 'Chaos Gambler'
                
            else:
                # 【D组：战术家】
                personality['w_island'] = 63
                personality['w_fragment'] = 1.0
                personality['role'] = 'Tactician (Golden)'

            # 封装参数 (V6.2接口)
            args = (
                TARGET_MAP, TARGET_VALS, ROWS, COLS, 
                BEAM_WIDTH, 'god', current_seed, TIME_LIMIT, personality
            )
            tasks.append(loop.run_in_executor(executor, _solve_process_hydra, args))
            
        # 等待结果
        results = loop.run_until_complete(asyncio.gather(*tasks))
        
        # 实时播报最高分
        round_best = 0
        round_best_role = ""
        
        for res in results:
            if res['score'] > round_best:
                round_best = res['score']
                round_best_role = res['personality']['role']
            
            if res['score'] > best_score_global:
                best_score_global = res['score']
                print(f"🔥 [新纪录!] {res['personality']['role']} | Score: {best_score_global} | Seed: {res['worker_id']}")
                
                # 哪怕是 148 也要存，我们要看路径！
                if best_score_global >= 148:
                    import json
                    filename = f"jailbreak_{best_score_global}_{res['worker_id']}.json"
                    with open(filename, "w") as f:
                        json.dump(res['path'], f)
                    print(f"💾 路径已保存: {filename}")

        print(f"   Round Best: {round_best} ({round_best_role}) | Global: {best_score_global}")

if __name__ == "__main__":
    run_miner()
    
    input("按 Enter 键启动破壁行动...")
    run_miner()
