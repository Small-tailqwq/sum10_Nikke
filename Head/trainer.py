"""
Sum10 Trainer based on Optuna (V6.2 Gambler Mode Edition)
文件名: trainer.py
描述: 针对特定盘面，利用贝叶斯优化自动寻找打破纪录的参数组合

V6.2 优化要点:
1. 模拟V4实战环境: Beam Width=1000 (V4级别算力)
2. 聚焦低惩罚参数: w_island 0-100, w_fragment 0.0-10.0
3. 关键洞察: w_fragment < 1 会触发2000随机噪音(赌徒模式)
4. 多样性探索: 包含纯贪婪(0,0)、微醺赌徒、战术家等多种组合

pip install optuna
"""

import optuna
import numpy as np
import asyncio
from concurrent.futures import ProcessPoolExecutor
import god_brain  # 导入你的主程序作为库
from god_brain import _solve_process_hydra # 直接调用核心求解器

# --- 1. 录入你的经典题目 ---
# 将你的题目文本转换为 0/1 掩码和 数值数组
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
    
    # 初始全是 1 (未消除)
    map_data = [1] * (rows * cols)
    return map_data, vals, rows, cols

TARGET_MAP, TARGET_VALS, ROWS, COLS = parse_map(raw_map_str)

# --- 2. 定义目标函数 (Objective Function) ---
# Optuna 会不断调用这个函数，传入 trial (尝试对象)，要求返回一个分数
def objective(trial):
    # 1. 让 Optuna 猜一组参数
    # V6.2 赌徒模式：根据教授处方，聚焦于低惩罚参数范围
    # 因为高分往往来自"敢于冒险"的性格
    
    # 孤岛权重范围：0-100 (V6.2经验：过高会导致畏手畏脚)
    w_island = trial.suggest_int('w_island', 0, 100)
    
    # 碎片/中心引力权重：0.0-10.0 (允许小数，精细调优)
    # V6.2关键：w_fragment控制随机噪音级别(< 1触发2000噪音)
    w_fragment = trial.suggest_float('w_fragment', 0.0, 10.0, step=0.5)
    
    # 构造性格参数
    personality = {
        'name': f"Trial-{trial.number}",
        'w_island': w_island,
        'w_fragment': w_fragment,
        'role': 'Optuna-Hunter'  # 猎杀最高分
    }
    
    # 2. 构造运行参数 - 模拟V4实战环境
    # V4历史最高分环境: 16核 + 1000 Beam Width
    # 为了复现V4荣光，必须用相同算力
    beam_width = 1000   # V4级别算力 (训练会慢，但更接近实战)
    seed = trial.number # 每次用不同的随机种子
    time_limit = 20.0   # 给足够时间让算法充分搜索 (V4是无时间限制的)
    
    args = (
        TARGET_MAP, TARGET_VALS, ROWS, COLS, 
        beam_width, 'god', seed, time_limit, personality
    )
    
    # 3. 运行求解器 (这里我们直接同步运行，不走 WebSocket)
    result = _solve_process_hydra(args)
    
    score = result['score']
    
    # 4. 返回分数 (Optuna 会试图最大化这个值)
    # 打印实时进度
    print(f"[Trial {trial.number:3d}] w_island={w_island:3d}, w_fragment={w_fragment:4.1f} => Score: {score}")
    return score

# --- 3. 启动主程序 ---
if __name__ == "__main__":
    print(">> [Optuna] 开始自动化参数寻找...")
    print(f">> 目标：突破现有记录 (针对当前 10x16 盘面)")
    
    # 创建一个 Study，方向是 "maximize" (最大化分数)
    study = optuna.create_study(direction='maximize')
    
    # 开始优化！
    # V4冲刺配置：
    # - n_trials=100: 因为每次trial用1000 beam width会很慢，100次已足够探索
    # - n_jobs=4: 不要用满16核，留一些给每个trial内部的并行计算
    #   (god_brain内部也会用多核，避免过度竞争CPU)
    # - timeout=7200: 2小时总时间限制，防止跑太久
    
    print(f">> 训练配置: Beam Width=1000 (V4级别), 每Trial限时20秒")
    print(f">> 并行度: 4个Trial同时运行, 预计总耗时: 约8-10分钟")
    
    study.optimize(objective, n_trials=100, n_jobs=4, timeout=7200)

    print("-" * 50)
    print(">> 训练结束！")
    print(f">> 历史最高分: {study.best_value} / 160 ({study.best_value/160*100:.2f}%)")
    print(">> 最佳参数组合:")
    print(study.best_params)
    
    # 把最佳参数打印出来，填回你的 god_brain.py
    best_p = study.best_params
    print(f"\n>> [关键洞察] 建议修改代码参数:")
    print(f"   w_island = {best_p['w_island']}")
    print(f"   w_fragment = {best_p['w_fragment']}")
    
    # V6.2赌徒模式分析
    if best_p['w_fragment'] < 1:
        print(f"\n>> [赌徒模式] w_fragment < 1, 此参数会触发2000随机噪音!")
        print(f"   这是一个'微醺赌徒'配置，算法会敢于冒险")
    elif best_p['w_island'] == 0 and best_p['w_fragment'] == 0:
        print(f"\n>> [V4复刻] 纯贪婪模式! 完全无视惩罚")
        print(f"   这复刻了V4的暴力美学 + V6的Numba速度")
    else:
        print(f"\n>> [战术家] 这是一个平衡型配置")
    
    # 显示前5名的参数组合
    print("\n>> 前5名参数组合:")
    try:
        trials_df = study.trials_dataframe().nlargest(5, 'value')
        print(trials_df[['number', 'value', 'params_w_island', 'params_w_fragment']])
    except ImportError:
        # pandas未安装，手动提取前5名
        all_trials = [(t.number, t.value, t.params) for t in study.trials if t.value is not None]
        all_trials.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Trial':<8} {'Score':<8} {'w_island':<12} {'w_fragment':<12}")
        print("-" * 50)
        for trial_num, score, params in all_trials[:5]:
            print(f"{trial_num:<8} {score:<8.1f} {params['w_island']:<12} {params['w_fragment']:<12.1f}")