# Sum10 God Brain (策略大脑)

## 🧠 简介
这是 Sum10 Nikke 项目的核心计算模块，负责接收棋盘数据、进行高强度的策略搜索，并输出最优解。它包含了一个基于 FastAPI 的后端服务和一个基于 Web 的可视化控制台。

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install fastapi uvicorn websockets
```

### 2. 启动大脑 (后端)
```bash
python Head/god_brain.py
```
*控制台显示 `Sum10 God Brain V7.1 ... 启动中` 即表示服务已就绪。*

### 3. 启动控制台 (前端)
直接在浏览器中打开 `Head/Head_web.html` 文件。

## ✨ 核心功能

*   **OCR 集成**：直接调用 `eyes` 模块进行屏幕截图和数字识别。
*   **☢️ 多线程竞速 (Race Mode)**：利用 `ProcessPoolExecutor` 开启多个并行进程，每个进程使用不同的随机种子进行狂暴搜索。
*   **S/L 大法 (Time Traveler)**：内置“时光倒流”机制，在残局阶段自动回滚并尝试新的路径，死磕 100% 清盘率。
*   **实时同步**：通过 WebSocket 实时推送计算进度和当前发现的最优解 (Best Rate)。
*   **内存保护**：计算压力全部在 Python 后端，前端仅负责展示，避免浏览器崩溃。

## 🎮 操作指南

1.  **连接**：打开网页后，确保 WebSocket 连接状态为绿色。
2.  **识别**：点击 **OCR 识别** 按钮，系统会自动截图并填入数字。
3.  **设置**：
    *   **Threads**: 建议设置为 CPU 核心数 (如 8 或 12)。
    *   **Beam Width**: 搜索广度，默认即可，机器性能好可调高。
4.  **开始**：点击 **Start**，观察进度条和得分。
5.  **执行**：计算完成后，点击 **执行** 按钮，神之手将接管鼠标进行操作。

## ⚠️ 注意事项
*   **神之手校准**：在执行操作前，必须先在网页上进行 **校准 (Calibrate)**，告诉程序棋盘在屏幕上的确切位置。
*   **管理员权限**：模拟鼠标操作可能需要以管理员身份运行终端。


---
*Powered by Gemini 3 Pro, debugged by small-tailqwq*
## 求解器性能与回归验证

候选扩展和评分已批量放入 Numba，只有进入下一层 beam 的候选才创建路径和
Python 状态对象。保留原有搜索顺序、启发式、重复候选和稳定排序，并修复了
Numba 随机数未被 worker seed 正确初始化的问题。首次 JIT 编译时间与热运行
时间需要分开衡量。

无需启动 OCR、网页或游戏窗口即可运行测试和基准，详见
[性能验证说明](../benchmarks/README.md)。本次改动优化运行效率，不保证全局最优解。


## 可选 CLEAR SEARCH（2026-10-02）

Web 控制台新增 `CLEAR SEARCH`，后端模式名 `complete`；原 classic / omni / god 保持原行为。新算法使用完整合法矩形扫描、精确位图去重、保留小数字的排序、数字配平软惩罚、定时重启和残局回退。找到全清或达到经过证明的上界即返回。

- 输入 0–9；0 表示空格，兼容旧 OCR/UI 把 0 标为 active 的输入；最多192格、63列
- `optimal=true` 仅在合法路径达到上界时成立；未达到上界只能称 best-found
- 时间限制按搜索层协作检查，不是硬实时截止；首次 Numba 编译也计时，可能超过预算
- 可提前在其他棋盘上预热工作进程；不要用被测棋盘提前预热
- 默认 Web 模式仍为原 God；新模式需主动选择

只计算、不调用 OCR 或游戏输入的命令：

```sh
python benchmarks/solve_certified.py board.json --beam 200 --seconds 3 --seed 17
```

`board.json` 为二维整数数组，或 `{"board": [[1, 9], [5, 5]]}`。冷/热计时、见证隔离、保留测试集和完整结果请见[实验报告](../benchmarks/CERTIFIED_RESULTS.md)。新的按钮布局未完成浏览器视觉验证；软件格式、坐标、WebSocket、故障处理和前端状态用无设备测试验证。详见 [COMPATIBILITY.md](COMPATIBILITY.md) 与 [CERTIFIED_RESULTS.md](../benchmarks/CERTIFIED_RESULTS.md)。
