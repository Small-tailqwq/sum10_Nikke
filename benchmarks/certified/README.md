# 可复现实验数据

完整结论见 [CERTIFIED_RESULTS.md](../CERTIFIED_RESULTS.md)，眼/手兼容性说明见 [COMPATIBILITY.md](../../Head/COMPATIBILITY.md)。

## 数据

- `corpus/`：16 个开发盘、32 个保留测试盘、3 个预热盘；所有 51 个盘都有独立复验的完整清盘见证
- `evaluation/`：打开保留测试集之前记录的算法哈希、设置和棋盘输入；机器本地绝对路径已改为仓库相对路径，源文件与棋盘哈希不变
- `frozen/`：原版、上轮优化版和 V4 的不可变代码快照
- `fast_solver_finalist.py`：实际参加封存测试的算法快照；正式接口位于 `Head/certified_solver.py`
- `results/`：每轮完整合法路径、实际耗时和汇总；`platform_publication.json` 记录实验平台
- `arena/`：公开 issue #1 原始棋盘与出处

求解进程只收到棋盘、beam、随机种子、预算；不读取生成器、见证或隐藏种子。这里公开见证供复核，不能再把这些已经看过的盘作为下一轮未见测试集。

## 复现

安装 `numpy numba pytest`，在仓库根目录运行：

```sh
python benchmarks/certified/benchmark_certified.py --boards benchmarks/certified/evaluation/all_boards.json --variants original,previous,candidate --candidate fast_solver_finalist.py --mode god --beam 200 --single-pass --seeds 17 97 --output reproduced.json
python benchmarks/certified/corpus/check_corpus.py
python benchmarks/certified/corpus/check_corpus.py --root benchmarks/certified/corpus/high_digit_extension
python -m pytest -q benchmarks/certified/test_independent_finalist.py
```

`--cold` 使用新进程和空 JIT 缓存计时；热测量只在其他棋盘上预热。完整时间边界和失败样本处理见报告。报告中的 3 秒预算是协作式截止，原版完整搜索会明显超时；请比较实际时间。

## 平台与限制

报告的 CPU 型号为 AMD EPYC 9V74 80-Core Processor；容器仅可见 9 个逻辑 CPU，CPU affinity 也是 9 个。CPU quota 文件未暴露，不能据此宣称拥有 9 个专属物理核心。测量使用 1 个求解进程、1 个 BLAS/Numba 线程，Python 3.12.14、NumPy 2.3.5、Numba 0.68.0、Linux。

保留测试集是四类有保证可解的构造分布，不是均匀随机盘，也不能保证与所有真实难盘等难。公开盘总和为 771，无法全清；159 是上界，测到的 158 仍是 best-found。没有测试 Windows 实机鼠标、真实 OCR 精度或实际游戏效果。
