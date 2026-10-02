# 旧眼、旧手与 CLEAR SEARCH 的接口兼容性

## 不变的接口

- OCR 仍调用原 `auto_capture_and_unwarp` 与 `Sum10Recognizer.recognize_board`；模型、截图、裁剪和识别流程不变
- OCR 矩阵按行展开，16 行 × 10 列；`START` 仍使用 `rows/cols/map/vals/beamWidth/threads/mode`
- `map[r * cols + c]` 与 `vals[r * cols + c]` 一一对应；关闭的格子忽略其旧数值
- 路径仍是包含边界的 `[r1, c1, r2, c2]`，没有转置、坐标交换或像素值混入
- 原 `GodHand.get_screen_pos`、校准、偏移、DIRECT_INPUT/pyautogui 拖拽和 `EXECUTE_PATH` / `EMERGENCY_EXECUTE` 协议不变
- 默认仍为 `god`；`classic` / `omni` / `god` 搜索保持原行为；新模式显式选择 `complete` / CLEAR SEARCH
- `BETTER_SOLUTION` 原有 `score/path/worker` 字段保留；新增 `optimal/full_clear/upper_bound/status` 只是附加元数据

## 本轮修复

1. 原 OCR 分类器可输出 0，旧 UI 也会把 0 标为 active。新适配器现在接受 0，并按空格处理，避免拒绝整盘；数字仍需为 0–9
2. Numba 与旧后端一样保持可选；没有 Numba 时使用同一算法的纯 Python 实现，但速度和截止延迟会明显变差
3. 脚本启动与包导入分别使用正确的 sibling/relative 导入，不再吞掉模块内部的 ImportError；双向模块别名兼容 Numba 磁盘缓存，避免切换启动方式后缓存报错
4. 求解进程异常会发送 `SOLVER_ERROR`；全失败时 `DONE.success=false`。没有非空合法路径时不启用神之手
5. 新一次 START 清空前次路径，避免全失败后误用旧结果。UI 根据已证上界标志区分最优结果和 best-found，删除无条件“Optimal path generated”提示

旧客户端可忽略新增字段；新客户端也接受没有新字段的旧 `DONE` 消息，但不会据此宣称最优。

## 验证范围

使用真实求解器和 AST 提取的原接口，配合 OCR、WebSocket、输入设备的 stub/mock 检查格式、坐标、路径、错误、混合成功/失败和进程派发。Node VM 检查前端脚本语法和关键消息/按钮状态，不连接浏览器或游戏。

这证明软件接口兼容性，不等于完成 Windows 实机与游戏验收。未运行真实截图/模型推理、鼠标/键盘动作；未验证游戏是否接受所有空角矩形手势。原坐标定位和 OCR 识别误差仍需在实际环境校准。发布前全量验证：Numba 424 项通过；禁用 JIT 424 项通过；另有 6 个 Node VM 前端场景、32 个眼/手专项测试（包含在 424 项中），并重跑两进程派发和 64 个封存输出逐项一致性。详情见 [验证记录](../benchmarks/certified/results/compatibility_validation.json)。
