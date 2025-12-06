# 截图工具使用说明

## 📁 文件说明

- `jietu.py` - 核心透视变换函数
- `coordinate_picker.py` - 坐标选择工具（首次使用）
- `auto_capture.py` - 自动截图工具（日常使用）
- `board_coordinates.txt` - 保存的坐标配置（自动生成）

## 🚀 快速开始

### 1. 安装依赖

```powershell
pip install opencv-python numpy Pillow pyautogui
```

### 2. 首次使用：选择坐标

```powershell
python eyes/coordinate_picker.py
```

运行后会提示选择模式：

#### 模式1：GUI模式（推荐）✨
1. 选择模式 `1`
2. 2秒后自动截取整个屏幕
3. 在弹出的窗口中，依次点击棋盘的**4个角**：
   - 第1个点：**左上角**
   - 第2个点：**右上角**
   - 第3个点：**右下角**
   - 第4个点：**左下角**
4. 点击完成后，按任意键继续
5. 可选：测试透视变换效果
6. 坐标自动保存到 `eyes/board_coordinates.txt`

**快捷键：**
- `r` - 重新选择坐标
- `q` - 退出程序

#### 模式2：延迟模式
1. 选择模式 `2`
2. 依次将鼠标移动到棋盘的4个角
3. 每个角有2秒倒计时，时间到自动记录鼠标位置
4. 坐标自动保存

### 3. 日常使用：自动截图

设置好坐标后，每次截图只需：

```powershell
python eyes/auto_capture.py
```

程序会：
1. 加载保存的坐标
2. 2秒延迟后截取屏幕
3. 自动校正棋盘
4. 保存到 `eyes/board_captured.png`
5. 显示预览窗口

## ⚠️ 重要提示

### 坐标一致性问题

**问题**：如果游戏窗口移动了位置，之前保存的坐标会失效！

**原因**：坐标是相对于整个屏幕的绝对位置，不是相对于游戏窗口。

**解决方案**：
1. **固定窗口位置**：每次玩游戏时，把游戏窗口放在同一个位置
2. **重新校准**：如果移动了窗口，重新运行 `coordinate_picker.py` 选择坐标

### 最佳实践

1. **首次设置时**：
   - 把游戏窗口移动到你习惯的位置
   - 运行 `coordinate_picker.py` 选择坐标
   - 记住这个窗口位置

2. **日常使用时**：
   - 确保游戏窗口在相同位置
   - 直接运行 `auto_capture.py` 截图

3. **如果截图不对**：
   - 检查游戏窗口位置是否改变
   - 重新运行 `coordinate_picker.py` 校准

## 📊 输出文件

所有文件都保存在 `eyes/` 目录下：

- `board_coordinates.txt` - 坐标配置文件
- `board_captured.png` - 自动截图的校正结果
- `board_warped.png` - coordinate_picker测试时的校正结果

## 🔧 高级用法

### 在代码中使用

```python
from eyes.jietu import unwarp_board
from eyes.auto_capture import load_coordinates
from PIL import ImageGrab
import cv2
import numpy as np

# 加载坐标
coords = load_coordinates()

# 截取屏幕
screenshot = ImageGrab.grab()
img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# 校正棋盘
warped = unwarp_board(img, coords)

# 使用 warped 进行后续处理...
```

### 调整输出尺寸

编辑 `jietu.py` 中的 `width, height` 参数：

```python
# 默认 400x640，可以改成你需要的尺寸
width, height = 400, 640  
```

## 🐛 常见问题

### Q: 为什么两次截图结果不一样？
A: 确保游戏窗口位置没有移动。坐标是屏幕绝对位置。

### Q: 点击窗口后坐标不准？
A: 确保在原始大小的窗口中点击，不要缩放窗口。

### Q: 能否自动检测窗口？
A: 当前版本需要手动固定窗口位置。未来可以考虑添加窗口检测功能。

### Q: 保存的图片在哪里？
A: 所有文件保存在 `eyes/` 目录下。

## 📝 文件结构

```
eyes/
├── jietu.py                    # 核心变换函数
├── coordinate_picker.py        # 坐标选择工具
├── auto_capture.py             # 自动截图工具
├── board_coordinates.txt       # 坐标配置（自动生成）
├── board_captured.png          # 截图结果（自动生成）
└── board_warped.png            # 测试结果（可选）
```
