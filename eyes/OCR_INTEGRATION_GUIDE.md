# OCR集成使用指南

## 🎯 功能概述

现在您可以通过一键OCR自动识别棋盘,无需手动输入数据!

## 📋 使用流程

### 1. 首次设置(仅需一次)

在使用OCR功能前,需要先设置截图坐标:

```bash
cd eyes
python coordinate_picker.py
```

**操作步骤:**
1. 打开游戏到棋盘界面
2. 运行脚本后按任意键截图
3. 依次点击棋盘的**四个角**:
   - 左上角 (Top-Left)
   - 右上角 (Top-Right)
   - 左下角 (Bottom-Left)
   - 右下角 (Bottom-Right)
4. 坐标将保存到 `board_coordinates.txt`

### 2. 启动后端服务

```bash
cd Head
python god_brain.py
```

**成功标志:**
- 看到 `✅ OCR模块已加载`
- 看到 `>> [系统] Numba 加速引擎已装载`
- 服务运行在 `http://localhost:8000`

### 3. 打开Web界面

在浏览器中打开 `Head/Head_web.html`

### 4. 使用OCR功能

#### 方法一:一键OCR(推荐)

1. **切换到游戏窗口**,确保棋盘完全可见
2. 点击Web界面的 **"📸 AUTO OCR CAPTURE"** 按钮
3. 系统会:
   - 倒数2秒提示
   - 自动截取屏幕
   - 识别棋盘数字
   - 自动加载数据到界面
4. 看到 "OCR COMPLETE" 提示后,点击 **"INITIALIZE"** 开始计算

#### 方法二:手动输入

如果OCR失败或想自定义数据:
1. 点击 **"DATA INPUT"** 
2. 粘贴或生成数据
3. 点击 **"LOAD DATA"**

## 🔍 技术细节

### 文件结构

```
eyes/
  ├── coordinate_picker.py      # 坐标设置工具
  ├── auto_capture.py           # 截图模块(已增强)
  └── Sum10_Labeling_Tool/
      ├── predict.py            # OCR识别模型
      └── sum10_model.pth       # 训练好的权重
  └── trainer/
      └── raw_images/           # OCR截图保存位置(自动创建)
          └── board_YYYYMMDD_HHMMSS.png

Head/
  ├── god_brain.py              # 后端服务(已集成OCR)
  └── Head_web.html             # Web界面(已添加OCR按钮)
```

### 数据流

```
用户点击OCR按钮
    ↓
前端发送 {cmd: 'RUN_OCR'} (WebSocket)
    ↓
后端 god_brain.py 接收命令
    ↓
调用 auto_capture_and_unwarp()
    → 截图保存到 raw_images/board_TIMESTAMP.png
    ↓
调用 Sum10Recognizer.recognize_board()
    → 识别出 16x10 矩阵
    ↓
转换为字符串 "1234567890..." (160位)
    ↓
发送 {type: 'OCR_RESULT', raw_data: "..."} (WebSocket)
    ↓
前端接收并自动加载
    → initGame(raw_data)
    → 显示在棋盘上
```

### 修改内容

#### `auto_capture.py`
- ✅ 添加 `use_timestamp` 参数:保存到 `raw_images/` 并使用时间戳命名
- ✅ 添加 `silent` 参数:静默模式,不弹窗显示
- ✅ 返回实际保存路径

#### `god_brain.py`
- ✅ 导入OCR相关模块 (`auto_capture`, `Sum10Recognizer`)
- ✅ 启动时加载OCR模型
- ✅ 添加 `RUN_OCR` WebSocket命令处理
- ✅ 使用异步线程池避免阻塞

#### `Head_web.html`
- ✅ 添加 **"📸 AUTO OCR CAPTURE"** 按钮
- ✅ 添加 `triggerOCR()` 函数
- ✅ 在 `handleServerMsg()` 中处理 `OCR_RESULT` 和 `OCR_ERROR`
- ✅ Toast提示显示OCR状态

## ⚠️ 常见问题

### Q: 点击OCR按钮没反应?
**A:** 检查:
1. `god_brain.py` 是否正在运行
2. 控制台是否显示 `✅ OCR模块已加载`
3. 浏览器控制台(F12)是否有WebSocket连接错误

### Q: 提示"截图失败,请先运行coordinate_picker.py"?
**A:** 说明坐标文件不存在,需要先运行 `eyes/coordinate_picker.py` 设置坐标

### Q: OCR识别不准确?
**A:** 可能原因:
- 游戏界面分辨率变化(需重新运行 `coordinate_picker.py`)
- 棋盘被遮挡或部分不可见
- 游戏字体与训练数据差异较大

解决方法:
- 重新校准坐标
- 确保截图时棋盘完整清晰
- 使用手动输入作为备选方案

### Q: 截图保存在哪里?
**A:** 所有OCR截图保存在 `eyes/trainer/raw_images/` 目录,文件名格式为 `board_YYYYMMDD_HHMMSS.png`

## 🚀 快速测试

```bash
# 1. 设置坐标(仅首次)
cd eyes
python coordinate_picker.py

# 2. 启动后端
cd ../Head
python god_brain.py

# 3. 打开浏览器
# 访问 Head/Head_web.html

# 4. 测试OCR
# - 切换到游戏窗口
# - 点击 "📸 AUTO OCR CAPTURE"
# - 等待识别完成
# - 点击 "INITIALIZE" 开始计算
```

## 💡 提示

- **截图时机**: 确保游戏棋盘清晰可见,无UI遮挡
- **倒计时**: OCR按钮点击后有2秒延迟,给你时间切换窗口
- **历史记录**: 所有截图都会保存,方便调试和复查
- **备选方案**: OCR失败时仍可使用"DATA INPUT"手动输入

---

✨ **享受一键识别的便利吧!**
