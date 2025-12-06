# Sum10 标注与训练工具箱 (Labeling & Training Tool)

## 🎯 简介
这是 Sum10 Nikke 项目的视觉核心，负责生产“眼睛”。
它包含了一整套从数据标注、模型训练到推理识别的完整工作流。

## 📦 依赖安装

```bash
pip install flask torch torchvision numpy pillow scikit-learn opencv-python
```

## 🛠️ 功能指南

### 1. 数据标注 (Labeling)
使用 Web 界面快速构建训练数据集。

1.  启动标注工具：
    ```bash
    python app.py
    ```
2.  浏览器访问 `http://127.0.0.1:5000`。
3.  **操作流程**：
    *   上传游戏截图。
    *   系统自动切分为小图。
    *   使用键盘 **1-9** 快速打标，**0** 或 **N** 标记为噪点/无效。
    *   数据会自动保存到 `dataset/` 目录下。

### 2. 模型训练 (Training)
当积累了足够的数据（建议每个数字 50+ 张）后，开始训练模型。

1.  运行训练脚本：
    ```bash
    python train.py
    ```
2.  脚本会自动：
    *   加载 `dataset/` 下的图片。
    *   进行数据增强和预处理 (Resize 64x64)。
    *   训练 CNN 模型 (100 Epochs)。
    *   保存最佳模型为 `sum10_model.pth`。

### 3. 推理识别 (Inference)
`predict.py` 提供了 `Sum10Recognizer` 类，被主程序 `Head/god_brain.py` 调用。
它负责将屏幕截图切分并识别为数字矩阵。

## 📂 文件结构

*   `app.py`: 标注工具 (Flask Web Server)。
*   `train.py`: 模型训练脚本。
*   `predict.py`: 推理与识别逻辑。
*   `dataset/`: 训练数据集存储位置。
*   `sum10_model.pth`: 训练好的模型文件。
*   `temp_crops/`: 标注过程中的临时文件。

## 🧠 模型说明
*   **架构**: 3层卷积 (Conv2d) + BatchNorm + 2层全连接。
*   **输入**: 64x64 灰度图。
*   **输出**: 10分类 (1-9 及 干扰项)。

---
*Powered by Gemini 3 Pro, debugged by small-tailqwq*