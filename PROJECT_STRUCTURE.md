# sum10_Nikke 项目架构

```
sum10_Nikke/
├── .gitignore
├── .venv/                  # Python 虚拟环境
├── debug_rois/             # 调试用 ROI 区域数据
├── eyes/                   # 视觉相关工具与模型训练
│   ├── auto_capture.py     # 自动截图采集
│   ├── board_captured.png  # 示例截图
│   ├── board_coordinates.txt
│   ├── coordinate_picker.py
│   ├── OCR_INTEGRATION_GUIDE.md
│   ├── README.md
│   ├── Sum10_Labeling_Tool/   # 标注与训练工具
│   │   ├── app.py
│   │   ├── board_captured.png
│   │   ├── dataset/
│   │   ├── debug_output/
│   │   ├── debug_predict.py
│   │   ├── predict.py
│   │   ├── README.md
│   │   ├── sum10_model.pth
│   │   ├── templates/
│   │   ├── temp_crops/
│   │   ├── train.py
│   │   └── __pycache__/
│   ├── trainer/                # 训练相关脚本和原始图片
│   │   └── raw_images/
│   └── __pycache__/
├── Head/                   # 策略、分析、训练等高级模块
│   ├── deep_dive.py
│   ├── god_brain.py
│   ├── god_brain_v4.py
│   ├── god_brain_v63.md
│   ├── Head_web.html
│   ├── README.md
│   ├── sum10_elite_data.jsonl
│   ├── trainer.py
│   ├── training_analysis.py
│   └── __pycache__/
├── model_data/             # 模型数据
├── README.md               # 项目说明
├── temp_crops/             # 临时图片
├── web_tool/               # Web 工具（FastAPI）
│   ├── app.py
│   ├── debug_vis_grid.png
│   ├── debug_vis_interactive.png
│   ├── README.md
│   ├── static/             # 静态资源
│   ├── templates/          # 网页模板
│   │   └── index.html
│   └── uploads/            # 上传文件
└── __pycache__/            # Python 缓存
```

> 说明：部分目录如 `model_data/`、`temp_crops/`、`uploads/`、`__pycache__/`、`debug_output/`、`raw_images/` 等为数据、缓存或临时文件夹，已在 `.gitignore` 排除。
