import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os

# === 配置 ===
MODEL_PATH = 'sum10_model.pth'
IMG_SIZE = 64
ROWS = 16
COLS = 10
CROP_RATIO = 0.8  # 必须与切图时的逻辑保持一致

# === 1. 模型定义 (必须与 V2版 train.py 完全一致) ===
class SimpleDigitNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleDigitNet, self).__init__()
        # 对应 V2 训练脚本的结构 (包含 BatchNorm)
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# === 2. 识别器类 ===
class Sum10Recognizer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Loading model on {self.device}...")
        
        self.model = SimpleDigitNet(num_classes=10).to(self.device)
        # 加载权重
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() 
        
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def predict_image(self, img_pil):
        img_gray = img_pil.convert('L')
        img_tensor = self.transform(img_gray).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            return predicted.item()

    def recognize_board(self, image_path):
        print(f"📸 Reading: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("无法读取图片")

        # 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 假设输入就是棋盘区域 (如果是全屏截图，请确保这里传入的是已经裁切好的棋盘区域，或者用之前的智能裁切逻辑)
        # 这里为了演示简单，假设图片已经是棋盘
        roi = gray 
        
        h, w = roi.shape
        matrix = [[0] * COLS for _ in range(ROWS)]
        
        print("🔍 Scanning board...")
        for r in range(ROWS):
            row_str = ""
            for c in range(COLS):
                # 浮点数坐标计算
                y1 = int(r * (h / ROWS))
                y2 = int((r + 1) * (h / ROWS))
                x1 = int(c * (w / COLS))
                x2 = int((c + 1) * (w / COLS))
                
                cell = roi[y1:y2, x1:x2]
                
                # 中心裁切 (Center Crop)
                ch, cw = cell.shape
                new_h = int(ch * CROP_RATIO)
                new_w = int(cw * CROP_RATIO)
                start_y = (ch - new_h) // 2
                start_x = (cw - new_w) // 2
                clean_cell = cell[start_y : start_y+new_h, start_x : start_x+new_w]
                
                pil_img = Image.fromarray(clean_cell)
                digit = self.predict_image(pil_img)
                matrix[r][c] = digit
                row_str += f"{digit:2d} "
            
            print(f"Row {r+1:02d}: {row_str}")
            
        return matrix

# === 测试入口 ===
if __name__ == '__main__':
    # 请确保项目目录下有一张图片叫 board_captured.png
    # 或者修改下面的路径指向你的任何一张截图
    TEST_IMAGE = 'board_captured.png' 
    
    if os.path.exists(TEST_IMAGE):
        recognizer = Sum10Recognizer(MODEL_PATH)
        result = recognizer.recognize_board(TEST_IMAGE)
        
        print("\n✅ Final Matrix for Solver:")
        print(result)
    else:
        print(f"⚠️ 找不到测试图片: {TEST_IMAGE}")