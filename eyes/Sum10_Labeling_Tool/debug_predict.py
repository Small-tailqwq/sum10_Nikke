import cv2
import numpy as np
import os

# === 配置 ===
# 找一张你的截图
TEST_IMAGE = 'board_captured.png' 
OUTPUT_DIR = 'debug_output'
IMG_SIZE = 64
ROWS = 16
COLS = 10
CROP_RATIO = 0.8  # 这就是罪魁祸首，因为它和数据集的处理方式不一致

os.makedirs(OUTPUT_DIR, exist_ok=True)

def debug_cut():
    if not os.path.exists(TEST_IMAGE):
        print(f"❌ 找不到 {TEST_IMAGE}")
        return

    img = cv2.imread(TEST_IMAGE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    print(f"正在生成预测时的视野... 请查看 {OUTPUT_DIR} 文件夹")

    for r in range(ROWS):
        for c in range(COLS):
            # 1. 浮点坐标切分
            y1 = int(r * (h / ROWS))
            y2 = int((r + 1) * (h / ROWS))
            x1 = int(c * (w / COLS))
            x2 = int((c + 1) * (w / COLS))
            
            cell = gray[y1:y2, x1:x2]
            
            # 2. 中心裁切 (Prediction 用的逻辑)
            ch, cw = cell.shape
            new_h = int(ch * CROP_RATIO)
            new_w = int(cw * CROP_RATIO)
            start_y = (ch - new_h) // 2
            start_x = (cw - new_w) // 2
            clean_cell = cell[start_y : start_y+new_h, start_x : start_x+new_w]
            
            # 3. 缩放
            final_cell = cv2.resize(clean_cell, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            
            # 保存
            cv2.imwrite(f"{OUTPUT_DIR}/r{r}_c{c}.png", final_cell)

if __name__ == '__main__':
    debug_cut()