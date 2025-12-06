import os
import cv2
import numpy as np
import base64
import shutil
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# 配置路径
TEMP_DIR = 'temp_crops'
DATASET_DIR = 'dataset'
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

def slice_game_board(image_path):
    print(f"--- 开始处理图片: {image_path} ---")
    img = cv2.imread(image_path)
    if img is None:
        return 0
    
    # 配置：竖屏游戏通常是 宽10 x 高16
    rows = 16
    cols = 10

    # 1. 智能识别部分 (如果你不想用智能识别，直接保留 roi = img 即可)
    roi = None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        board_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(board_contour)
        img_area = img.shape[0] * img.shape[1]
        
        if area > img_area * 0.1:
            x, y, w, h = cv2.boundingRect(board_contour)
            # 这里简单做个边距保护，防止智能识别贴着边太紧
            margin = 2 
            y1 = max(0, y + margin)
            y2 = min(img.shape[0], y + h - margin)
            x1 = max(0, x + margin)
            x2 = min(img.shape[1], x + w - margin)
            roi = img[y1:y2, x1:x2]
            print("✅ 智能识别生效")

    # 2. 兜底逻辑
    if roi is None:
        print("🔄 使用全图模式")
        roi = img

    # ================= ❌ 删除：全局边缘内缩 (这里删掉了之前的 margin code) =================
    # 原因：防止因为截图不对称导致的整体偏移。所有的去边工作交给下面的 Center Crop 完成。
    # =================================================================================

    # 强制转灰度
    if len(roi.shape) == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    h, w = roi.shape[:2] # 获取当前裁切区域的精确宽高

    # 配置输出
    TARGET_SIZE = (64, 64)
    # 如果你的截图边缘干扰很大，可以把这个值调小，比如 0.8 或 0.75
    # 0.8 表示只取格子中间 80% 的区域，这样容错率更高
    CROP_RATIO = 0.8 

    count = 0
    timestamp = int(os.path.getmtime(image_path))
    
    for r in range(rows):
        for c in range(cols):
            # ================= 🆕 修改：使用浮点数计算绝对坐标 =================
            # 这样消除了累积误差。无论 r 多大，坐标都是相对于总高度的精确比例。
            y1 = int(r * (h / rows))
            y2 = int((r + 1) * (h / rows))
            x1 = int(c * (w / cols))
            x2 = int((c + 1) * (w / cols))
            # ===============================================================
            
            # 1. 粗切
            raw_cell = roi[y1:y2, x1:x2]
            
            if raw_cell.shape[0] < 5 or raw_cell.shape[1] < 5:
                continue

            # 2. 中心裁切 (Center Crop)
            cell_h_px, cell_w_px = raw_cell.shape[:2]
            new_h = int(cell_h_px * CROP_RATIO)
            new_w = int(cell_w_px * CROP_RATIO)
            
            start_y = (cell_h_px - new_h) // 2
            start_x = (cell_w_px - new_w) // 2
            
            clean_cell = raw_cell[start_y : start_y+new_h, start_x : start_x+new_w]
            
            # 3. 统一尺寸
            final_cell = cv2.resize(clean_cell, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                
            filename = f"crop_{timestamp}_r{r}_c{c}.png"
            cv2.imwrite(os.path.join(TEMP_DIR, filename), final_cell)
            count += 1
            
    print(f"🎉 处理完成，共生成 {count} 张切片")
    return count

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload_and_cut', methods=['POST'])
def upload_and_cut():
    print("收到上传请求...")
    file = request.files.get('file')
    if not file:
        return jsonify({'success': False, 'error': 'No file'})
    
    temp_path = os.path.join(TEMP_DIR, 'temp_board.png')
    file.save(temp_path)
    
    try:
        count = slice_game_board(temp_path)
        # 切割完删除原大图，节省空间
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if count == 0:
             return jsonify({'success': False, 'error': '未检测到有效格子，请尝试裁剪掉截图边缘的干扰内容'})

        return jsonify({'success': True, 'count': count})
    except Exception as e:
        print(f"Error stack: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/unsigned_images')
def get_unsigned():
    images = []
    if not os.path.exists(TEMP_DIR):
        return jsonify([])
    
    # 按文件名排序，确保顺序对劲
    files = sorted(os.listdir(TEMP_DIR), key=lambda x: os.path.getmtime(os.path.join(TEMP_DIR, x)))
    
    for filename in files:
        if not filename.endswith('.png'): continue
        if 'crop_' not in filename: continue # 只读取切片文件
        
        filepath = os.path.join(TEMP_DIR, filename)
        with open(filepath, "rb") as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
            images.append({
                'filename': filename,
                'data': f"data:image/png;base64,{b64}"
            })
    return jsonify(images)

@app.route('/api/label_batch', methods=['POST'])
def label_batch():
    data = request.json
    items = data.get('items', [])
    success_count = 0
    errors = []
    
    print(f"收到提交: {len(items)} 个样本")
    
    for item in items:
        filename = item['filename']
        label = item['label']
        
        src = os.path.join(TEMP_DIR, filename)
        dst_folder = os.path.join(DATASET_DIR, label)
        os.makedirs(dst_folder, exist_ok=True)
        dst = os.path.join(dst_folder, filename)
        
        try:
            if os.path.exists(src):
                shutil.move(src, dst)
                success_count += 1
            else:
                print(f"文件不存在: {src}")
        except Exception as e:
            print(f"移动失败 {filename}: {e}")
            errors.append(filename)
            
    return jsonify({'success': True, 'count': success_count, 'errors': errors})

if __name__ == '__main__':
    print("启动标注工具... http://127.0.0.1:5000")
    app.run(debug=True, port=5000)