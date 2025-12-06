"""
自动截图工具 - 使用预设坐标自动截取并校正棋盘
"""
import cv2
import numpy as np
from PIL import ImageGrab
import time
import os

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def unwarp_board(img, corners):
    """
    对图像进行透视变换校正
    Args:
        img: 原始图像
        corners: 4个角点坐标 [(x,y), (x,y), (x,y), (x,y)] (左上, 右上, 右下, 左下)
    Returns:
        校正后的图像
    """
    # 确保角点是 float32 类型
    pts1 = np.array(corners, dtype=np.float32)
    
    # 解包四个点
    (tl, tr, br, bl) = pts1
    
    # 计算宽度 (取上下边长的最大值)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算高度 (取左右边长的最大值)
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 定义目标点坐标
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(pts1, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped


def load_coordinates(filename='board_coordinates.txt'):
    """从文件加载坐标"""
    try:
        filepath = os.path.join(SCRIPT_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('coordinates'):
                    coords_str = line.split('=')[1].strip()
                    coords = eval(coords_str)
                    return coords
    except FileNotFoundError:
        print(f"错误: 未找到坐标文件 {filename}")
        print("请先运行 coordinate_picker.py 来选择棋盘坐标")
    return None


def auto_capture_and_unwarp(coords=None, save_path=None, use_timestamp=False, silent=False):
    """
    自动截取屏幕并校正棋盘
    
    Args:
        coords: 棋盘四个角的坐标，如果为None则从文件加载
        save_path: 保存路径，如果为None则不保存
        use_timestamp: 是否使用时间戳命名（保存到 trainer/raw_images/）
        silent: 静默模式，不显示窗口和打印信息
    
    Returns:
        校正后的棋盘图像, 实际保存路径（如果保存了）
    """
    if coords is None:
        coords = load_coordinates()
        if coords is None:
            return None, None
    
    # 截取屏幕
    screenshot = ImageGrab.grab()
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    
    # 校正棋盘
    warped = unwarp_board(img, coords)

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # 保存(可选)
    actual_save_path = None
    if save_path or use_timestamp:
        if use_timestamp:
            # 使用时间戳命名，保存到 trainer/raw_images/
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_images_dir = os.path.join(SCRIPT_DIR, 'trainer', 'raw_images')
            os.makedirs(raw_images_dir, exist_ok=True)
            actual_save_path = os.path.join(raw_images_dir, f'board_{timestamp}.png')
        else:
            actual_save_path = save_path

        if actual_save_path:
            cv2.imwrite(actual_save_path, warped)
            if not silent:
                print(f"已保存到 {actual_save_path}")
    
    return warped, actual_save_path


def main():
    print("=== 自动截图工具 ===")
    
    # 加载坐标
    coords = load_coordinates()
    if coords is None:
        return
    
    print(f"已加载坐标: {coords}")
    
    # 延迟截图
    print("\n2秒后将截取屏幕...")
    time.sleep(2)
    
    # 截取并校正
    save_path = os.path.join(SCRIPT_DIR, 'board_captured.png')
    warped, actual_path = auto_capture_and_unwarp(coords, save_path=save_path)
    
    if warped is not None:
        # 显示结果
        cv2.imshow('校正后的棋盘', warped)
        print("\n按任意键退出...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
