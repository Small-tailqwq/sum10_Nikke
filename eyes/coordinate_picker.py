import cv2
import numpy as np
import pyautogui
import time
import os
from PIL import ImageGrab

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class CoordinatePicker:
    def __init__(self):
        self.points = []
        self.img = None
        self.display_img = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"已选择第 {len(self.points)} 个点: ({x}, {y})")
            
            # 在图像上标记点
            cv2.circle(self.display_img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.display_img, str(len(self.points)), (x+10, y+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 如果有多个点，画线连接
            if len(self.points) > 1:
                cv2.line(self.display_img, self.points[-2], self.points[-1], (255, 0, 0), 2)
            
            # 如果选够4个点，闭合区域
            if len(self.points) == 4:
                cv2.line(self.display_img, self.points[-1], self.points[0], (255, 0, 0), 2)
                print("\n已选择4个点，按任意键继续...")
            
            cv2.imshow('选择棋盘四个角 (左上→右上→右下→左下)', self.display_img)
    
    def capture_and_select(self):
        """截取屏幕并让用户选择4个点"""
        print("2秒后将截取屏幕...")
        time.sleep(2)
        
        # 截取整个屏幕
        screenshot = ImageGrab.grab()
        self.img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        self.display_img = self.img.copy()
        
        print("\n请依次点击棋盘的4个角:")
        print("1. 左上角")
        print("2. 右上角")
        print("3. 右下角")
        print("4. 左下角")
        print("\n按 'r' 重新选择, 按 'q' 退出")
        
        cv2.namedWindow('选择棋盘四个角 (左上→右上→右下→左下)')
        cv2.setMouseCallback('选择棋盘四个角 (左上→右上→右下→左下)', self.mouse_callback)
        
        while True:
            cv2.imshow('选择棋盘四个角 (左上→右上→右下→左下)', self.display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('r'):  # 重新选择
                self.points = []
                self.display_img = self.img.copy()
                print("\n已重置，请重新选择4个点")
            elif key == ord('q'):  # 退出
                cv2.destroyAllWindows()
                return None
            elif len(self.points) == 4 and key != 255:  # 选够4个点且按了任意键
                cv2.destroyAllWindows()
                return self.points
    
    def save_coordinates(self, filename='board_coordinates.txt'):
        """保存坐标到文件"""
        if len(self.points) == 4:
            filepath = os.path.join(SCRIPT_DIR, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("# 棋盘四个角的坐标 (左上, 右上, 右下, 左下)\n")
                f.write(f"coordinates = {self.points}\n")
            print(f"\n坐标已保存到 {filepath}")
            print(f"坐标: {self.points}")
            return True
        return False


def load_coordinates(filename='board_coordinates.txt'):
    """从文件加载坐标"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('coordinates'):
                    # 提取坐标
                    coords_str = line.split('=')[1].strip()
                    coords = eval(coords_str)  # 简单方法，生产环境建议用ast.literal_eval
                    return coords
    except FileNotFoundError:
        print(f"未找到文件 {filename}")
    return None


def main():
    print("=== 棋盘坐标选择工具 ===")
    print("\n选择模式:")
    print("1. GUI模式 - 截图后用鼠标点击选择4个角")
    print("2. 延迟模式 - 2秒倒计时后记录鼠标位置(需要手动点4次)")
    
    choice = input("\n请选择模式 (1/2): ").strip()
    
    if choice == '1':
        # GUI模式
        picker = CoordinatePicker()
        points = picker.capture_and_select()
        
        if points:
            print(f"\n最终坐标: {points}")
            picker.save_coordinates()
            
            # 询问是否测试透视变换
            test = input("\n是否测试透视变换? (y/n): ").strip().lower()
            if test == 'y':
                from jietu import unwarp_board
                warped = unwarp_board(picker.img, points)
                cv2.imshow('校正后的棋盘', warped)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
                # 询问是否保存
                save = input("\n是否保存校正后的图像? (y/n): ").strip().lower()
                if save == 'y':
                    save_path = os.path.join(SCRIPT_DIR, 'board_warped.png')
                    cv2.imwrite(save_path, warped)
                    print(f"已保存为 {save_path}")
    
    elif choice == '2':
        # 延迟模式
        points = []
        for i in range(4):
            corner_names = ['左上角', '右上角', '右下角', '左下角']
            print(f"\n请将鼠标移动到棋盘的{corner_names[i]}")
            for j in range(2, 0, -1):
                print(f"{j}...", end=' ', flush=True)
                time.sleep(1)
            
            pos = pyautogui.position()
            points.append((pos.x, pos.y))
            print(f"\n已记录: ({pos.x}, {pos.y})")
        
        print(f"\n最终坐标: {points}")
        
        # 保存坐标
        filepath = os.path.join(SCRIPT_DIR, 'board_coordinates.txt')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# 棋盘四个角的坐标 (左上, 右上, 右下, 左下)\n")
            f.write(f"coordinates = {points}\n")
        print(f"坐标已保存到 {filepath}")


if __name__ == '__main__':
    main()
