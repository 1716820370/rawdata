import cv2
import sys

# --- 1. 配置区域 ---
# --- 请将这里修改为您的视频文件路径 ---
VIDEO_PATH = 'test_5.mp4'
# -----------------

WINDOW_NAME = "请点击四个角点进行校准 (顺序: 左上 -> 右上 -> 右下 -> 左下)，然后按 'q' 键退出"

def mouse_callback(event, x, y, flags, param):
    """鼠标点击回调函数，用于记录坐标"""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"坐标: [{x}, {y}]")

# 读取视频的第一帧
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"错误：无法打开视频文件 '{VIDEO_PATH}'")
    print("请检查：")
    print("1. 文件路径是否正确。")
    print("2. 文件是否损坏。")
    print("3. 是否安装了正确的OpenCV解码器。")
    sys.exit()

success, frame = cap.read()

if not success:
    print("错误：成功打开了视频文件，但无法读取第一帧。文件可能为空或已损坏。")
    cap.release()
    sys.exit()

# 创建窗口并设置鼠标回调
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

print("请在弹出的窗口中，按 左上 -> 右上 -> 右下 -> 左下 的顺序依次点击矩形区域的四个角点。")
print("坐标将显示在控制台中。完成后，请按键盘上的 'q' 键关闭窗口。")

while True:
    cv2.imshow(WINDOW_NAME, frame)
    # 等待按键，如果按下 'q' 则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
print("\n坐标拾取完成。请将上面打印的4个坐标复制到主程序中。")
