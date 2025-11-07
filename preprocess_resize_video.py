import cv2
import os

# =================================================================================
# ---                           配置区域 (请在此处修改)                         ---
# =================================================================================

# 1. 输入和输出文件名
# --- 请将这里的文件名改成您的高清视频文件名 ---
input_video_name = '8.mp4'
# ---------------------------------------------
# 这是处理后输出的优化版视频文件名
output_video_name = 'test_6.mp4'

# 2. 目标宽度
# 您希望视频被缩小到多宽？1280 是一个很好的通用值 (720p 宽度)。
# 如果您的原始视频是竖屏，您可能需要设置 TARGET_HEIGHT。
TARGET_WIDTH = 1280

# =================================================================================
# ---                           主程序逻辑                                  ---
# =================================================================================

def resize_video():
    # 检查输入文件是否存在
    if not os.path.exists(input_video_name):
        print(f"错误: 找不到输入文件 '{input_video_name}'。请确保文件名正确，且文件与脚本在同一个文件夹下。")
        return

    # 1. 打开原始视频文件
    cap = cv2.VideoCapture(input_video_name)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 '{input_video_name}'。")
        return

    # 2. 获取原始视频的属性
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 3. 计算新的、保持长宽比的尺寸
    aspect_ratio = original_height / original_width
    new_width = TARGET_WIDTH
    new_height = int(new_width * aspect_ratio)

    print(f"准备处理视频: '{input_video_name}'")
    print(f"原始尺寸: {original_width}x{original_height}, @ {fps:.2f} FPS")
    print(f"目标尺寸: {new_width}x{new_height}, @ {fps:.2f} FPS")

    # 4. 定义视频编码器并创建 VideoWriter 对象用于写入新视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, fps, (new_width, new_height))

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break # 视频读取结束

        # 5. 将当前帧缩小到目标尺寸
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # 6. 将缩小后的帧写入输出文件
        out.write(resized_frame)

        frame_count += 1
        # 打印进度
        progress = (frame_count / total_frames) * 100
        print(f"\r处理中... 进度: {progress:.2f}% ({frame_count}/{total_frames})", end="")

    # 7. 释放所有资源
    cap.release()
    out.release()
    print(f"\n处理完成! 优化后的视频已保存为: '{output_video_name}'")

# 当直接运行这个脚本时，执行 resize_video 函数
if __name__ == '__main__':
    resize_video()
