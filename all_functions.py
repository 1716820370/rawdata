import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import csv
import time as system_time

# ======================================================================================
# --- 1. 配置区域 (请根据您的实际情况修改) ---
# ======================================================================================

# --- 模型和文件路径 ---
MODEL_PATH = 'yolov8n.pt'
INPUT_VIDEO_PATH = 'test_5.mp4'
OUTPUT_VIDEO_PATH = 'traffic_analysis_result_5.mp4'
OUTPUT_CSV_PATH = 'traffic_analysis_data_5.csv'

# --- 核心新增配置：数据统计周期 (单位：秒) ---
TIME_INTERVAL_SECONDS = 10 

# --- 透视变换校准 ---
SOURCE_POINTS = np.float32([[414, 517], [865, 530], [976, 677], [319, 644]])
REAL_WORLD_DISTANCE_WIDTH_METERS = 3.7*3.5 
REAL_WORLD_DISTANCE_HEIGHT_METERS = 12.19  #中国6m，美国12.19m
TARGET_WIDTH_PIXELS = int(REAL_WORLD_DISTANCE_WIDTH_METERS * 50)
TARGET_HEIGHT_PIXELS = int(REAL_WORLD_DISTANCE_HEIGHT_METERS * 50)
DESTINATION_POINTS = np.float32([
    [0, 0],
    [TARGET_WIDTH_PIXELS, 0],
    [TARGET_WIDTH_PIXELS, TARGET_HEIGHT_PIXELS],
    [0, TARGET_HEIGHT_PIXELS]
])

# --- 新增配置：模型二所需的参数 ---
ALPHA = 1.0
BETA = 1.0
GAMMA = 1.0
N_MAX = 30

# --- 绘图和字体设置 ---
BOX_COLOR = (0, 255, 255)
FONT_THICKNESS = 2
try:
    font_label = ImageFont.truetype("simhei.ttf", 18, encoding="utf-8")
    font_display = ImageFont.truetype("simhei.ttf", 24, encoding="utf-8")
except IOError:
    print("未找到中文字体'simhei.ttf'，将使用默认字体。")
    font_label = ImageFont.load_default()
    font_display = ImageFont.load_default()

# ======================================================================================
# --- 2. 程序初始化 ---
# ======================================================================================

model = YOLO(MODEL_PATH)
M = cv2.getPerspectiveTransform(SOURCE_POINTS, DESTINATION_POINTS)
PIXELS_PER_METER = TARGET_HEIGHT_PIXELS / REAL_WORLD_DISTANCE_HEIGHT_METERS

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (frame_width, frame_height))

FLOW_DETECTION_LINE_Y = int(frame_height * 0.75) 
flow_counted_ids = set()

interval_start_time = system_time.time()
interval_flow = 0
interval_speeds = []
interval_vehicle_counts = []

vehicle_tracking = {}

try:
    csv_file = open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Timestamp (s)', 'Traffic Flow (q)', 'Avg Speed (v, km/h)', 
                         'Occupancy (O)', 'TPI (gamma)', 'Congestion Index (D)'])
except Exception as e:
    print(f"警告: 无法创建CSV文件. 错误: {e}")
    csv_writer = None

# ======================================================================================
# --- 3. 主程序循环 ---
# ======================================================================================

frame_id = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_id += 1
    current_time_sec = frame_id / FPS

    results = model.track(frame, persist=True, tracker="bytetrack.yaml")
    
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # 在Pillow图像上绘制可视化线条
    draw.line([(0, FLOW_DETECTION_LINE_Y), (frame_width, FLOW_DETECTION_LINE_Y)], fill=(255, 0, 0), width=2)
    # 将Numpy点集转换为Pillow可绘制的格式
    poly_points = [tuple(p) for p in SOURCE_POINTS]
    draw.polygon(poly_points, outline=(0, 255, 0), width=2)


    current_frame_vehicle_count = len(results[0].boxes) if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None else 0
    interval_vehicle_counts.append(current_frame_vehicle_count)

    if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id in [2, 3, 5, 7]:
                vehicle_type = "车辆"
                
                bottom_center_point = np.float32([[(x1 + x2) / 2, y2]])
                warped_point = cv2.perspectiveTransform(bottom_center_point.reshape(1, 1, 2), M)
                warped_x, warped_y = warped_point[0][0]

                if track_id not in vehicle_tracking:
                    vehicle_tracking[track_id] = {'positions': [], 'speeds': [], 'final_speed': 0}
                vehicle_tracking[track_id]['positions'].append((warped_x, warped_y, current_time_sec))

                history = vehicle_tracking[track_id]['positions']
                if len(history) >= 2:
                    x_new, y_new, time_new = history[-1]
                    x_old, y_old, time_old = history[0]
                    time_diff = time_new - time_old
                    if time_diff > 0:
                        pixel_distance = np.sqrt((x_new - x_old)**2 + (y_new - y_old)**2)
                        real_distance_meters = pixel_distance / PIXELS_PER_METER
                        speed_ms = real_distance_meters / time_diff
                        speed_kmh = speed_ms * 3.6
                        
                        vehicle_tracking[track_id]['speeds'].append(speed_kmh)
                        if len(vehicle_tracking[track_id]['speeds']) > 5:
                            vehicle_tracking[track_id]['speeds'].pop(0)
                        
                        avg_speed = np.mean(vehicle_tracking[track_id]['speeds'])
                        vehicle_tracking[track_id]['final_speed'] = avg_speed
                
                if y2 > FLOW_DETECTION_LINE_Y and track_id not in flow_counted_ids:
                    interval_flow += 1
                    flow_counted_ids.add(track_id)

                speed_to_display = vehicle_tracking[track_id]['final_speed']
                label = f"ID:{track_id} {speed_to_display:.1f} km/h"
                draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=FONT_THICKNESS)
                draw.text((x1, y1 - 20), label, font=font_label, fill=BOX_COLOR)

    if system_time.time() - interval_start_time >= TIME_INTERVAL_SECONDS:
        
        current_speeds = [data['final_speed'] for data in vehicle_tracking.values() if data['final_speed'] > 0]
        avg_speed_v = np.mean(current_speeds) if current_speeds else 0.0
        flow_q = interval_flow
        occupancy_o = np.mean(interval_vehicle_counts) if interval_vehicle_counts else 0.0
        
        denominator_gamma = (flow_q * avg_speed_v) + 0.00001
        tpi_gamma = (occupancy_o**2 * 10) / denominator_gamma
        
        term1 = ALPHA * (1 / (avg_speed_v + 0.00001))
        term2 = BETA * (occupancy_o / N_MAX)
        term3 = GAMMA * occupancy_o
        congestion_d = term1 + term2 + term3

        if csv_writer:
            csv_writer.writerow([f"{current_time_sec:.2f}", flow_q, f"{avg_speed_v:.2f}",
                                 f"{occupancy_o:.2f}", f"{tpi_gamma:.2f}", f"{congestion_d:.2f}"])
        
        interval_start_time = system_time.time()
        interval_flow = 0
        interval_vehicle_counts = []
        flow_counted_ids.clear()
        vehicle_tracking.clear()

    # --- 绘制实时统计信息到画面 (已修正) ---
    # 修正逻辑：必须先在Pillow图像上绘制所有文本，然后再转换回OpenCV格式
    display_text = f"周期内流量: {interval_flow} | 画面车辆数: {current_frame_vehicle_count}"
    draw.text((20, 20), display_text, font=font_display, fill=(255, 255, 255))
    
    # 将Pillow图像转回OpenCV格式用于显示和写入
    # 修正拼写：COLOR_RGB_BGR -> COLOR_RGB2BGR
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    
    out.write(frame)
    cv2.imshow("交通指数分析系统", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======================================================================================
# --- 4. 程序收尾 ---
# ======================================================================================
cap.release()
out.release()
if csv_writer:
    csv_file.close()
cv2.destroyAllWindows()
print(f"处理完成！视频已保存至 {OUTPUT_VIDEO_PATH}, 数据已保存至 {OUTPUT_CSV_PATH}")
