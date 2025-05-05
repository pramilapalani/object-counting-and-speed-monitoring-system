import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
import pandas as pd

model = YOLO("yolov8n.pt")
REGIONS = {
    "entry": ((50, 200), (300, 400)),
    "exit": ((400, 100), (600, 300))
}

unique_ids = set()
region_counts = defaultdict(set)
speeds = {}
prev_positions = {}
log_data = []

def draw_regions(frame):
    for name, ((x1, y1), (x2, y2)) in REGIONS.items():
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def estimate_speed(object_id, current_pos, fps):
    prev_pos = prev_positions.get(object_id)
    if prev_pos:
        distance = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))
        speed = distance * fps * 0.05
        speeds[object_id] = speed
    prev_positions[object_id] = current_pos

def count_in_region(object_id, centroid):
    for name, ((x1, y1), (x2, y2)) in REGIONS.items():
        if x1 <= centroid[0] <= x2 and y1 <= centroid[1] <= y2:
            region_counts[name].add(object_id)

def process_video(video_path):
    global log_data
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml")[0]
        annotated_frame = results.plot()
        draw_regions(annotated_frame)

        if results.boxes.id is not None:
            for box, track_id in zip(results.boxes.xyxy, results.boxes.id):
                id = int(track_id.item())
                x1, y1, x2, y2 = map(int, box[:4])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                height = y2 - y1

                unique_ids.add(id)
                estimate_speed(id, (cx, cy), fps)
                count_in_region(id, (cx, cy))

                speed = speeds.get(id, 0)
                log_data.append({
                    "timestamp": datetime.now(),
                    "frame_id": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    "object_id": id,
                    "region": [r for r in REGIONS if id in region_counts[r]],
                    "speed_kmph": speed
                })

                # Highlighted speed display
                speed_text = f"{speed:.1f} km/h"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                (text_width, text_height), _ = cv2.getTextSize(speed_text, font, font_scale, font_thickness)
                text_x = int(cx - text_width / 2)
                text_y = int(cy + height / 2 + 25)  # Below bounding box

                # Draw background rectangle
                cv2.rectangle(annotated_frame,
                              (text_x - 5, text_y - text_height - 5),
                              (text_x + text_width + 5, text_y + 5),
                              (0, 0, 0),  # Black background
                              -1)  # Filled rectangle

                # Draw speed text
                cv2.putText(annotated_frame, speed_text,
                            (text_x, text_y),
                            font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

        cv2.putText(annotated_frame, f"Total Objects: {len(unique_ids)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        y_offset = 60
        for region, ids in region_counts.items():
            cv2.putText(annotated_frame, f"{region.capitalize()} Count: {len(ids)}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 30

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    os.makedirs("logs", exist_ok=True)
    pd.DataFrame(log_data).to_csv("logs/object_stats.csv", index=False)
