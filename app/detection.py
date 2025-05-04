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
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)

                unique_ids.add(id)
                estimate_speed(id, (cx, cy), fps)
                count_in_region(id, (cx, cy))

                log_data.append({
                    "timestamp": datetime.now(),
                    "frame_id": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    "object_id": id,
                    "region": [r for r in REGIONS if id in region_counts[r]],
                    "speed_kmph": speeds.get(id, 0)
                })

                cv2.putText(annotated_frame, f"ID:{id} {speeds.get(id, 0):.1f} km/h", (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

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
