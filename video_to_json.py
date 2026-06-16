from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import json

model = YOLO("yolo26x.pt")

video_path = "film.mp4"
cap = cv2.VideoCapture(video_path)

tracks = defaultdict(list)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with tqdm(total=total_frames, desc="Postęp detekcji", unit="klatka") as pbar:
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            result = model.track(frame, persist=True, verbose=False)[0]

            if result.boxes and result.boxes.is_track:
                track_ids = result.boxes.id.int().cpu().tolist()

                for box, track_id in zip(result.boxes, track_ids):
                    x, y, w, h = box.xywh[0].cpu().tolist()
                    conf = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    detection = {
                        "bbox": [x, y, w, h],
                        "confidence": conf,
                        "class_name": class_name,
                        "class_id": class_id
                    }
                    tracks[track_id].append(detection)

            pbar.update(1)
        else:
            break

cap.release()

with open("trackin.json", "w", encoding="utf-8") as f:
    json.dump(tracks, f, indent=4, ensure_ascii=False)

print(f"\nSukces! Wszystkie detekcje zostały zapisane.")