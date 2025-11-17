from ultralytics import YOLO
import cv2
import os
import csv
import numpy as np
import time
import zipfile
import requests
from sort import Sort
import shutil
import psutil

# ================================================================
# FOLDERS + FIXED OUTPUT FILENAMES
# ================================================================
os.makedirs("Outputs", exist_ok=True)
os.makedirs("foregrounds", exist_ok=True)
os.makedirs("csv_logs", exist_ok=True)

CSV_PATH = "csv_logs/detections_metadata.csv"
ZIP_PATH = "Outputs/foregrounds.zip"
UPLOAD_URL = "http://192.168.31.30:5000/upload"

# Clear old files
if os.path.exists(CSV_PATH): os.remove(CSV_PATH)
if os.path.exists(ZIP_PATH): os.remove(ZIP_PATH)
shutil.rmtree("foregrounds")
os.makedirs("foregrounds")


# ================================================================
# MODEL + TRACKER
# ================================================================
model = YOLO("weights/yolov8n.pt")
class_names = model.names
tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.3)


# ================================================================
# VIDEO INPUT + OUTPUT
# ================================================================
cap = cv2.VideoCapture("data/hehe2.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("Outputs/output.mp4", fourcc, fps, (width, height))


# ================================================================
# CSV INIT
# ================================================================
csv_file = open(CSV_PATH, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "label", "track_id", "confidence", "x", "y", "w", "h"])


# ================================================================
# IOU CALCULATOR
# ================================================================
def bbox_iou_numpy(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    x11, y11, x12, y12 = np.split(a[:, :4], 4, axis=1)
    x21, y21, x22, y22 = np.split(b[:, :4], 4, axis=1)

    xa = np.maximum(x11, x21.T)
    ya = np.maximum(y11, y21.T)
    xb = np.minimum(x12, x22.T)
    yb = np.minimum(y12, y22.T)

    inter = np.clip(xb - xa, 0, None) * np.clip(yb - ya, 0, None)
    area1 = (x12-x11) * (y12-y11)
    area2 = (x22-x21) * (y22-y21)

    return inter / (area1 + area2.T - inter + 1e-6)


CONF_THRESH = 0.5
frame_num = 0
prev_gray = None

yolo_times, full_times, cpu_usages = [], [], []
detection_counts = {cls: 0 for cls in class_names.values()}

while cap.isOpened():
    start_full = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    # ==========================================================
    # REPLACED: Foreground Motion Filter (your new one)
    # ==========================================================
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        _, fg_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        moving_ratio = np.sum(fg_mask > 0) / (fg_mask.shape[0] * fg_mask.shape[1])

        # Skip frame if almost no movement
        if moving_ratio < 0.001:
            prev_gray = gray
            continue

    prev_gray = gray
    # ==========================================================

    # ---------- YOLOv8n inference ----------
    start_yolo = time.time()
    results = model(frame, verbose=False)
    yolo_time = (time.time() - start_yolo) * 1000  # ms
    yolo_times.append(yolo_time)

    detections = []
    det_labels = []
    det_confs = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(len(boxes)):
            if confs[i] > CONF_THRESH:
                x1, y1, x2, y2 = boxes[i]
                label = class_names[cls_ids[i]]
                detections.append([x1, y1, x2, y2, confs[i]])
                det_labels.append(label)
                det_confs.append(confs[i])

    if len(detections) > 0:
        detections = np.array(detections)
        tracks = tracker.update(detections)

        for j, track in enumerate(tracks):
            x1, y1, x2, y2, track_id = track.astype(int)
            w, h = x2 - x1, y2 - y1
            label = det_labels[j] if j < len(det_labels) else "unknown"
            conf = det_confs[j] if j < len(det_confs) else 0.0
            detection_counts[label] += 1

            # ---------- Save foreground crop ----------
            os.makedirs("foregrounds", exist_ok=True)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size > 0:
                filename = f"{frame_num:04d}_{int(track_id)}_{label}.jpg"
                filepath = os.path.join("foregrounds", filename)
                cv2.imwrite(filepath, crop)

            # ---------- Write to CSV ----------
            csv_writer.writerow([frame_num, label, int(track_id), f"{conf:.2f}", x1, y1, w, h])

            # ---------- Draw tracking box ----------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}-{int(track_id)} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)
    full_times.append(time.time() - start_full)
    cpu_usages.append(psutil.cpu_percent(interval=None))

cap.release()
out.release()
csv_file.close()


# ================================================================
# CLEANUP
# ================================================================
cap.release()
out.release()
csv_file.close()


# ================================================================
# ZIP FOREGROUNDS
# ================================================================
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk("foregrounds"):
        for f in files:
            zipf.write(os.path.join(root, f))

'''
# ================================================================
# UPLOAD TO SERVER
# ================================================================
files_to_send = [
    "Outputs/foregrounds.zip",
    "csv_logs/detections_metadata.csv"
]

for file_path in files_to_send:
    with open(file_path, "rb") as f:
        r = requests.post(
                UPLOAD_URL,
            files={"file": (file_path.split("/")[-1], f)},
            timeout=None   # <-- IMPORTANT
        )
    print("Uploaded:", file_path, "->", r.text)
'''

