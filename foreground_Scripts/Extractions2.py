import cv2
import os
import csv
import numpy as np
import urllib.request
import time
import psutil  # For CPU and RAM usage
from sort import Sort
import shutil
import ultralytics

# ==============================
# Step 1: Download YOLOv4-tiny if not present
# ==============================

if not os.path.exists("yolov4-tiny.weights"):
    print("Downloading yolov4-tiny.weights...")
    urllib.request.urlretrieve(
        "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
        "yolov4-tiny.weights"
    )

if not os.path.exists("yolov4-tiny.cfg"):
    print("Downloading yolov4-tiny.cfg...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        "yolov4-tiny.cfg"
    )

if not os.path.exists("coco.names"):
    print("Downloading coco.names...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
        "coco.names"
    )

# ==============================
# Step 2: Load model
# ==============================
net = cv2.dnn.readNetFromDarknet("yolov4-tiny.cfg", "yolov4-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open("coco.names", "r") as f:
    class_names = [c.strip() for c in f.readlines()]

output_layers = net.getUnconnectedOutLayersNames()
tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.3)

# ==============================
# Step 3: Video input/output
# ==============================
video_path = "data/Blacktivities.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception("❌ Could not open video")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("Outputs/output_v4tiny.mp4", fourcc, fps, (width, height))

# ==============================
# Step 4: Create directories
# ==============================
os.makedirs("Outputs", exist_ok=True)
os.makedirs("foregrounds", exist_ok=True)
os.makedirs("csv_logs", exist_ok=True)

# ==============================
# Step 5: CSV metadata setup
# ==============================
csv_path = os.path.join("csv_logs", "BLM_metadata.csv")
csv_file = open(csv_path, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["frame", "label", "track_id", "confidence", "x", "y", "w", "h"])

# ==============================
# Step 6: Processing loop
# ==============================
CONF_THRESH = 0.5
frame_num = 0
prev_gray = None

yolo_times, full_times, cpu_usages = [], [], []
detection_counts = {cls: 0 for cls in class_names}

while cap.isOpened():
    start_full = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    # ---------- Foreground filter ----------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        _, fg_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        moving_ratio = np.sum(fg_mask > 0) / (fg_mask.shape[0] * fg_mask.shape[1])
        if moving_ratio < 0.001:
            prev_gray = gray
            continue
    prev_gray = gray

    # ---------- YOLOv4-tiny inference ----------
    start_yolo = time.time()
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    yolo_time = (time.time() - start_yolo) * 1000  # ms
    yolo_times.append(yolo_time)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONF_THRESH:
                cx, cy, w, h = (det[0:4] * np.array([width, height, width, height])).astype('int')
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, 0.4)

    if len(idxs) > 0:
        detections = []
        det_labels = []
        det_confs = []
        for i in idxs.flatten():
            x, y, w, h = boxes[i]
            conf = confidences[i]
            label = class_names[class_ids[i]]

            detections.append([x, y, x + w, y + h, conf])
            det_labels.append(label)
            det_confs.append(conf)

        detections = np.array(detections)
        tracks = tracker.update(detections)

        for j, track in enumerate(tracks):
            x1, y1, x2, y2, track_id = track.astype(int)
            w, h = x2 - x1, y2 - y1
            label = det_labels[j] if j < len(det_labels) else "unknown"
            conf = det_confs[j] if j < len(det_confs) else 0.0
            detection_counts[label] += 1

            # ---------- Save foreground crop ----------
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size > 0:
                filename = f"{frame_num:04d}_{int(track_id)}_{label}.jpg"
                filepath = os.path.join("foregrounds/Blacktivities", filename)
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

# ==============================
# Step 7: Performance report
# ==============================
frame_count = frame_num
avg_yolo_time = np.mean(yolo_times)
avg_full_time = np.mean(full_times)
avg_fps = frame_count / np.sum(full_times)
cpu_usage = np.mean(cpu_usages)
memory_usage = psutil.virtual_memory().percent
filtered_counts = {cls: count for cls, count in detection_counts.items() if count > 0}

# ==============================
# Step 8: Zip foregrounds
# ==============================
zip_path = "Outputs/foregrounds.zip"
if os.path.exists("foregrounds") and len(os.listdir("foregrounds")) > 0:
    print("\nZipping all detected foregrounds...")
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', "foregrounds")
    print(f"Foregrounds zipped at: {zip_path}")
else:
    print("\nNo cropped detections found to zip.")

# ==============================
# Step 9: Print summary
# ==============================
print("\n========== Performance Report ==========")
print(f"Frames Processed: {frame_count}")
print(f"Avg YOLO Inference Time: {avg_yolo_time:.2f} ms")
print(f"Avg Full Pipeline Time: {avg_full_time:.2f} s")
print(f"Avg FPS: {avg_fps:.2f}")
print(f"CPU Usage: {cpu_usage:.2f}%")
print(f"RAM Usage: {memory_usage:.2f}%")
print(f"Per-Class Detection Counts: {filtered_counts}")
print(f"CSV saved at: {csv_path}")
# ==============================