import os
import glob
import cv2
import numpy as np
import pandas as pd
import shutil
import zipfile
import tempfile
from collections import defaultdict



def detect_static_objects(
    csv_path,
    fps=30,
    threshold_minutes=2,
    dist_threshold=5,
    motion_tolerance=2.0,
    min_static_ratio=0.5,
    end_window_seconds=15,
    min_confidence=0.65,
    debug=False
):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {"frame", "label", "track_id", "x", "y", "w", "h", "confidence"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain: {required_cols}")

    # Keep only reasonable detections
    df = df[df["confidence"] >= min_confidence]
    df = df.dropna(subset=["frame", "track_id", "x", "y", "w", "h"])
    df["label"] = df["label"].astype(str).str.lower()

    threshold_frames = int(threshold_minutes * 60 * fps)
    end_window_frames = int(end_window_seconds * fps)

    # Determine last frame in video
    last_frame = int(df["frame"].max())

    results = {}

    # Process objects grouped by (track_id, label) pair
    for (tid, lbl), group in df.groupby(["track_id", "label"]):
        group = group.sort_values("frame")

        frames = group["frame"].astype(int).to_numpy()
        xs = group["x"].astype(float).to_numpy()
        ys = group["y"].astype(float).to_numpy()
        ws = group["w"].astype(float).to_numpy()
        hs = group["h"].astype(float).to_numpy()

        if len(frames) < 4:
            continue  

        cx = xs + ws / 2.0
        cy = ys + hs / 2.0

        overall_span = frames[-1] - frames[0]
        spatial_disp = np.hypot(cx.max() - cx.min(), cy.max() - cy.min())

        static_flag = False

        # Frame-to-frame motion
        displacements = np.hypot(np.diff(cx), np.diff(cy))
        if len(displacements) > 0:
            mean_disp = displacements.mean()
            stable_ratio = np.mean(displacements < motion_tolerance)
        else:
            mean_disp, stable_ratio = 0, 1

        # Core conditions
        if overall_span >= threshold_frames and spatial_disp < dist_threshold:
            static_flag = True
        elif stable_ratio >= min_static_ratio and mean_disp < motion_tolerance:
            stable_frames = frames[:-1][displacements < motion_tolerance]
            if len(stable_frames) > 2:
                stable_span = stable_frames[-1] - stable_frames[0]
                if stable_span >= threshold_frames:
                    static_flag = True

        # End window condition – new requirement
        still_in_final_window = (last_frame - frames[-1]) <= end_window_frames
        if still_in_final_window:
            static_flag = True  

        if static_flag:
            avg_x = int(round(xs.mean()))
            avg_y = int(round(ys.mean()))
            avg_w = int(round(ws.mean()))
            avg_h = int(round(hs.mean()))

            results[(tid, lbl)] = {
                "track_id": int(tid),
                "label": lbl,
                "first_frame": int(frames[0]),
                "last_frame": int(frames[-1]),
                "bbox_avg": [avg_x, avg_y, avg_w, avg_h]
            }

            if debug:
                print(f"✔ static object: track {tid}, label '{lbl}'")

    return results


'''


def detect_static_objects(
    csv_path,
    fps=30,
    threshold_minutes=2,
    dist_threshold=5,
    motion_tolerance=2.0,
    min_static_ratio=0.5
):
    """
    Detect truly static objects from sparse detections CSV.
    Finds long-lived or gap-consistent tracks where motion is minimal.

    Args:
        csv_path: path to CSV with columns [frame,label,track_id,x,y,w,h]
        fps: frames per second of source video
        threshold_minutes: minimum duration in minutes for static consideration
        dist_threshold: max spatial displacement (pixels)
        motion_tolerance: max avg per-frame motion to consider stable
        min_static_ratio: fraction of frames that must remain under tolerance

    Returns:
        List[Dict]: [{track_id, label, first_frame, last_frame, bbox_avg}]
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required_cols = {"frame", "label", "track_id", "x", "y", "w", "h"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    threshold_frames = int(threshold_minutes * 60 * fps)
    results = []

    for tid, group in df.groupby("track_id"):
        group = group.sort_values("frame")
        frames = group["frame"].astype(int).to_numpy()
        xs = group["x"].astype(float).to_numpy()
        ys = group["y"].astype(float).to_numpy()
        ws = group["w"].astype(float).to_numpy()
        hs = group["h"].astype(float).to_numpy()
        labels = group["label"].astype(str).to_numpy()

        cx = xs + ws / 2.0
        cy = ys + hs / 2.0

        # Calculate overall span and dispersion
        overall_span = frames[-1] - frames[0]
        spatial_disp = np.sqrt((cx.max() - cx.min())**2 + (cy.max() - cy.min())**2)

        static_flag = False

        # Compute frame-to-frame displacement
        displacements = np.sqrt(np.diff(cx)**2 + np.diff(cy)**2)
        if len(displacements) > 0:
            mean_disp = np.mean(displacements)
            stable_ratio = np.mean(displacements < motion_tolerance)
        else:
            mean_disp, stable_ratio = 0, 1

        # Condition A: long duration + small movement overall
        if overall_span >= threshold_frames and spatial_disp < dist_threshold:
            static_flag = True

        # Condition B: stable motion region (foreground actually static)
        elif stable_ratio > min_static_ratio and mean_disp < motion_tolerance:
            # Check longest stable segment
            stable_frames = frames[:-1][displacements < motion_tolerance]
            if len(stable_frames) > 0:
                duration = stable_frames[-1] - stable_frames[0]
                if duration >= threshold_frames:
                    static_flag = True

        # Condition C: large frame gaps with little positional change
        if not static_flag and len(frames) > 1:
            gaps = frames[1:] - frames[:-1]
            for i, gap in enumerate(gaps):
                if gap >= threshold_frames:
                    d = np.hypot(cx[i+1] - cx[i], cy[i+1] - cy[i])
                    if d < dist_threshold:
                        static_flag = True
                        break

        if static_flag:
            avg_x = int(np.round(xs.mean()))
            avg_y = int(np.round(ys.mean()))
            avg_w = int(np.round(ws.mean()))
            avg_h = int(np.round(hs.mean()))
            label = labels[-1]
            results.append({
                "track_id": int(tid),
                "label": label,
                "first_frame": int(frames[0]),
                "last_frame": int(frames[-1]),
                "bbox_avg": [avg_x, avg_y, avg_w, avg_h]
            })

    return results



def update_background_from_crops(
    static_objects,
    background_path,
    foregrounds_dir,
    output_path,
    detections_csv,
    fps=30,
    threshold_minutes=2,
    dist_threshold=5,
    try_padding_lengths=(4,5,0)
):
    """
    Selects the most stable (truly static) frame that lies BETWEEN long detection gaps.
    This frame is where the object was stationary and YOLO's motion triggers were silent.

    Args:
        static_objects: list of dicts (from detect_static_objects)
        background_path: path to background image
        foregrounds_dir: directory containing crops (<frame>_<track_id>_<label>.jpg)
        output_path: final background save path
        detections_csv: CSV with columns [frame,track_id,label,x,y,w,h]
        fps, threshold_minutes, dist_threshold: same thresholds used in detect_static_objects
        try_padding_lengths: filename padding options
    """
    bg = cv2.imread(background_path)
    if bg is None:
        raise FileNotFoundError(f"Could not open background: {background_path}")
    h_bg, w_bg = bg.shape[:2]

    df = pd.read_csv(detections_csv)
    df.columns = [c.strip() for c in df.columns]
    threshold_frames = int(threshold_minutes * 60 * fps)

    for obj in static_objects:
        tid = obj["track_id"]
        label = obj["label"]
        x_avg, y_avg, w, h = obj["bbox_avg"]

        subset = df[df["track_id"] == tid].sort_values("frame")
        frames = subset["frame"].astype(int).to_numpy()
        xs = subset["x"].astype(float).to_numpy()
        ys = subset["y"].astype(float).to_numpy()
        ws = subset["w"].astype(float).to_numpy()
        hs = subset["h"].astype(float).to_numpy()

        cx = xs + ws / 2.0
        cy = ys + hs / 2.0

        selected_frame = None

        # Identify long temporal gaps and select a frame roughly in the middle
        if len(frames) > 1:
            gaps = frames[1:] - frames[:-1]
            for i, gap in enumerate(gaps):
                if gap >= threshold_frames:
                    mid_frame = (frames[i] + frames[i+1]) // 2
                    # choose the recorded frame closest to mid_frame
                    closest_idx = np.argmin(np.abs(frames - mid_frame))
                    selected_frame = int(frames[closest_idx])
                    break

        # Fallback: if no gap found, use the frame with minimal motion from average
        if selected_frame is None:
            cx_avg, cy_avg = np.mean(cx), np.mean(cy)
            dists = np.sqrt((cx - cx_avg)**2 + (cy - cy_avg)**2)
            selected_frame = int(frames[np.argmin(dists)])

        # Try to locate crop file
        crop_path = None
        for pad in try_padding_lengths:
            if pad > 0:
                fname = f"{selected_frame:0{pad}d}_{tid}_{label}.jpg"
            else:
                fname = f"{selected_frame}_{tid}_{label}.jpg"
            p = os.path.join(foregrounds_dir, fname)
            if os.path.exists(p):
                crop_path = p
                break

        if crop_path is None:
            print(f"⚠️ Crop not found for track {tid} ({label}), frame {selected_frame}")
            continue

        crop = cv2.imread(crop_path)
        if crop is None:
            print(f"⚠️ Could not load crop {crop_path}")
            continue

        # Paste coordinates, clamped
        x1, y1 = max(0, x_avg), max(0, y_avg)
        x2, y2 = min(x_avg + w, w_bg), min(y_avg + h, h_bg)
        if x1 >= x2 or y1 >= y2:
            print(f"⚠️ Invalid bbox for {tid}: {x1,y1,x2,y2}")
            continue

        crop_resized = cv2.resize(crop, (x2 - x1, y2 - y1))
        bg[y1:y2, x1:x2] = crop_resized
        print(f"✅ Pasted track {tid} ({label}) from frame {selected_frame} at {x1,y1,x2,y2}")

    cv2.imwrite(output_path, bg)
    print(f"✅ Updated background saved to: {output_path}")
    return output_path




def overlay_static_object_on_video(csv_path, zip_path, background_path, output_path, fps=15):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    print(f"📦 Extracted foregrounds to temporary folder: {temp_dir}")

    background = cv2.imread(background_path)
    if background is None:
        raise ValueError(f"❌ Could not read background image from: {background_path}")
    h, w, _ = background.shape

    df = pd.read_csv(csv_path)
    frame_ids = sorted(df['frame'].unique())

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_id in frame_ids:
        frame_img = background.copy()
        detections = df[df['frame'] == frame_id]

        for _, row in detections.iterrows():
            x, y, w_box, h_box = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
            track_id = int(row['track_id'])
            cls = row['label']

            crop_filename = f"{frame_id}_{track_id}_{cls}.jpg"
            crop_path = os.path.join(temp_dir, crop_filename)

            if not os.path.exists(crop_path):
                continue

            crop = cv2.imread(crop_path)
            if crop is None:
                continue

            crop_resized = cv2.resize(crop, (w_box, h_box))
            try: 
                frame_img[y:y+h_box, x:x+w_box] = crop_resized
            except:
                pass

        out.write(frame_img)

    out.release()
    shutil.rmtree(temp_dir)
    print(f"✅ Rebuilt video saved to: {output_path}")

'''




import cv2
import os
import numpy as np
import pandas as pd
from collections import defaultdict



def update_background_from_crops(
        static_objects,            # <-- coming from previous function
        background_path,
        foregrounds_dir,
        output_path,
        detections_csv,
        fps,
        static_threshold_minutes,   # user-defined
        end_window_seconds=15       # fixed as requested
    ):

    # --- Load detection CSV ---
    df = pd.read_csv(detections_csv)

    # Create unique object key = label + track_id
    df["unique_obj"] = df["label"] + "_" + df["track_id"].astype(str)

    # --- Convert static objects list into trackable set ---
    # static_objects list format expected: [(label, track_id), ...]
    static_object_keys = [f"{lbl}_{tid}" for (lbl, tid) in static_objects]

    if len(static_object_keys) == 0:
        print("[INFO] No static objects found. Saving original background.")
        bg = cv2.imread(background_path)
        cv2.imwrite(output_path, bg)
        return

    # --- Determine the last frame of video ---
    max_frame = df["frame"].max()
    end_window_start = max_frame - (end_window_seconds * fps)

    # --- Determine survivor objects based on end window ---
    survivors = []  # list of tuples: (unique_obj, last_seen_frame)
    static_frame_thresh = static_threshold_minutes * 60 * fps

    for obj in static_object_keys:
        obj_frames = sorted(df[df["unique_obj"] == obj]["frame"].tolist())

        if len(obj_frames) < 2:
            continue

        duration = obj_frames[-1] - obj_frames[0]

        # Condition 1: must have met static duration
        if duration < static_frame_thresh:
            continue

        # Condition 2: must be present near end
        if any(frame >= end_window_start for frame in obj_frames):
            survivors.append((obj, obj_frames[-1]))

    if len(survivors) == 0:
        print("[INFO] No end-window survivors. Saving background unchanged.")
        bg = cv2.imread(background_path)
        cv2.imwrite(output_path, bg)
        return

    # Sort by last appearance (later appearance = higher priority)
    survivors = sorted(survivors, key=lambda x: x[1], reverse=True)

    # --- Load background ---
    background = cv2.imread(background_path)
    final_bg = background.copy()
    H, W = final_bg.shape[:2]

    # --- Superimpose survivors by priority ---
    for obj, _ in survivors:
        # Pick latest frame bbox
        obj_rows = df[df["unique_obj"] == obj].sort_values(by="frame", ascending=False)
        row = obj_rows.iloc[0]

        x, y, w, h = map(int, [row['x'], row['y'], row['w'], row['h']])

        # Clip to bounds
        x, y = max(0, x), max(0, y)
        w, h = min(w, W - x), min(h, H - y)

        crop_path = os.path.join(foregrounds_dir, obj + ".jpg")
        if not os.path.exists(crop_path):
            print(f"[WARN] Missing crop => {crop_path}")
            continue

        crop = cv2.imread(crop_path)
        if crop is None:
            print(f"[WARN] Unable to load => {crop_path}")
            continue

        crop = cv2.resize(crop, (w, h))

        # Paste - automatically resolves overlap due to sorted priority
        final_bg[y:y+h, x:x+w] = crop

    cv2.imwrite(output_path, final_bg)
    print(f"[INFO] Final background saved to: {output_path}")







def overlay_static_object_on_video(csv_path, foregrounds_dir, background_path, output_path, fps=15, debug=False):
    # ---- Load background image ----
    background = cv2.imread(background_path)
    if background is None:
        raise ValueError(f"❌ Could not read background image: {background_path}")
    H, W, _ = background.shape
    print(f"📐 Background size: {W}x{H}")
    
    # ---- Load detections CSV ----
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Debug: Print column names to verify
    print(f"📊 CSV columns: {df.columns.tolist()}")
    print(f"📊 First few rows:\n{df.head()}")
    
    frame_ids = sorted(df['frame'].unique())
    print(f"🎞 Total frames to rebuild → {len(frame_ids)}")
    print(f"🎞 Frame range: {min(frame_ids)} to {max(frame_ids)}")
    
    # ---- Best crop finder for given (frame, track_id, label) ----
    def find_best_crop(frame, tid, label):
        label = str(label).lower()
        best_candidate = None
        best_dist = float('inf')
        
        for fname in os.listdir(foregrounds_dir):
            f = fname.lower()
            if not (f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")):
                continue
            
            parts = f.split("_")
            if len(parts) < 3:
                continue
            
            # Extract frame #, track_id, and label from filename
            try:
                f_frame = int(parts[0])
            except:
                continue
            
            f_tid = parts[1]
            f_label = parts[2].split(".")[0]
            
            # Match same track_id and label
            if str(f_tid) == str(tid) and f_label == label:
                dist = abs(f_frame - frame)
                if dist < best_dist:
                    best_dist = dist
                    best_candidate = os.path.join(foregrounds_dir, fname)
        
        return best_candidate
    
    # ---- Create video writer ----
    # Try multiple codecs in order of preference
    codecs_to_try = [
        ('avc1', '.mp4'),  # H.264 - best quality
        ('mp4v', '.mp4'),  # MPEG-4
        ('XVID', '.avi'),  # Xvid
        ('MJPG', '.avi'),  # Motion JPEG - most compatible
    ]
    
    out = None
    used_codec = None
    final_output_path = output_path
    
    for codec, ext in codecs_to_try:
        try:
            # Adjust extension if needed
            if not output_path.lower().endswith(ext):
                final_output_path = os.path.splitext(output_path)[0] + ext
            
            fourcc = cv2.VideoWriter_fourcc(*codec)
            temp_out = cv2.VideoWriter(final_output_path, fourcc, fps, (W, H))
            
            if temp_out.isOpened():
                out = temp_out
                used_codec = codec
                print(f"✓ Using codec: {codec} (output: {final_output_path})")
                break
            else:
                temp_out.release()
        except:
            continue
    
    if out is None or not out.isOpened():
        raise ValueError(
            f"❌ Could not create video writer with any codec.\n"
            f"   Tried: {[c[0] for c in codecs_to_try]}\n"
            f"   Path: {output_path}\n"
            f"   Make sure the directory exists and you have write permissions."
        )
    
    print(f"🎥 Video writer initialized: {fps} fps, {W}x{H}")
    
    # ---- Reconstruct each frame ----
    frames_written = 0
    objects_pasted = 0
    
    for frame_id in frame_ids:
        frame_img = background.copy()
        detections = df[df['frame'] == frame_id]
        
        if debug:
            print(f"\n🔧 Processing frame {frame_id} with {len(detections)} detections")
        
        for idx, row in detections.iterrows():
            x, y = int(row['x']), int(row['y'])
            w_box, h_box = int(row['w']), int(row['h'])
            tid = row['track_id']
            label = row['label']
            
            if debug:
                print(f"  📦 Object: id={tid}, label={label}, bbox=({x},{y},{w_box},{h_box})")
            
            # Find matching crop
            crop_path = find_best_crop(frame_id, tid, label)
            if crop_path is None:
                if debug:
                    print(f"  ⚠️ Missing crop → frame={frame_id}, id={tid}, label={label}")
                continue
            
            # Load crop image
            crop = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)  # Load with alpha if present
            if crop is None:
                if debug:
                    print(f"  ⚠️ Failed to load crop → {crop_path}")
                continue
            
            if debug:
                print(f"  ✓ Loaded crop: {crop_path} (shape: {crop.shape})")
            
            # Resize crop to match bounding box
            crop_resized = cv2.resize(crop, (w_box, h_box))
            
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w_box), min(H, y + h_box)
            
            # Calculate actual paste dimensions
            paste_w = x2 - x1
            paste_h = y2 - y1
            
            if paste_w <= 0 or paste_h <= 0:
                if debug:
                    print(f"  ⚠️ Invalid paste region: ({x1},{y1}) to ({x2},{y2})")
                continue
            
            # Crop the resized image if it extends beyond frame
            crop_x1 = max(0, -x)
            crop_y1 = max(0, -y)
            crop_x2 = crop_x1 + paste_w
            crop_y2 = crop_y1 + paste_h
            
            crop_to_paste = crop_resized[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Handle alpha channel if present
            if crop_to_paste.shape[2] == 4:
                # Has alpha channel
                alpha = crop_to_paste[:, :, 3:4] / 255.0
                crop_rgb = crop_to_paste[:, :, :3]
                
                # Alpha blending
                background_region = frame_img[y1:y2, x1:x2]
                blended = (crop_rgb * alpha + background_region * (1 - alpha)).astype(np.uint8)
                frame_img[y1:y2, x1:x2] = blended
            else:
                # No alpha channel - direct paste
                frame_img[y1:y2, x1:x2] = crop_to_paste
            
            objects_pasted += 1
            
            if debug:
                print(f"  ✅ Pasted crop at ({x1},{y1}) to ({x2},{y2})")
        
        # Write frame to video
        out.write(frame_img)
        frames_written += 1
        
        if frames_written % 10 == 0:
            print(f"  📹 Written {frames_written}/{len(frame_ids)} frames...")
    
    out.release()
    
    print(f"\n✅ Video reconstruction complete!")
    print(f"   📹 Frames written: {frames_written}")
    print(f"   📦 Objects pasted: {objects_pasted}")
    print(f"   💾 Saved to: {final_output_path}\n")
    
    return final_output_path


# Example usage with debugging enabled:
# overlay_static_object_on_video(
#     csv_path="detections.csv",
#     foregrounds_dir="foreground_crops/",
#     background_path="background.jpg",
#     output_path="reconstructed_video.mp4",
#     fps=15,
#     debug=True
# )