
import os
import glob
import cv2
import numpy as np
import pandas as pd
import shutil
import zipfile
import tempfile
from collections import defaultdict






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
        
    
    out.release()
    
    print(f"\n✅ Video reconstruction complete!")
    print(f"   📹 Frames written: {frames_written}")
    print(f"   📦 Objects pasted: {objects_pasted}")
    print(f"   💾 Saved to: {final_output_path}\n")
    
    return final_output_path




video_path = "data/hehe2.mp4"
get_fps = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS))
overlay_static_object_on_video(
    csv_path="csv_logs/detections_metadata.csv",
    foregrounds_dir="foregrounds",
    background_path="Outputs/static_background2.png",
    output_path="reconstucted_video2.mp4",
    fps=int(get_fps))