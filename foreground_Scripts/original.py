import os
import cv2
import glob
import numpy as np
import pandas as pd

def detect_static_objects(csv_path, fps, threshold_seconds=30, dist_threshold=5):

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    required_cols = {"frame", "label", "track_id", "x", "y", "w", "h"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain: {required_cols}")

    threshold_frames = int(threshold_seconds * fps)
    static_entries, static_exits = [], []

    # --- Detect static entries (object stays still)
    for tid, group in df.groupby("track_id"):
        group = group.sort_values("frame")
        frames = group["frame"].to_numpy()
        x, y, w, h = group[["x", "y", "w", "h"]].to_numpy().T

        cx = x + w / 2
        cy = y + h / 2
        span = frames[-1] - frames[0]
        disp = np.sqrt((cx.max() - cx.min())*2 + (cy.max() - cy.min())*2)

        if span >= threshold_frames and disp < dist_threshold:
            static_entries.append({
                "track_id": int(tid),
                "label": group["label"].iloc[-1],
                "first_frame": int(frames[0]),
                "last_frame": int(frames[-1]),
                "bbox_avg": [
                    int(np.mean(x)), int(np.mean(y)),
                    int(np.mean(w)), int(np.mean(h))
                ]
            })

    # --- Detect static exits (object leaves the frame)
    max_frame = int(df["frame"].max())
    last_seen = df.groupby("track_id")["frame"].max().to_dict()
    for tid, last_frame in last_seen.items():
        if (max_frame - last_frame) >= threshold_frames:
            row = df[df["track_id"] == tid].iloc[-1]
            static_exits.append({
                "track_id": int(tid),
                "label": row["label"],
                "disappear_frame": int(last_frame),
                "bbox_last": [
                    int(row["x"]), int(row["y"]),
                    int(row["w"]), int(row["h"])
                ]
            })

    return {"entries": static_entries, "exits": static_exits}


def update_background_from_crops(static_data, background_image_path,
                                 clean_background_path, crops_dir,
                                 output_path):
    """
    Updates the background with:
     
    """
    bg = cv2.imread(background_image_path)
    clean_bg = cv2.imread(clean_background_path)
    if bg is None or clean_bg is None:
        raise FileNotFoundError("Background or clean background not found.")

    h_bg, w_bg = bg.shape[:2]

    # --- Handle static entries (object stays)
    for obj in static_data["entries"]:
        tid = obj["track_id"]
        label = obj["label"]
        last_frame = obj["last_frame"]
        x, y, w, h = obj["bbox_avg"]

        # find crop
        crop_path = None
        for pad in (4, 5, 0):
            name = f"{last_frame:0{pad}d}{tid}{label}.jpg" if pad else f"{last_frame}{tid}{label}.jpg"
            p = os.path.join(crops_dir, name)
            if os.path.exists(p):
                crop_path = p
                break

        if not crop_path:
            matches = glob.glob(os.path.join(crops_dir, f"*{tid}{label}.jpg"))
            if matches:
                crop_path = sorted(matches)[-1]

        if not crop_path:
            print(f"⚠ No crop found for {label} (ID {tid})")
            continue

        crop = cv2.imread(crop_path)
        if crop is None:
            continue

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_bg, x + w), min(h_bg, y + h)
        crop = cv2.resize(crop, (x2 - x1, y2 - y1))
        bg[y1:y2, x1:x2] = crop
        print(f"🟢 Added static object {label} (ID {tid})")

    # --- Handle static exits (object leaves)
    for obj in static_data["exits"]:
        tid = obj["track_id"]
        label = obj["label"]
        x, y, w, h = obj["bbox_last"]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_bg, x + w), min(h_bg, y + h)
        bg[y1:y2, x1:x2] = clean_bg[y1:y2, x1:x2]
        print(f"🔴 Restored background for {label} (ID {tid})")

    cv2.imwrite(output_path, bg)
    print(f"✅ Updated background saved to {output_path}")
    return output_path

