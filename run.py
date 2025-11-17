


from static_update2 import detect_static_objects, update_background_from_crops, overlay_static_object_on_video

import cv2


# Example
video_path = "data/hehe.mp4"
get_fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)



SO = detect_static_objects('csv_logs/detections_metadata2.csv', get_fps, 2, 5)


update_background_from_crops(
    SO,
    background_path="Outputs/static_background.png",
    foregrounds_dir="foregrounds/",
    output_path="Outputs/updated_background.png",
    detections_csv="csv_logs/detections_metadata2.csv",
    fps=int(get_fps),
    static_threshold_minutes=2,


)


overlay_static_object_on_video(
    csv_path="csv_logs/detections_metadata.csv",
    foregrounds_dir="foregrounds",
    background_path="Outputs/static_background2.png",
    output_path="reconstucted_video2.mp4",
    fps=int(get_fps))