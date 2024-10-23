import supervision as sv
import numpy as np
from ultralytics import YOLO
import torch

VIDEO_PATH = "example_video.MOV"

model = YOLO("yolov8n.pt")
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
tracker = sv.ByteTrack()
smoother = sv.DetectionsSmoother()

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)


def process_frame(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame, device="mps")[0]

    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    detections = smoother.update_with_detections(detections)

    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

    annotated_frame = bounding_box_annotator.annotate(
        scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)

    return annotated_frame

sv.process_video(source_path=VIDEO_PATH,
                 target_path=f"result.mp4", callback=process_frame)

sv.process_video(source_path=VIDEO_PATH,
                 target_path=f"result.mp4", callback=process_frame)
