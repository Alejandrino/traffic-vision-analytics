import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

model = YOLO("yolov8n_openvino_model", task="detect")
frame = np.zeros((720, 1280, 3), dtype=np.uint8)

res = model.track(frame, persist=True, classes=[2], verbose=False)[0]
det = sv.Detections.from_ultralytics(res)
print("Detections with tracker_id:", det.tracker_id)
