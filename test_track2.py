import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import urllib.request

model = YOLO('yolov8n_openvino_model', task='detect')
urllib.request.urlretrieve('https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg', 'bus.jpg')
frame = cv2.imread('bus.jpg')
res = model.track(frame, persist=True, classes=[5], verbose=False)[0]
det = sv.Detections.from_ultralytics(res)
print('Detections tracker_id:', det.tracker_id)
