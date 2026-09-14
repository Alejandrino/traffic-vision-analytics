import numpy as np
import supervision as sv

print("Supervision version:", sv.__version__)

start = sv.Point(x=0, y=100)
end = sv.Point(x=200, y=100)
line_zone = sv.LineZone(start=start, end=end)

print(f"LineZone created: start={start}, end={end}")

# frame 1: box above the line
det1 = sv.Detections(
    xyxy=np.array([[10, 10, 50, 50]]),
    tracker_id=np.array([1])
)
in1, out1 = line_zone.trigger(det1)
print(f"Frame 1: trigger returned {in1}, {out1} | counts: {line_zone.in_count}, {line_zone.out_count}")

# frame 2: box crosses the line
det2 = sv.Detections(
    xyxy=np.array([[10, 110, 50, 150]]),
    tracker_id=np.array([1])
)
in2, out2 = line_zone.trigger(det2)
print(f"Frame 2: trigger returned {in2}, {out2} | counts: {line_zone.in_count}, {line_zone.out_count}")
