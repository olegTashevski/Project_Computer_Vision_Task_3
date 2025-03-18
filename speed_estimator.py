from collections import defaultdict, deque
import numpy as np
import cv2
import math
import supervision as sv
MPS_TO_KPH = 3.6
class SpeedEstimator:
    def __init__(self, fps, source, target):
        self.coordinates = defaultdict(lambda: deque(maxlen=fps))
        self.view_transformer = ViewTansformer(source, target)
        self._fps = fps
    def get_speed_estimates(self, detections):
        speed_labels = []
        source_points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        target_points = self.view_transformer.transform_points(points=source_points).astype(int)

        for tracker_id, [x, y] in zip(detections.tracker_id, target_points):
            self.coordinates[tracker_id].append((x, y))

            if len(self.coordinates[tracker_id]) < self._fps / 2:
                speed_labels.append(f"#{tracker_id}")
            else:
                last_xy = self.coordinates[tracker_id][0]
                dx, dy = abs(x - last_xy[0]), abs(y - last_xy[1])
                ds = math.sqrt(dx**2 + dy**2)
                time = len(self.coordinates[tracker_id]) / self._fps

                speed_labels.append(f"#{tracker_id} {ds / time * MPS_TO_KPH} km/h")

        return speed_labels

    def reset(self):
        self.coordinates = defaultdict(lambda: deque(maxlen=self._fps))

class ViewTansformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)