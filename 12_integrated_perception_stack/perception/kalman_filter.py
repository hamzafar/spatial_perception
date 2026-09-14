import numpy as np

class KalmanFilter:

    def __init__(self):
        
        self.state = None
        self.covariance = None
        
    def initiate(self, bbox):

        measurement = self.xyxy_to_xyah(bbox)

        self.state = np.zeros(8, dtype=np.float32)
        self.state[:4] = measurement

        self.covariance = np.eye(8, dtype=np.float32)

    def predict(self):
        pass

    def update(self, bbox):
        pass

    def xyxy_to_xyah(self, bbox):

        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1

        center_x = x1 + width / 2.0
        center_y = y1 + height / 2.0

        aspect_ratio = width / height

        return np.array([
            center_x,
            center_y,
            aspect_ratio,
            height
        ], dtype=np.float32)

    def xyah_to_xyxy(self, state):

        center_x, center_y, aspect_ratio, height = state[:4]

        width = aspect_ratio * height

        x1 = center_x - width / 2.0
        y1 = center_y - height / 2.0
        x2 = center_x + width / 2.0
        y2 = center_y + height / 2.0

        return np.array([
            x1,
            y1,
            x2,
            y2
        ], dtype=np.float32)