from kalman_filter import KalmanFilter

class Track:

    def __init__(
        self,
        track_id,
        bbox,
        class_id,
        confidence
    ):
        self.track_id = track_id
        self.bbox = bbox
        self.class_id = class_id
        self.confidence = confidence

        self.kalman = KalmanFilter()