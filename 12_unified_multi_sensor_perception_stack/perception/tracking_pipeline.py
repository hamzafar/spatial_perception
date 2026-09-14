from track import Track

class TrackingPipeline:

    def __init__(self):

        self.tracks = []
        self.next_track_id = 0

    def update(self, detections):

        iou_threshold = 0.3

        # Predict all tracks once
        for track in self.tracks:
            track.kalman.predict()

        for detection in detections:

            best_track = None
            best_iou = 0.0

            # Find the best matching existing track
            for track in self.tracks:
                
                iou = self.compute_iou(track.bbox, detection["bbox"])

                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_track is not None and best_iou >= iou_threshold:

                # Update existing track
                best_track.bbox = detection["bbox"]
                best_track.class_id = detection["class_id"]
                best_track.confidence = detection["confidence"]

            else:

                # Create a new track
                track = Track(
                    track_id=self.next_track_id,
                    bbox=detection["bbox"],
                    class_id=detection["class_id"],
                    confidence=detection["confidence"]
                )

                self.tracks.append(track)
                self.next_track_id += 1

        return self.tracks

    def compute_iou(self, box1, box2):
        """
        Compute Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]

        Returns:
            IoU value between 0.0 and 1.0
        """

        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0

        return intersection / union