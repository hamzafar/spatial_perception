class DetectionPipeline:

    def __init__(self):
        self.bridge = None
        self.model = None

    def validate(self):
        assert self.bridge is not None
        assert self.model is not None

    def convert_ros_to_cv(self, image_msg):
        return self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

    def detect(self, image):

        results = self.model(image, verbose=False)

        return image, results[0]

    def extract_detections(self, result):

        detections = []

        for box in result.boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            detection = {
                "bbox": [
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ],
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0])
            }

            detections.append(detection)
            
        return detections

    def extract_detections(self, results):

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        return boxes, scores, classes

