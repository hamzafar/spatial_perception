import numpy as np
import cv2

class DetectionPipeline:

    def __init__(self):
        self.bridge = None
        self.model = None

        self.EGO_HOOD_POLYGON = np.array([
            [175, 480],
            [240, 390],
            [400, 390],
            [465, 480]
        ], dtype=np.int32)
        
        self.scale_polygon(self.EGO_HOOD_POLYGON, 1.10)

    def validate(self):
        assert self.bridge is not None
        assert self.model is not None

    def convert_ros_to_cv(self, image_msg):
        return self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

    def scale_polygon(self, polygon, scale=1.05):
        center = polygon.mean(axis=0)
        self.EGO_HOOD_POLYGON = np.round((polygon - center) * scale + center).astype(np.int32)


    def remove_hood(self, image):
        image = image.copy()
        cv2.fillPoly(image, [self.EGO_HOOD_POLYGON], (0, 0, 0))
        return image  

    def detect(self, image):

        inference_image = self.remove_hood(image)

        results = self.model(inference_image, verbose=False)

        return image, results[0]


    def detect_batch(self, images):
        """images: list of BGR numpy arrays, in a fixed order.
        Returns a list of Results in the same order.
        The original (unmasked) images are unchanged, as in detect()."""

        inference_images = [self.remove_hood(img) for img in images]

        results = self.model.predict(inference_images, verbose=False)

        assert len(results) == len(images), (
            f"Expected {len(images)} results, got {len(results)}"
        )

        return list(results)
        
    def extract_detections(self, results):

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        return boxes, scores, classes

