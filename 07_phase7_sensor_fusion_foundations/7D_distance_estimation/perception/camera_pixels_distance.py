import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node

from sensor_msgs.msg import Image

from ultralytics import YOLO


class CameraDistanceEstimator(Node):

    def __init__(self):

        super().__init__(
            "camera_distance_estimator"
        )

        self.model = YOLO(
            "/home/hamza/ros2_cv_ws/scripts/phase6/engines/yolov8m-seg-trt-int8.engine"
        )

        self.bridge = CvBridge()

        #
        # Camera intrinsics
        #

        self.fy = 320.0

        #
        # Approximate object heights (meters)
        #

        self.object_heights = {

            "person": 1.7,

            "car": 1.5,

            "truck": 3.5,

            "bus": 3.2,

            "motorcycle": 1.2,

            "bicycle": 1.2
        }

        self.subscription = self.create_subscription(
            Image,
            "/carla/ego_vehicle/rgb_front/image",
            self.image_callback,
            10
        )

        self.get_logger().info(
            "Camera distance estimator started"
        )

    def image_callback(
        self,
        image_msg
    ):

        frame = self.bridge.imgmsg_to_cv2(
            image_msg,
            desired_encoding="bgr8"
        )

        results = self.model(
            frame,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        frame = result.plot()

        if result.masks is None:
            cv2.imshow(
                "Camera Distance Estimation",
                frame
            )

            cv2.waitKey(1)

            return

        yolo_masks = result.masks.data.cpu().numpy()

        resized_masks = []

        for yolo_mask in yolo_masks:

            resized_mask = cv2.resize(
                yolo_mask.astype(np.uint8),
                (
                    frame.shape[1],
                    frame.shape[0]
                ),
                interpolation=cv2.INTER_NEAREST
            )

            resized_masks.append(
                resized_mask
            )

        for idx, yolo_mask in enumerate(resized_masks):

            cls_id = int(
                result.boxes.cls[idx]
            )

            class_name = self.model.names[
                cls_id
            ]

            #
            # Skip classes without known height
            #

            if class_name not in self.object_heights:
                continue

            #
            # Measure mask height
            #

            ys, xs = np.where(
                yolo_mask > 0
            )

            if len(ys) == 0:
                continue

            mask_height = (
                ys.max()
                -
                ys.min()
            )

            if mask_height <= 0:
                continue

            #
            # Monocular distance
            #

            real_height = self.object_heights[
                class_name
            ]

            distance_m = (
                real_height
                *
                self.fy
            ) / mask_height

            #
            # Draw distance
            #

            box = result.boxes.xyxy[
                idx
            ].cpu().numpy()

            x1, y1, x2, y2 = box

            label = (
                f"{distance_m:.1f}m"
            )

            cv2.putText(
                frame,
                label,
                (
                    int(x1),
                    int(y1) - 25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "Camera Distance Estimation",
            frame
        )

        cv2.waitKey(
            1
        )


def main():

    rclpy.init()

    node = CameraDistanceEstimator()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    cv2.destroyAllWindows()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":

    main()