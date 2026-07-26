import cv2
import rclpy
from cv_bridge import CvBridge

from rclpy.node import Node

from sensor_msgs.msg import Image

from message_filters import Subscriber, TimeSynchronizer

import numpy as np
from ultralytics import YOLO

class MultiCameraPerception(Node):

    def __init__(self):

        super().__init__("multi_camera_perception")

        self.bridge = CvBridge()


        self.display_window_width = 1600
        self. display_window_height = 900

        cv2.namedWindow(
            "Phase 8 - Multi Camera Perception",
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            "Phase 8 - Multi Camera Perception",
            self.display_window_width,
            self.display_window_height
        )

        self.model = YOLO(
            "/home/hamza/ros2_cv_ws/scripts/phase6/engines/yolov8m-seg-trt-int8.engine"
        )

        # ---------------------------------
        # Subscribers
        # ---------------------------------

        self.front_sub = Subscriber(
            self,
            Image,
            "/carla/ego_vehicle/rgb_front/image"
        )

        self.rear_sub = Subscriber(
            self,
            Image,
            "/carla/ego_vehicle/rgb_rear/image"
        )

        self.left_sub = Subscriber(
            self,
            Image,
            "/carla/ego_vehicle/rgb_left/image"
        )

        self.right_sub = Subscriber(
            self,
            Image,
            "/carla/ego_vehicle/rgb_right/image"
        )

        self.sync = TimeSynchronizer(
            [
                self.front_sub,
                self.rear_sub,
                self.left_sub,
                self.right_sub
            ],
            queue_size=10
        )

        self.sync.registerCallback(
            self.synchronized_callback
        )

        self.get_logger().info(
            "Multi-camera perception node started."
        )

    # -------------------------------------------------

    def synchronized_callback(
        self,
        front_msg,
        rear_msg,
        left_msg,
        right_msg
    ):

        front = self.convert_ros_to_cv(front_msg)
        rear = self.convert_ros_to_cv(rear_msg)
        left = self.convert_ros_to_cv(left_msg)
        right = self.convert_ros_to_cv(right_msg)


        front, rear, left, right = self.yolo_detection(
            front,
            rear,
            left,
            right
        )

        stitched = self.stitch_images(
            front,
            rear,
            left,
            right
        )

        self.display_result(
            stitched
        )

    # -------------------------------------------------

    def convert_ros_to_cv(
        self,
        image_msg
    ):

        return self.bridge.imgmsg_to_cv2(
            image_msg,
            desired_encoding="bgr8"
        )

    # -------------------------------------------------

    def yolo_detection(
        self,
        front,
        rear,
        left,
        right
    ):
        front = self.detect_image(front)

        rear = self.detect_image(rear)

        left = self.detect_image(left)

        right = self.detect_image(right)

        return (
            front,
            rear,
            left,
            right
        )


    def detect_image(
        self,
        image
    ):

        results = self.model.predict(
            source=image,
            verbose=False
        )

        annotated = results[0].plot()

        return annotated
    
    # -------------------------------------------------

    def stitch_images(
        self,
        front,
        rear,
        left,
        right
    ):

        top = cv2.hconcat(
            [
                left,
                front,
                right
            ]
        )

        bottom = cv2.hconcat(
            [
                right,
                rear,
                left
            ]
        )

        stitched = cv2.vconcat(
            [
                top,
                bottom
            ]
        )

        return stitched

    # -------------------------------------------------

    def display_result(
        self,
        image
    ):

        cv2.imshow(
            "Phase 8 - Multi Camera Perception",
            image
        )

        cv2.waitKey(1)


def main():

    rclpy.init()

    node = MultiCameraPerception()

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
