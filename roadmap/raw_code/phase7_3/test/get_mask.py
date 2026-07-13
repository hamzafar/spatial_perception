import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO

import cv2
import time
import torch

import numpy as np
import os

# /home/hamza/ros2_cv_ws/scripts/phase6/engines/yolov8m-seg-trt-fp16.engine
TRT_INT8 = "/home/hamza/ros2_cv_ws/scripts/phase6/engines/yolov8m-seg-trt-int8.engine"
TRT_FP16 = "/home/hamza/ros2_cv_ws/scripts/phase6/engines/yolov8m-seg-trt-fp16.engine"
PYT_FP32 = "/home/hamza/ros2_cv_ws/scripts/phase6/test/yolov8m-seg.pt"

class YoloSegDetector(Node):

    def __init__(self):

        super().__init__('yolo_seg_detector')

        # ==========================
        # User Settings
        # ==========================
        self.show_fps = True

        # FPS counters
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0.0
        
        # =======================
        # FPS calculatoin
        # ========================
        self.fps_history = []

        self.benchmark_start = time.time()
        self.benchmark_duration = 60


        self.bridge = CvBridge()

        # # Force CPU
        # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # Yolo Segmentation  model
        self.model = YOLO(TRT_INT8)

        self.get_logger().info(
            f"Loaded model: {self.model.ckpt_path}"
        )

        self.get_logger().info(
            "YOLOv8m-seg TRT model loaded"
        )

        self.subscription = self.create_subscription(
            Image,
            "/carla/ego_vehicle/rgb_front/image",
            self.image_callback,
            10
        )

        self.get_logger().info(
            "YOLOv8 Segmentation node started"
        )

    def image_callback(self, msg):

        # ROS Image -> OpenCV
        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8"
        )

        # YOLO inference
        results = self.model(
            frame,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        print("Boxes:", result.boxes is not None)
        print("Masks:", result.masks is not None)

        if result.masks is not None:
            print("Mask shape:", result.masks.data.shape)

        
        result = results[0]


        if result.boxes is not None:
            print("Num boxes:", len(result.boxes))

        if result.masks is not None:
            print("Num masks:", len(result.masks.data))
        else:
            print("NO MASKS")

        if result.masks is not None:

            masks = result.masks.data.cpu().numpy()

            combined_mask = np.zeros(
                masks[0].shape,
                dtype=np.uint8
            )

            for mask in masks:
                combined_mask |= mask.astype(np.uint8)

            cv2.imshow(
                "Combined Mask",
                combined_mask * 255
            )
        # cv2.waitKey(1)


        # Draw segmentation masks, boxes, labels
        annotated_frame = results[0].plot()

        # ==========================
        # FPS Calculation
        # ==========================
        self.frame_count += 1

        elapsed = time.time() - self.start_time

        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed

            self.fps_history.append(self.fps)

            self.frame_count = 0

            # Avoid drift
            self.start_time += elapsed


        # ==========================
        # FPS MIN, MAX, AVG
        # ==========================
        if (
                time.time() - self.benchmark_start
                >= self.benchmark_duration
                and len(self.fps_history) > 0
            ):

            min_fps = min(self.fps_history)

            max_fps = max(self.fps_history)

            avg_fps = (
                sum(self.fps_history)
                / len(self.fps_history)
            )

            self.get_logger().info(
                f"60s Benchmark | "
                f"Samples: {len(self.fps_history)} | "
                f"Min FPS: {min_fps:.2f} | "
                f"Avg FPS: {avg_fps:.2f} | "
                f"Max FPS: {max_fps:.2f}"
            )

            self.fps_history.clear()

            self.benchmark_start = time.time() 

        # ==========================
        # Draw FPS
        # ==========================
        if self.show_fps:
            cv2.putText(
                annotated_frame,
                f"FPS: {self.fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        # ==========================
        # Display
        # ==========================
        cv2.imshow(
            "YOLOv8 Segmentation",
            annotated_frame
        )

        cv2.waitKey(1)

        # input('imhere')


def main(args=None):

    rclpy.init(args=args)

    node = YoloSegDetector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    rclpy.shutdown()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()