import cv2
import rclpy
import numpy as np

from cv_bridge import CvBridge

from message_filters import (
    Subscriber,
    TimeSynchronizer,
)

from rclpy.node import Node

from sensor_msgs.msg import (
    Image,
    PointCloud2
)

from sensor_msgs_py import point_cloud2

from ultralytics import YOLO

from yolo_detection_pipeline import DetectionPipeline
from radar_perception_pipeline import RadarPerceptionPipeline
from display_pipeline_radar_perception import DisplayPipeline

class WorldObjectsMotion(Node):

    def __init__(self):

        super().__init__("world_objects_motion")

        self.bridge = CvBridge()


        self.front_sub = Subscriber(
            self,
            Image,
            "/carla/ego_vehicle/rgb_front/image"
        )

        self.model = YOLO("/home/hamza/ros2_cv_ws/models/yolo26m-seg.engine")

        self.image_width = 640
        self.image_height = 480

        self.fx = self.image_width / 2.0    # 320.0
        self.fy = self.image_width / 2.0    # 320.0
        self.cx = self.image_width / 2.0    # 320.0
        self.cy = self.image_height / 2.0   # 240.0


        # Initialize detection pipeline
        self.pipeline_detection = DetectionPipeline()

        self.pipeline_detection.bridge = self.bridge
        self.pipeline_detection.model = self.model

        self.pipeline_detection.validate()
    
        # Initialize radar perception pipeline
        self.pipeline_radar = RadarPerceptionPipeline()

        self.pipeline_radar.velocity_threshold = 0.5
        self.pipeline_radar.max_assoc_distance_px = 50

        self.pipeline_radar.fx = self.fx
        self.pipeline_radar.fy = self.fy
        self.pipeline_radar.cx = self.cx
        self.pipeline_radar.cy = self.cy
        self.pipeline_radar.width = self.image_width
        self.pipeline_radar.height = self.image_height

        R = np.array([
            [0.0,  1.0,  0.0],
            [0.0,  0.0, -1.0],
            [1.0,  0.0,  0.0],
        ], dtype=np.float32)

        t = np.array([
            0.0,
            1.4,
            3.5,
        ], dtype=np.float32)

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t

        self.pipeline_radar.T_radar_to_cam = T

        # display pipeline
        self.pipeline_display = DisplayPipeline()
        self.pipeline_display.class_names = self.model.names
        self.pipeline_display.validate()

        self.radar_sub = Subscriber(
            self,
            PointCloud2,
            "/carla/ego_vehicle/radar"
        )

        self.sync = TimeSynchronizer(
            [
                self.front_sub,
                self.radar_sub,
            ],
            queue_size=10
        )

        self.sync.registerCallback(
            self.synchronized_callback
        )

        self.get_logger().info(
            "World motion motion node started."
        )

    # -------------------------------------------------

    def synchronized_callback(
        self,
        front_msg,
        radar_msg
    ):  

        front = self.pipeline_detection.convert_ros_to_cv(front_msg)

        front, results = self.pipeline_detection.detect(front)

        boxes, scores, classes = self.pipeline_detection.extract_detections(results)

        radar_points = self.pipeline_radar.convert_ros_to_numpy(radar_msg)

        objects = self.pipeline_radar.process(
            boxes=boxes,
            scores=scores,
            classes=classes,
            radar_targets=radar_points
        )

        #
        # Draw all YOLO detections
        #

        # front = self.pipeline_display.draw_yolo_detections(
        #     front,
        #     boxes,
        #     scores,
        #     classes
        # )

        front = self.pipeline_display.draw_radar_points(
            front,
            self.pipeline_radar.projected_points
        )
        
        self.pipeline_display.display(
            front,
            objects
        )



def main():

    rclpy.init()

    node = WorldObjectsMotion()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    cv2.destroyAllWindows()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":

    main()