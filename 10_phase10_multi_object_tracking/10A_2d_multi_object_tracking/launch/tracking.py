import cv2
import rclpy
from cv_bridge import CvBridge

from rclpy.node import Node

from sensor_msgs.msg import Image

from message_filters import Subscriber, TimeSynchronizer


from sensor_msgs.msg import Image


from ultralytics import YOLO

import numpy as np

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


from tracking_display_pipeline import TrackingDisplayPipeline

from detection_pipeline import DetectionPipeline

from tracking_pipeline import TrackingPipeline


ROOT = Path(__file__).resolve().parent
BYTE_TRACK_ROOT = ROOT / "ByteTrack"

sys.path.insert(0, str(BYTE_TRACK_ROOT))

from yolox.tracker.byte_tracker import BYTETracker

from types import SimpleNamespace

class Tracking(Node):

    def __init__(self):

        super().__init__("tracking")
    
        self.bridge = CvBridge()


        self.display_window_width = 640
        self.display_window_height = 480


        self.image_width = 640
        self.image_height = 480
   
        self.model = YOLO("/home/hamza/ros2_cv_ws/models/yolo26m-seg.engine")

        # Initialize detection pipeline
        self.pipeline_detection = DetectionPipeline()

        self.pipeline_detection.bridge = self.bridge
        self.pipeline_detection.model = self.model

        self.pipeline_detection.validate()


        # Initialize display_pipeline
        self.pipeline_display = TrackingDisplayPipeline()
        
        self.pipeline_display.image_width = self.image_width
        self.pipeline_display.image_height = self.image_height


        self.pipeline_display.initialize_display(display_window_width = self.display_window_width, 
            display_window_height= self.display_window_height)


        # Initialize tracking pipeline

        self.pipeline_tracking = TrackingPipeline()


        # Initialize ByteTracker

        args = SimpleNamespace(
            track_thresh=0.5,
            track_buffer=30,
            match_thresh=0.8,
            mot20=False,
        )

        self.tracker = BYTETracker(args, frame_rate=10)
        
        # ---------------------------------
        # Subscribers
        # ---------------------------------

        self.front_sub = Subscriber(
            self,
            Image,
            "/carla/ego_vehicle/rgb_front/image"
        )

        self.sync = TimeSynchronizer(
            [
                self.front_sub,
            ],
            queue_size=10
        )

        self.sync.registerCallback(
            self.synchronized_callback
        )

        self.get_logger().info(
            "Tracking node started."
        )

    # -------------------------------------------------

    def synchronized_callback(
        self,
        front_msg,
    ):

        front = self.pipeline_detection.convert_ros_to_cv(front_msg)

        front, results = self.pipeline_detection.detect(front)

        boxes, scores, classes = self.pipeline_detection.extract_detections(results)


        # boxes = results.boxes.xyxy.cpu().numpy()
        # scores = results.boxes.conf.cpu().numpy()
        # classes = results.boxes.cls.cpu().numpy()

        dets = np.concatenate(
            [boxes, scores.reshape(-1, 1)],
            axis=1
        )


        online_targets = self.tracker.update(
            dets,
            (self.image_height, self.image_width),
            (self.image_height, self.image_width),
                )
     

        # self.pipeline_display.display(results.plot())

        # image = results.plot()


        # Use the original image
        image = front.copy()

        image = self.pipeline_display.draw_tracks(
            image,
            online_targets
        )

        self.pipeline_display.display(image)


    # -------------------------------------------------

  

def main():

    rclpy.init()

    node = Tracking()

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
