import cv2
import rclpy
from cv_bridge import CvBridge

from rclpy.node import Node

from sensor_msgs.msg import Image

from message_filters import Subscriber, TimeSynchronizer

from sensor_msgs.msg import PointCloud2

from sensor_msgs.msg import Image


from ultralytics import YOLO

import numpy as np

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


from tracking_3d_display_pipeline import Tracking3DDisplayPipeline

from perception_3d_pipeline import Perception3DPipeline

import os
import sys

AB3DMOT_ROOT = os.path.join(
    os.path.dirname(__file__),
    "third_party",
    "AB3DMOT"
)

if AB3DMOT_ROOT not in sys.path:
    sys.path.insert(0, AB3DMOT_ROOT)

from ab3dmot_tracker import AB3DMOTTracker


# ROOT = Path(__file__).resolve().parent
# BYTE_TRACK_ROOT = ROOT / "ByteTrack"

# sys.path.insert(0, str(BYTE_TRACK_ROOT))

# from yolox.tracker.byte_tracker import BYTETracker

from types import SimpleNamespace

class Tracking(Node):

    def __init__(self):

        super().__init__("tracking")
    
        self.bridge = CvBridge()


        self.display_window_width = 640
        self.display_window_height = 480


        self.image_width = 640
        self.image_height = 480

        self.fx = 320
        self.fy = 320
        self.cx = 320
        self.cy = 240
   
        self.model = YOLO("/home/hamza/ros2_cv_ws/models/yolo26m-seg.engine")

        # Initialize 3d_perception_pipeline
        self.pipeline_perception = Perception3DPipeline()

        self.pipeline_perception.bridge = self.bridge
        self.pipeline_perception.model = self.model

        self.pipeline_perception.cameras = {
            "front": {
                "yaw": 0.0
            }
        }

        self.pipeline_perception.fx = self.fx
        self.pipeline_perception.fy = self.fy
        self.pipeline_perception.cx = self.cx
        self.pipeline_perception.cy = self.cy

        self.pipeline_perception.validate()


        # Initialize display_pipeline
        self.pipeline_3d_display = Tracking3DDisplayPipeline()
        
        self.pipeline_3d_display.image_width = self.image_width
        self.pipeline_3d_display.image_height = self.image_height

        self.pipeline_3d_display.fx = self.fx
        self.pipeline_3d_display.fy = self.fy
        self.pipeline_3d_display.cx = self.cx
        self.pipeline_3d_display.cy = self.cy

        self.pipeline_3d_display.initialize_display(display_window_width = self.display_window_width, 
            display_window_height= self.display_window_height)


        # Initialize ab3dmot tracker
        self.ab3dmot_tracker = AB3DMOTTracker()
      
        # ---------------------------------
        # Subscribers
        # ---------------------------------

        self.front_sub = Subscriber(
            self,
            Image,
            "/carla/ego_vehicle/rgb_front/image"
        )

        self.lidar_sub = Subscriber(
            self,
            PointCloud2,
            "/carla/ego_vehicle/lidar"
        )

        self.sync = TimeSynchronizer(
            [
                self.front_sub,
                self.lidar_sub,
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
        lidar_msg
        ):

        
        image = self.pipeline_perception.convert_ros_to_cv(front_msg)

        lidar = self.pipeline_perception.convert_ros_to_numpy(lidar_msg)

        lidar = lidar[:, :3] # three channels

        image, results = self.pipeline_perception.detect_image(image)

        image, u, v, projected_points = \
            self.pipeline_perception.project_lidar(
                image,
                lidar,
                "front"
            )

        object_clouds = self.pipeline_perception.extract_object_clouds(
                image,
                u,
                v,
                projected_points,
                results
            )

        detections_3d = self.pipeline_perception.generate_3d_detections(
                object_clouds,
                results
            )

        tracks = self.ab3dmot_tracker.update(
            detections_3d
        )

        self.pipeline_3d_display.draw_3d_boxes(
            image,
            tracks
        )

        # self.pipeline_3d_display.draw_3d_boxes_only(
        #     image,
        #     detections_3d
        # )

        self.pipeline_3d_display.display(image)    


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
