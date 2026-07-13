import cv2
import rclpy
from cv_bridge import CvBridge

from rclpy.node import Node

from sensor_msgs.msg import Image

from message_filters import Subscriber, TimeSynchronizer

import numpy as np

from sensor_msgs.msg import (
    Image,
    PointCloud2
)

from sensor_msgs_py import point_cloud2

from ultralytics import YOLO

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from perception_3d_pipeline import Perception3DPipeline

from display_pipeline import DisplayPipeline

from object_association_pipeline import ObjectAssociationPipeline

class UniqueObjectsLocalization(Node):

    def __init__(self):

        super().__init__("multi_camera_perception")

        self.bridge = CvBridge()


        self.display_window_width = 640*3
        self.display_window_height = 480*2

        self.cameras = {

            "front": {
                "yaw": 0
            },

            "rear": {
                "yaw": 180
            },

            "left": {
                "yaw": -90
            },

            "right": {
                "yaw": 90
            }

        }

        self.image_width = 640
        self.image_height = 480

        self.bev_width = 640 * 4
        self.bev_height = 480 * 2
        self.scale = 10.0  # pixels / meter

        # self.fx = self.image_width / 2
        # self.fy = self.image_width / 2

        # self.cx = self.image_width / 2
        # self.cy = self.image_height / 2

        self.fx = self.fy = 320
        
        self.cx = 320
        self.cy = 240 

        self.model = YOLO("/home/hamza/ros2_cv_ws/scripts/phase6/engines/yolov8m-seg-trt-int8.engine")


        # Initialize 3d_perception class variables
        self.pipeline_3d = Perception3DPipeline()

        self.pipeline_3d.bridge = self.bridge
        self.pipeline_3d.model = self.model

        self.pipeline_3d.cameras = self.cameras

        self.pipeline_3d.fx = self.fx
        self.pipeline_3d.fy = self.fy
        self.pipeline_3d.cx = self.cx
        self.pipeline_3d.cy = self.cy

        self.pipeline_3d.validate()


        # Initialize display_pipeline
        self.pipeline_display = DisplayPipeline()
        
        self.pipeline_display.image_width = self.image_width
        self.pipeline_display.image_height = self.image_height

        self.pipeline_display.bev_width = self.bev_width
        self.pipeline_display.bev_height = self.bev_height
        self.pipeline_display.scale = self.scale  # pixels / meter

        self.pipeline_display.initialize_display(display_window_width = self.display_window_width, 
            display_window_height= self.display_window_height)

        
        # Initialize object_association_pipeline
        self.pipeline_object_association = ObjectAssociationPipeline()
        self.pipeline_object_association.association_distance = 10
        self.pipeline_object_association.bearing_overlap_margin = 30


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

        self.lidar_sub = Subscriber(
            self,
            PointCloud2,
            "/carla/ego_vehicle/lidar"
        )

        self.sync = TimeSynchronizer(
            [
                self.front_sub,
                self.rear_sub,
                self.left_sub,
                self.right_sub,
                self.lidar_sub
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
        right_msg,
        lidar_msg
    ):

        front = self.pipeline_3d.convert_ros_to_cv(front_msg)
        rear = self.pipeline_3d.convert_ros_to_cv(rear_msg)
        left = self.pipeline_3d.convert_ros_to_cv(left_msg)
        right = self.pipeline_3d.convert_ros_to_cv(right_msg)


        (
            front, rear, left, right,

            front_result, rear_result, left_result, right_result

        ) = self.pipeline_3d.yolo_detection(

            front,
            rear,
            left,
            right

        )

        lidar = self.pipeline_3d.convert_ros_to_numpy(lidar_msg)

        front, front_world_obj = self.process_camera_lidar(image=front, lidar=lidar, result=front_result, camera_name="front")

        rear, rear_world_obj = self.process_camera_lidar(image=rear, lidar=lidar, result=rear_result, camera_name="rear")

        left, left_world_obj = self.process_camera_lidar(image=left, lidar=lidar, result=left_result, camera_name="left")

        right, right_world_obj = self.process_camera_lidar(image=right, lidar=lidar, result=right_result, camera_name="right")

        front, rear, left, right = self.pipeline_display.draw_camera_names(front, rear, left, right)

        world_objects = ( front_world_obj + rear_world_obj + left_world_obj + right_world_obj)


        world_objects = self.pipeline_object_association.associate(world_objects)

        bev = self.pipeline_display.draw_bev(world_objects)

        stitched = self.stitch_images(
            front,
            rear,
            left,
            right,
            bev
        )

        self.pipeline_display.display_result(
            stitched
        )

    # -------------------------------------------------

    def process_camera_lidar(
        self,
        image,
        lidar,
        result,
        camera_name 
    ):

        image, u, v, projected_points = self.pipeline_3d.project_lidar(
            image,
            lidar,
            camera_name
        )

        object_clouds = self.pipeline_3d.extract_object_clouds(
            image,
            u,
            v,
            projected_points,
            result
        )

        image, world_objects = self.pipeline_3d.process_object_clouds_and_distance(
            image,
            object_clouds,
            camera_name #duplicate_removal
        )

        return image, world_objects
    # --------------------------------------------------------------

    def stitch_images(
        self,
        front,
        rear,
        left,
        right,
        bev
    ):

        top = cv2.hconcat(
            [
                left,
                front,
                right,
                rear
            ]
        )

        stitched = cv2.vconcat(
            [
                top,
                bev
            ]
        )

        return stitched


    # ------------------------------------------------


def main():

    rclpy.init()

    node = UniqueObjectsLocalization()

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
