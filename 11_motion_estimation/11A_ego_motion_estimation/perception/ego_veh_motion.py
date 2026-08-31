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
    Imu,
    NavSatFix,
)

from gnss_pipeline import GNSSPipeline

from imu_pipeline import IMUPipeline

from display_pipeline import DisplayPipeline

class EgoVehicleMotion(Node):

    def __init__(self):

        super().__init__("ego_vehicle_motion")

        self.bridge = CvBridge()

        self.display_window_width = 640
        self.display_window_height = 480

        self.image_width = 640
        self.image_height = 480

        self.previous_frame = None

        self.gnss_pipeline = GNSSPipeline()

        self.imu_pipeline = IMUPipeline()

        self.pipeline_display = DisplayPipeline()
        self.pipeline_display.initialize_display()

        self.front_sub = Subscriber(
            self,
            Image,
            "/carla/ego_vehicle/rgb_front/image"
        )

        self.imu_sub = Subscriber(
            self,
            Imu,
            "/carla/ego_vehicle/imu"
        )

        self.gnss_sub = Subscriber(
            self,
            NavSatFix,
            "/carla/ego_vehicle/gnss"
        )

        self.sync = TimeSynchronizer(
            [
                self.front_sub,
                self.imu_sub,
                self.gnss_sub
            ],
            queue_size=10
        )

        self.sync.registerCallback(
            self.synchronized_callback
        )

        self.get_logger().info(
            "Ego vehicle motion node started."
        )

    # -------------------------------------------------

    def synchronized_callback(
        self,
        front_msg,
        imu_msg,
        gnss_msg
    ):  

        front = self.bridge.imgmsg_to_cv2(
            front_msg,
            desired_encoding="bgr8"
        )

        timestamp = (
            gnss_msg.header.stamp.sec +
            gnss_msg.header.stamp.nanosec * 1e-9
        )

        gnss = self.gnss_pipeline.process(
            gnss_msg.latitude,
            gnss_msg.longitude,
            gnss_msg.altitude,
            timestamp
        )

        imu = self.imu_pipeline.process(
            imu_msg,
            gnss["speed"]
        )


        self.pipeline_display.display(
            front,
            gnss,
            imu
        )

        current_frame = int(front_msg.header.frame_id.split("_")[-1])

        if self.previous_frame is not None:
            if current_frame < self.previous_frame:

                self.get_logger().info(
                    "Replay restarted. Resetting display."
                )

                self.pipeline_display.reset()
            
        self.previous_frame = current_frame
        

def main():

    rclpy.init()

    node = EgoVehicleMotion()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    cv2.destroyAllWindows()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":

    main()