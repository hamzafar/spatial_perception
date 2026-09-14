import rclpy
import cv2
import numpy as np
import math

from pathlib import Path

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
from std_msgs.msg import Header

# from radar_msgs.msg import RadarScan

from dataset_synchronizer import DatasetSynchronizer

import pandas as pd
from sensor_msgs.msg import Imu, NavSatFix

# =====================================
# USER CONFIGURATION
# =====================================

DATASET_PATH = "/mnt/e/autonomous_vision/code_backup/syc_cam_lidar_ladar_imu_gnss/recordings/session_phase12_64ch_112k_20260811_231009"

TARGET_WIDTH = 640
TARGET_HEIGHT = 480

REPLAY_FPS = 3

# =====================================


class MultiCameraLidarReplay(Node):

    def __init__(self):

        super().__init__("multi_camera_lidar_replay")

        self.bridge = CvBridge()

        self.front_pub = self.create_publisher(
            Image,
            "/carla/ego_vehicle/rgb_front/image",
            10
        )

        self.rear_pub = self.create_publisher(
            Image,
            "/carla/ego_vehicle/rgb_rear/image",
            10
        )

        self.left_pub = self.create_publisher(
            Image,
            "/carla/ego_vehicle/rgb_left/image",
            10
        )

        self.right_pub = self.create_publisher(
            Image,
            "/carla/ego_vehicle/rgb_right/image",
            10
        )

        self.lidar_pub = self.create_publisher(
            PointCloud2,
            "/carla/ego_vehicle/lidar",
            10
        )

        self.radar_pub = self.create_publisher(
            PointCloud2,
            "/carla/ego_vehicle/radar",
            10
        )

        self.imu_pub = self.create_publisher(
            Imu,
            "/carla/ego_vehicle/imu",
            10
        )

        self.gnss_pub = self.create_publisher(
            NavSatFix,
            "/carla/ego_vehicle/gnss",
            10
        )

        dataset = Path(DATASET_PATH)

        self.synchronizer = DatasetSynchronizer()
        self.synchronizer.dataset_path = dataset
        self.sync_dict = self.synchronizer.run()

        self.front_dir = dataset / "front"
        self.rear_dir = dataset / "rear"
        self.left_dir = dataset / "left"
        self.right_dir = dataset / "right"
        self.lidar_dir = dataset / "lidar"
        self.radar_dir = dataset / "radar"

        self.imu_csv = pd.read_csv(dataset / "imu" / "imu.csv")
        self.gnss_csv = pd.read_csv(dataset / "gnss" / "gnss.csv")

        # with open(dataset / "timestamps.csv", "r") as f:
        #     self.rows = list(csv.DictReader(f))

        self.frame_index = 0
        
        self.timer = self.create_timer(
            1.0 / REPLAY_FPS,
            self.publish_frame
        )

        self.get_logger().info(
            f"Loaded {len(self.sync_dict)} synchronized frames."
        )

    def load_image(self, directory, frame_id):

        file = directory / f"frame_{frame_id:06d}.jpg"

        image = cv2.imread(str(file))

        if image is None:
            raise FileNotFoundError(file)

        return cv2.resize(
            image,
            (TARGET_WIDTH, TARGET_HEIGHT)
        )

    def publish_image(self, image, publisher, stamp, frame_name):

        msg = self.bridge.cv2_to_imgmsg(
            image,
            encoding="bgr8"
        )

        msg.header.stamp = stamp
        msg.header.frame_id = frame_name

        publisher.publish(msg)

    def publish_radar(
        self,
        radar_frame,
        stamp
    ):
        """
        Publish one radar frame.
        """

        radar_points = np.load(
            self.radar_dir /
            f"frame_{radar_frame:06d}.npy"
        )

        header = Header()

        header.stamp = stamp
        header.frame_id = f"radar_{radar_frame}"

        fields = [
            PointField(
                name="depth",
                offset=0,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name="azimuth",
                offset=4,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name="altitude",
                offset=8,
                datatype=PointField.FLOAT32,
                count=1
            ),
            PointField(
                name="velocity",
                offset=12,
                datatype=PointField.FLOAT32,
                count=1
            ),
        ]

        cloud = point_cloud2.create_cloud(
            header,
            fields,
            radar_points
        )

        self.radar_pub.publish(cloud)

    def publish_imu(
        self,
        imu_frame,
        stamp
    ):

        imu_data = self.imu_csv[
            self.imu_csv["frame_id"] == imu_frame
        ].iloc[0]

        msg = Imu()

        msg.header.stamp = stamp
        msg.header.frame_id = "imu"

        msg.linear_acceleration.x = imu_data["acc_x"]
        msg.linear_acceleration.y = imu_data["acc_y"]
        msg.linear_acceleration.z = imu_data["acc_z"]

        msg.angular_velocity.x = imu_data["gyro_x"]
        msg.angular_velocity.y = imu_data["gyro_y"]
        msg.angular_velocity.z = imu_data["gyro_z"]

        yaw = imu_data["compass"]

        msg.orientation.x = 0.0
        msg.orientation.y = 0.0
        msg.orientation.z = math.sin(yaw / 2.0)
        msg.orientation.w = math.cos(yaw / 2.0)

        self.imu_pub.publish(msg)

    def publish_gnss(
        self,
        gnss_frame,
        stamp
    ):

        gnss_data = self.gnss_csv[
            self.gnss_csv["frame_id"] == gnss_frame
        ].iloc[0]

        msg = NavSatFix()

        msg.header.stamp = stamp
        msg.header.frame_id = "gnss"

        msg.latitude = gnss_data["latitude"]
        msg.longitude = gnss_data["longitude"]
        msg.altitude = gnss_data["altitude"]

        self.gnss_pub.publish(msg)


    def publish_frame(self):

        # frame_id = self.frame_index + 1

        mapping = self.sync_dict[self.frame_index]

        front_frame = mapping["front"] + 1
        rear_frame  = mapping["rear"] + 1
        left_frame  = mapping["left"] + 1
        right_frame = mapping["right"] + 1
        lidar_frame = mapping["lidar"] + 1
        radar_frame = mapping["radar"] + 1
        imu_frame = mapping["imu"] + 1
        gnss_frame = mapping["gnss"] +1 

        stamp = self.get_clock().now().to_msg()

        try:

            front = self.load_image(
                self.front_dir,
                front_frame
            )

            rear = self.load_image(
                self.rear_dir,
                rear_frame
            )

            left = self.load_image(
                self.left_dir,
                left_frame
            )

            right = self.load_image(
                self.right_dir,
                right_frame
            )

            

        except Exception as e:

            self.get_logger().error(
                str(e)
            )

            self.frame_index = (
                self.frame_index + 1
            ) % len(self.sync_dict)

            return

        self.publish_image(
            front,
            self.front_pub,
            stamp,
            f"front_camera_{front_frame}"
        )

        self.publish_image(
            rear,
            self.rear_pub,
            stamp,
            f"rear_camera_{rear_frame}"
        )

        self.publish_image(
            left,
            self.left_pub,
            stamp,
            f"left_camera_{left_frame}"
        )

        self.publish_image(
            right,
            self.right_pub,
            stamp,
            f"right_camera_{right_frame}"
        )

        lidar_points = np.load(
            self.lidar_dir /
            f"frame_{lidar_frame:06d}.npy"
        )

        header = Header()

        header.stamp = stamp
        header.frame_id = f"lidar_{lidar_frame}"


        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        cloud = point_cloud2.create_cloud(
            header,
            fields,
            lidar_points
        )

        self.lidar_pub.publish(cloud)
        
        self.publish_radar(radar_frame, stamp)

        self.publish_imu(imu_frame, stamp)

        self.publish_gnss(gnss_frame, stamp)

        # input('wait please') # debug

        if self.frame_index % 50 == 0:

            self.get_logger().info(
                f"Published synchronized replay frame {self.frame_index}"
            )

        self.frame_index = (
            self.frame_index + 1
        ) % len(self.sync_dict)
    
        # input('imhere')



def main():

    rclpy.init()

    node = MultiCameraLidarReplay()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
