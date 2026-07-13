import rclpy
import cv2
import csv
import numpy as np

from pathlib import Path

from rclpy.node import Node

from sensor_msgs.msg import (
    Image,
    PointCloud2,
    PointField
)

from sensor_msgs_py import point_cloud2

from cv_bridge import CvBridge

from std_msgs.msg import Header


# =====================================
# USER CONFIG
# =====================================

DATASET_PATH = (
    "/home/hamza/ros2_cv_ws/scripts/phase7/phase7_3/recordings/session_20260620_003441"
)

REPLAY_FPS = 10

# =====================================


class CameraLidarReplay(Node):

    def __init__(self):

        super().__init__(
            "camera_lidar_replay"
        )

        self.bridge = CvBridge()

        self.image_pub = (
            self.create_publisher(
                Image,
                "/carla/ego_vehicle/rgb_front/image",
                10
            )
        )

        self.lidar_pub = (
            self.create_publisher(
                PointCloud2,
                "/carla/ego_vehicle/lidar",
                10
            )
        )

        self.images_dir = (
            Path(DATASET_PATH)
            / "images"
        )

        self.lidar_dir = (
            Path(DATASET_PATH)
            / "lidar"
        )

        self.rows = []

        with open(
            Path(DATASET_PATH)
            / "timestamps.csv",
            "r"
        ) as f:

            reader = csv.DictReader(
                f
            )

            self.rows = list(
                reader
            )

        self.frame_index = 0

        self.timer = (
            self.create_timer(
                1.0 / REPLAY_FPS,
                self.publish_frame
            )
        )

        self.get_logger().info(
            f"Loaded "
            f"{len(self.rows)} frames"
        )

    def publish_frame(self):


        if (
            self.frame_index
            >= len(self.rows)
        ):

            self.get_logger().info(
                "Restarting replay..."
            )

            self.frame_index = 0

            return


        frame_id = (
            self.frame_index + 1
        )

        image_file = (
            self.images_dir
            /
            f"frame_{frame_id:06d}.jpg"
        )

        lidar_file = (
            self.lidar_dir
            /
            f"frame_{frame_id:06d}.npy"
        )

        # ==========================
        # Image
        # ==========================

        frame = cv2.imread(
            str(image_file)
        )

        if frame is None:

            self.get_logger().warning(
                f"Missing image "
                f"{image_file}"
            )

            self.frame_index += 1

            return

        frame = cv2.resize(
            frame,
            (640, 480)
        )

        image_msg = (
            self.bridge.cv2_to_imgmsg(
                frame,
                encoding="bgr8"
            )
        )

        stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        image_msg.header.stamp = stamp

        self.image_pub.publish(
            image_msg
        )

        # ==========================
        # LiDAR
        # ==========================

        points = np.load(
            lidar_file
        )

        pc_header = Header()

        pc_header.stamp = stamp

        pc_header.frame_id = (
            "lidar"
        )

        cloud_msg = (
            point_cloud2.create_cloud_xyz32(
                pc_header,
                points[:, :3]
            )
        )

        self.lidar_pub.publish(
            cloud_msg
        )

        # print(points.shape) #debug
        # print(points[:5])   #debug

        if (
            frame_id % 50 == 0
        ):

            self.get_logger().info(
                f"Published "
                f"{frame_id}"
            )

        self.frame_index += 1


def main():

    rclpy.init()

    node = (
        CameraLidarReplay()
    )

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":

    main()
