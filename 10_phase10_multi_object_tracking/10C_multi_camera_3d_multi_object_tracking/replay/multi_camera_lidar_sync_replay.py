import rclpy
import cv2
import numpy as np

from pathlib import Path

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge
from std_msgs.msg import Header

from dataset_synchronizer import DatasetSynchronizer

# =====================================
# USER CONFIGURATION
# =====================================


# DATASET_PATH = "/home/hamza/ros2_cv_ws/scripts/phase9/recordings/session_20260704_223058" #dataset for phase9.2
#DATASET_PATH = "/home/hamza/ros2_cv_ws/scripts/phase8/recordings/session_20260626_000617" #dataset for phase9.1

DATASET_PATH = "/mnt/e/autonomous_vision/code_backup/syc_cam_lidar_ladar_imu_gnss/recordings/session_phase11_64ch_112k_20260804_160441"

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

        dataset = Path(DATASET_PATH)

        self.synchronizer = DatasetSynchronizer()
        self.synchronizer.dataset_path = dataset
        self.sync_dict = self.synchronizer.run()

        self.front_dir = dataset / "front"
        self.rear_dir = dataset / "rear"
        self.left_dir = dataset / "left"
        self.right_dir = dataset / "right"
        self.lidar_dir = dataset / "lidar"

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

    def publish_frame(self):

        # frame_id = self.frame_index + 1

        mapping = self.sync_dict[self.frame_index]

        front_frame = mapping["front"] + 1
        rear_frame  = mapping["rear"] + 1
        left_frame  = mapping["left"] + 1
        right_frame = mapping["right"] + 1
        lidar_frame = mapping["lidar"] + 1

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

            lidar_points = np.load(
                self.lidar_dir /
                f"frame_{lidar_frame:06d}.npy"
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

        # input('wait please') # debug

        if self.frame_index % 50 == 0:

            self.get_logger().info(
                f"Published synchronized replay frame {self.frame_index + 1}"
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
