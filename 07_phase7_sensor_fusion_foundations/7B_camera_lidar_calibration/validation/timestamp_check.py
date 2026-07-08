import rclpy

from rclpy.node import Node

from sensor_msgs.msg import (
    Image,
    PointCloud2
)


class SyncChecker(Node):

    def __init__(self):

        super().__init__(
            "sync_checker"
        )

        self.latest_image_ts = None
        self.latest_lidar_ts = None

        self.create_subscription(
            Image,
            "/carla/ego_vehicle/rgb_front/image",
            self.camera_callback,
            10
        )

        self.create_subscription(
            PointCloud2,
            "/carla/ego_vehicle/lidar",
            self.lidar_callback,
            10
        )

        # self.create_timer(
        #     1.0,
        #     self.timer_callback
        # )

        self.get_logger().info(
            "Waiting for camera and lidar data..."
        )

    def camera_callback(
        self,
        msg
    ):

        self.latest_image_ts = (
            msg.header.stamp.sec
            +
            msg.header.stamp.nanosec
            * 1e-9
        )

        print(
            f"Camera: {self.latest_image_ts:.6f}"
        )

    def lidar_callback(
        self,
        msg
    ):

        self.latest_lidar_ts = (
            msg.header.stamp.sec
            +
            msg.header.stamp.nanosec
            * 1e-9
        )

        print(
            f"LiDAR : {self.latest_lidar_ts:.6f}"
        )

        delta_ms = abs(
            self.latest_image_ts
            -
            self.latest_lidar_ts
        ) * 1000.0

        # print(
        #     f"Delta : "
        #     f"{delta_ms:.2f} ms"
        # )

        print(
            "-" * 40
        )

    def timer_callback(
        self
    ):

        if (
            self.latest_image_ts is None
            or
            self.latest_lidar_ts is None
        ):
            return

        delta_ms = abs(
            self.latest_image_ts
            -
            self.latest_lidar_ts
        ) * 1000.0

        print(
            f"Camera: "
            f"{self.latest_image_ts:.6f}"
        )

        print(
            f"LiDAR : "
            f"{self.latest_lidar_ts:.6f}"
        )

        print(
            f"Delta : "
            f"{delta_ms:.2f} ms"
        )

        print(
            "-" * 40
        )


def main():

    rclpy.init()

    node = SyncChecker()

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