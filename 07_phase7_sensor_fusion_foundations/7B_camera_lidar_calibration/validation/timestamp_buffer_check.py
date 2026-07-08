import rclpy

from rclpy.node import Node

from sensor_msgs.msg import (
    Image,
    PointCloud2
)

from message_filters import (
    Subscriber,
    TimeSynchronizer
)


class SyncChecker(Node):

    def __init__(self):

        super().__init__(
            "sync_checker"
        )

        self.get_logger().info(
            "Waiting for synchronized camera and lidar data..."
        )

        self.image_sub = Subscriber(
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
                self.image_sub,
                self.lidar_sub
            ],
            queue_size=10
        )

        self.sync.registerCallback(
            self.sync_callback
        )

    def get_timestamp(
        self,
        msg
    ):

        return (
            msg.header.stamp.sec
            +
            msg.header.stamp.nanosec
            * 1e-9
        )

    def sync_callback(
        self,
        image_msg,
        lidar_msg
    ):

        image_ts = self.get_timestamp(
            image_msg
        )

        lidar_ts = self.get_timestamp(
            lidar_msg
        )

        delta_ms = abs(
            image_ts
            -
            lidar_ts
        ) * 1000.0

        print(
            f"Image TS : {image_ts:.6f}"
        )

        print(
            f"LiDAR TS : {lidar_ts:.6f}"
        )

        print(
            f"Delta    : {delta_ms:.3f} ms"
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