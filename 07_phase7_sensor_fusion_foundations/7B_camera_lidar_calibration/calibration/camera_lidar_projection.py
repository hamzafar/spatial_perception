import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge

from rclpy.node import Node

from sensor_msgs.msg import (
    Image,
    PointCloud2
)

from sensor_msgs_py import point_cloud2

from message_filters import (
    Subscriber,
    TimeSynchronizer
)


class CameraLidarProjection(Node):

    def __init__(self):

        super().__init__(
            "camera_lidar_projection"
        )

        self.bridge = CvBridge()

        self.fx = 320.0
        self.fy = 320.0

        self.cx = 320.0
        self.cy = 240.0

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

        self.get_logger().info(
            "Camera-LiDAR projection started"
        )

    def sync_callback(
        self,
        image_msg,
        lidar_msg
    ):

        frame = self.bridge.imgmsg_to_cv2(
            image_msg,
            desired_encoding="bgr8"
        )

        points = np.array([
            [p[0], p[1], p[2]]
            for p in point_cloud2.read_points(
                lidar_msg,
                field_names=(
                    "x",
                    "y",
                    "z"
                ),
                skip_nans=True
            )
        ])

        if len(points) == 0:
            return

        #
        # LiDAR -> Camera translation
        #

        points[:, 0] -= 1.5

        #
        # CARLA axes -> Camera axes
        #

        X = points[:, 1]
        Y = -points[:, 2]
        Z = points[:, 0]

        #
        # Keep points in front
        #

        mask = Z > 0

        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]

        if len(Z) == 0:
            return

        #
        # Projection
        #

        u = (
            self.fx * X / Z
            +
            self.cx
        )

        v = (
            self.fy * Y / Z
            +
            self.cy
        )

        #
        # Image bounds
        #

        mask = (
            (u >= 0)
            &
            (u < frame.shape[1])
            &
            (v >= 0)
            &
            (v < frame.shape[0])
        )

        depths = Z[mask]

        u = u[mask].astype(np.int32)
        v = v[mask].astype(np.int32)

        #
        # Draw points
        #

        for px, py, depth in zip(
            u,
            v,
            depths
        ):

            if depth < 10:
                color = (0, 0, 255)      # red

            elif depth < 20:
                color = (0, 255, 255)    # yellow

            else:
                color = (0, 255, 0)      # green

            cv2.circle(
                frame,
                (px, py),
                2,
                color,
                -1
            )

        # for px, py in zip(
        #     u,
        #     v
        # ):

        #     cv2.circle(
        #         frame,
        #         (px, py),
        #         2,
        #         (0, 255, 0),
        #         -1
        #     )

        # cv2.putText(
        #     frame,
        #     f"Projected Points: {len(u)}",
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (0, 255, 0),
        #     2
        # )

        cv2.imshow(
            "Camera-LiDAR Projection",
            frame
        )

        cv2.waitKey(
            1
        )


def main():

    rclpy.init()

    node = CameraLidarProjection()

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