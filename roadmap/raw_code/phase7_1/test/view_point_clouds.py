import rclpy
import numpy as np
import open3d as o3d

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class LidarViewer(Node):

    def __init__(self):

        super().__init__(
            "lidar_open3d_viewer"
        )

        self.subscription = (
            self.create_subscription(
                PointCloud2,
                "/carla/ego_vehicle/lidar",
                self.lidar_callback,
                10
            )
        )

        self.vis = (
            o3d.visualization.Visualizer()
        )

        self.vis.create_window(
            window_name="CARLA LiDAR"
        )

        self.pcd = (
            o3d.geometry.PointCloud()
        )

        self.geometry_added = False

        self.get_logger().info(
            "Waiting for LiDAR data..."
        )

    def lidar_callback(
        self,
        msg
    ):

        points = np.array(
            list(
                point_cloud2.read_points(
                    msg,
                    field_names=(
                        "x",
                        "y",
                        "z"
                    ),
                    skip_nans=True
                )
            ),
            dtype=np.float32
        )

        if len(points) == 0:
            return

        self.pcd.points = (
            o3d.utility.Vector3dVector(
                points
            )
        )

        if not self.geometry_added:

            self.vis.add_geometry(
                self.pcd
            )

            self.geometry_added = True

        else:

            self.vis.update_geometry(
                self.pcd
            )

        self.vis.poll_events()
        self.vis.update_renderer()

        self.get_logger().info(
            f"Points: {len(points)}"
        )


def main():

    rclpy.init()

    node = LidarViewer()

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    node.vis.destroy_window()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()