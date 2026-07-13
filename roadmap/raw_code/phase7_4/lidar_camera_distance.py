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

from ultralytics import YOLO


class CameraLidarDistance(Node):

    def __init__(self):

        super().__init__(
            "camera_lidar_distance"
        )

        self.model = YOLO(
            "/home/hamza/ros2_cv_ws/scripts/phase6/engines/yolov8m-seg-trt-int8.engine"
        )

        self.bridge = CvBridge()

        #
        # Camera Intrinsics
        #

        self.fx = 320.0
        self.fy = 320.0

        self.cx = 320.0
        self.cy = 240.0

        #
        # Approximate object heights (meters)
        #

        self.object_heights = {

            "person": 1.7,

            "car": 1.5,

            "truck": 2.5,

            "bus": 3.2,

            "motorcycle": 1.2,

            "bicycle": 1.2
        }

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
            "Camera + LiDAR distance estimator started"
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

        results = self.model(
            frame,
            imgsz=640,
            verbose=False
        )

        result = results[0]

        frame = result.plot()

        if result.masks is None:
            cv2.imshow(
                "Camera + LiDAR Distance",
                frame
            )

            cv2.waitKey(1)

            return

        #
        # Resize masks
        #

        yolo_masks = result.masks.data.cpu().numpy()

        resized_masks = []

        for yolo_mask in yolo_masks:

            resized_mask = cv2.resize(
                yolo_mask.astype(np.uint8),
                (
                    frame.shape[1],
                    frame.shape[0]
                ),
                interpolation=cv2.INTER_NEAREST
            )

            resized_masks.append(
                resized_mask
            )

        #
        # Read LiDAR
        #

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

        original_points = points.copy()

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

        front_mask = Z > 0

        X = X[front_mask]
        Y = Y[front_mask]
        Z = Z[front_mask]

        original_points = original_points[
            front_mask
        ]

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

        image_mask = (
            (u >= 0)
            &
            (u < frame.shape[1])
            &
            (v >= 0)
            &
            (v < frame.shape[0])
        )

        u = u[
            image_mask
        ].astype(
            np.int32
        )

        v = v[
            image_mask
        ].astype(
            np.int32
        )

        original_points = original_points[
            image_mask
        ]

        #
        # Per-object processing
        #

        for idx, yolo_mask in enumerate(
            resized_masks
        ):

            #
            # Class
            #

            cls_id = int(
                result.boxes.cls[idx]
            )

            class_name = self.model.names[
                cls_id
            ]

            #
            # -----------------------
            # LiDAR Distance
            # -----------------------
            #

            inside_mask = (
                yolo_mask[v, u] > 0
            )

            object_points = (
                original_points[
                    inside_mask
                ]
            )

            object_u = u[
                inside_mask
            ]

            object_v = v[
                inside_mask
            ]

            lidar_distance = None

            if len(object_points) >= 3:

                distances = np.linalg.norm(
                    object_points,
                    axis=1
                )

                lidar_distance = (
                    np.percentile(
                        distances,
                        10
                    )
                )

            #
            # -----------------------
            # Camera Distance
            # -----------------------
            #

            camera_distance = None

            if (
                class_name
                in
                self.object_heights
            ):

                ys, xs = np.where(
                    yolo_mask > 0
                )

                if len(ys) > 0:

                    mask_height = (
                        ys.max()
                        -
                        ys.min()
                    )

                    if mask_height > 0:

                        real_height = (
                            self.object_heights[
                                class_name
                            ]
                        )

                        camera_distance = (
                            real_height
                            *
                            self.fy
                        ) / mask_height

                            #
            # -----------------------
            # Fuse Distance
            # -----------------------
            #

            fusion_distance = None

            if (
                lidar_distance is not None
                and
                camera_distance is not None
            ):

                fusion_distance = (
                    0.8 * lidar_distance
                    +
                    0.2 * camera_distance
                )

            elif lidar_distance is not None:

                fusion_distance = lidar_distance

            elif camera_distance is not None:

                fusion_distance = camera_distance



            #
            # Draw distances
            #

            box = result.boxes.xyxy[
                idx
            ].cpu().numpy()

            x1, y1, x2, y2 = box

            
            lidar_text = "L:N/A"

            if lidar_distance is not None:

                lidar_text = (
                    f"L:{lidar_distance:.1f}m"
                )


            camera_text = "C:N/A"

            if camera_distance is not None:

                camera_text = (
                    f"C:{camera_distance:.1f}m"
                )


            fusion_text = "F:N/A"

            if fusion_distance is not None:

                fusion_text = (
                    f"F:{fusion_distance:.1f}m"
                )


 
            cv2.putText(
                frame,
                lidar_text,
                (int(x1), int(y1) - 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,0,0),
                2
            )

            cv2.putText(
                frame,
                camera_text,
                (int(x1), int(y1) - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                fusion_text,
                (int(x1), int(y1) - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2
            )


            #
            # Draw LiDAR points
            #

            for px, py in zip(
                object_u,
                object_v
            ):

                cv2.circle(
                    frame,
                    (
                        px,
                        py
                    ),
                    2,
                    (
                        0,
                        255,
                        0
                    ),
                    -1
                )

        cv2.imshow(
            "Camera + LiDAR Distance",
            frame
        )

        # cv2.waitKey(
        #     1
        # )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):
            cv2.waitKey(0)


        key = cv2.waitKey(1) & 0xFF

def main():

    rclpy.init()

    node = CameraLidarDistance()

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