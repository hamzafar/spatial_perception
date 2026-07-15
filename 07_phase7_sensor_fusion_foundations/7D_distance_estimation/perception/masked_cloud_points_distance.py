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


class MaskLidarProjection(Node):

    def __init__(self):

        super().__init__(
            "mask_lidar_projection"
        )

        self.model = YOLO(
            "/home/hamza/ros2_cv_ws/scripts/phase6/engines/yolov8m-seg-trt-int8.engine"
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
            "Masked-LiDAR projection started"
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
            return

        yolo_masks = result.masks.data.cpu().numpy()

        resized_masks = []

        for yolo_mask in yolo_masks:

            yolo_mask = cv2.resize(
                yolo_mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            resized_masks.append(yolo_mask)


        masked_u = []
        masked_v = []

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

        original_points = points.copy()

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

        front_mask  = Z > 0

        X = X[front_mask ]
        Y = Y[front_mask ]
        Z = Z[front_mask ]

        original_points = original_points[front_mask]

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

        image_mask  = (
            (u >= 0)
            &
            (u < frame.shape[1])
            &
            (v >= 0)
            &
            (v < frame.shape[0])
        )

        depths = Z[image_mask]

        u = u[image_mask].astype(np.int32)
        v = v[image_mask].astype(np.int32)

        original_points = original_points[image_mask]


        #
        # Draw points
        #


        # # all projected points
        # for px, py in zip(u, v):
        #     cv2.circle(frame, (px, py), 1, (0,0,255), -1)

        # # masked points
        # for px, py in zip(masked_u, masked_v):
        #     cv2.circle(frame, (px, py), 3, (0,255,0), -1)

        

        # for yolo_mask in resized_masks:

        #     for px, py in zip(u, v):

        #         if yolo_mask[py, px] > 0:

        #             masked_u.append(px)
        #             masked_v.append(py)

        for idx, yolo_mask in enumerate(resized_masks):

            inside_mask = yolo_mask[v, u] > 0

            object_points = original_points[inside_mask]

            object_u = u[inside_mask]
            object_v = v[inside_mask]


            if len(object_points) < 5:
                continue


            distances = np.linalg.norm(
                object_points,
                axis=1
            )

            # distance_m = np.percentile(
            #     distances,
            #     10
            # )


            min_distance = np.min(
                distances
            )

            p10_distance = np.percentile(
                distances,
                10
            )

            median_distance = np.median(
                distances
            )


            cls_id = int(result.boxes.cls[idx])

            class_name = self.model.names[cls_id]

            box = result.boxes.xyxy[idx].cpu().numpy()

            x1, y1, x2, y2 = box

            label = (
                f"m:{min_distance:.1f}/" # m:min
                f"p:{p10_distance:.1f}/" # p:p10
                f"d:{median_distance:.1f}" # d:med
            )

            # label = f"{distance_m:.1f}m" ## original distance

            # label = (
            #     f"{distance_m:.1f}m "
            #     f"({len(object_points)})"
            # )  ### dubugging purpose

            cv2.putText(
                frame,
                label,
                (int(x1), int(y1) - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )



            # print(
            #     f"{class_name}: "
            #     f"{distance_m:.2f} m "
            #     f"({len(object_points)} points)"
            # )

            for px, py in zip(object_u, object_v):

                cv2.circle(
                    frame,
                    (px, py),
                    2,
                    (0,255,0),
                    -1
                )
            


        # for px, py in zip(masked_u, masked_v):

        #     cv2.circle(
        #         frame,
        #         (px, py),
        #         2,
        #         (0, 255, 0),
        #         -1
        #     )


        cv2.imshow(
            "Masked-LiDAR Projection",
            frame
        )

        cv2.waitKey(
            1
        )


def main():

    rclpy.init()

    node = MaskLidarProjection()

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