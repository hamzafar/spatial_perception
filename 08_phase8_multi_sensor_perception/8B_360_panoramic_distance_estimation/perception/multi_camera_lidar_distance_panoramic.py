import cv2
import rclpy
from cv_bridge import CvBridge

from rclpy.node import Node

from sensor_msgs.msg import Image

from message_filters import Subscriber, TimeSynchronizer

import numpy as np

from sensor_msgs.msg import (
    Image,
    PointCloud2
)

from sensor_msgs_py import point_cloud2

from ultralytics import YOLO

from panoramic_stitcher import PanoramicStitcher

class MultiCameraPerception(Node):

    def __init__(self):

        super().__init__("multi_camera_perception")

        self.bridge = CvBridge()


        self.display_window_width = 1280
        self. display_window_height = 920

        self.cameras = {

            "front": {
                "yaw": 0
            },

            "rear": {
                "yaw": 180
            },

            "left": {
                "yaw": -90
            },

            "right": {
                "yaw": 90
            }

        }

        self.image_width = 640
        self.image_height = 480

        # self.fx = self.image_width / 2
        # self.fy = self.image_width / 2

        # self.cx = self.image_width / 2
        # self.cy = self.image_height / 2

        self.fx = self.fy = 320
        
        self.cx = 320
        self.cy = 240 

        self.model = YOLO("/home/hamza/ros2_cv_ws/scripts/phase6/engines/yolov8m-seg-trt-int8.engine")

        self.panoramic_class = PanoramicStitcher(
            image_width=self.image_width,
            image_height=self.image_height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            camera_yaws={
                name: cfg["yaw"]
                for name, cfg in self.cameras.items()
            }
        )

        cv2.namedWindow(
            "Phase 8 - Multi Camera Perception",
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            "Phase 8 - Multi Camera Perception",
            self.display_window_width,
            self.display_window_height
        )


        # ---------------------------------
        # Subscribers
        # ---------------------------------

        self.front_sub = Subscriber(
            self,
            Image,
            "/carla/ego_vehicle/rgb_front/image"
        )

        self.rear_sub = Subscriber(
            self,
            Image,
            "/carla/ego_vehicle/rgb_rear/image"
        )

        self.left_sub = Subscriber(
            self,
            Image,
            "/carla/ego_vehicle/rgb_left/image"
        )

        self.right_sub = Subscriber(
            self,
            Image,
            "/carla/ego_vehicle/rgb_right/image"
        )

        self.lidar_sub = Subscriber(
            self,
            PointCloud2,
            "/carla/ego_vehicle/lidar"
        )

        self.sync = TimeSynchronizer(
            [
                self.front_sub,
                self.rear_sub,
                self.left_sub,
                self.right_sub,
                self.lidar_sub
            ],
            queue_size=10
        )

        self.sync.registerCallback(
            self.synchronized_callback
        )

        self.get_logger().info(
            "Multi-camera perception node started."
        )

    # -------------------------------------------------

    def synchronized_callback(
        self,
        front_msg,
        rear_msg,
        left_msg,
        right_msg,
        lidar_msg
    ):

        front = self.convert_ros_to_cv(front_msg)
        rear = self.convert_ros_to_cv(rear_msg)
        left = self.convert_ros_to_cv(left_msg)
        right = self.convert_ros_to_cv(right_msg)


        (
            front, rear, left, right,

            front_result, rear_result, left_result, right_result

        ) = self.yolo_detection(

            front,
            rear,
            left,
            right

        )

        lidar = self.convert_ros_to_numpy(lidar_msg)

        object_labels = [] 

        front, u, v, projected_points = self.project_lidar(front, lidar, "front")

        front_clouds = self.extract_object_clouds(front, u, v, projected_points, front_result)

        front = self.process_object_clouds_and_distance(front, front_clouds)
        # front = self.draw_distance_legend(front)
        # object_labels += self.collect_object_labels(front_clouds, "front")

        rear, u, v, projected_points = self.project_lidar(rear, lidar, "rear")

        rear_clouds = self.extract_object_clouds(rear, u, v, projected_points, rear_result)

        rear = self.process_object_clouds_and_distance(rear, rear_clouds)
        # object_labels += self.collect_object_labels(rear_clouds, "rear")
        
        left, u, v, projected_points = self.project_lidar(left, lidar, "left")

        left_clouds = self.extract_object_clouds(left, u, v, projected_points, left_result)
        
        left = self.process_object_clouds_and_distance(left, left_clouds)
        # object_labels += self.collect_object_labels(left_clouds, "left")

        right, u, v, projected_points = self.project_lidar(right, lidar, "right")

        right_clouds = self.extract_object_clouds(right, u, v, projected_points, right_result)

        right = self.process_object_clouds_and_distance(right, right_clouds)
        # object_labels += self.collect_object_labels(right_clouds, "right")


        # front = self.draw_camera_name(front, "FRONT")
        # rear = self.draw_camera_name(rear, "REAR")
        # left = self.draw_camera_name(left, "LEFT")
        # right = self.draw_camera_name(right, "RIGHT")


        panoramic_images = self.panoramic_class.convert_panoramic_images(
            front,
            rear,
            left,
            right
        )

        panoramic_images = self.panoramic_class.draw_compass_strip(
            panoramic_images,
            camera_yaws={name: cfg["yaw"] for name, cfg in self.cameras.items()}
        )        


        # for azimuth_rad, distance in object_labels:
        #     panoramic_images = self.panoramic_class.draw_distance_pill(
        #         panoramic_images,
        #         azimuth_rad,
        #         distance
        #     )

        panoramic_images = self.panoramic_class.draw_ego_marker(panoramic_images)
        panoramic_images = self.panoramic_class.draw_distance_legend(panoramic_images)

        self.display_result(
            panoramic_images
        )

    # -------------------------------------------------

    def convert_ros_to_cv(
        self,
        image_msg
    ):

        return self.bridge.imgmsg_to_cv2(
            image_msg,
            desired_encoding="bgr8"
        )

 
    def stitch_images(
        self,
        front,
        rear,
        left,
        right
    ):

        top = cv2.hconcat(
            [
                left,
                front,
                right
            ]
        )

        bottom = cv2.hconcat(
            [
                right,
                rear,
                left
            ]
        )

        stitched = cv2.vconcat(
            [
                top,
                bottom
            ]
        )

        return stitched

    # -------------------------------------------------
    
    def detect_image(
        self,
        image
    ):

        results = self.model(
            image,
            imgsz=640,
            verbose=False
        )

        annotated = results[0].plot()

        annotated = results[0].plot(boxes=False)

        return annotated, results[0]


    def yolo_detection(
        self,
        front,
        rear,
        left,
        right
    ):

        front, front_result = self.detect_image(front)

        rear, rear_result = self.detect_image(rear)

        left, left_result = self.detect_image(left)

        right, right_result = self.detect_image(right)

        return (

            front,
            rear,
            left,
            right,

            front_result,
            rear_result,
            left_result,
            right_result

        )

    # -------------------------------------------------

    def convert_ros_to_numpy(
        self,
        lidar_msg
    ):

        points = np.asarray(
            list(
                point_cloud2.read_points(
                    lidar_msg,
                    field_names=("x", "y", "z"),
                    skip_nans=True
                )
            )
        )

        xyz = np.stack(
            [
                points["x"],
                points["y"],
                points["z"]
            ],
            axis=1
        ).astype(np.float32)

        return xyz

    # -------------------------------------------------

    def project_lidar(
        self,
        image,
        lidar,
        camera_name
    ):

        points = lidar.copy()

        #
        # Camera configuration
        #

        yaw = np.deg2rad(
            self.cameras[camera_name]["yaw"]
        )

        #
        # Rotate LiDAR into camera frame
        #

        R = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw),  np.cos(yaw), 0.0],
                [0.0,          0.0,         1.0]
            ],
            dtype=np.float32
        )

        points = points @ R.T

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
            return image

        #
        # Projection
        #

        u = self.fx * X / Z + self.cx
        v = self.fy * Y / Z + self.cy

        #
        # Keep points inside image
        #

        h, w = image.shape[:2]

        valid = (
            (u >= 0) &
            (u < w) &
            (v >= 0) &
            (v < h)
        )

        u = u[valid].astype(np.int32)
        v = v[valid].astype(np.int32)


        # image = self.draw_projected_points(image, u, v)

        # return image
        return (
            image,
            u,
            v,
            points[mask][valid]
        )


    def extract_object_clouds(
        self,
        image,
        u,
        v,
        projected_points,
        results
    ):

        if results.masks is None:
            return []

        # masks = results.masks.data.cpu().numpy()

        masks = []

        for mask in results.masks.data.cpu().numpy():

            mask = cv2.resize(
                mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            masks.append(mask)        

        object_clouds  = []

        for mask in masks:

            cloud = []

            for i, (x, y) in enumerate(zip(u, v)):

                if mask[int(y), int(x)] > 0:

                    cloud.append(
                        {
                            "pixel": (int(x), int(y)),
                            "xyz": projected_points[i]
                        }
                    )

            # object_clouds.append(cloud)

            object_clouds.append(
                {
                    "cloud": cloud,
                    "box": results.boxes.xyxy.cpu().numpy()[len(object_clouds)]
                }
            )


        return object_clouds 

    # ------------------------------------------------

    def process_object_clouds(
        self,
        object_cloud
    ):

        if len(object_cloud) == 0:

            return np.empty(
                (0, 3),
                dtype=np.float32
            )

        xyz = np.array(
            [
                point["xyz"]
                for point in object_cloud
            ],
            dtype=np.float32
        )

        return xyz


    def estimate_distance(
        self,
        xyz_points
    ):

        if len(xyz_points) == 0:

            return None

        distances = np.linalg.norm(
            xyz_points,
            axis=1
        )

        distance = np.percentile(
            distances,
            10
        )

        return float(distance)


    def process_object_clouds_and_distance(
        self,
        image,
        object_clouds
    ):

        for obj in object_clouds:

            # image = self.draw_object_cloud(
            #     image,
            #     obj["cloud"]
            # )

            image = self.draw_distance(
                image,
                obj,
                self.estimate_distance(
                    self.process_object_clouds(
                        obj["cloud"]
                    )
                )
            )

        return image

    # -------------------------------------------------

    def draw_camera_name(
        self,
        image,
        camera_name
    ):

        text_size, _ = cv2.getTextSize(
            camera_name,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            2
        )

        x = (image.shape[1] - text_size[0]) // 2
        y = 30

        cv2.putText(
            image,
            camera_name,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (249, 100, 50),
            2
        )

        return image

    def draw_distance(
        self,
        image,
        obj,
        distance
    ):

        if distance is None:
            return image

        x1, y1, x2, y2 = obj["box"]

        x = int((x1 + x2) / 2)
        y = int(y1)

        # Distance color
        if distance < 10:
            color = (0, 0, 255)      # Red (Danger)
        elif distance < 20:
            color = (0, 255, 255)    # Yellow (Warning)
        else:
            color = (0, 255, 0)      # Green (Safe)


        cv2.putText(
            image,
            f"{distance:.1f} m",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        return image


    def draw_object_cloud(
        self,
        image,
        object_cloud
    ):

        for point in object_cloud:

            x, y = point["pixel"]

            cv2.circle(
                image,
                (x, y),
                2,
                (0, 255, 0),
                -1
            )

        return image

    def draw_projected_points(
        self,
        image,
        u,
        v
    ):

        for x, y in zip(u, v):

            cv2.circle(
                image,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )

        return image



    def project_lidar(
        self,
        image,
        lidar,
        camera_name
    ):
 
        points = lidar.copy()
 
        #
        # Camera configuration
        #
 
        yaw = np.deg2rad(
            self.cameras[camera_name]["yaw"]
        )
 
        #
        # Rotate LiDAR into camera frame
        #
 
        R = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw),  np.cos(yaw), 0.0],
                [0.0,          0.0,         1.0]
            ],
            dtype=np.float32
        )
 
        points = points @ R.T
 
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
            return image
 
        #
        # Projection
        #
 
        u = self.fx * X / Z + self.cx
        v = self.fy * Y / Z + self.cy
 
        #
        # Keep points inside image
        #
 
        h, w = image.shape[:2]
 
        valid = (
            (u >= 0) &
            (u < w) &
            (v >= 0) &
            (v < h)
        )
 
        u = u[valid].astype(np.int32)
        v = v[valid].astype(np.int32)
 
 
        # image = self.draw_projected_points(image, u, v)
 
        # return image
        return (
            image,
            u,
            v,
            points[mask][valid]
        )

    def draw_distance_legend(
        self,
        image
    ):

        legend = [
            ("Safe", (0, 255, 0)),
            ("Caution", (0, 255, 255)),
            ("Danger", (0, 0, 255))
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2

        spacing = 25
        padding = 10
        y = 55

        # Compute legend width
        total_width = 0

        for label, _ in legend:

            (w, h), _ = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            total_width += 12 + 8 + w + spacing

        total_width -= spacing

        box_width = total_width + 2 * padding
        box_height = 35

        x0 = (image.shape[1] - box_width) // 2
        y0 = 35

        # Semi-transparent background
        overlay = image.copy()

        cv2.rectangle(
            overlay,
            (x0, y0),
            (x0 + box_width, y0 + box_height),
            (40, 40, 40),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.6,
            image,
            0.4,
            0,
            image
        )

        x = x0 + padding

        for label, color in legend:

            cv2.circle(
                image,
                (x + 6, y - 4),
                5,
                color,
                -1
            )

            text_x = x + 18

            # Outline
            cv2.putText(
                image,
                label,
                (text_x, y),
                font,
                font_scale,
                (0, 0, 0),
                4,
                cv2.LINE_AA
            )

            # Colored text
            cv2.putText(
                image,
                label,
                (text_x, y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )

            (w, _), _ = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            x += 12 + 8 + w + spacing

        return image


    def display_result(
        self,
        image
    ):

        cv2.imshow(
            "Phase 8 - Multi Camera Perception",
            image
        )

        cv2.waitKey(1)


def main():

    rclpy.init()

    node = MultiCameraPerception()

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