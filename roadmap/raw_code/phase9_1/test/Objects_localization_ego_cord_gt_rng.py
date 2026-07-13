import cv2
import rclpy
from cv_bridge import CvBridge

from rclpy.node import Node

from sensor_msgs.msg import Image

from message_filters import Subscriber, TimeSynchronizer

import numpy as np
import pandas as pd

from sensor_msgs.msg import (
    Image,
    PointCloud2
)

from sensor_msgs_py import point_cloud2

from ultralytics import YOLO

class MultiCameraPerception(Node):

    def __init__(self):

        super().__init__("multi_camera_perception")

        self.bridge = CvBridge()


        self.display_window_width = 640*3
        self. display_window_height = 480*2

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

        self.bev_weidth = 640 * 4
        self.bev_height = 480 * 2
        self.scale = 10.0  # pixels / meter

        # self.fx = self.image_width / 2
        # self.fy = self.image_width / 2

        # self.cx = self.image_width / 2
        # self.cy = self.image_height / 2

        self.fx = self.fy = 320
        
        self.cx = 320
        self.cy = 240 

        self.gt_max_distance = 25.0 # ground truth objects ranges

        self.gt_z_min = -2.0 # ground truth z min range
        self.gt_z_max = 2.0 # ground truth z max range
        
        self.model = YOLO("/home/hamza/ros2_cv_ws/scripts/phase6/engines/yolov8m-seg-trt-int8.engine")

        self.csv_path = "/home/hamza/ros2_cv_ws/scripts/phase9/recordings/session_20260704_223058/ground_truth.csv"

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


        front, u, v, projected_points = self.project_lidar(front, lidar, "front")

        front_clouds = self.extract_object_clouds(front, u, v, projected_points, front_result)

        front, front_world_obj = self.process_object_clouds_and_distance(front, front_clouds)


        rear, u, v, projected_points = self.project_lidar(rear, lidar, "rear")

        rear_clouds = self.extract_object_clouds(rear, u, v, projected_points, rear_result)

        rear, rear_world_obj = self.process_object_clouds_and_distance(rear, rear_clouds)

        
        left, u, v, projected_points = self.project_lidar(left, lidar, "left")

        left_clouds = self.extract_object_clouds(left, u, v, projected_points, left_result)
        
        left, left_world_obj = self.process_object_clouds_and_distance(left, left_clouds)


        right, u, v, projected_points = self.project_lidar(right, lidar, "right")

        right_clouds = self.extract_object_clouds(right, u, v, projected_points, right_result)

        right, right_world_obj = self.process_object_clouds_and_distance(right, right_clouds)


        front = self.draw_camera_name(front, "FRONT")
        rear = self.draw_camera_name(rear, "REAR")
        left = self.draw_camera_name(left, "LEFT")
        right = self.draw_camera_name(right, "RIGHT")

        world_objects_lidar = ( front_world_obj + rear_world_obj + left_world_obj + right_world_obj) # from lidar

        frame_id = int(front_msg.header.frame_id.split("_")[-1])

        world_objects_gt = self.load_ground_truth_frame(self.csv_path, frame_id) # from ground truth csv      

        # bev = self.draw_bev(world_objects_lidar)

        bev_comp = self.draw_bev_comparison(world_objects_lidar, world_objects_gt, frame_id)

        stitched = self.stitch_images(
            front,
            rear,
            left,
            right,
            bev_comp
        )

        self.display_result(
            stitched
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
        right,
        bev
    ):

        top = cv2.hconcat(
            [
                left,
                front,
                right,
                rear
            ]
        )

        stitched = cv2.vconcat(
            [
                top,
                bev
            ]
        )

        return stitched

    # -------------------------------------------------

    def estimate_object_position(
        self,
        obj_cloud
    ):
        """
        Estimate object position using the
        front surface of the LiDAR cluster.
        """

        xyz_points = self.process_object_clouds(
            obj_cloud
        )

        if len(xyz_points) == 0:
            return None

        #
        # Distance of each LiDAR point
        #

        distances = np.linalg.norm(
            xyz_points,
            axis=1
        )

        #
        # Front surface threshold
        #

        threshold = np.percentile(
            distances,
            10
        )

        #
        # Keep only the front-most points
        #

        front_points = xyz_points[
            distances <= threshold
        ]

        #
        # Fallback
        #

        if len(front_points) == 0:
            front_points = xyz_points

        #
        # Representative position
        #

        position = np.mean(
            front_points,
            axis=0
        )

        return position


    # def estimate_object_position(
    #     self,
    #     obj_cloud
    # ):
    #     xyz_points = self.process_object_clouds(
    #         obj_cloud
    #     )

    #     if len(xyz_points) == 0:
    #         return None

    #     centroid = np.mean(
    #         xyz_points,
    #         axis=0
    #     )

    #     return centroid

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

        points = lidar.copy() # Camera frame
        ego_points = lidar.copy() # Ego frame

        #
        # Camera configuration
        #

        yaw = -np.deg2rad(
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
            ego_points[mask][valid]
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

            class_id = int(results.boxes.cls[len(object_clouds)])
            class_name = self.model.names[class_id]

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

            MIN_LIDAR_POINTS = 5

            if len(cloud) < MIN_LIDAR_POINTS:
                continue

            object_clouds.append(
                {
                    "cloud": cloud,
                    "box": results.boxes.xyxy.cpu().numpy()[len(object_clouds)],
                    "class_name": class_name
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

        world_objects = []

        for obj in object_clouds:

            image = self.draw_object_cloud(
                image,
                obj["cloud"]
            )

            distance = self.estimate_distance(
                    self.process_object_clouds(
                        obj["cloud"]
                    )
                )

            # image = self.draw_distance(
            #     image,
            #     obj,
            #     distance
            # )

            position = self.estimate_object_position(
                obj["cloud"]
            )

            if position is not None:
                x, y, z = position

                # if obj["class_name"] == "bench":
                #     print(position) #debug

                world_objects.append(
                    {
                        "class": obj["class_name"],
                        "position": position,
                        "distance": distance
                    }
                )
            # print(
            #     f"x={x:.2f}, y={y:.2f}, z={z:.2f}, distance={distance:.2f} m"
            # )
            # print(world_objects)
        # input('imhere')    

        return image, world_objects

    # -------------------------------------------------

    def load_ground_truth_frame(
        self,
        csv_path,
        frame_id
    ):
        """
        Load one frame from ground_truth.csv
        and convert it into world_objects
        compatible with draw_bev().
        """

        df = pd.read_csv(csv_path)

        frame_df = df[
            df["frame_id"] == frame_id
        ]

        world_objects = []

        for _, row in frame_df.iterrows():

            # Distance filter
            if row["distance"] > self.gt_max_distance:
                continue

            # Height filter
            if (
                row["ego_z"] < self.gt_z_min
                or
                row["ego_z"] > self.gt_z_max
            ):
                continue

            world_objects.append(
                {
                    "class": row["yolo_class"],

                    "position": np.array(
                        [
                            row["ego_x"],
                            row["ego_y"],
                            row["ego_z"]
                        ],
                        dtype=np.float32
                    ),

                    "distance": float(
                        row["distance"]
                    ),

                    "actor_id": int(
                        row["actor_id"]
                    ),

                    "lidar_visible": bool(
                        row["lidar_visible"]
                    )
                }
            )

        return world_objects

    # -------------------------------------------------
    def draw_bev_comparison(
        self,
        lidar_objects,
        gt_objects,
        frame_id
    ):

        bev = np.zeros(
            (self.bev_height, self.bev_weidth, 3),
            dtype=np.uint8
        )

        center_x = bev.shape[1] // 2
        center_y = bev.shape[0] // 2

        #
        # Ego Vehicle
        #

        cv2.rectangle(
            bev,
            (center_x - 12, center_y - 20),
            (center_x + 12, center_y + 20),
            (255, 255, 255),
            -1
        )

        cv2.arrowedLine(
            bev,
            (center_x, center_y),
            (center_x, center_y - 35),
            (0, 255, 255),
            2,
            tipLength=0.4
        )

        cv2.putText(
            bev,
            "EGO",
            (center_x - 15, center_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        #
        # Ground Truth (Green)
        #

        for obj in gt_objects:

            x, y, _ = obj["position"]

            px = int(center_x + y * self.scale)
            py = int(center_y - x * self.scale)

            cv2.circle(
                bev,
                (px, py),
                7,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                bev,
                obj["class"],
                (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        #
        # LiDAR (Red)
        #

        for obj in lidar_objects:

            x, y, _ = obj["position"]

            px = int(center_x + y * self.scale)
            py = int(center_y - x * self.scale)

            cv2.circle(
                bev,
                (px, py),
                4,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                bev,
                obj["class"],
                (px + 5, py + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        #
        # Legend
        #

        cv2.putText(
            bev,
            "GT",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            bev,
            "LiDAR",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            bev,
            "FrameID: " + str(frame_id),
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )
        return bev


    def draw_bev(
        self,
        world_objects
    ):

        bev = np.zeros(
            (self.bev_height, self.bev_weidth, 3),
            dtype=np.uint8
        )

        center_x = bev.shape[1] // 2
        center_y = bev.shape[0] // 2

        self.class_colors = {
            "person": (0, 255, 255),
            "car": (0, 255, 0),
            "truck": (0, 0, 255),
            "bus": (255, 0, 0),
            "motorcycle": (255, 255, 0),
            "bicycle": (255, 0, 255),
        }


        # Ego vehicle
        cv2.rectangle(
            bev,
            (center_x - 12, center_y - 20),
            (center_x + 12, center_y + 20),
            (255, 255, 255),
            -1
        )

        # Heading arrow
        cv2.arrowedLine(
            bev,
            (center_x, center_y),
            (center_x, center_y - 35),
            (0, 255, 255),
            2,
            tipLength=0.4
        )

        # Label
        cv2.putText(
            bev,
            "EGO",
            (center_x - 15, center_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # # Ego vehicle
        # cv2.circle(
        #     bev,
        #     (center_x, center_y),
        #     8,
        #     (0, 255, 255),
        #     -1
        # )

        for obj in world_objects:

            color = self.class_colors.get(
                obj["class"],
                (255, 255, 255)
            )            

            x, y, _ = obj["position"]

            px = int(center_x + y * self.scale)
            py = int(center_y - x * self.scale)

            cv2.circle(
                bev,
                (px, py),
                5,
                color,
                -1
            )

            cv2.putText(
                bev,
                obj["class"],
                (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, # was 0.4
                color,
                2 # was 1
            )

            # cv2.circle(
            #     bev,
            #     (px, py),
            #     5,
            #     (0, 255, 0),
            #     -1
            # )

            # cv2.putText(
            #     bev,
            #     obj["class"],
            #     (px + 5, py - 5),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.4,
            #     (255, 255, 255),
            #     1
            # )

        return bev    

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
            (0, 255, 255),
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

        cv2.putText(
            image,
            f"{distance:.1f} m",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
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
