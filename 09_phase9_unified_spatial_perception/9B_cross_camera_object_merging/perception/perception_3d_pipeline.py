import cv2
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
from ultralytics import YOLO

from sensor_msgs.msg import PointCloud2

class Perception3DPipeline:

    def __init__(self):
        self.bridge = None
        self.model = None

        self.cameras = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

    def validate(self):
        assert self.bridge is not None
        assert self.model is not None
        assert self.cameras is not None

        assert self.fx is not None
        assert self.fy is not None
        assert self.cx is not None
        assert self.cy is not None

    def convert_ros_to_cv(
        self,
        image_msg
    ):

        return self.bridge.imgmsg_to_cv2(
            image_msg,
            desired_encoding="bgr8"
        )


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

            object_clouds.append(
                {
                    "cloud": cloud,
                    "box": results.boxes.xyxy.cpu().numpy()[len(object_clouds)],
                    "class_name": class_name
                }
            )

        return object_clouds 

    def process_object_clouds_and_distance(
        self,
        image,
        object_clouds,
        camera_name #duplicate_removal
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

            position = self.estimate_object_position(
                obj["cloud"]
            )

            if position is not None:
                x, y, z = position

                world_objects.append(
                    {
                        "class": obj["class_name"],
                        "camera": camera_name, #duplicate_removal
                        "position": position,
                        "distance": distance
                    }
                )

        return image, world_objects

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



    def process_camera(
        self,
        image,
        lidar,
        result,
        camera_name
    ):
        """
        project_lidar()
            ↓
        extract_object_clouds()
            ↓
        process_object_clouds_and_distance()
        """
        pass