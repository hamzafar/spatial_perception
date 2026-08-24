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

        self.MIN_OBJECT_POINTS = None

        self.EGO_HOOD_POLYGON = np.array([
            [175, 480],
            [240, 390],
            [400, 390],
            [465, 480]
        ], dtype=np.int32)
        
        self.scale_polygon(self.EGO_HOOD_POLYGON, 1.10)

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

    def scale_polygon(self, polygon, scale=1.05):
        center = polygon.mean(axis=0)
        self.EGO_HOOD_POLYGON = np.round((polygon - center) * scale + center).astype(np.int32)


    def remove_hood(self, image):
        image = image.copy()
        cv2.fillPoly(image, [self.EGO_HOOD_POLYGON], (0, 0, 0))
        return image  


    def detect_image(self, image):

        original_image = image.copy()

        inference_image = self.remove_hood(image)

        results = self.model(
            inference_image,
            imgsz=640,
            verbose=False
        )

        annotated = results[0].plot(img=original_image)

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
                    "class_name": class_name,
                    "score": float(
                        results.boxes.conf[len(object_clouds)].cpu().item()
                    )
                }
            )

        return object_clouds 


    def process_object_clouds_and_distance(
        self,
        image,
        object_clouds,
        camera_name
    ):

        world_objects = []

        for obj in object_clouds:

            if len(obj["cloud"]) < self.MIN_OBJECT_POINTS:
                continue

            image = self.draw_object_cloud(
                image,
                obj["cloud"]
            )

            #
            # Convert LiDAR cluster once
            #

            xyz_points = self.process_object_clouds(
                obj["cloud"]
            )

            #
            # Estimate object attributes
            #

            distance = self.estimate_distance(
                xyz_points
            )

            position = self.estimate_object_position(
                xyz_points
            )

            dimensions = self.estimate_object_dimensions(
                xyz_points
            )

            yaw = self.estimate_object_yaw(
                xyz_points
            )

            if position is None:
                continue

            world_objects.append(
                {
                    "class": obj["class_name"],
                    "camera": camera_name,

                    # BEV visualization
                    "position": position,
                    "distance": distance,

                    # Future 3D MOT
                    "center": position,
                    "score": obj["score"],
                    "dimensions": dimensions,
                    "yaw": yaw,
                    
                    # Optional
                    "cloud": obj["cloud"]
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


    def estimate_object_dimensions(
        self,
        xyz_points
    ):
        """
        Estimate object dimensions from the LiDAR point cloud.

        Returns
        -------
        tuple
            (length, width, height)
        """

        # xyz_points = self.process_object_clouds(
        #     object_cloud
        # )

        if len(xyz_points) == 0:
            return None

        min_xyz = np.min(
            xyz_points,
            axis=0
        )

        max_xyz = np.max(
            xyz_points,
            axis=0
        )

        length = float(max_xyz[0] - min_xyz[0])
        width = float(max_xyz[1] - min_xyz[1])
        height = float(max_xyz[2] - min_xyz[2])

        return (
            length,
            width,
            height
        )

    def estimate_object_yaw(
        self,
        xyz_points
    ):
        """
        Estimate object yaw (heading) from the LiDAR point cloud.

        Returns
        -------
        float
            Yaw in radians (ego frame).
        """

        # xyz_points = self.process_object_clouds(
        #     object_cloud
        # )

        if len(xyz_points) < 2:
            return 0.0

        #
        # Use ground-plane coordinates only
        #

        xy = xyz_points[:, :2]

        #
        # Center the points
        #

        xy = xy - np.mean(
            xy,
            axis=0
        )

        #
        # PCA
        #

        covariance = np.cov(
            xy,
            rowvar=False
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        #
        # Principal axis
        #

        principal_axis = eigenvectors[
            :,
            np.argmax(eigenvalues)
        ]

        #
        # Heading
        #

        yaw = np.arctan2(
            principal_axis[1],
            principal_axis[0]
        )

        return float(yaw)


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
        xyz_points
    ):
        """
        Estimate object position using the
        front surface of the LiDAR cluster.
        """

        # xyz_points = self.process_object_clouds(
        #     obj_cloud
        # )

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