import cv2
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2
from ultralytics import YOLO

from sensor_msgs.msg import PointCloud2
from sklearn.cluster import DBSCAN


class Perception3DPipeline:

    def __init__(self):
        self.bridge = None
        self.model = None

        self.cameras = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.MIN_POINTS = 3

        self.DBSCAN_EPS = 1.5          # meters
        self.DBSCAN_MIN_SAMPLES = 3

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
                    field_names=("x", "y", "z", "intensity"),
                    skip_nans=True
                )
            )
        )

        xyzi = np.stack(
            [
                points["x"],
                points["y"],
                points["z"],
                points["intensity"]
            ],
            axis=1
        ).astype(np.float32)

        return xyzi


    # def convert_ros_to_numpy(
    #     self,
    #     lidar_msg
    # ):

    #     points = np.asarray(
    #         list(
    #             point_cloud2.read_points(
    #                 lidar_msg,
    #                 field_names=("x", "y", "z"),
    #                 skip_nans=True
    #             )
    #         )
    #     )

    #     xyz = np.stack(
    #         [
    #             points["x"],
    #             points["y"],
    #             points["z"]
    #         ],
    #         axis=1
    #     ).astype(np.float32)

    #     return xyz

    def scale_polygon(self, polygon, scale=1.05):
        center = polygon.mean(axis=0)
        self.EGO_HOOD_POLYGON = np.round((polygon - center) * scale + center).astype(np.int32)

    def remove_hood(self, image):
        image = image.copy()
        cv2.fillPoly(image, [self.EGO_HOOD_POLYGON], (0, 0, 0))
        return image  


    def detect_image(
        self,
        image
    ):
        inference_image = self.remove_hood(image)

        results = self.model(
            inference_image,
            imgsz=640,
            verbose=False
        )

        # annotated = results[0].plot()
        annotated = image.copy()

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

        # for mask in masks:
        for mask_idx, mask in enumerate(masks):

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

        xyz = self.filter_dbscan(xyz)

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

    def generate_3d_detections(
        self,
        object_clouds,
        results
    ):
        """
        Generate 3D bounding boxes for AB3DMOT.

        Returns:
        [
            {
                "class": str,
                "score": float,
                "center": np.ndarray(3,),
                "dimensions": (length, width, height),
                "yaw": float
            },
            ...
        ]
        """

        boxes_3d = []

        for i, obj in enumerate(object_clouds):

            xyz = self.process_object_clouds(
                obj["cloud"]
            )
    
            if len(xyz) == 0:
                continue

            #
            # Skip sparse detections
            #

            if len(xyz) < self.MIN_POINTS:
                continue

            #
            # Center
            #

            center = self.estimate_object_position(
                obj["cloud"]
            )

            if center is None:
                continue

            #
            # Dimensions
            #

            min_xyz = np.min(xyz, axis=0)
            max_xyz = np.max(xyz, axis=0)

            length = float(max_xyz[0] - min_xyz[0])
            width  = float(max_xyz[1] - min_xyz[1])
            height = float(max_xyz[2] - min_xyz[2])


            #
            # Initial heading
            # (improved later)
            #

            yaw = self.estimate_yaw(xyz)
            # yaw = 0.0

            #
            # Confidence
            #

            score = float(
                results.boxes.conf[i].cpu().item()
            )

            #
            # Class
            #

            class_id = int(
                results.boxes.cls[i].cpu().item()
            )

            class_name = self.model.names[class_id]

            boxes_3d.append(
                {
                    "class": class_name,
                    "score": score,
                    "center": center,
                    "dimensions": (
                        length,
                        width,
                        height
                    ),
                    "yaw": yaw
                }
            )


        return boxes_3d    


    def estimate_yaw(self, xyz):

        if xyz is None or len(xyz) < 5:
            return 0.0

        # Ground plane
        xy = xyz[:, :2]

        # Initial centroid
        centroid = np.mean(xy, axis=0)

        # Remove outliers
        dist = np.linalg.norm(xy - centroid, axis=1)
        mask = dist < np.percentile(dist, 95)
        xy = xy[mask]

        # Too few points after filtering
        if len(xy) < 3:
            return 0.0

        # Recompute centroid
        centroid = np.mean(xy, axis=0)

        # Center data
        xy_centered = xy - centroid

        # PCA
        cov = np.cov(xy_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

        yaw = np.arctan2(
            principal_axis[1],
            principal_axis[0]
        )

        return float(yaw)


    def filter_dbscan(self, xyz):
        """
        Remove outliers from an object's LiDAR point cloud using DBSCAN.

        Args:
            xyz: (N,3) numpy array

        Returns:
            Filtered (N,3) numpy array containing only the largest cluster.
        """

        if len(xyz) < self.MIN_POINTS:
            return np.empty((0, 3), dtype=np.float32)

        clustering = DBSCAN(
            eps=self.DBSCAN_EPS,
            min_samples=self.DBSCAN_MIN_SAMPLES
        ).fit(xyz)

        # print(f"Points: {len(xyz)}, Labels: {np.unique(clustering.labels_)}")

        labels = clustering.labels_

        #
        # Remove noise
        #

        valid = labels != -1

        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        xyz = xyz[valid]
        labels = labels[valid]

        #
        # Keep largest cluster
        #

        unique_labels, counts = np.unique(
            labels,
            return_counts=True
        )

        largest_cluster = unique_labels[np.argmax(counts)]

        return xyz[labels == largest_cluster]

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