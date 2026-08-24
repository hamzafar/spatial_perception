import numpy as np
from sensor_msgs_py import point_cloud2


class RadarPerceptionPipeline:

    def __init__(self):

        self.velocity_threshold = None #velocity_threshold
        self.max_assoc_distance_px = None #max_assoc_distance_px

        self.fx = None #camera_intrinsics["fx"]
        self.fy = None #camera_intrinsics["fy"]
        self.cx = None #camera_intrinsics["cx"]
        self.cy = None #camera_intrinsics["cy"]
        self.width = None #camera_intrinsics["width"]
        self.height = None #camera_intrinsics["height"]

        self.T_radar_to_cam = None #radar_to_camera_transform

    # -------------------------------------------------

    def convert_ros_to_numpy(self, radar_msg):
        radar_points = point_cloud2.read_points(
            radar_msg,
            field_names=("depth", "azimuth", "altitude", "velocity"),
            skip_nans=True
        )
        return np.column_stack([
            radar_points["depth"],
            radar_points["azimuth"],
            radar_points["altitude"],
            radar_points["velocity"],
        ])

    # -------------------------------------------------

    def spherical_to_cartesian(self, radar_target):
        """
        CARLA radar convention: depth (range), azimuth, altitude in radians.
        Returns point in radar's local frame (x forward, y right, z up).
        """
        depth, azimuth, altitude, _ = radar_target

        x = depth * np.cos(altitude) * np.cos(azimuth)
        y = depth * np.cos(altitude) * np.sin(azimuth)
        z = depth * np.sin(altitude)

        return np.array([x, y, z, 1.0])

    # -------------------------------------------------

    def project_to_image(self, point_radar_frame):
        """
        Transforms a radar-frame point into the camera frame and projects
        it to pixel coordinates using a pinhole model.

        Returns (u, v, depth_in_cam) or None if the point is behind the camera.
        """
        point_cam = self.T_radar_to_cam @ point_radar_frame

        X, Y, Z = point_cam[0], point_cam[1], point_cam[2]

        if Z <= 0.0:
            return None

        u = (self.fx * X / Z) + self.cx
        v = (self.fy * Y / Z) + self.cy

        return u, v, Z

    # -------------------------------------------------

    def associate_targets(self, boxes, scores, classes, radar_targets):
        """
        Projects each radar target into the image and matches it to the
        bounding box whose center it lands closest to (within a pixel
        tolerance). One radar target per box (closest wins).
        """

        detections = [
            {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(scores[i]),
                "class_id": int(classes[i]),
            }
            for i, (x1, y1, x2, y2) in enumerate(boxes)
        ]

        # Project all radar points once
        # projected = []

        # for blinking effect
        self.projected_points = []

        projected = self.projected_points

        for radar_target in radar_targets:
            point_radar_frame = self.spherical_to_cartesian(radar_target)
            proj = self.project_to_image(point_radar_frame)

            if proj is None:
                continue

            u, v, depth_cam = proj

            if 0 <= u < self.width and 0 <= v < self.height:
                projected.append((u, v, radar_target))

        associations = []

        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]

            best_target = None
            best_dist = self.max_assoc_distance_px

            for u, v, radar_target in projected:
                if not (x1 <= u <= x2 and y1 <= v <= y2):
                    continue

                # Prefer the point closest to the box center, in case
                # multiple radar returns land inside the same box.
                box_cx = (x1 + x2) / 2.0
                box_cy = (y1 + y2) / 2.0
                dist = np.hypot(u - box_cx, v - box_cy)

                if dist < best_dist:
                    best_dist = dist
                    best_target = radar_target

            if best_target is not None:
                associations.append((detection, best_target))

        return associations

    # -------------------------------------------------

    def compute_range(self, radar_target):
        return float(radar_target[0])

    def compute_bearing(self, radar_target):
        return float(np.degrees(radar_target[1]))

    def compute_radial_velocity(self, radar_target):
        return float(radar_target[3])

    def estimate_motion_state(self, velocity):
        if velocity < -self.velocity_threshold:
            return "Approaching"
        elif velocity > self.velocity_threshold:
            return "Receding"
        return "Stationary"

    # -------------------------------------------------

    def process(self, boxes, scores, classes, radar_targets):
        objects = []

        associations = self.associate_targets(boxes, scores, classes, radar_targets)

        for detection, radar_target in associations:
            obj = {
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
                "class_id": detection["class_id"],
                "range": self.compute_range(radar_target),
                "bearing": self.compute_bearing(radar_target),
                "velocity": self.compute_radial_velocity(radar_target),
            }
            obj["motion"] = self.estimate_motion_state(obj["velocity"])
            objects.append(obj)

        return objects