import numpy as np


class PerceptionUtils:
    def __init__(self):
        pass

    def normalize_bev_class(self, cls):

        if cls == "person":
            return "person"

        if cls in ["car"]:
            return "vehicle"

        if cls in ["truck", "bus"]:
            return "truck"

        if cls in ["bicycle", "motorcycle"]:
            return "cyclist"

        return None

    def attach_track_ids(
        self,
        world_objects,
        online_targets,
        camera_prefix
    ):
        if not online_targets:
            return world_objects

        track_ids = np.array(
            [int(t.track_id) for t in online_targets]
        )

        tlwh = np.array(
            [t.tlwh for t in online_targets],
            dtype=np.float64
        )

        tx1 = tlwh[:, 0]
        ty1 = tlwh[:, 1]
        tw = tlwh[:, 2]
        th = tlwh[:, 3]

        trk_x1 = tx1
        trk_y1 = ty1
        trk_x2 = tx1 + tw
        trk_y2 = ty1 + th

        trk_areas = tw * th

        for obj in world_objects:

            ox1, oy1, ox2, oy2 = map(
                float,
                obj["box"]
            )

            obj_area = (
                (ox2 - ox1) *
                (oy2 - oy1)
            )

            ix1 = np.maximum(ox1, trk_x1)
            iy1 = np.maximum(oy1, trk_y1)
            ix2 = np.minimum(ox2, trk_x2)
            iy2 = np.minimum(oy2, trk_y2)

            inter = (
                np.maximum(0.0, ix2 - ix1) *
                np.maximum(0.0, iy2 - iy1)
            )

            union = (
                obj_area +
                trk_areas -
                inter
            )

            iou = np.where(
                union > 0,
                inter / union,
                0.0
            )

            best_idx = int(np.argmax(iou))

            if iou[best_idx] >= 0.3:
                obj["id"] = (
                    f"{camera_prefix}_{track_ids[best_idx]}"
                )

        return world_objects

    def prepare_bev_objects(self, world_objects):

        bev_objects = []

        for obj in world_objects:

            bev_class = self.normalize_bev_class(
                obj["class"]
            )

            if bev_class is None:
                continue

            x, y, _ = obj["position"]

            bev_objects.append({
                "id": obj["id"],
                "cls": bev_class,
                "x": float(x),
                "y": float(y),
                "distance": float(obj["distance"])
            })

        return bev_objects

    def attach_radar_data(
        self,
        world_objects,
        radar_objects,
        iou_threshold=0.3
    ):

        if not world_objects or not radar_objects:
            return world_objects

        radar_boxes = np.asarray(
            [r["bbox"] for r in radar_objects],
            dtype=np.float64
        )

        rx1 = radar_boxes[:, 0]
        ry1 = radar_boxes[:, 1]
        rx2 = radar_boxes[:, 2]
        ry2 = radar_boxes[:, 3]

        radar_areas = (
            (rx2 - rx1) *
            (ry2 - ry1)
        )

        for obj in world_objects:

            ox1, oy1, ox2, oy2 = np.asarray(
                obj["box"],
                dtype=np.float64
            )

            obj_area = (
                (ox2 - ox1) *
                (oy2 - oy1)
            )

            ix1 = np.maximum(ox1, rx1)
            iy1 = np.maximum(oy1, ry1)
            ix2 = np.minimum(ox2, rx2)
            iy2 = np.minimum(oy2, ry2)

            inter = (
                np.maximum(0.0, ix2 - ix1) *
                np.maximum(0.0, iy2 - iy1)
            )

            union = (
                obj_area +
                radar_areas -
                inter
            )

            iou = np.where(
                union > 0,
                inter / union,
                0.0
            )

            best_idx = int(np.argmax(iou))
            best_iou = float(iou[best_idx])

            if (
                best_iou > 0.0
                and best_iou >= iou_threshold
            ):

                radar = radar_objects[best_idx]

                obj["radar"] = {
                    "range": float(radar["range"]),
                    "bearing": float(radar["bearing"]),
                    "velocity": float(radar["velocity"]),
                    "motion": radar["motion"],
                }

        return world_objects

    def prepare_nearest_objects(self, world_objects, timestamp):

        nearest_objects = []

        labels = {
            "vehicle": "Car",
            "person": "Pedestrian",
            "truck": "Truck",
            "cyclist": "Cyclist",
        }

        for obj in world_objects:

            if "id" not in obj:
                continue

            cls = self.normalize_bev_class(obj["class"])

            if cls is None:
                continue

            radar = obj.get("radar")

            if radar:
                speed_mps = round(
                    abs(float(radar["velocity"])), 3
                )
                motion = radar["motion"].lower()
            else:
                speed_mps, motion = self.estimate_track_motion(obj, timestamp)

            nearest_objects.append({
                "id": obj["id"],
                "cls": cls,
                "label": labels[cls],
                "dist_m": round(float(obj["distance"]), 3),
                "speed_mps": speed_mps,
                "motion": motion,
            })

        nearest_objects.sort(
            key=lambda obj: obj["dist_m"]
        )

        return nearest_objects

    def estimate_track_speed(self, obj, timestamp):
        if not hasattr(self, "_track_history"):
            self._track_history = {}

        track_id = obj["id"]
        position = np.asarray(obj["position"], dtype=np.float64)

        previous = self._track_history.get(track_id)

        # Store current observation
        self._track_history[track_id] = (
            position.copy(),
            float(timestamp)
        )

        if previous is None:
            return None

        previous_position, previous_timestamp = previous

        dt = float(timestamp) - previous_timestamp

        if dt <= 0:
            return None

        displacement = np.linalg.norm(
            position[:2] - previous_position[:2]
        )

        speed_mps = displacement / dt

        return round(float(speed_mps), 3)

    def estimate_track_motion(self, obj, timestamp):
        if not hasattr(self, "_track_history"):
            self._track_history = {}

        track_id = obj["id"]
        position = np.asarray(obj["position"], dtype=np.float64)

        previous = self._track_history.get(track_id)

        self._track_history[track_id] = (
            position.copy(),
            float(timestamp)
        )

        if previous is None:
            return None, "unknown"

        previous_position, previous_timestamp = previous

        dt = float(timestamp) - previous_timestamp

        if dt <= 0:
            return None, "unknown"

        # Ego-frame X: forward/backward motion
        velocity_x = (
            position[0] - previous_position[0]
        ) / dt

        speed_mps = abs(velocity_x)

        threshold = 0.2

        if velocity_x < -threshold:
            motion = "approaching"
        elif velocity_x > threshold:
            motion = "receding"
        else:
            motion = "stationary"

        return round(float(speed_mps), 3), motion

    def count_objects(self, world_objects):
        counts = {
            "vehicle": 0,
            "person": 0,
            "cyclist": 0,
        }

        for obj in world_objects:
            cls = self.normalize_bev_class(obj["class"])

            if cls in ["vehicle", "truck"]:
                counts["vehicle"] += 1

            elif cls == "person":
                counts["person"] += 1

            elif cls == "cyclist":
                counts["cyclist"] += 1

        return counts