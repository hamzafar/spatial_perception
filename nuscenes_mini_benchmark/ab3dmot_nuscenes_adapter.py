import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from easydict import EasyDict
from pyquaternion import Quaternion

THIRD_PARTY = Path.home() / "ros2_cv_ws/scripts/third_party"
AB3DMOT_ROOT = THIRD_PARTY / "AB3DMOT"
XINSHUO_ROOT = THIRD_PARTY / "Xinshuo_PyToolbox"

for path in (AB3DMOT_ROOT, XINSHUO_ROOT):
    sys.path.insert(0, str(path))

from AB3DMOT_libs.model import AB3DMOT


CLASS_NAMES = {
    0: "car", 1: "truck", 2: "construction_vehicle", 3: "bus",
    4: "trailer", 5: "barrier", 6: "motorcycle", 7: "bicycle",
    8: "pedestrian", 9: "traffic_cone",
}

SUPPORTED_CLASSES = {
    0: "Car", 1: "Truck", 3: "Bus", 4: "Trailer",
    6: "Motorcycle", 7: "Bicycle", 8: "Pedestrian",
}

# Detection score cutoffs applied BEFORE tracking. Starting values: tune them
# by checking per-frame counts and unique IDs per class in the printed summary.
DEFAULT_SCORE_THRESHOLDS = {
    0: 0.30,  # car
    1: 0.30,  # truck
    3: 0.30,  # bus
    4: 0.30,  # trailer
    6: 0.30,  # motorcycle
    7: 0.30,  # bicycle
    8: 0.50,  # pedestrian
}


def make_tracker(category, log_handle):
    cfg = EasyDict(
        dataset="nuScenes", det_name="centerpoint", vis=False,
        ego_com=False, affi_pro=False, num_hypo=1, score_threshold=0.0,
    )
    return AB3DMOT(
        cfg=cfg, cat=category, calib=None, oxts=None,
        img_dir=None, vis_dir=None, hw=None, log=log_handle, ID_init=0,
    )


def prediction_to_global(nusc, sample_token, box):
    """Convert a BEVFusion LiDAR box to global geometric center, size and yaw."""
    sample = nusc.get("sample", sample_token)
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])

    x, y, z_bottom, dx, dy, dz, yaw = map(float, box[:7])
    # BEVFusion z is the bottom height; nuScenes translation uses box center.
    center_lidar = np.array([x, y, z_bottom + dz / 2.0], dtype=np.float64)

    q_sensor = Quaternion(cs["rotation"])
    q_pose = Quaternion(pose["rotation"])
    center_ego = q_sensor.rotate(center_lidar) + np.asarray(cs["translation"])
    center_global = q_pose.rotate(center_ego) + np.asarray(pose["translation"])

    q_box = Quaternion(axis=[0, 0, 1], angle=yaw)
    q_global = q_pose * q_sensor * q_box
    yaw_global = float(q_global.yaw_pitch_roll[0])
    return center_global, (dx, dy, dz), yaw_global


def global_to_tracker(center, dimensions, yaw):
    """
    Convert a global z-up box to AB3DMOT's camera-like representation.
    AB3DMOT expects [h, w, l, x, y, z, yaw], with y at the bottom center
    and positive down. `center` is the geometric center in global coordinates.
    """
    gx, gy, gz = map(float, center)
    dx, dy, dz = map(float, dimensions)
    bottom_z = gz - dz / 2.0
    return [dz, dy, dx, gx, -bottom_z, gy, -float(yaw)]


def tracker_to_global(row):
    """Convert an AB3DMOT row back to a global geometric-center box."""
    h, w, length, x, y, z, yaw, track_id = map(float, row[:8])
    return {
        "track_id": int(track_id),
        # Tracker y is down-positive bottom-center; restore global center z.
        "translation": [x, z, -y + h / 2.0],
        "size": [length, w, h],
        "yaw": -yaw,
    }


class NuScenesAB3DMOTAdapter:
    def __init__(self, nusc, predictions, score_thresholds=None):
        self.nusc = nusc
        self.predictions = predictions
        self.score_thresholds = dict(DEFAULT_SCORE_THRESHOLDS)
        if score_thresholds:
            self.score_thresholds.update(score_thresholds)

    def run(
        self,
        scenes,
        output_path="ab3dmot_tracks_global.json",
        log_path="ab3dmot_nuscenes.log",
    ):
        all_results = {}
        processed_frames = 0
        dets_before = defaultdict(int)   # class_id -> detections before filtering
        dets_after = defaultdict(int)    # class_id -> detections fed to tracker
        unique_ids = defaultdict(set)    # (scene, class_id) -> track ids

        with open(log_path, "w", encoding="utf-8") as log:
            for scene_name, frames in scenes.items():
                # New tracker bank for every scene; IDs reset between scenes.
                trackers = {
                    class_id: make_tracker(category, log)
                    for class_id, category in SUPPORTED_CLASSES.items()
                }

                for frame_index, frame in enumerate(frames):
                    token = frame["sample_token"]
                    pred = self.predictions.get(token)

                    if pred is None:
                        boxes = np.empty((0, 9), dtype=np.float32)
                        scores = np.empty((0,), dtype=np.float32)
                        labels = np.empty((0,), dtype=np.int32)
                    else:
                        boxes = pred["boxes_3d"]
                        scores = pred["scores_3d"]
                        labels = pred["labels_3d"].astype(np.int32)

                    frame_tracks = []
                    for class_id, tracker in trackers.items():
                        thr = self.score_thresholds.get(class_id, 0.0)
                        class_mask = labels == class_id
                        dets_before[class_id] += int(class_mask.sum())
                        # Score filter BEFORE tracking.
                        selected = np.flatnonzero(class_mask & (scores >= thr))
                        dets_after[class_id] += len(selected)
                        det_rows, det_scores = [], []

                        for idx in selected:
                            center, dims, yaw = prediction_to_global(
                                self.nusc, token, boxes[idx]
                            )
                            det_rows.append(global_to_tracker(center, dims, yaw))
                            det_scores.append(float(scores[idx]))

                        dets = (
                            np.asarray(det_rows, dtype=np.float32).reshape(-1, 7)
                            if det_rows else np.empty((0, 7), dtype=np.float32)
                        )
                        info = np.asarray(det_scores, dtype=np.float32).reshape(-1, 1)

                        # Call once per frame, including frames with no detections.
                        results, _ = tracker.track(
                            {"dets": dets, "info": info}, frame_index, scene_name
                        )

                        for row in results[0]:
                            record = tracker_to_global(row)
                            record.update({
                                "sample_token": token,
                                "scene_name": scene_name,
                                "frame_index": frame_index,
                                "class_id": class_id,
                                "class_name": CLASS_NAMES[class_id],
                            })
                            frame_tracks.append(record)
                            unique_ids[(scene_name, class_id)].add(record["track_id"])

                    all_results[token] = frame_tracks
                    processed_frames += 1

                print(f"Processed {scene_name}: {len(frames)} frames")

        output_path = Path(output_path)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "dataset": "nuScenes-mini",
                    "tracker": "official AB3DMOT",
                    "parameter_profile": "nuScenes/centerpoint",
                    "coordinate_frame": "global",
                    "frames": processed_frames,
                    "supported_class_ids": sorted(SUPPORTED_CLASSES),
                    "excluded_class_ids": sorted(
                        set(CLASS_NAMES) - set(SUPPORTED_CLASSES)
                    ),
                    "score_thresholds": {
                        CLASS_NAMES[c]: t for c, t in self.score_thresholds.items()
                        if c in SUPPORTED_CLASSES
                    },
                },
                "results": all_results,
            }, f)

        total_tracks = sum(len(v) for v in all_results.values())
        print(f"Frames processed: {processed_frames}")
        print(f"Output track records: {total_tracks}")
        print("Excluded class IDs: [2, 5, 9]")
        print("\nPer-class summary (detections kept / total, unique track IDs):")
        for class_id in SUPPORTED_CLASSES:
            ids = sum(len(v) for (s, c), v in unique_ids.items() if c == class_id)
            print(
                f"  {CLASS_NAMES[class_id]:<11} thr={self.score_thresholds.get(class_id, 0.0):.2f}  "
                f"dets {dets_after[class_id]}/{dets_before[class_id]}  "
                f"unique IDs {ids}"
            )
        print(f"\nSaved: {output_path.resolve()}")
        print(f"Tracker log: {Path(log_path).resolve()}")
        return all_results