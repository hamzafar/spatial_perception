import numpy as np
from easydict import EasyDict as edict

import os
import sys

ROOT = os.path.dirname(__file__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

THIRD_PARTY_ROOT = os.path.join(ROOT, "third_party")

AB3DMOT_ROOT = os.path.join(THIRD_PARTY_ROOT, "AB3DMOT")
XINSHUO_ROOT = os.path.join(THIRD_PARTY_ROOT, "Xinshuo_PyToolbox")


for path in [AB3DMOT_ROOT, XINSHUO_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from AB3DMOT_libs.model import AB3DMOT


class AB3DMOTTracker:

    def __init__(self):

        ROOT = os.path.dirname(__file__)

        log_path = os.path.join(ROOT, "ab3dmot.log")
        self.log = open(log_path, "w")

        self.frame = 0
        self.sequence_name = "carla"

        #
        # Tracker configuration
        #

        self.cfg = edict()

        self.cfg.dataset = "KITTI"
        self.cfg.det_name = "pointrcnn"

        self.cfg.vis = False
        self.cfg.ego_com = False
        self.cfg.affi_pro = False

        self.cfg.num_hypo = 1
        self.cfg.score_threshold = 0.0

        #
        # Create tracker
        #

        self.tracker = AB3DMOT(
            cfg=self.cfg,
            cat="Car",
            calib=None,
            oxts=None,
            img_dir=None,
            vis_dir=None,
            hw=None,
            log=self.log,
            ID_init=0
        )

    def reset(self):

        self.frame = 0

        self.tracker = AB3DMOT(
            cfg=self.cfg,
            cat="Car",
            calib=None,
            oxts=None,
            img_dir=None,
            vis_dir=None,
            hw=None,
            log=self.log,
            ID_init=0
        )

    def convert_detections(
        self,
        detections
    ):

        dets = []
        info = []

        for detection in detections:

            center = detection["center"]

            length, width, height = detection["dimensions"]

            yaw = detection["yaw"]

            score = detection["score"]

            class_name = detection["class"]

            dets.append([
                height,
                width,
                length,
                center[0],
                center[1],
                center[2],
                yaw
            ])

            info.append([
                score
            ])

        if len(dets) == 0:

            dets = np.empty((0, 7), dtype=np.float32)
            info = np.empty((0, 1), dtype=np.float32)

        else:

            dets = np.asarray(
                dets,
                dtype=np.float32
            )

            info = np.asarray(
                info,
                dtype=np.float32
            )

        return {
            "dets": dets,
            "info": info
        }

    def parse_tracks(
        self,
        results,
        detections
    ):

        tracks = []

        if len(results) == 0:
            return tracks

        if len(results[0]) == 0:
            return tracks

        for row in results[0]:

            h = row[0]
            w = row[1]
            l = row[2]

            x = row[3]
            y = row[4]
            z = row[5]

            yaw = row[6]

            track_id = int(row[7])

            score = float(row[8])

            tracks.append({

                "track_id": track_id,

                "class": "car",

                "score": score,

                "center": np.array(
                    [x, y, z],
                    dtype=np.float32
                ),

                "dimensions": (
                    l,
                    w,
                    h
                ),

                "yaw": yaw
            })

        return tracks

    def update(
        self,
        detections
    ):

        dets_all = self.convert_detections(
            detections
        )

        results, _ = self.tracker.track(
            dets_all,
            self.frame,
            self.sequence_name
        )

        self.frame += 1

        tracks = self.parse_tracks(
            results,
            detections
        )

        return tracks