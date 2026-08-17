import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
THIRD_PARTY = os.path.join(ROOT, "third_party")

OPENPCDET_ROOT = os.path.join(THIRD_PARTY, "OpenPCDet")

if OPENPCDET_ROOT not in sys.path:
    sys.path.insert(0, OPENPCDET_ROOT)

import copy
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from inference_dataset import InferenceDataset

class PointPillarsDetector:

    def __init__(
        self,
        config_path,
        checkpoint_path,
        device="cuda",
        score_threshold=0.25
    ):

        self.score_threshold = score_threshold
        self.device = device

        self.logger = common_utils.create_logger()

        self.logger.info("----------------------------------------")
        self.logger.info("Loading PointPillars Detector...")
        self.logger.info("----------------------------------------")

        ###################################################
        # Configuration
        ###################################################

        self.cfg = copy.deepcopy(cfg)

        cfg_from_yaml_file(
            config_path,
            self.cfg
        )

        ###################################################
        # Dataset
        ###################################################

        self.dataset = InferenceDataset(
            dataset_cfg=self.cfg.DATA_CONFIG,
            class_names=self.cfg.CLASS_NAMES,
            logger=self.logger
        )

        ###################################################
        # Build Network
        ###################################################

        self.model = build_network(
            model_cfg=self.cfg.MODEL,
            num_class=len(self.cfg.CLASS_NAMES),
            dataset=self.dataset
        )

        ###################################################
        # Load Weights
        ###################################################

        self.model.load_params_from_file(
            filename=checkpoint_path,
            logger=self.logger,
            to_cpu=(device == "cpu")
        )

        ###################################################
        # Device
        ###################################################

        if device == "cuda":

            self.model.cuda()

        self.model.eval()

        self.logger.info("----------------------------------------")
        self.logger.info("PointPillars Ready")
        self.logger.info("----------------------------------------")

    def detect(
        self,
        points
    ):
        """
        Run PointPillars inference.

        Parameters
        ----------
        points : np.ndarray
            Shape (N, 4)
            [x, y, z, intensity]

        Returns
        -------
        dict
            {
                "boxes": ...,
                "scores": ...,
                "labels": ...
            }
        """

        batch_dict = self.preprocess(points)

        pred_dicts = self.infer(batch_dict)

        detections = self.postprocess(pred_dicts)

        return detections

    def preprocess(
        self,
        points
    ):
        """
        Convert raw LiDAR points into an OpenPCDet batch.

        Parameters
        ----------
        points : np.ndarray
            Shape (N, 4)
            [x, y, z, intensity]

        Returns
        -------
        dict
            OpenPCDet batch dictionary.
        """

        data_dict = self.dataset.prepare_pointcloud(points)

        batch_dict = self.dataset.collate_batch(
            [data_dict]
        )

        load_data_to_gpu(
            batch_dict
        )

        return batch_dict


    @torch.no_grad()
    def infer(
        self,
        batch_dict
    ):
        """
        Run PointPillars forward inference.

        Parameters
        ----------
        batch_dict : dict
            Preprocessed OpenPCDet batch.

        Returns
        -------
        list
            Prediction dictionaries.
        """

        pred_dicts, _ = self.model(batch_dict)

        return pred_dicts

        
    def postprocess(self, pred_dicts):

        pred = pred_dicts[0]

        boxes = pred["pred_boxes"].cpu().numpy()
        scores = pred["pred_scores"].cpu().numpy()
        labels = pred["pred_labels"].cpu().numpy()

        label_map = {
            1: "Car",
            2: "Pedestrian",
            3: "Cyclist"
        }

        detections = []

        for box, score, label in zip(boxes, scores, labels):

            if float(score) < self.score_threshold:
                continue

            detections.append({

                "center": box[:3],

                "dimensions": (
                    box[3],   # length
                    box[4],   # width
                    box[5]    # height
                ),

                "yaw": box[6],

                "score": float(score),

                "class": label_map.get(int(label), "Unknown")
            })
        return detections