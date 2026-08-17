from pathlib import Path

import numpy as np

from pcdet.datasets import DatasetTemplate

class InferenceDataset(DatasetTemplate):
    """
    Lightweight dataset used only for inference.

    It accepts a NumPy point cloud directly instead of reading
    KITTI .bin files from disk.
    """

    def __init__(
        self,
        dataset_cfg,
        class_names,
        logger
    ):

        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=Path("."),
            logger=logger
        )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise NotImplementedError(
            "InferenceDataset does not support indexing."
        )

    def prepare_pointcloud(
        self,
        points
    ):
        """
        Parameters
        ----------
        points : numpy.ndarray
            Shape (N,4)
            [x,y,z,intensity]
        """

        input_dict = {
            "points": points.astype(np.float32),
            "frame_id": 0
        }

        return self.prepare_data(input_dict)
