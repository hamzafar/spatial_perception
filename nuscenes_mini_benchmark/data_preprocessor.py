
from pathlib import Path
import numpy as np
from nuscenes.nuscenes import NuScenes

class NuScenesDataPreprocessor:
    REQUIRED_KEYS = {
        "sample_index",
        "sample_token",
        "boxes_3d",
        "scores_3d",
        "labels_3d",
    }

    def __init__(self, prediction_dir):
        self.prediction_dir = Path(prediction_dir)

    def load_predictions(self):
        """Load and validate all BEVFusion prediction files."""
        files = sorted(self.prediction_dir.rglob("*.npz"))

        if not files:
            raise FileNotFoundError(
                f"No .npz files found in: {self.prediction_dir}"
            )

        predictions = {}

        for path in files:
            with np.load(path, allow_pickle=False) as data:
                missing = self.REQUIRED_KEYS - set(data.files)
                if missing:
                    raise ValueError(
                        f"{path.name}: missing keys {sorted(missing)}"
                    )

                sample_index = int(data["sample_index"].item())
                sample_token = str(data["sample_token"].item())
                boxes = data["boxes_3d"].copy()
                scores = data["scores_3d"].copy()
                labels = data["labels_3d"].copy()

            # Validate dimensions
            if boxes.ndim != 2 or boxes.shape[1] != 9:
                raise ValueError(
                    f"{path.name}: expected boxes shape (N, 9), "
                    f"got {boxes.shape}"
                )

            n = len(boxes)
            if scores.shape != (n,) or labels.shape != (n,):
                raise ValueError(
                    f"{path.name}: inconsistent array shapes"
                )

            if sample_token in predictions:
                raise ValueError(
                    f"Duplicate sample token: {sample_token}"
                )

            if not (
                np.isfinite(boxes).all()
                and np.isfinite(scores).all()
            ):
                raise ValueError(
                    f"Non-finite values in {path.name}"
                )

            predictions[sample_token] = {
                "sample_index": sample_index,
                "boxes_3d": boxes,
                "scores_3d": scores,
                "labels_3d": labels,
                "filename": path.name,
            }

        print(f"Loaded prediction files: {len(predictions)}")
        print(
            "Total detections:",
            sum(len(p["boxes_3d"]) for p in predictions.values())
        )

        return predictions
    

    def load_nuscenes(self, dataroot, version="v1.0-mini"):
        """Load the nuScenes-mini dataset."""

        nusc = NuScenes(
            version=version,
            dataroot=str(dataroot),
            verbose=False,
        )

        print(f"Loaded nuScenes version: {version}")
        print(f"Scenes: {len(nusc.scene)}")
        print(f"Samples: {len(nusc.sample)}")

        return nusc


    def organize_scenes(self, nusc, predictions):
        """Organize all samples chronologically within each scene."""

        scenes = {}

        for scene in nusc.scene:
            scene_name = scene["name"]
            frames = []
            sample_token = scene["first_sample_token"]

            while sample_token:
                sample = nusc.get("sample", sample_token)

                frames.append({
                    "sample_token": sample_token,
                    "timestamp": sample["timestamp"],
                    "ground_truth_annotations": sample["anns"],
                    "prediction": predictions.get(sample_token),
                })

                sample_token = sample["next"]

            frames.sort(key=lambda frame: frame["timestamp"])
            scenes[scene_name] = frames

        total_frames = sum(len(frames) for frames in scenes.values())
        missing = sum(
            frame["prediction"] is None
            for frames in scenes.values()
            for frame in frames
        )

        print(f"Scenes organized: {len(scenes)}")
        print(f"Total frames: {total_frames}")
        print(f"Frames missing predictions: {missing}")

        return scenes


