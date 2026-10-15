
from pathlib import Path
import numpy as np

# Prediction directory in WSL
PRED_DIR = Path("/mnt/e/autonomous_vision/code_backup/scripts/learned-bev/bevfusion_lightweight_predictions")

REQUIRED_KEYS = {
    "sample_index",
    "sample_token",
    "boxes_3d",
    "scores_3d",
    "labels_3d",
}


def load_predictions(pred_dir):
    pred_dir = Path(pred_dir)
    files = sorted(pred_dir.rglob("*.npz"))

    if not files:
        raise FileNotFoundError(
            f"No .npz files found in: {pred_dir}"
        )

    predictions = {}

    for path in files:
        with np.load(path, allow_pickle=False) as data:
            missing = REQUIRED_KEYS - set(data.files)
            if missing:
                raise ValueError(
                    f"{path.name}: missing keys {sorted(missing)}"
                )

            sample_index = int(data["sample_index"].item())
            sample_token = str(data["sample_token"].item())
            boxes = data["boxes_3d"].copy()
            scores = data["scores_3d"].copy()
            labels = data["labels_3d"].copy()

        # Validate array shapes
        if boxes.ndim != 2 or boxes.shape[1] != 9:
            raise ValueError(
                f"{path.name}: expected boxes shape (N, 9), "
                f"got {boxes.shape}"
            )

        n = len(boxes)
        if scores.shape != (n,) or labels.shape != (n,):
            raise ValueError(
                f"{path.name}: inconsistent boxes/scores/labels"
            )

        if sample_token in predictions:
            raise ValueError(
                f"Duplicate sample token: {sample_token}"
            )

        if not (
            np.isfinite(boxes).all()
            and np.isfinite(scores).all()
        ):
            raise ValueError(f"Non-finite values in {path.name}")

        predictions[sample_token] = {
            "sample_index": sample_index,
            "boxes_3d": boxes,
            "scores_3d": scores,
            "labels_3d": labels,
            "filename": path.name,
        }

    return predictions


if __name__ == "__main__":
    predictions = load_predictions(PRED_DIR)

    print(f"Prediction files loaded: {len(predictions)}")
    print(f"Unique sample tokens: {len(predictions)}")

    total_detections = sum(
        len(p["boxes_3d"]) for p in predictions.values()
    )
    print(f"Total detections: {total_detections}")

    # Display one example
    token, sample = next(iter(predictions.items()))
    print("\nExample sample:")
    print("Token:", token)
    print("Index:", sample["sample_index"])
    print("Boxes:", sample["boxes_3d"].shape)
    print("Scores:", sample["scores_3d"].shape)
    print("Labels:", sample["labels_3d"].shape)

    assert len(predictions) == 404, (
        f"Expected 404 unique samples, got {len(predictions)}"
    )

    print("\nPASS: all prediction files loaded and validated.")
