
from data_preprocessor import NuScenesDataPreprocessor


def main():
    pred_dir = (
        "/mnt/e/autonomous_vision/code_backup/scripts/"
        "learned-bev/bevfusion_lightweight_predictions"
    )

    # Initialize preprocessor
    preprocessor = NuScenesDataPreprocessor(pred_dir)

    # Load all predictions
    predictions = preprocessor.load_predictions()

    # Access an example sample
    token = next(iter(predictions))
    sample = predictions[token]

    print("Sample token:", token)
    print("Sample index:", sample["sample_index"])
    print("Boxes shape:", sample["boxes_3d"].shape)
    print("Scores shape:", sample["scores_3d"].shape)
    print("Labels shape:", sample["labels_3d"].shape)


if __name__ == "__main__":
    main()
