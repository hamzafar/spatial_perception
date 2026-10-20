
from pathlib import Path
from data_preprocessor import NuScenesDataPreprocessor
from ab3dmot_nuscenes_adapter import NuScenesAB3DMOTAdapter


def main():
    # Paths
    prediction_dir = Path(
        "/mnt/e/autonomous_vision/code_backup/scripts/"
        "learned-bev/bevfusion_lightweight_predictions"
    )

    nuscenes_root = Path(
        "/mnt/e/autonomous_vision/learned-perception/"
        "nuScenes-mini-dataset"
    )

    nuscenes_version = "v1.0-mini"

    # Load predictions
    preprocessor = NuScenesDataPreprocessor(prediction_dir)
    predictions = preprocessor.load_predictions()

    nusc = preprocessor.load_nuscenes(
        dataroot="/mnt/e/autonomous_vision/learned-perception/nuScenes-mini-dataset",
        version="v1.0-mini",
        )

    scenes = preprocessor.organize_scenes(nusc, predictions)

    adapter = NuScenesAB3DMOTAdapter(nusc, predictions)
    # tracks = adapter.run(scenes)


    first_scene = next(iter(scenes))
    pilot_scenes = {first_scene: scenes[first_scene]}

    print(f"Pilot scene: {first_scene}")
    print(f"Pilot frames: {len(pilot_scenes[first_scene])}")

    tracks = adapter.run(pilot_scenes)

    # first_scene = next(iter(scenes))
    # first_frame = scenes[first_scene][0]

    # print("First scene:", first_scene)
    # print("First frame timestamp:", first_frame["timestamp"])
    # print("Ground-truth annotations:",
    #     len(first_frame["ground_truth_annotations"]))
    # print("Predictions available:",
    #     first_frame["prediction"] is not None)


    print(f"Loaded {len(predictions)} prediction samples.")

    # Dataset paths are ready for the next preprocessing step
    print("nuScenes root:", nuscenes_root)
    print("nuScenes version:", nuscenes_version)


if __name__ == "__main__":
    main()
