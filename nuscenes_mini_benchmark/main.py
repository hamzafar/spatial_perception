
from pathlib import Path
from datetime import datetime
import logging

from nuscenes.utils.splits import create_splits_scenes

from data_preprocessor import NuScenesDataPreprocessor
from ab3dmot_nuscenes_adapter import NuScenesAB3DMOTAdapter


def main():
    # Dataset and prediction paths
    prediction_dir = Path(
        "/mnt/e/autonomous_vision/code_backup/scripts/"
        "learned-bev/bevfusion_lightweight_predictions"
    )

    nuscenes_root = Path(
        "/mnt/e/autonomous_vision/learned-perception/"
        "nuScenes-mini-dataset"
    )
    nuscenes_version = "v1.0-mini"

    # Output paths
    output_dir = Path("outputs") / "mini_val"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"ab3dmot_tracks_mini_val_{run_id}.json"
    log_path = output_dir / f"ab3dmot_mini_val_{run_id}.log"
    main_log_path = output_dir / f"main_mini_val_{run_id}.log"

    # Log both this script's messages and console output.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(main_log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger("mini_val")

    logger.info("Starting AB3DMOT mini_val run")
    logger.info("Dataset root: %s", nuscenes_root)
    logger.info("Prediction directory: %s", prediction_dir)

    # Load predictions and nuScenes
    preprocessor = NuScenesDataPreprocessor(prediction_dir)
    predictions = preprocessor.load_predictions()

    nusc = preprocessor.load_nuscenes(
        dataroot=str(nuscenes_root),
        version=nuscenes_version,
    )

    scenes = preprocessor.organize_scenes(nusc, predictions)
    logger.info("Loaded prediction samples: %d", len(predictions))
    logger.info("Scenes available: %s", sorted(scenes.keys()))

    # Resolve the official mini_val scene names.
    mini_val_names = set(create_splits_scenes()["mini_val"])
    mini_val_scenes = {
        name: scenes[name]
        for name in sorted(mini_val_names)
        if name in scenes
    }

    missing_scenes = sorted(mini_val_names - set(mini_val_scenes))
    if missing_scenes:
        raise RuntimeError(
            f"Official mini_val scenes missing from organized data: "
            f"{missing_scenes}"
        )

    if not mini_val_scenes:
        raise RuntimeError("No mini_val scenes found.")

    logger.info("Selected mini_val scenes: %s", list(mini_val_scenes))

    for scene_name, frames in mini_val_scenes.items():
        logger.info("Scene %s: %d frames", scene_name, len(frames))

    # Run AB3DMOT only on mini_val.
    adapter = NuScenesAB3DMOTAdapter(nusc, predictions)

    tracks = adapter.run(
        mini_val_scenes,
        output_path=str(json_path),
        log_path=str(log_path),
    )

    logger.info("Tracking finished.")
    logger.info("Scenes processed: %s", list(mini_val_scenes))
    logger.info("Total frames: %d", sum(
        len(frames) for frames in mini_val_scenes.values()
    ))
    logger.info("Sample tokens with track results: %d", len(tracks))
    logger.info("Track JSON: %s", json_path.resolve())
    logger.info("AB3DMOT log: %s", log_path.resolve())
    logger.info("Main log: %s", main_log_path.resolve())

    print("\n=== MINI_VAL COMPLETE ===")
    print(f"Scenes:      {list(mini_val_scenes)}")
    print(f"Track JSON:  {json_path.resolve()}")
    print(f"Tracker log: {log_path.resolve()}")
    print(f"Main log:    {main_log_path.resolve()}")


if __name__ == "__main__":
    main()
