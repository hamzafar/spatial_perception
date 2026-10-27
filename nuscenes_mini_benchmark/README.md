# nuScenes-Mini Perception Benchmark

A benchmarking and evaluation workspace for analyzing **BEVFusion 3D detection predictions on nuScenes-mini**, with temporal tracking using **AB3DMOT** and evaluation using **HOTA** and **MOTA**.

BEVFusion inference is performed separately in the companion **Learned Perception** repository using Kaggle notebooks. This repository consumes the exported predictions and evaluates their tracking performance locally.

## Table of Contents

- [Objectives](#objectives)
- [Pipeline](#pipeline)
- [Repository Structure](#repository-structure)
- [Current Implementation](#current-implementation)
- [Evaluation Results](#evaluation-results)
- [Running the Benchmark](#running-the-benchmark)
- [Planned Detection Error Analysis](#planned-detection-error-analysis)
- [Related Project](#related-project)
- [Scope and Limitations](#scope-and-limitations)

## Objectives

- Load and validate exported BEVFusion predictions.
- Prepare nuScenes-mini samples and organize them chronologically by scene.
- Generate temporally associated 3D tracks using AB3DMOT.
- Evaluate tracking performance using HOTA and MOTA.
- Analyze results by object class.
- Extend the benchmark with detection metrics and systematic error analysis.

## Pipeline

```text
BEVFusion Inference
(Kaggle — Learned Perception)
          |
          v
Exported 3D Predictions (.npz)
          |
          v
Prediction Loading and Validation
          |
          v
nuScenes-Mini Scene Organization
          |
          v
3D Coordinate Transformation
(LiDAR -> Ego -> Global)
          |
          v
AB3DMOT Tracking
          |
          v
Global Track Predictions (.json)
          |
          v
HOTA / MOTA Evaluation
          |
          v
Per-Class Metrics and Error Analysis
```

## Repository Structure

```text
nuscenes_mini_benchmark/
├── main.py
├── data_preprocessor.py
├── ab3dmot_nuscenes_adapter.py
├── evaluate_hota_mota_nuscenes.py
├── evaluate_tracks.py
├── track_metrics.py
├── test_ab3dmot_import.py
├── hota_mota_mini_val.json
├── outputs/
│   └── mini_val/
└── error_analysis/
    ├── 01_map_evaluation.ipynb
    ├── 02_nds_evaluation.ipynb
    ├── 03_detection_error_analysis.ipynb
    └── 04_tracking_error_analysis.ipynb
```

> The `error_analysis/` notebooks are planned additions; create them as the corresponding analyses are implemented.

## Current Implementation

### 1. Prediction preprocessing

**`data_preprocessor.py`**

- Loads exported `.npz` prediction files.
- Validates required arrays and their dimensions.
- Checks for duplicate sample tokens and non-finite values.
- Loads nuScenes-mini and organizes samples by scene and timestamp.

Expected prediction arrays:

| Field | Description |
|---|---|
| `sample_index` | Sample index |
| `sample_token` | nuScenes sample identifier |
| `boxes_3d` | 3D predicted boxes |
| `scores_3d` | Detection confidence scores |
| `labels_3d` | Predicted class IDs |

The current loader expects `boxes_3d` to have shape `(N, 9)`.

### 2. 3D tracking

**`ab3dmot_nuscenes_adapter.py`**

- Converts BEVFusion LiDAR-frame box predictions into global coordinates.
- Applies configurable, class-specific confidence thresholds before tracking.
- Runs AB3DMOT independently for each supported object class.
- Resets trackers between scenes.
- Exports track IDs, classes, global translations, dimensions, yaw, and frame metadata.

Supported classes: car, truck, bus, trailer, motorcycle, bicycle, and pedestrian. Construction vehicles, barriers, and traffic cones are excluded from tracking.

### 3. Tracking evaluation

**`evaluate_tracks.py`** and **`track_metrics.py`**

The evaluation reports:

| Metric | Description |
|---|---|
| **HOTA** | Combined detection and association performance |
| **DetA** | Detection accuracy |
| **AssA** | Association accuracy |
| **LocA** | Localization accuracy under the implemented similarity definition |
| **MOTA** | Tracking accuracy based on false negatives, false positives, and identity switches |
| **MOTP** | Mean matched-object center distance, in meters |
| **TP, FP, FN, IDSW** | Diagnostic tracking counts |

The evaluator supports per-class results and a class-averaged summary. It also implements nuScenes-style class mapping, range filtering, and ground-truth filtering based on LiDAR/radar point availability.

> **Metric interpretation:** HOTA here uses distance-based similarity across multiple thresholds. MOTA uses center-distance matching with a configurable threshold. These results should **not** be described as official nuScenes tracking benchmark scores.

**`evaluate_hota_mota_nuscenes.py`** is an additional, simpler baseline evaluator using a fixed center-distance threshold. Its HOTA variant is not interchangeable with the more complete implementation in `evaluate_tracks.py`.

## Evaluation Results

The saved `hota_mota_mini_val.json` reports the following class-averaged results for scenes `scene-0103` and `scene-0916`:

| Metric | Result |
|---|---:|
| HOTA | 0.5142 |
| DetA | 0.4859 |
| AssA | 0.5701 |
| LocA | 0.8490 |
| MOTA | 0.5265 |

These are results from the saved evaluation report, not a claim of official nuScenes leaderboard performance. Consult the JSON file for per-class results and evaluation settings.

## Running the Benchmark

### Requirements

- Python
- NumPy, SciPy
- nuScenes devkit
- EasyDict, PyQuaternion
- AB3DMOT and Xinshuo_PyToolbox (third-party)

The scripts currently reference the third-party repositories under:

```text
~/ros2_cv_ws/scripts/third_party/
├── AB3DMOT/
└── Xinshuo_PyToolbox/
```

Update these paths if your local installation differs.

### Step 1 — Configure dataset and prediction paths

Open `main.py` and update `prediction_dir` and `nuscenes_root` to match your local environment.

- `prediction_dir` must contain the exported `.npz` files.
- `nuscenes_root` must contain the nuScenes-mini data and its `v1.0-mini` metadata.

### Step 2 — Generate tracks

Run from the benchmark directory:

```bash
python3 main.py
```

The script selects the official `mini_val` scenes and writes timestamped track JSON and log files under `outputs/mini_val/`.

### Step 3 — Evaluate HOTA and MOTA

Use the generated track JSON:

```bash
python3 evaluate_tracks.py \
  --dataroot /path/to/nuScenes-mini \
  --tracks outputs/mini_val/ab3dmot_tracks_mini_val_TIMESTAMP.json \
  --split mini_val \
  --out outputs/mini_val/hota_mota.json
```

Replace the dataset path and timestamped filename with your actual paths. The evaluator writes the metrics to JSON and prints the per-class and mean results.

## Planned Detection Error Analysis

The next stage is to evaluate the original BEVFusion detections before tracking.

1. **Mean Average Precision (mAP)** — Evaluate 3D detection quality against nuScenes ground truth, including per-class average precision and precision-recall behavior.
2. **nuScenes Detection Score (NDS)** — Evaluate the overall detection score and inspect its component metrics to identify weaknesses in localization, scale, orientation, velocity, and attribute estimation.
3. **Detection error analysis** — Investigate false positives, missed objects, localization errors, class confusion, and performance by class and distance.
4. **Tracking error analysis** — Investigate identity switches, track fragmentation, missed tracks, false tracks, and differences between detection quality and temporal association quality.

These analyses require the original BEVFusion detection predictions and the matching nuScenes ground-truth annotations. HOTA/MOTA track outputs alone are insufficient to calculate mAP or NDS.

## Related Project

**Learned Perception** — the companion repository for BEVFusion inference and prediction generation using Kaggle notebooks.

This repository focuses on downstream evaluation and error analysis rather than model inference or training.

## Scope and Limitations

This is a research and engineering benchmark using nuScenes-mini. Results depend on the exported predictions, confidence thresholds, class mappings, filtering rules, coordinate transformations, tracker settings, and evaluation protocol.

The objective is to establish a reproducible baseline and identify the next improvements needed in 3D detection and multi-object tracking.
