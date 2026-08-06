# 9B — Cross-Camera Object Merging

## Objective

Develop a **cross-camera object association pipeline** to identify and merge duplicate object detections observed by overlapping camera views into a unified object representation.

This milestone extends the unified spatial perception framework by introducing cross-camera data association, duplicate removal, and multi-camera perception fusion using geometric reasoning and optimal assignment.

**Question addressed:**

> **Which detections from different cameras correspond to the same physical object?**

---

## Architecture

```text
        Front  Rear  Left  Right Cameras
                    +
                 LiDAR Sensor
                      │
                      ▼
          Multi-Camera Object Detection
                      │
                      ▼
          Camera–LiDAR Projection
                      │
                      ▼
      Object-Level Point Association
                      │
                      ▼
      Ego Coordinate Localization
                      │
                      ▼
     Cross-Camera Candidate Generation
                      │
                      ▼
     Bearing-Based Overlap Filtering
                      │
                      ▼
      Hungarian Data Association
                      │
                      ▼
       Duplicate Object Merging
                      │
                      ▼
      Unified Object Representation
                      │
                      ▼
    Bird's-Eye View Visualization
```

---

## Project Structure

```text
9B_cross_camera_object_merging/
├── assets/
│   ├── gifs/
│   └── images/
├── config/
├── dev/
│   └── unique_objects_localization_ego_coord.py
├── launch/
├── perception/
│   ├── display_pipeline.py
│   ├── object_association_pipeline.py
│   ├── perception_3d_pipeline.py
│   └── unique_objects_localization_ego_coord.py
├── replay/
│   └── multi_camera_lidar_replay.py
└── README.md
```

---

## Key Components

### Cross-Camera Candidate Generation

Generated association candidates between object detections originating from different camera views.

Completed:

- Candidate generation across adjacent cameras
- Semantic class consistency filtering
- Camera-pair association
- Cross-camera object candidate creation

---

### Bearing-Based Overlap Filtering

Reduced false associations by considering only detections located inside overlapping camera fields of view.

Completed:

- Ego-coordinate bearing computation
- Camera overlap validation
- Bearing-based candidate filtering
- Configurable overlap margin

---

### Cross-Camera Object Association

Associated duplicate detections using geometric similarity and optimal assignment.

Completed:

- Camera-pair grouping
- Euclidean centroid distance computation
- Cost matrix construction
- Hungarian assignment
- Association threshold validation

---

### Duplicate Object Merging

Merged duplicate detections into a unified object representation.

Completed:

- Position averaging
- Distance averaging
- Camera source aggregation
- Merged object generation
- Unified object representation

Object representation:

```text
Unique Object
├── Class
├── Ego-frame Position (x, y, z)
├── LiDAR Distance
├── Contributing Cameras
└── Merged Flag
```

---

### Bird's-Eye View Visualization

Visualized both raw detections and merged objects within the same Bird's-Eye View.

Completed:

- Camera-colored raw detections
- White merged object visualization
- Ego vehicle visualization
- Heading visualization
- BEV legend
- Cross-camera association verification

---

## Engineering Outcome

Successfully developed a **cross-camera object association framework** by combining bearing-based overlap filtering, Hungarian assignment, and duplicate object merging into a unified spatial perception pipeline.

This milestone provides the perception foundation required for:

- Unified multi-camera object representation
- 360° multi-object tracking
- Temporal object association
- Dynamic scene understanding

---

## Deliverable

<h3 align="center">Cross-Camera Object Association</h3>

<p align="center">
  <img src="../../assets/gifs/phase9B_pipeline.gif" width="700"/>
</p>

<p align="center">
Cross-camera duplicate object association and merging using bearing-based overlap filtering and Hungarian assignment to produce a unified object representation.
</p>

---

## Technologies

- CARLA 0.9.15
- ROS2 Humble
- YOLOv8 Instance Segmentation
- OpenCV
- PointCloud2
- NumPy
- SciPy (Hungarian Assignment)
- Python
- CV Bridge
- CycloneDDS

---

## Outcome

Phase 9B successfully extended the unified spatial perception framework by introducing cross-camera object association and duplicate merging. Using bearing-based overlap filtering, centroid-distance cost matrices, and Hungarian assignment, duplicate observations from overlapping camera views were merged into unified object representations. The resulting perception pipeline produces a consistent 360° representation of surrounding objects, providing the foundation for persistent multi-object tracking and higher-level scene understanding.