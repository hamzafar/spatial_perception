# 8A — 360° Camera–LiDAR Perception

## Objective

Develop a synchronized **360° Camera–LiDAR perception pipeline** by integrating four RGB cameras and a LiDAR sensor within CARLA.

This milestone introduces deterministic recording, offline replay, multi-camera perception, and surround-view generation to establish a scalable foundation for 360° robotic perception.

**Question addressed:**

> **How can synchronized multi-camera and LiDAR data be combined into a unified 360° perception pipeline?**

---

## Architecture

```text
        Front  Rear  Left  Right Cameras
                    +
                 LiDAR Sensor
                      │
                      ▼
          Deterministic Recording
                      │
                      ▼
               Offline Replay
                      │
                      ▼
          Multi-Camera Perception
                      │
                      ▼
      360° Image Stitching + LiDAR Projection
                      │
                      ▼
         Unified 360° Surround Perception
```

---

## Project Structure

```text
8A_360_camera_lidar_perception/
├── assets/
├── config/
├── dev/
│   ├── multi_camera_lidar_perception_stitching_45.py
│   ├── multi_camera_perception.py
│   ├── multi_camera_perception_stitching.py
│   └── surround_view.py
├── launch/
├── perception/
│   └── multi_camera_lidar_perception_stitching.py
├── recording/
│   └── sync_360_camera_lidar_recorder.py
├── replay/
│   └── multi_camera_lidar_replay.py
└── README.md
```

---

## Key Components

### 360° Multi-Camera Perception

Implemented a synchronized perception pipeline using four RGB cameras covering the complete surroundings of the ego vehicle.

Completed:

- Front camera integration
- Rear camera integration
- Left camera integration
- Right camera integration
- Multi-camera synchronization
- Unified surround-view generation

---

### Deterministic Dataset Recording

Implemented a synchronized multi-camera and LiDAR recording pipeline in CARLA.

Completed:

- CARLA synchronous recording
- Four RGB camera recording
- LiDAR point cloud recording
- Frame-level timestamp logging
- Deterministic dataset generation

Dataset structure:

```text
session_xxx/
├── front/
├── rear/
├── left/
├── right/
├── lidar/
└── timestamps.csv
```

---

### Offline Replay Pipeline

Implemented a replay publisher capable of reproducing synchronized multi-camera and LiDAR streams without CARLA.

Completed:

- ROS2 Image publishers
- ROS2 PointCloud2 publisher
- Shared ROS timestamps
- Adjustable replay FPS
- Infinite replay loop

---

### Unified 360° Perception

Validated the complete perception pipeline using recorded datasets.

Completed:

- Multi-camera synchronization verification
- 360° image stitching
- LiDAR projection onto stitched images
- Repeatable offline perception workflow

---

## Engineering Outcome

Successfully established a synchronized **360° Camera–LiDAR perception framework** by combining deterministic recording, offline replay, multi-camera perception, and LiDAR projection.

This milestone provides the perception foundation required for:

- Multi-camera object association
- 360° object tracking
- Spatial scene understanding
- Multi-modal perception

---

## Deliverable

<p align="center">
  <img src="../../assets/gifs/phase8A_pipeline.gif" width="700"/>
</p>

<p align="center">
Offline 360° multi-camera replay with synchronized LiDAR projection and unified surround perception.
</p>

---

## Technologies

- CARLA 0.9.15
- ROS2 Humble
- OpenCV
- PointCloud2
- NumPy
- Python
- CV Bridge
- CycloneDDS

---

## Outcome

Phase 8A successfully established a synchronized **360° Camera–LiDAR perception pipeline** by integrating four RGB cameras and LiDAR into a deterministic recording and offline replay framework. The resulting system enables repeatable multi-camera perception experiments with unified surround-view generation and LiDAR projection, providing the foundation for subsequent multi-camera object association, tracking, and multi-modal perception.