# 7C — 2D–3D Association

## Objective

Associate 2D object detections from the RGB camera with 3D LiDAR points to generate object-specific point clouds.

This milestone combines instance segmentation and LiDAR projection to determine which LiDAR points belong to each detected object while establishing a deterministic offline replay framework for repeatable perception experiments.

**Question addressed:**

> **Which LiDAR points belong to each detected object?**

---

## Architecture

```text
RGB Camera + LiDAR
          │
          ▼
Deterministic Recording
          │
          ▼
Offline Replay
          │
          ▼
YOLOv8 Segmentation
          │
          ▼
LiDAR Projection
          │
          ▼
2D–3D Association
          │
          ▼
Object-specific Point Clouds
```

---

## Project Structure

```text
7C_2d_3d_association/
├── assets/
├── config/
├── launch/
├── perception/
│   └── masked_cloud_points.py
├── recording/
│   └── sync_camera_lidar_recorder.py
├── replay/
│   └── lidar_camera_publisher.py
└── README.md
```

---

## Key Components

### Object-Level 2D–3D Association

Associated projected LiDAR points with YOLOv8 segmentation masks to extract object-specific point clouds.

Completed:

- YOLOv8m-seg TensorRT inference
- Segmentation mask extraction
- LiDAR point projection
- Object-specific LiDAR point filtering
- Point cloud visualization

---

### Deterministic Dataset Recording

Implemented a synchronized camera–LiDAR recording pipeline in CARLA.

Completed:

- CARLA synchronous recording
- RGB image recording
- LiDAR point cloud recording
- Frame-level timestamp logging
- Deterministic dataset generation

Dataset structure:

```text
session_xxx/
├── images/
├── lidar/
└── timestamps.csv
```

---

### Offline Replay Pipeline

Implemented a replay publisher capable of reproducing synchronized camera and LiDAR streams without CARLA.

Completed:

- ROS2 Image publisher
- ROS2 PointCloud2 publisher
- Shared ROS timestamps
- Adjustable replay FPS
- Infinite replay loop

---

### Replay Validation

Validated the complete perception pipeline using recorded datasets.

Completed:

- Camera–LiDAR synchronization verification
- Existing perception pipeline validation
- Repeatable offline perception workflow

---

## Engineering Outcome

Successfully established object-level camera–LiDAR association by combining instance segmentation with projected LiDAR points.

This milestone provides the perception foundation required for:

- Object distance estimation
- Multi-modal perception
- Unified spatial perception
- Multi-camera perception

---

## Deliverable

<p align="center">
  <img src="../../assets/gifs/phase7C_pipeline.gif" width="700"/>
</p>

<p align="center">
Offline camera–LiDAR replay with object-specific LiDAR point cloud association.
</p>

---

## Technologies

- CARLA 0.9.15
- ROS2 Humble
- YOLOv8m-seg TensorRT INT8
- OpenCV
- PointCloud2
- NumPy
- Python
- CycloneDDS

---

## Outcome

Phase 7C successfully established object-level 2D–3D association by combining synchronized RGB images and LiDAR point clouds through YOLOv8 instance segmentation. The addition of a deterministic recording and offline replay framework enabled repeatable perception experiments independent of the CARLA simulator, providing the foundation for robust object distance estimation in the next milestone.