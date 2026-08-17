# 10B — 3D Multi-Object Tracking

## Objective

Develop a modular **3D multi-object tracking pipeline** by combining LiDAR-based 3D object detection with persistent object tracking using AB3DMOT.

This milestone introduces both geometry-based and deep learning-based 3D detection while maintaining a common tracking, visualization, and offline replay framework.

**Question addressed:**

> **How can 3D objects detected from LiDAR maintain persistent identities across consecutive frames?**

---

## Architecture

```text
                 Front Camera
                      +
                  LiDAR Sensor
                      │
                      ▼
            Synchronized Sensor Data
                      │
                      ▼
              3D Object Detection
                 /          \
                /            \
               ▼              ▼
      Geometry-Based       PointPillars
        Detection          3D Detection
               \              /
                \            /
                 ▼          ▼
              3D Detections
                      │
                      ▼
                  AB3DMOT
                      │
                      ▼
             Persistent 3D Tracks
                      │
                      ▼
          Camera Projection + Display
```

---

## Project Structure

```text
10B_3d_multi_object_tracking/
├── assets/
│   ├── gifs/
│   └── images/
├── config/
├── dev/
├── launch/
│   ├── tracking_3d_1cam.py
│   └── tracking_pointpillars_1cam_pipeline.py
├── perception/
│   ├── ab3dmot_tracker.py
│   ├── detection_pipeline.py
│   ├── inference_dataset.py
│   ├── perception_3d_pipeline.py
│   ├── pointpillars_detector.py
│   └── tracking_3d_display_pipeline.py
├── replay/
│   └── multi_camera_lidar_replay.py
└── README.md
```

---

## Key Components

### Camera–LiDAR 3D Perception

Integrated synchronized front-camera and LiDAR data into the 3D perception pipeline.

Completed:

- Front RGB camera integration
- LiDAR integration
- Camera–LiDAR synchronization
- LiDAR point cloud projection
- YOLO instance segmentation
- Object-specific LiDAR point extraction
- 3D point cloud filtering

---

### Geometry-Based 3D Detection

Developed a geometry-based 3D object detection pipeline using object-specific LiDAR point clouds.

Completed:

- DBSCAN point cloud outlier removal
- Front-surface LiDAR localization
- 3D object position estimation
- 3D object dimension estimation
- PCA-based yaw estimation
- AB3DMOT-compatible detection generation

Detection format:

```text
Detection
├── Class
├── Score
├── Center (x, y, z)
├── Dimensions (length, width, height)
└── Yaw
```

---

### PointPillars 3D Detection

Integrated the OpenPCDet PointPillars detector as a deep learning-based alternative to the geometry-based detector.

Completed:

- Pretrained PointPillars inference
- OpenPCDet integration
- LiDAR preprocessing
- XYZI point representation
- Batch preparation
- GPU inference
- Prediction postprocessing
- 3D bounding box decoding
- Confidence score extraction
- Semantic class extraction
- AB3DMOT detection format conversion

The detector can therefore be replaced without changing the downstream tracking framework.

---

### 3D Multi-Object Tracking

Integrated the official AB3DMOT framework for persistent 3D object tracking.

Completed:

- AB3DMOT integration
- Frame-by-frame 3D data association
- Track management
- Persistent 3D Track ID assignment
- AB3DMOT output parsing
- Internal Track object generation
- Persistent identity maintenance across consecutive frames

---

### Modular Detection and Tracking

Designed the perception stack so that different 3D detectors can operate with the same tracking framework.

Supported detection paths:

```text
Geometry-Based 3D Detection
              │
              ▼
           AB3DMOT
              │
              ▼
        3D Tracking
```

and:

```text
PointPillars 3D Detection
              │
              ▼
           AB3DMOT
              │
              ▼
        3D Tracking
```

This enables direct comparison of geometry-based and PointPillars-based detection while keeping tracking and visualization consistent.

---

### 3D Tracking Visualization

Projected tracked 3D bounding boxes from the LiDAR coordinate frame onto the synchronized front camera image.

Completed:

- 3D cuboid generation
- 3D bounding box projection
- Object class visualization
- Persistent Track ID visualization
- Camera-view tracking visualization

The visualization supports both:

- Geometry-based 3D detections
- PointPillars-based 3D detections
- AB3DMOT tracked objects

---

### Deterministic Offline Replay

Validated the complete 3D detection and tracking pipeline using synchronized offline replay data.

Completed:

- ROS2 camera replay
- ROS2 LiDAR replay
- Synchronized sensor replay
- Repeatable 3D detection evaluation
- Repeatable tracking evaluation
- Geometry-based vs PointPillars comparison using the same tracking framework

---

## Engineering Outcome

Successfully developed a modular **3D multi-object tracking framework** combining LiDAR-based 3D detection with persistent object tracking through AB3DMOT.

The system supports interchangeable geometry-based and PointPillars-based 3D detectors while maintaining a common tracking and visualization pipeline.

This milestone provides the foundation required for:

- 360° 3D multi-object tracking
- Multi-camera 3D tracking
- Object motion estimation
- Temporal spatial reasoning
- Dynamic scene understanding

---

## Deliverable

<h3 align="center">3D Multi-Object Tracking</h3>

<p align="center">
  <img src="../../assets/gifs/phase10B_pipeline.gif" width="700"/>
</p>

<p align="center">
Persistent 3D object identities maintained across consecutive LiDAR frames using PointPillars 3D detection and AB3DMOT tracking with synchronized front-camera visualization.
</p>

---

## Technologies

- CARLA 0.9.15
- ROS2 Humble
- YOLO26m Instance Segmentation
- OpenPCDet
- PointPillars
- AB3DMOT
- OpenCV
- PointCloud2
- NumPy
- PyTorch
- Python
- CV Bridge
- CycloneDDS

---

## External Dependencies

This phase integrates the following third-party frameworks:

- **AB3DMOT** — 3D multi-object tracking
- **OpenPCDet** — LiDAR-based 3D object detection
- **PointPillars** — pretrained 3D object detection model
- **Xinshuo_PyToolbox** — dependency used by AB3DMOT

These dependencies are maintained separately from the `spatial_perception` repository and are not modified as part of this phase.

---

## Scope

Phase 10B focuses on **3D multi-object detection and tracking using a single front camera and LiDAR sensor**.

The following are outside the scope of this milestone:

- 360° 3D tracking
- Cross-camera 3D identity association
- Velocity estimation
- Trajectory prediction
- Time-To-Collision
- Behavior prediction
- Planning or control

These capabilities belong to subsequent perception phases.

---

## Outcome

Phase 10B successfully extended the perception stack from 2D temporal tracking to **3D multi-object tracking** by integrating both geometry-based and PointPillars-based LiDAR 3D detection with AB3DMOT. The modular architecture allows interchangeable 3D detectors to use the same tracking and visualization framework, while deterministic offline replay enables repeatable evaluation and comparison of 3D detection and tracking performance.