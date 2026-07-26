# 8B — 360° Panoramic Distance Estimation

## Objective

Develop a unified **360° panoramic distance estimation pipeline** by combining synchronized multi-camera object detection, LiDAR projection, and object-level distance estimation within a panoramic surround-view representation.

This milestone extends the previous 360° Camera–LiDAR perception framework by introducing panoramic image generation and distance-aware visualization, enabling intuitive surround perception for autonomous robotic systems.

**Question addressed:**

> **How can object distances from synchronized multi-camera and LiDAR data be represented within a unified 360° panoramic view?**

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
          LiDAR Point Projection
                      │
                      ▼
     Object-Level Point Association
                      │
                      ▼
       LiDAR Distance Estimation
                      │
                      ▼
      Cylindrical Panoramic Stitching
                      │
                      ▼
     360° Distance-Aware Visualization
```

---

## Project Structure

```text
8B_360_panoramic_distance_estimation/
├── assets/
│   ├── gifs/
│   └── images/
├── config/
├── dev/
│   ├── multi_camera_perception_detection.py
│   ├── multi_camera_lidar_detection_360.py
│   └── multi_camera_lidar_distance.py
├── launch/
├── perception/
│   ├── multi_camera_lidar_distance_panoramic.py
│   └── panoramic_stitcher.py
├── replay/
│   └── multi_camera_lidar_replay.py
└── README.md
```

---

## Key Components

### Multi-Camera Object Detection

Implemented synchronized object detection across four RGB cameras using YOLO instance segmentation.

Completed:

- Front camera object detection
- Rear camera object detection
- Left camera object detection
- Right camera object detection
- Multi-camera synchronization
- Unified multi-camera visualization

---

### Camera–LiDAR Projection

Projected LiDAR points into each synchronized camera image and associated them with detected objects.

Completed:

- Camera–LiDAR projection
- Instance mask filtering
- Object-specific LiDAR point extraction
- Multi-camera point association

---

### LiDAR Distance Estimation

Estimated the distance to detected objects using associated LiDAR point clouds.

Completed:

- Object-level point cloud filtering
- Robust distance estimation
- Distance-aware object visualization
- Multi-camera distance estimation

---

### 360° Panoramic Visualization

Generated a unified cylindrical panoramic representation of the surrounding environment with integrated distance information.

Completed:

- Cylindrical panoramic stitching
- Seam blending between camera views
- Compass overlay
- Ego-vehicle indicator
- Distance-aware panoramic visualization

---

## Engineering Outcome

Successfully developed a **360° panoramic distance estimation framework** by integrating synchronized multi-camera perception, LiDAR projection, object-level point association, distance estimation, and panoramic visualization into a unified perception pipeline.

This milestone provides the perception foundation required for:

- Cross-camera object association
- 360° multi-object tracking
- Spatial scene understanding
- Unified world representation

---

## Deliverable

<p align="center">
  <img src="../../assets/gifs/phase8B_pipeline.gif" width="700"/>
</p>

<p align="center">
360° panoramic perception with synchronized multi-camera object detection, LiDAR projection, and distance-aware visualization.
</p>

---

## Technologies

- CARLA 0.9.15
- ROS2 Humble
- YOLO Instance Segmentation
- OpenCV
- PointCloud2
- NumPy
- Python
- CV Bridge
- CycloneDDS

---

## Outcome

Phase 8B successfully extended the synchronized **360° Camera–LiDAR perception pipeline** by introducing panoramic visualization and LiDAR-based object distance estimation. The resulting system transforms synchronized multi-camera observations into a unified distance-aware panoramic representation, providing enhanced situational awareness and establishing a robust foundation for cross-camera object association, multi-object tracking, and higher-level spatial perception.