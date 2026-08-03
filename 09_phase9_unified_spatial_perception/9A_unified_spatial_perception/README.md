# 9A — Unified Spatial Perception

## Objective

Develop a **unified spatial perception pipeline** by transforming synchronized multi-camera detections and LiDAR observations into a common ego-centric world representation.

This milestone extends the previous panoramic perception pipeline by localizing detected objects from all four cameras into the ego coordinate frame and generating a unified Bird's-Eye View (BEV) for spatial scene understanding.

**Question addressed:**

> **How can objects detected from multiple synchronized cameras be localized into a unified ego-centric world representation?**

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
     Unified World Representation
                      │
                      ▼
       Bird's-Eye View Generation
```

---

## Project Structure

```text
9A_unified_spatial_perception/
├── assets/
│   ├── gifs/
│   └── images/
├── config/
├── dev/
│   ├── Objects_localization_ego_coordinate.py
│   ├── Objects_localization_ego_coordinate_gt.py
│   └── Objects_localization_ego_coordinate_values.py
├── launch/
├── perception/
│   └── Objects_localization_ego_coordinate.py
├── replay/
│   └── multi_camera_lidar_replay.py
└── README.md
```

---

## Key Components

### Unified Multi-Camera Perception

Integrated synchronized detections from four RGB cameras into a common perception pipeline.

Completed:

- Front camera object detection
- Rear camera object detection
- Left camera object detection
- Right camera object detection
- Multi-camera synchronization
- Unified perception processing

---

### Camera–LiDAR Fusion

Projected LiDAR points into each camera view and transformed object observations into a common ego-centric coordinate frame.

Completed:

- Camera intrinsic projection
- Camera extrinsic transformation
- Inverse camera yaw correction
- Camera-to-ego coordinate transformation
- Object-specific LiDAR point extraction
- Multi-camera coordinate transformation validation

---

### Ego Coordinate Localization

Localized detected objects into a unified ego-centric coordinate frame.

Completed:

- Object localization in ego coordinates
- Front-surface localization using 10th percentile LiDAR points
- LiDAR distance estimation
- Unified object representation

Object representation:

```text
Object
├── Class
├── Ego-frame Position (x, y, z)
├── LiDAR Distance
├── Camera Source
└── BEV Position
```

---

### Bird's-Eye View Generation

Generated a unified Bird's-Eye View representing the spatial positions of detected objects relative to the ego vehicle.

Completed:

- Ego vehicle visualization
- Object visualization
- Ego-frame localization
- Unified multi-camera BEV generation
- Coordinate transformation validation
- BEV-based localization verification

---

### Bird's-Eye View Generation & Coordinate Transformation Validation

Generated a unified Bird's-Eye View (BEV) by transforming object detections from all four cameras into a common ego-centric coordinate frame.

During development, BEV visualization was used to identify incorrect object localization caused by camera orientation differences. Camera-specific yaw corrections were introduced for the left and right cameras, resulting in consistent object placement across all camera views.

Completed:

- Ego vehicle visualization
- Unified multi-camera BEV generation
- Ego-frame object localization
- Camera-specific coordinate transformation
- Left camera yaw correction
- Right camera yaw correction
- BEV-based localization verification
- Unified world representation validation

--- 

## Engineering Outcome

Successfully developed a **unified spatial perception framework** by combining synchronized multi-camera perception, Camera–LiDAR fusion, inverse camera yaw correction, ego-coordinate localization, and Bird's-Eye View generation into a common world representation.

This milestone provides the perception foundation required for:

- Cross-camera object association
- Unified object tracking
- Spatial scene understanding
- Multi-object reasoning

---

## Deliverable

<h3 align="center">BEV Generation</h3>

<p align="center">
  <img src="../../assets/gifs/phase9A_pipeline.gif" width="700"/>
</p>

<p align="center">
Unified spatial perception with ego-coordinate object localization and Bird's-Eye View generation from synchronized multi-camera and LiDAR observations.
</p>


<h3 align="center">Camera Yaw Correction & BEV Validation</h3>

<p align="center">
  <img src="../../assets/gifs/phase9A_1_pipeline.gif" width="700"/>
</p>

<p align="center">
Bird's-Eye View visualization used to validate ego-coordinate transformations and correct left and right camera yaw, resulting in consistent object localization across all synchronized camera views.
</p>

---

## Technologies

- CARLA 0.9.15
- ROS2 Humble
- YOLOv8 Instance Segmentation
- OpenCV
- PointCloud2
- NumPy
- Python
- CV Bridge
- CycloneDDS

---

## Outcome

Phase 9A successfully transformed synchronized multi-camera detections into a unified ego-centric world representation by integrating Camera–LiDAR fusion, inverse camera yaw correction, ego-coordinate localization, and Bird's-Eye View generation. The resulting spatial perception pipeline enables consistent localization of surrounding objects across all camera views, providing the foundation for higher-level perception tasks such as cross-camera object association and unified scene understanding.