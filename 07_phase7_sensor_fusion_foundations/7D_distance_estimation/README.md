# 7D — Object Distance Estimation

## Objective

Estimate object distances by combining monocular camera geometry and LiDAR measurements within a unified perception pipeline.

This milestone extends the object-specific LiDAR point clouds generated in Phase 7C by computing camera-based, LiDAR-based, and fused distance estimates for every detected object.

**Question addressed:**

> **How far away is each detected object?**

---

## Architecture

```text
RGB Camera + LiDAR
          │
          ▼
YOLOv8 Segmentation
          │
          ▼
2D–3D Association
          │
          ▼
Object-specific LiDAR Point Clouds
          │
          ▼
Distance Estimation
     ┌───────────────┐
     │ Monocular     │
     │ LiDAR         │
     │ Sensor Fusion │
     └───────────────┘
          │
          ▼
Distance-aware Object Detection
```

---

## Project Structure

```text
7D_distance_estimation/
├── assets/
├── config/
├── launch/
├── perception/
│   ├── camera_pixels_distance.py
│   ├── lidar_camera_distance.py
│   └── masked_cloud_points_distance.py
└── README.md
```

---

## Key Components

### Monocular Camera Distance

Estimated object distance using the pinhole camera model and class-specific object height priors.

Completed:

- Camera intrinsic calibration
- Segmentation mask height extraction
- Class-specific object height estimation
- Monocular distance computation

---

### LiDAR Distance Estimation

Estimated object distances directly from object-specific LiDAR point clouds.

Completed:

- Object-specific LiDAR point extraction
- Euclidean distance computation
- Robust 10th-percentile distance estimation
- Sparse point filtering

---

### Camera–LiDAR Sensor Fusion

Combined camera and LiDAR measurements into a unified distance estimate.

Completed:

- Weighted camera–LiDAR fusion
- Multi-modal distance estimation
- Object-level distance visualization
- Unified perception pipeline integration

---

## Engineering Outcome

Successfully established object-level metric distance estimation by combining monocular vision and LiDAR measurements.

This milestone provides the spatial understanding required for:

- Multi-camera perception
- Unified world representation
- Object localization
- Spatial scene understanding

---

## Deliverable

<p align="center">
  <img src="../../assets/gifs/phase7D_pipeline.gif" width="700"/>
</p>

<p align="center">
Comparison of monocular, LiDAR, and camera–LiDAR fusion distance estimation for synchronized RGB and LiDAR perception.
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

Phase 7D successfully established object-level metric distance estimation by combining monocular camera geometry with LiDAR measurements through a unified perception pipeline. The resulting system augments each detected object with camera-based, LiDAR-based, and fused distance estimates, completing the sensor fusion foundation required for multi-camera spatial perception in the next phase.