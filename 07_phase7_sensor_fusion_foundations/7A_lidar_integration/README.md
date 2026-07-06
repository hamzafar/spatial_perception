# 7A — LiDAR Integration

## Objective

Extend the perception stack beyond monocular vision by integrating LiDAR into the existing ROS2 perception pipeline.

This milestone establishes the first step toward 3D spatial perception by publishing synchronized point cloud data from CARLA through ROS2 for downstream sensor fusion.

**Question addressed:**

> **What is this object and how far away is it in the 3D world?**

---

## Architecture

```text
CARLA Simulator
        │
        ▼
LiDAR Sensor
        │
        ▼
ROS Bridge
        │
        ▼
PointCloud2
        │
        ▼
RViz2 Visualization
```

---

## Key Components

### LiDAR Integration

Integrated a 3D LiDAR sensor with the CARLA ego vehicle.

Completed:

- LiDAR sensor configuration
- LiDAR attachment to ego vehicle
- Point cloud generation
- ROS2 PointCloud2 publishing

---

### Point Cloud Visualization

Validated real-time visualization using RViz2.

Completed:

- Point cloud visualization
- Coordinate frame validation
- Live sensor monitoring

---

## Engineering Outcome

Successfully established the LiDAR perception pipeline required for subsequent camera–LiDAR fusion.

This milestone provides the spatial sensing capability required for:

- Camera–LiDAR calibration
- 2D–3D object association
- Object distance estimation
- Multi-modal perception

---

## Deliverable

<p align="center">
  <img src="../../assets/gifs/phase7A_pipeline.gif" width="700"/>
</p>

<p align="center">
CARLA RGB Camera and LiDAR integration with synchronized ROS2 visualization.
</p>

---

## Technologies

- CARLA 0.9.15
- ROS2 Humble
- PointCloud2
- RViz2
- Python
- OpenCV
- CycloneDDS

---

## Outcome

Phase 7A successfully extended the perception stack from monocular RGB vision to synchronized camera and LiDAR sensing, providing the foundation for spatial perception and the remaining sensor fusion milestones.