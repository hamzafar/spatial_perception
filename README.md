# Spatial Perception

# Spatial Perception

A robotics perception engineering project focused on building 3D spatial understanding using synchronized RGB cameras and LiDAR.

Built using CARLA, ROS2, OpenCV, and YOLOv8, the project progresses from camera–LiDAR integration to multi-camera perception, unified spatial understanding, and 360° environmental perception.

This project builds upon the 2D perception stack developed in the companion repository:

**ROS2 Autonomous Perception Stack** — [2D Perception](https://github.com/hamzafar/autonomous_perception)

---

---

## Technology Stack

| Category | Technologies |
|-----------|-------------|
| Simulation | CARLA 0.9.15 |
| Robotics Middleware | ROS2 Humble |
| Computer Vision | OpenCV |
| Object Detection | YOLOv8m-seg TensorRT INT8 |
| 3D Sensor | LiDAR |
| Programming Language | Python |
| Communication | CycloneDDS |
| Environment | Windows 11 + WSL2 Ubuntu 22.04 |

---

## System Architecture

```text
RGB Cameras + LiDAR
          │
          ▼
Sensor Synchronization
          │
          ▼
Multi-Modal Perception
          │
          ▼
Spatial Understanding
          │
          ▼
Unified World Representation
```

---

## Completed Phases

### 🚧 Phase 7 — Sensor Fusion Foundations

Established the foundation for spatial perception by integrating LiDAR with RGB cameras, calibrating multi-sensor geometry, associating 2D detections with 3D point clouds, and estimating object distances.

####  Milestones

- ✅ **7A — LiDAR Integration**
    -  📁 [View Phase 7A](07_phase7_sensor_fusion_foundations/7A_lidar_integration/)
    
- 🚧 **7B — Camera–LiDAR Calibration**
- 🚧 **7C — 2D–3D Association**
- 🚧 **7D — Object Distance Estimation**



---

### 🚧 Phase 8 — Multi-Sensor Perception

Documentation will be published after project milestone release.

📁 [View Phase 8](08_phase8_multi_sensor_perception)

---

### 🚧 Phase 9 — Unified Spatial Perception

Documentation will be published after project milestone release.

📁 [View Phase 9](09_phase9_unified_spatial_perception)

---

## Demonstration

<h2 align="center">Phase 7 — Sensor Fusion Foundations</h2>

<p align="center">
  <img src="assets/gifs/phase7A_pipeline.gif" width="700"/>
</p>

<p align="center">
RGB camera and LiDAR integration establishing the foundation for 3D spatial perception.
</p>

---

## Future Work

Planned areas of development include:

- Multi-camera perception
- 360° environment perception
- Unified world representation
- Bird's-Eye View generation
- Multi-camera overlap resolution
- Spatial perception evaluation

---

## Project Journal

Detailed planning, development notes, experiments, and engineering decisions are maintained separately:

- `roadmap/README.md`