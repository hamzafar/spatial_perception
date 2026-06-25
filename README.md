Prerequisite:

This project builds upon the 2D perception stack developed in:

ROS2 Autonomous Perception Stack — [2D Perception](https://github.com/hamzafar/autonomous_perception)

# Phase 7 — Sensor Fusion & 3D Perception Foundations

## Objectives

Extend the perception stack beyond monocular vision by integrating LiDAR data and basic sensor fusion techniques.

Question: What is this object and how far away is it in 3D World?


## 7.1 LiDAR Integration

### Tasks

- Add LiDAR sensor to CARLA vehicle (completed)
- Publish LiDAR point clouds through ROS2 (completed)
- Visualize point cloud data (completed)


### Focus Areas

- Point cloud processing

### Deliverable

<p align="center">
  <img src="assets/phase7A_pipeline.gif" width="700"/>
</p>

<p align="center">
  CARLA Online RGB Camera + LiDAR Integration
</p>

# 7.2 Camera–LiDAR Calibration & Projection

## Tasks

- Extract camera intrinsic parameters (completed)
- Extract LiDAR extrinsic parameters (completed)
- Validate camera–LiDAR synchronization (completed)
- Transform LiDAR points into camera coordinates (completed)
- Project LiDAR points onto image plane (completed)
- Visualize projected LiDAR points on RGB images (completed)

## Completed

- ✅ Camera intrinsics extracted from ROS2 CameraInfo
- ✅ Camera matrix validated
- ✅ LiDAR extrinsics derived from CARLA sensor configuration
- ✅ ROS2 TimeSynchronizer implemented
- ✅ Exact camera–LiDAR timestamp synchronization verified
- ✅ PointCloud2 parsing implemented
- ✅ LiDAR → camera coordinate transformation implemented
- ✅ Perspective projection implemented
- ✅ Image bounds filtering implemented
- ✅ OpenCV overlay visualization implemented
- ✅ Depth-colored projection visualization implemented

## Camera Configuration

Topic:
- `/carla/ego_vehicle/rgb_front/image`

Resolution:
- 640 × 480

FOV:
- 90°

Position:
- x = -1.5
- y = 0.0
- z = 2.4

## LiDAR Configuration

Topic:
- `/carla/ego_vehicle/lidar`

Position:
- x = 0.0
- y = 0.0
- z = 2.4

Parameters:
- Range: 50 m
- Channels: 32
- Points/sec: 56000
- Rotation Frequency: 10 Hz

## Camera Intrinsics

fx = 320.0

fy = 320.0

cx = 320.0

cy = 240.0

Camera Matrix:

| 320 | 0 | 320 |
|------|------|------|
| 0 | 320 | 240 |
| 0 | 0 | 1 |

## Synchronization Validation

Used:

- ROS2 `message_filters.TimeSynchronizer`

Results:

```text
Image TS : 3045.320203
LiDAR TS : 3045.320203
Delta    : 0.000 ms
```

### Deliverable

<p align="center">
  <img src="assets/phase7B_pipeline.gif" width="700"/>
</p>

<p align="center">
  CARLA Online RGB Camera + LiDAR Integration
</p>


## 7.3 2D–3D Association

### Tasks

- Run YOLOv8m-seg TensorRT perception pipeline (completed)
- Associate projected LiDAR points with detected objects (completed)
- Filter object-specific point clusters using segmentation masks (completed)
- Record synchronized camera–LiDAR data (completed)
- Build deterministic camera–LiDAR dataset (completed)
- Implement ROS2 camera–LiDAR replay publisher (completed)
- Validate object association on recorded datasets (completed)

## Completed

- ✅ YOLOv8m-seg TensorRT inference integrated
- ✅ Segmentation mask extraction implemented
- ✅ Projected LiDAR points associated with segmented objects
- ✅ Object-specific LiDAR point filtering implemented
- ✅ CARLA synchronous recording pipeline implemented
- ✅ Synchronized camera–LiDAR dataset generation implemented
- ✅ Camera image recording implemented
- ✅ LiDAR point cloud recording implemented
- ✅ Frame-level timestamp logging implemented
- ✅ Deterministic offline replay pipeline implemented
- ✅ ROS2 Image publisher implemented
- ✅ ROS2 PointCloud2 publisher implemented
- ✅ Adjustable replay FPS implemented
- ✅ Exact camera–LiDAR timestamp synchronization during replay verified
- ✅ Existing perception pipeline validated on replayed datasets
- ✅ Repeatable offline testing workflow established


### Deliverable
<p align="center">
  <img src="assets/phase7C_pipeline.gif" width="700"/>
</p>

<p align="center">
  Offline detection point clouds
</p>

## 7.4 Monocular vs LiDAR vs Sensor Fusion Distance

### Tasks

- Compute Monocular Camera Distance (completed)
- Compute LiDAR Distance (completed)
- Compute Sensor Fusion (Camera + LiDAR) Distance (completed)

### Completed

- ✅ Monocular camera distance estimation implemented
- ✅ LiDAR distance estimation using object-specific LiDAR point clusters implemented
- ✅ Camera and LiDAR distance visualization integrated into perception pipeline
- ✅ Weighted camera–LiDAR fusion distance estimation implemented
- ✅ Object-level distance estimation validated on synchronized camera–LiDAR streams
- ✅ End-to-end RGB + LiDAR distance estimation pipeline established

### Focus Areas

- Monocular geometric distance estimation
- LiDAR-based object range estimation
- Camera–LiDAR sensor fusion
- Object-level metric scene understanding
- Multi-modal perception validation



## 7.5 3D Semantic Bird Eye View(BEV)

### Tasks

- Deterministic 4 cameras and LiDAR Recording (completed)
- Generate 360 degree camera view
- Project Lidar points on 360 Image View
- Perform detection and extract objects point clouds
- Estimate Distance of surronding objects
- Generate BEV
- Unified world representation
- Object localization in world coordinates
- Scene-level perception analysis

### Goal

Transition from 2D perception toward 3D scene understanding.


## 7.6 Benchmarking & Analysis

### Comparison

| Pipeline | Detection | Distance Estimation | Spatial Awareness |
|-----------|-----------|-----------|-----------|
| Camera Only | ✓ | ✗ | Limited |
| Camera + LiDAR | ✓ | ✓ | Improved |

### Metrics

- Distance estimation accuracy
- Sensor synchronization stability
- Fusion processing latency
- Perception throughput


## Deliverables

- ROS2 LiDAR integration
- Camera-LiDAR calibration pipeline
- Distance-aware object detection
- Sensor fusion perception pipeline
- Basic 3D perception framework
- Fusion benchmarking report
- Multi-modal perception demonstration


## Outcome

Transition from camera-only perception toward multi-modal robotics perception.
---

# Phase 8 — Multi-Camera Perception

## Objectives

Expand perception coverage using multiple synchronized cameras.

Question: Can I perceive my surroundings in all directions in 3D World?


## Scope

### Camera Configuration

- Front camera
- Rear camera
- Left camera
- Right camera

### Tasks

- Multi-camera ROS2 integration
- Camera synchronization
- Multi-stream visualization
- Overlapping field-of-view analysis
- Unified perception visualization
- Project Lidar points on 360 Image View
- Perform detection and extract objects point clouds
- Estimate Distance of surronding objects

### Focus Areas

- 360° scene awareness
- Multi-camera architecture
- Perception scalability
- Sensor synchronization


## Deliverables

- Multi-camera ROS2 pipeline
- 360° perception visualization
- Multi-camera benchmarking report
- Multi-camera perception demonstration


## Outcome

Expand perception coverage from a single viewpoint to full-surround awareness.

---
# Phase 9 — Advanced Multi-Modal Perception

## Objectives

Combine camera, LiDAR, and multi-camera perception into a unified perception system.

Question : Where is everything relative to me in a unified world representation?

## Scope

### Tasks

- Multi-camera and LiDAR synchronization
- Multi-modal data association
- Bird's-Eye View generation
- Unified world representation
- Object localization in world coordinates
- Scene-level perception analysis

### Focus Areas

- Advanced sensor fusion
- 3D scene understanding
- Spatial reasoning
- Autonomous perception architecture


## Deliverables

- Multi-camera + LiDAR fusion pipeline
- Bird's-Eye View visualization
- Unified perception framework
- Multi-modal benchmarking report


## Outcome

Build a complete multi-modal perception stack resembling modern autonomous systems.

---

# Phase 10 — Edge Inference Readiness & Deployment

## Objectives

Deploy the optimized perception stack on embedded edge hardware.


## Target Platforms

- NVIDIA Jetson Nano
- NVIDIA Jetson Xavier NX
- NVIDIA Jetson Orin Nano


## 10.1 Deployment Preparation

### Tasks

- Containerize perception pipeline
- Prepare deployment scripts
- Package ROS2 perception nodes
- Validate TensorRT deployment workflow


## 10.2 Edge Optimization

### Tasks

- Optimize memory usage
- Optimize power consumption
- Tune TensorRT inference settings
- Analyze thermal behavior
- Evaluate deployment constraints


## 10.3 Edge Benchmarking

### Comparison

| Platform | FPS | Latency | GPU Utilization | Memory Usage |
|-----------|-----|----------|----------------|--------------|
| Desktop GPU | | | | |
| Jetson Nano | | | | |
| Xavier NX | | | | |
| Orin Nano | | | | |

### Metrics

- Real-time FPS
- End-to-end latency
- Memory footprint
- Power efficiency
- Thermal stability


## 10.4 Deployment Validation

### Tasks

- Continuous perception testing
- Long-duration stability testing
- Resource monitoring
- Failure analysis


## Deliverables

- Edge deployment workflow
- TensorRT deployment package
- Containerized perception stack
- Edge benchmarking report
- Embedded deployment guide


## Outcome

Deploy a robotics perception system on real edge hardware with validated real-time performance.

---
# Phase 11 — ViT-Based Detection Extension

## Tasks
- Integrate transformer-based detector

## Comparison
- Accuracy
- FPS
- Latency
- Edge suitability

## Deliverable
- CNN vs ViT perception comparison

---

# Final Deliverables

- GitHub repository
- Demo video
- Benchmark report
- ROS2 modular perception stack
- CNN vs ViT comparison
