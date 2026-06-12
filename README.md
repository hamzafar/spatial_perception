Prerequisite:

This project builds upon the 2D perception stack developed in:

ROS2 Autonomous Perception Stack — [2D Perception](https://github.com/hamzafar/autonomous_perception)

# Phase 7 — Sensor Fusion & 3D Perception Foundations

## Objectives

Extend the perception stack beyond monocular vision by integrating LiDAR data and basic sensor fusion techniques.


## 7.1 LiDAR Integration

### Tasks

- Add LiDAR sensor to CARLA vehicle
- Publish LiDAR point clouds through ROS2
- Visualize point cloud data
- Validate camera-LiDAR synchronization

### Focus Areas

- Point cloud processing
- ROS2 sensor integration
- Multi-sensor synchronization


## 7.2 Camera–LiDAR Calibration

### Tasks

- Extract camera intrinsic parameters
- Extract LiDAR extrinsic parameters
- Transform LiDAR points into camera coordinates
- Project LiDAR points onto image plane

### Goal

Create a unified camera-LiDAR representation.


## 7.3 2D–3D Association

### Tasks

- Run YOLOv8 perception pipeline
- Associate LiDAR points with detected objects
- Filter object-specific point clusters
- Estimate object distance

### Outputs

- Object Class
- Bounding Box
- Estimated Distance

### Example

Car: 18.4 m

Truck: 27.1 m

Pedestrian: 12.3 m


## 7.4 Sensor Fusion Pipeline

### Tasks

- Fuse camera detections with LiDAR measurements
- Generate distance-aware detections
- Evaluate fusion robustness
- Analyze fusion performance

### Focus Areas

- Multi-modal perception
- Detection enhancement
- Scene understanding


## 7.5 3D Perception Foundations

### Tasks

- Generate Bird's-Eye View representation
- Visualize projected point clouds
- Estimate object positions
- Build spatial awareness pipeline

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
- Cross-camera object tracking
- Overlapping field-of-view analysis
- Unified perception visualization

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
