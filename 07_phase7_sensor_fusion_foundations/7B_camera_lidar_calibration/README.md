# 7B — Camera–LiDAR Calibration

## Objective

Establish geometric correspondence between the RGB camera and LiDAR sensors through calibration, synchronization, and point projection.

This milestone transforms LiDAR points into the camera coordinate frame and projects them onto RGB images, providing the geometric foundation required for camera–LiDAR sensor fusion.

**Question addressed:**

> **Which LiDAR point corresponds to each image pixel?**

---

## Architecture

```text
RGB Camera
        │
        │
LiDAR Point Cloud
        │
        ▼
Time Synchronization
        │
        ▼
Coordinate Transformation
        │
        ▼
Perspective Projection
        │
        ▼
RGB Image + Projected LiDAR
```

---

## Project Structure

```text
7B_camera_lidar_calibration/
├── assets/
├── calibration/
│   └── camera_lidar_projection.py
├── config/
│   └── ego_rgb_lidar.json
├── launch/
├── validation/
│   ├── timestamp_buffer_check.py
│   └── timestamp_check.py
└── README.md
```

---

## Key Components

### Camera Calibration

Extracted camera intrinsic parameters from the ROS2 CameraInfo topic.

Completed:

- Camera intrinsic extraction
- Camera matrix validation
- Perspective projection parameter validation

---

### LiDAR Calibration

Derived LiDAR extrinsic parameters from the CARLA sensor configuration.

Completed:

- Sensor pose validation
- Coordinate frame alignment
- LiDAR-to-camera coordinate transformation

---

### Sensor Synchronization

Implemented synchronized acquisition of RGB images and LiDAR point clouds.

Completed:

- ROS2 `message_filters.TimeSynchronizer`
- Exact timestamp synchronization
- Synchronization validation

---

### LiDAR Projection

Projected LiDAR points onto synchronized RGB images.

Completed:

- PointCloud2 parsing
- Perspective projection
- Image bounds filtering
- Depth-colored visualization

---

## Camera Configuration

| Parameter | Value |
|-----------|------:|
| Topic | `/carla/ego_vehicle/rgb_front/image` |
| Resolution | 640 × 480 |
| Field of View | 90° |
| Position | x = -1.5, y = 0.0, z = 2.4 |

---

## LiDAR Configuration

| Parameter | Value |
|-----------|------:|
| Topic | `/carla/ego_vehicle/lidar` |
| Range | 50 m |
| Channels | 32 |
| Points per Second | 56,000 |
| Rotation Frequency | 10 Hz |
| Position | x = 0.0, y = 0.0, z = 2.4 |

---

## Camera Intrinsics

| Parameter | Value |
|-----------|------:|
| fx | 320.0 |
| fy | 320.0 |
| cx | 320.0 |
| cy | 240.0 |

Camera Matrix

```text
320   0   320
  0 320   240
  0   0     1
```

---

## Synchronization Validation

| Metric | Result |
|--------|-------:|
| Synchronization Method | ROS2 TimeSynchronizer |
| Image Timestamp | 3045.320203 |
| LiDAR Timestamp | 3045.320203 |
| Timestamp Difference | **0.000 ms** |

---

## Engineering Outcome

Successfully established geometric calibration between the RGB camera and LiDAR sensors.

This milestone provides the calibration foundation required for:

- 2D–3D object association
- LiDAR point filtering
- Object distance estimation
- Multi-modal perception

---

## Deliverable

<p align="center">
  <img src="../../assets/gifs/phase7B_pipeline.gif" width="700"/>
</p>

<p align="center">
LiDAR point cloud projected onto synchronized RGB camera images.
</p>

---

## Technologies

- CARLA 0.9.15
- ROS2 Humble
- OpenCV
- PointCloud2
- CameraInfo
- Python
- CycloneDDS

---

## Outcome

Phase 7B successfully established camera–LiDAR geometric calibration by synchronizing both sensors, transforming LiDAR points into the camera coordinate frame, and projecting them onto RGB images. This calibration provides the foundation for robust 2D–3D object association and subsequent distance estimation.