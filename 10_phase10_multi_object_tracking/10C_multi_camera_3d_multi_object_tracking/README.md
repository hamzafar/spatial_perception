# 10.3 — 360° Geometric 3D Multi-Object Tracking (AB3DMOT)

## Objective

Extend the existing single-camera 3D tracking pipeline to a **360° multi-camera 3D tracking system** by fusing four synchronized cameras with LiDAR, removing cross-camera duplicate detections, and maintaining persistent object identities using AB3DMOT.

This milestone introduces geometry-based cross-camera association on top of the existing 3D detection and tracking framework, enabling full surround coverage instead of a single forward-facing field of view.

**Question addressed:**

> **How can 3D objects detected independently from four overlapping cameras be merged into a single, duplicate-free set of persistent 360° object tracks?**

---

## Architecture

```text
                 4 Synchronized Cameras
                            +
                       LiDAR Sensor
                            │
                            ▼
                 Synchronized Sensor Data
                            │
                            ▼
                     YOLO26m-seg
                            │
                            ▼
                    LiDAR Projection
                            │
                            ▼
             Segmentation Mask ↔ LiDAR Association
                            │
                            ▼
               Object-specific 3D Point Clouds
                            │
                            ▼
              3D Position / Dimensions / Yaw
                            │
                            ▼
                Four Camera World Objects
                            │
                            ▼
              Cross-Camera Duplicate Association
                            │
                            ▼
                   Unified Unique Objects
                            │
                            ▼
                         AB3DMOT
                            │
                            ▼
                Persistent 3D Track IDs
                            │
                            ▼
                  360° BEV Visualization
```

---

## Project Structure

```text
10C_360_3d_multi_object_tracking/
├── assets/
│   ├── gifs/
│   └── images/
├── config/
├── dev/
├── launch/
│   └── tracking_3d_360_pipeline.py
├── perception/
│   ├── ab3dmot_tracker.py
│   ├── object_association_pipeline.py
│   ├── perception_3d_pipeline.py
│   └── tracking_3d_display_pipeline.py
├── replay/
│   └── multi_camera_lidar_replay.py
└── README.md
```

---

## Sensor Configuration

Four cameras:

- Front — yaw 0°
- Rear — yaw 180°
- Left — yaw -90°
- Right — yaw 90°

Camera resolution:

```text
640 × 480
```

Camera intrinsics:

```text
fx = 320
fy = 320
cx = 320
cy = 240
```

LiDAR:

- 32 channels
- 50 m range
- Ego-centric point cloud
- Synchronized with all four camera streams

---

## Key Components

### Four-Camera 3D Perception

Extended the existing `Perception3DPipeline` to run independently across all four cameras, each producing its own set of 3D world objects from synchronized LiDAR data.

Completed:

- Four-camera synchronized perception
- LiDAR projection per camera (using camera yaw and intrinsics)
- YOLO26m-seg segmentation
- Segmentation mask ↔ LiDAR association
- Object-specific point cloud extraction
- DBSCAN point cloud filtering

---

### 3D Object Representation

Each camera produces `world_objects` with the following structure:

```python
{
    "class": class_name,
    "camera": camera_name,
    "position": position,
    "distance": distance,
    "cloud": object_cloud,
    "score": score,
    "dimensions": dimensions,
    "yaw": yaw
}
```

Coordinate convention:

```text
+X = forward
-X = rear
+Y = left
-Y = right
+Z = up
```

---

### 3D Position, Dimensions, and Yaw

**Position** is estimated from the front surface of the LiDAR cluster:

1. Calculate Euclidean distance of LiDAR points.
2. Calculate the 10th percentile distance.
3. Keep points at or below that threshold.
4. Average those points.

**Dimensions** are derived from the filtered point cloud:

```text
length = max(X) - min(X)
width  = max(Y) - min(Y)
height = max(Z) - min(Z)
```

**Yaw** is estimated using PCA on the object point cloud, taking the principal horizontal direction as the dominant orientation, stored in radians.

---

### Cross-Camera Duplicate Association

Introduced the `ObjectAssociationPipeline` to merge duplicate observations of the same physical object seen by overlapping cameras.

```python
world_objects = (
    front_world_obj +
    rear_world_obj +
    left_world_obj +
    right_world_obj
)

unique_objects = self.pipeline_object_association.associate(
    world_objects
)
```

Association process:

```text
World Objects
    │
    ▼
Candidate Generation
    │
    ▼
Same Class + Different Camera
    │
    ▼
Bearing Overlap Filtering
    │
    ▼
Camera Pair Grouping
    │
    ▼
Euclidean Position Cost Matrix
    │
    ▼
Hungarian Assignment
    │
    ▼
Distance Threshold
    │
    ▼
Duplicate Merging
    │
    ▼
unique_objects
```

Camera overlap pairs:

```text
front ↔ left
front ↔ right
left  ↔ rear
rear  ↔ right
```

Configurable parameters:

```python
association_distance = 7
bearing_overlap_margin = 30
```

Merged objects contain:

```python
{
    "class": ...,
    "camera": [...],
    "position": ...,
    "distance": ...,
    "source_positions": [...],
    "merged": True
}
```

Unmatched objects retain their original single-camera representation.

Completed:

- Cross-camera candidate generation
- Bearing-based overlap filtering
- Camera pair grouping
- Euclidean position cost matrix
- Hungarian assignment
- Distance-thresholded duplicate merging
- Unified unique-object generation

---

### 360° 3D Multi-Object Tracking

Unified detections from `ObjectAssociationPipeline` are passed into the existing `AB3DMOTTracker`, which maintains persistent 3D identities across the full 360° field of view.

```python
tracks = self.ab3dmot_tracker.update(
    unique_objects
)
```

AB3DMOT detection format:

```text
height
width
length
x
y
z
yaw
```

Tracker output:

```python
{
    "track_id": track_id,
    "class": ...,
    "score": ...,
    "center": np.array([x, y, z]),
    "dimensions": (l, w, h),
    "yaw": yaw
}
```

Association happens **before** tracking — raw four-camera detections are never fed directly into AB3DMOT, since overlapping cameras would otherwise produce duplicate tracks.

Completed:

- Unified detection conversion to AB3DMOT format
- AB3DMOT update on merged 360° detections
- Persistent Track ID assignment across the full surround view
- Replay-aware tracker reset

---

### 360° BEV Visualization

Extended the display pipeline to render a unified ego-centric bird's-eye view covering all four cameras.

The BEV shows:

- Ego vehicle
- Ego heading
- Tracked objects across the full 360° perimeter
- Persistent Track IDs
- Deterministic per-track color, derived from `track_id`

```text
ID #1 → Blue
ID #2 → Green
ID #3 → Red
ID #4 → Cyan
...
```

Track IDs displayed in the BEV are the actual AB3DMOT `track_id` values — no new IDs are generated in the display pipeline.

Completed:

- 360° BEV rendering
- Ego vehicle and heading visualization
- Track visualization across all four camera sectors
- Deterministic Track-ID color coding

---

### Deterministic Offline Replay

Validated the complete 360° pipeline using synchronized four-camera + LiDAR offline replay data.

Replay dataset:

```text
front/
rear/
left/
right/
lidar/
timestamps.csv
```

A `DatasetSynchronizer` builds a synchronization dictionary from timestamps, and the replay node uses it to select matching frames across all sensors:

```python
mapping = self.sync_dict[self.frame_index]

front_frame = mapping["front"] + 1
rear_frame = mapping["rear"] + 1
left_frame = mapping["left"] + 1
right_frame = mapping["right"] + 1
lidar_frame = mapping["lidar"] + 1
```

All published sensor messages share the same ROS timestamp so ROS2 `TimeSynchronizer` pairs them correctly. Replay restart resets the AB3DMOT tracker so Track IDs begin again from the initial ID.

Completed:

- ROS2 four-camera replay
- ROS2 LiDAR replay
- Full sensor-suite synchronized replay
- Repeatable 360° detection and tracking evaluation

---

## Software Architecture

```text
Replay Node
    │
    ▼
Perception3DPipeline
    │
    ▼
ObjectAssociationPipeline
    │
    ▼
AB3DMOTTracker
    │
    ▼
DisplayPipeline
```

| Module | Responsibility |
|---|---|
| **Replay Node** | Load synchronized replay data, publish ROS2 sensor messages, handle looping/reset |
| **Perception3DPipeline** | Image conversion, YOLO inference, LiDAR projection, mask association, point-cloud filtering, position/dimensions/yaw estimation, world-object generation |
| **ObjectAssociationPipeline** | Cross-camera candidate generation, bearing filtering, Hungarian matching, duplicate merging, unified object generation |
| **AB3DMOTTracker** | Detection conversion, AB3DMOT update, track management, persistent Track IDs, tracker reset |
| **DisplayPipeline** | BEV rendering, ego visualization, track visualization, Track-ID labels and colors |

The ROS2 node itself remains lightweight, delegating all logic to the pipeline modules above.

---

## Engineering Outcome

Successfully extended the perception stack from single-camera 3D tracking to **360° geometric 3D multi-object tracking**, combining four synchronized cameras and LiDAR with geometry-based cross-camera duplicate association and AB3DMOT.

This milestone provides the foundation required for:

- Full surround object motion estimation
- Multi-camera trajectory prediction
- Time-To-Collision across all directions
- Behavior prediction over a complete 360° scene
- Downstream planning and control

---

## Deliverable

<h3 align="center">360° 3D Multi-Object Tracking</h3>

<p align="center">
  <img src="../../assets/gifs/phase10C_pipeline.gif" width="700"/>
</p>

<p align="center">
Persistent 3D object identities maintained across four synchronized cameras and LiDAR, using geometry-based cross-camera association and AB3DMOT tracking with 360° BEV visualization.
</p>

---

## Technologies

- CARLA 0.9.15
- ROS2 Humble
- WSL2 Ubuntu 22.04
- Python 3.10
- Ultralytics YOLO26m-seg
- TensorRT FP16
- AB3DMOT
- OpenCV
- PointCloud2
- NumPy
- SciPy (Hungarian assignment)
- CV Bridge
- CycloneDDS

---

## External Dependencies

This phase integrates the following third-party frameworks:

- **AB3DMOT** — 3D multi-object tracking
- **Xinshuo_PyToolbox** — dependency used by AB3DMOT

These dependencies are maintained separately from the `spatial_perception` repository and are not modified as part of this phase.

---

## Scope

Phase 10.3 focuses on **360° geometric 3D multi-object detection and tracking using four synchronized cameras and a single LiDAR sensor**.

The following are outside the scope of this milestone:

- Deep appearance-based cross-camera ReID
- Velocity estimation
- Trajectory prediction
- Time-To-Collision
- Behavior prediction
- Planning or control

These capabilities belong to subsequent perception phases.

---

## Important Limitation

The Phase 10.3 tracker provides persistent IDs after cross-camera duplicate merging, but this is **not** a fully learned cross-camera ReID system.

The association is primarily geometric, relying on:

- Ego-frame position
- Semantic class
- Camera overlap
- Bearing
- Euclidean distance
- Hungarian assignment

This should therefore be described as **geometry-based cross-camera association + 3D tracking**, rather than deep appearance-based cross-camera identity tracking.

---

## Outcome

Phase 10.3 successfully extended the perception stack from single-camera 3D tracking to **360° multi-camera 3D multi-object tracking** by fusing four synchronized cameras with LiDAR, applying geometry-based cross-camera duplicate association, and maintaining persistent identities through AB3DMOT. The modular architecture keeps detection, association, tracking, and visualization fully decoupled, while deterministic offline replay enables repeatable evaluation of the complete 360° pipeline.
