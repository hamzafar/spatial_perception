# 11B — Surrounding Object Motion Estimation (Radar)

## Objective

Develop a **surrounding object motion estimation pipeline** by fusing synchronized Front Camera and Radar data to detect nearby traffic objects and estimate their range, bearing, radial velocity, and motion state.

This milestone introduces radar-camera geometric fusion: radar targets are converted from spherical to Cartesian coordinates, transformed into the camera frame, projected into image space, and associated with YOLO object detections. The result is a per-object motion classification (Approaching / Receding / Stationary) rendered directly on the camera feed.

**Question addressed:**

> **How can synchronized camera and radar measurements be geometrically fused to estimate the range, bearing, and radial velocity of surrounding objects?**

---

## Architecture

```text
                    Front Camera
                          +
                        Radar
                          │
                          ▼
                 Synchronized Sensor Data
                          │
                          ▼
                YOLO TensorRT Detection
                          │
                          ▼
                  Radar Processing
             (Spherical → Cartesian)
                          │
                          ▼
              Radar → Camera Transformation
                          │
                          ▼
                  Radar Projection
                  (Image Coordinates)
                          │
                          ▼
             Camera–Radar Association
                          │
                          ▼
              Object Motion Estimation
        (Range · Bearing · Radial Velocity · State)
                          │
                          ▼
                    Visualization
```

---

## Project Structure

```text
11B_surrounding_object_motion_estimation/
├── assets/
│   ├── gifs/
│   └── images/
├── config/
├── dev/
│   └── dataset_synchronizer.py
├── perception/
│   ├── yolo_detection_pipeline.py
│   ├── radar_perception_pipeline.py
│   ├── display_pipeline_radar_perception.py
│   └── world_objs_motion.py
├── replay/
│   └── multi_sensors_replay.py
└── README.md
```

---

## Key Components

### Synchronized Multi-Sensor Replay

`DatasetSynchronizer` loads `timestamps.csv` for every recorded stream (front, rear, left, right, lidar, radar, imu, gnss) and builds a frame-aligned mapping by matching each front-camera timestamp to the nearest timestamp of every other sensor via `np.argmin(|Δt|)`.

`multi_sensors_replay.py` is the deterministic offline replay node. It:

- Publishes the four RGB camera feeds, LiDAR, and Radar as ROS2 `Image` / `PointCloud2` messages
- Publishes IMU (`sensor_msgs/Imu`) and GNSS (`sensor_msgs/NavSatFix`) from recorded CSVs
- Encodes radar points as a custom `PointCloud2` with `depth`, `azimuth`, `altitude`, `velocity` fields
- Uses the synchronized frame mapping from `DatasetSynchronizer` to keep every sensor stream frame-accurate
- Loops back to frame `0` once the recorded session is exhausted

Completed:

- Front/rear/left/right camera replay
- LiDAR replay
- Radar replay
- IMU replay
- GNSS replay
- Deterministic, timestamp-synchronized offline replay

---

### YOLO TensorRT Detection

`DetectionPipeline` (`yolo_detection_pipeline.py`) wraps the TensorRT-accelerated YOLO model:

- Converts the incoming ROS `Image` message to an OpenCV BGR frame
- Masks out the ego-vehicle hood using a scaled polygon (`EGO_HOOD_POLYGON`) so the hood is never fed to the detector, while the original (unmasked) frame is still returned for display
- Runs inference and extracts `boxes`, `scores`, and `classes` as NumPy arrays

---

### Radar Processing

`RadarPerceptionPipeline` (`radar_perception_pipeline.py`) implements the CARLA-radar-to-camera geometric fusion:

1. **ROS → NumPy** — converts the radar `PointCloud2` into an `(N, 4)` array of `[depth, azimuth, altitude, velocity]`
2. **Spherical → Cartesian** — converts each radar return into the radar's local frame (x forward, y right, z up)
3. **Radar → Camera transform** — applies the extrinsic `T_radar_to_cam` (rotation + translation) to bring the point into the camera frame
4. **Pinhole projection** — projects the camera-frame point to pixel coordinates `(u, v)` using `fx, fy, cx, cy`, discarding points behind the camera
5. **Association** — matches each projected radar point to the YOLO bounding box whose center it falls closest to, within `max_assoc_distance_px`, so at most one radar target is assigned per detection
6. **Estimation** — computes range (depth), bearing (azimuth in degrees), and radial velocity per associated object, then classifies motion state against `velocity_threshold`

Motion states:

```text
Approaching   → velocity < -threshold
Receding      → velocity > +threshold
Stationary    → otherwise
```

Two pipeline versions are included:

| File | Behavior |
|---|---|
| `radar_perception_pipeline.py` | Baseline association; returns object dicts without the matched pixel location |
| `radar_perception_pipeline-V2.py` | Also returns the associated radar pixel `(u, v)` as `radar_point`, enabling the display layer to draw an exact association line/marker instead of falling back to the box center |

---

### Visualization

`DisplayPipeline` (`display_pipeline_radar_perception.py`) renders the fused perception output on the camera image:

- Header bar with a live object count
- Color-coded bounding boxes (Red = Approaching, Blue = Receding, Green = Stationary)
- A compact label chip per object (`CLASS  RANGE m  ⇅VELOCITY`) sized to its text and auto-nudged upward to avoid overlapping neighboring labels (`_place_label`)
- A leader line connecting each object's actual radar association point (or box center as fallback) to its label chip, drawn black-then-white for contrast against any background
- Radar beams: translucent rays traced from the projected radar sensor origin (`radar_origin`, falling back to bottom-center of frame) to every in-view radar target, blended at low opacity, plus a marker at the origin
- A legend explaining the motion color/arrow convention

An earlier iteration, `display_pipeline_radar_perception copy.py`, is kept for reference. It used a simpler fixed-offset label layout (no overlap avoidance) and rendered radar returns as plain colored dots rather than beams from the sensor origin.

---

### World Objects Motion Node

`world_objs_motion.py` (`WorldObjectsMotion`) is the ROS2 node that ties the three pipelines together:

- Subscribes to the front camera and radar topics using `message_filters.TimeSynchronizer` for exact-time synchronized callbacks
- Loads the TensorRT YOLO engine and configures the camera intrinsics (`fx = fy = 320`, `cx = 320`, `cy = 240` for a 640×480 image)
- Builds the radar→camera extrinsic transform `T_radar_to_cam` from a fixed rotation/translation
- On each synchronized callback: runs detection → radar conversion → radar/camera fusion → radar beam overlay → full display render

---

## Software Architecture

```text
Replay Node
     │
     ▼
Sensor Synchronization (TimeSynchronizer)
     │
     ├──────────────────────┐
     ▼                      ▼
DetectionPipeline      RadarPerceptionPipeline
     │                      │
     └──────────┬───────────┘
                ▼
      WorldObjectsMotion Node
                │
                ▼
          DisplayPipeline
```

| Module | Responsibility |
|---|---|
| **Replay Node** | Load recorded sensor data and publish ROS2 messages |
| **DatasetSynchronizer** | Align recorded sensor frames using nearest-timestamp matching |
| **DetectionPipeline** | ROS image conversion, hood masking, YOLO TensorRT inference, detection extraction |
| **RadarPerceptionPipeline** | Radar message conversion, spherical-to-Cartesian conversion, radar-to-camera transform, image projection, camera-radar association, range/bearing/velocity/motion-state estimation |
| **WorldObjectsMotion Node** | Subscribe to synchronized camera + radar, coordinate pipeline execution |
| **DisplayPipeline** | Draw detections, radar projections, association lines, motion overlays, legend |

The architecture keeps computation inside pipeline classes while the ROS2 node coordinates synchronized inputs and outputs, consistent with the project's existing modular design.

---

## Engineering Outcome

Successfully developed a **surrounding-object motion estimation framework** combining synchronized camera and radar measurements through geometric sensor fusion within the existing modular ROS2 perception architecture.

The system estimates, per detected object:

- Class
- Image location
- Range
- Bearing
- Radial velocity
- Motion state

This milestone extends the project from ego-only motion reasoning (Phase 11A) to simultaneous reasoning about **both ego vehicle dynamics and surrounding object dynamics**, using synchronized multi-sensor perception.

The resulting object motion estimates provide a foundation for:

- Ego-relative object motion compensation
- Multi-object motion tracking over time
- Time-to-collision estimation
- Dynamic scene understanding
- Behavior and trajectory prediction

---

## Deliverable

<h3 align="center">Surrounding Object Motion Estimation</h3>

<p align="center">
  <img src="../../assets/gifs/phase11B_pipeline.gif" width="350"/>
</p>

<p align="center">
Object range, bearing, radial velocity, and motion state estimated from synchronized camera and radar data, with association lines and radar beams visualized on the front-camera view.
</p>

---

## Technologies

- CARLA 0.9.15
- ROS2 Humble
- YOLO (TensorRT engine)
- OpenCV
- NumPy
- Python
- CV Bridge
- CycloneDDS

---

## External Dependencies

This phase uses the following external software components:

- **Ultralytics YOLO / TensorRT** — accelerated object detection inference
- **message_filters.TimeSynchronizer** — synchronized camera/radar message processing
- **sensor_msgs_py.point_cloud2** — radar/LiDAR `PointCloud2` encoding and decoding

These dependencies provide detection inference and message synchronization while the radar-camera fusion and motion-estimation logic remain implemented within the project pipelines.

---

## Scope

Phase 11B focuses on **surrounding object motion estimation using synchronized Front Camera and Radar data**.

The following are outside the scope of this milestone:

- Ego-motion compensation of object tracks
- Multi-frame object tracking / ID persistence
- Object trajectory prediction
- Time-to-collision computation
- Behavior prediction
- Planning or control

These capabilities belong to subsequent perception phases.

---

## Outcome

Phase 11B successfully introduced **surrounding object motion estimation** into the perception stack by geometrically fusing synchronized camera and radar measurements within the existing modular ROS2 architecture. Radar returns are converted to Cartesian coordinates, transformed into the camera frame, projected into image space, and associated with YOLO detections to yield per-object range, bearing, and radial velocity, from which a motion state (Approaching / Receding / Stationary) is classified and visualized. Combined with Phase 11A's ego motion estimation, the project now supports simultaneous reasoning about both ego vehicle dynamics and surrounding object dynamics using synchronized multi-sensor perception.
