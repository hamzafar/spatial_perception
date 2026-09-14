# 12 — Unified Multi-Sensor Perception Stack (Camera · LiDAR · Radar · IMU · GNSS)

## Objective

Develop a **unified real-time perception stack** that fuses four RGB cameras, LiDAR, Radar, IMU, and GNSS into a single synchronized pipeline, producing a de-duplicated set of tracked world objects (with class, 3D position, distance, radar-derived motion) alongside ego-vehicle motion, and streaming all of it to a live web dashboard.

This milestone consolidates the project's earlier per-sensor phases (2D detection, camera–LiDAR 3D localization, radar range/bearing/velocity fusion, GNSS/IMU ego motion) into one synchronized ROS2 node, adds persistent multi-object identity via ByteTrack, and exposes the entire perception state through a WebSocket-driven dashboard and a frame-accurate offline replay tool.

**Question addressed:**

> **How can camera, LiDAR, radar, IMU, and GNSS be fused inside one synchronized ROS2 pipeline to produce a single, tracked, motion-aware world model — and made observable in real time?**

---

## Architecture

```text
              Front / Rear / Left / Right Cameras
                    +  Radar  +  LiDAR
                    +  IMU    +  GNSS
                          │
                          ▼
              TimeSynchronizer (8-way sync)
                          │
                          ▼
              YOLO26m-seg TensorRT Detection
              (per camera, ego-hood masked)
                          │
                          ▼
                ByteTrack (per-camera IDs)
                          │
                          ▼
              Camera → LiDAR 3D Localization
              (per-object position + distance)
                          │
                          ▼
              Track-ID Attachment (IoU match)
                          │
                          ▼
              Radar Fusion (front camera)
        (range · bearing · radial velocity · motion)
                          │
                          ▼
                 Unified world_objects
                          │
              ┌───────────┼────────────┐
              ▼           ▼            ▼
      Nearest-Object   BEV Prep   Object Counting
       Preparation
              │           │            │
              └───────────┴────────────┘
                          │
                          ▼
              GNSS + IMU Ego Motion
        (position · heading · speed · state)
                          │
                          ▼
          Dashboard Payload Assembly + Metrics
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
       DashboardBridge        PerceptionRecorder
         (WebSocket)          (CSV, no images)
                 │
                 ▼
        Live Web Dashboard (HTML/JS)
```

---

## Project Structure

```text
12_unified_perception_stack/
├── assets/
│   ├── gifs/
│   └── images/
├── config/
├── dashboard/
│   ├── perception-dashboard.html
│   └── perception-dashboard-v1.html
├── dev/
│   └── dataset_synchronizer.py
├── perception/
│   ├── perception_node.py
│   ├── yolo_detection_pipeline.py
│   ├── tracking_pipeline.py
│   ├── track.py
│   ├── kalman_filter.py
│   ├── perception_3d_pipeline.py
│   ├── radar_perception_pipeline.py
│   ├── gnss_pipeline.py
│   ├── imu_pipeline.py
│   ├── perception_utils.py
│   ├── unique_objects_localization_ego_coord.py
│   ├── dashboard_adapter.py
│   ├── dashboard_bridge.py
│   ├── dashboard_metrics.py
│   └── recorder_perception.py
├── replay/
│   └── multi_sensors_replay.py
└── README.md
```

---

## Key Components

### Synchronized Multi-Sensor Replay

`DatasetSynchronizer` (`dataset_synchronizer.py`) loads `timestamps.csv` for every recorded stream (front, rear, left, right, lidar, radar, imu, gnss) and builds a frame-aligned mapping by matching each front-camera timestamp to the nearest timestamp of every other sensor via `np.argmin(|Δt|)`.

`multi_sensors_replay.py` is the deterministic offline replay node. It:

- Publishes all four RGB camera feeds, LiDAR, and Radar as ROS2 `Image` / `PointCloud2` messages
- Publishes IMU (`sensor_msgs/Imu`) and GNSS (`sensor_msgs/NavSatFix`) from recorded CSVs
- Encodes radar points as a custom `PointCloud2` with `depth`, `azimuth`, `altitude`, `velocity` fields
- Uses the synchronized frame mapping from `DatasetSynchronizer` to keep every sensor stream frame-accurate
- Loops back to frame `0` once the recorded session is exhausted

---

### YOLO TensorRT Detection

`DetectionPipeline` (`yolo_detection_pipeline.py`) wraps the TensorRT-accelerated YOLO26m-seg model:

- Converts the incoming ROS `Image` message to an OpenCV BGR frame
- Masks out the ego-vehicle hood using a scaled polygon (`EGO_HOOD_POLYGON`) so the hood is never fed to the detector, while the original (unmasked) frame is still returned for display
- Runs inference and extracts `boxes`, `scores`, and `classes` as NumPy arrays, per camera

---

### Multi-Object Tracking (ByteTrack)

Each of the four cameras (front, rear, left, right) is assigned its own `BYTETracker` instance inside `perception_node.py`, giving every detection a persistent, camera-local track ID (`F_*`, `R_*`, `L_*`, `RT_*`) across frames.

- `reset_trackers()` re-initializes all four trackers (and `BaseTrack._count`) whenever a replay loop restart or timestamp rollback is detected, keeping track IDs from a previous loop from bleeding into the next
- `PerceptionUtils.attach_track_ids()` matches each 3D-localized object back to its tracker output by IoU, so downstream world objects carry a stable ID rather than a raw per-frame detection index

A separate, simpler `TrackingPipeline` / `Track` / `KalmanFilter` set of classes implements a lightweight from-scratch IoU tracker (constant-position Kalman state, `predict()`/`update()` stubs) used for earlier experimentation and as a dependency-free fallback to ByteTrack.

**Known limitation:** track IDs are currently **camera-local** — the same physical object seen by two cameras gets two different IDs. Global cross-camera identity association is not yet implemented.

---

### Camera–LiDAR 3D Localization

`Perception3DPipeline` (`perception_3d_pipeline.py`) projects LiDAR into each camera view and extracts a 3D position per detected object:

1. **ROS → NumPy** — converts the LiDAR `PointCloud2` into an `(N, 3)` XYZ array
2. **Per-camera rotation** — rotates LiDAR points into each camera's frame using its mounted `yaw` (front/rear/left/right), then applies the LiDAR→camera translation and axis remap
3. **Pinhole projection** — projects the transformed points to pixel coordinates `(u, v)`, discarding points behind the camera or outside the image bounds
4. **Mask-cloud extraction** — for every YOLO segmentation mask, collects the projected LiDAR points that fall inside it, producing a per-object point cloud
5. **Distance + position estimation** — takes the closest 10th-percentile of points per object as its "front surface," reporting both a scalar distance and a mean 3D position from that front-surface subset

---

### Radar Fusion

`RadarPerceptionPipeline` (`radar_perception_pipeline.py`) implements the CARLA-radar-to-camera geometric fusion for the front camera:

1. **ROS → NumPy** — converts the radar `PointCloud2` into an `(N, 4)` array of `[depth, azimuth, altitude, velocity]`
2. **Spherical → Cartesian** — converts each radar return into the radar's local frame (x forward, y right, z up)
3. **Radar → Camera transform** — applies the extrinsic `T_radar_to_cam` (rotation + translation) to bring the point into the camera frame
4. **Pinhole projection** — projects the camera-frame point to pixel coordinates `(u, v)` using `fx, fy, cx, cy`, discarding points behind the camera
5. **Association** — matches each projected radar point to the YOLO bounding box whose center it falls closest to, within `max_assoc_distance_px`, so at most one radar target is assigned per detection
6. **Estimation** — computes range (depth), bearing (azimuth in degrees), and radial velocity per associated object, then classifies motion state against `velocity_threshold`

```text
Approaching   → velocity < -threshold
Receding      → velocity > +threshold
Stationary    → otherwise
```

`PerceptionUtils.attach_radar_data()` then merges these radar objects back onto the corresponding camera-LiDAR world objects by IoU, so a single object dict can carry both a LiDAR-based position/distance and a radar-based range/bearing/velocity/motion.

---

### Ego Vehicle Motion (GNSS + IMU)

- `GNSSPipeline` (`gnss_pipeline.py`) converts each `(lat, lon, alt)` fix to a local ENU position relative to the first fix received (`set_reference`), then differentiates consecutive positions over elapsed time to estimate ego speed
- `IMUPipeline` (`imu_pipeline.py`) extracts linear acceleration, yaw rate, and heading (from quaternion `z, w`) from the IMU message, and classifies a discrete motion state — `Stationary` / `Moving Forward`, `Accelerating` / `Braking`, `Turning Left` / `Turning Right` — by thresholding speed, longitudinal acceleration, and yaw rate

---

### Unified World Objects & Perception Utilities

`PerceptionUtils` (`perception_utils.py`) is the shared post-processing layer that turns raw per-camera detections into the project's central representation, `world_objects`:

- **Class normalization** — collapses raw YOLO classes into four BEV categories: `car → vehicle`, `truck/bus → truck`, `person → person`, `bicycle/motorcycle → cyclist`
- **Track-ID attachment** — IoU-matches 3D-localized objects to ByteTrack output per camera
- **Radar attachment** — IoU-matches radar-derived range/bearing/velocity/motion onto front-camera objects
- **BEV preparation** — reduces each object to `{id, cls, x, y, distance}` for the bird's-eye-view panel
- **Nearest-object preparation** — sorts objects by distance and reports `{id, cls, label, dist_m, speed_mps, motion}` per object, using radar velocity where available and a LiDAR temporal fallback (`estimate_track_motion`) otherwise
- **Temporal speed/motion fallback** — when no radar is associated, estimates approaching/receding/stationary from frame-to-frame displacement of a track's ego-frame X position, thresholded at `0.2 m/s`
- **Object counting** — tallies `vehicle`, `person`, `cyclist` counts across all cameras

Motion resolution priority:

```text
Radar available    → radar range / bearing / velocity / motion
Radar unavailable   → LiDAR temporal position-delta speed / motion
Otherwise           → speed = None, motion = "unknown"
```

---

### Live Dashboard Pipeline

- `DashboardAdapter` (`dashboard_adapter.py`) matches ByteTrack output to raw detections (vectorized IoU), JPEG-encodes each camera frame to base64, and normalizes box coordinates to `[cx, cy, w, h]` fractions of image size for the dashboard
- `DashboardMetrics` (`dashboard_metrics.py`) computes smoothed FPS (EMA of instantaneous frame time), CPU utilization (`psutil`), and GPU utilization (`pynvml`, optional)
- `DashboardBridge` (`dashboard_bridge.py`) runs a background-thread WebSocket server (`ws://localhost:8765`) and exposes a thread-safe `push(dict)` that any ROS callback can call; pushes are partial updates — only changed top-level/nested keys need to be sent, nothing else is overwritten
- `perception-dashboard.html` / `perception-dashboard-v1.html` render the pushed state: four tracked camera feeds, BEV, nearest-objects panel with a motion legend, live object counts, ego speed/heading, GNSS trajectory trail, FPS/latency/GPU/CPU, sensor-health indicators, and a replay/model footer
  - Perception geometry (camera intrinsics, box normalization) is computed at ~640×480 regardless of the dashboard's own display resolution, so dashboard-only resizing never affects underlying perception math

---

### Perception Recording

`PerceptionRecorder` (`recorder_perception.py`) captures the exact dashboard payload stream to disk for offline analysis:

- Writes one row per frame to `perception.csv` inside a timestamped session folder
- Strips base64 camera images before saving (keeps boxes, IDs, classes, confidence, and all other fields) to keep recordings lightweight
- Deep-copies each payload before mutation so later dashboard updates cannot alter already-recorded frames
- Buffers frames in memory and flushes a single CSV write once `max_frame` is reached or `close()` is called, with dict/list fields JSON-serialized per cell

---

### Perception Node

`perception_node.py` (`PerceptionStack`) is the ROS2 node that ties every pipeline together:

- Subscribes to all 8 sensor topics (4 cameras, radar, LiDAR, IMU, GNSS) using a single `message_filters.TimeSynchronizer`
- Detects replay loop resets / timestamp rollbacks from the front-camera frame ID and re-initializes all four ByteTrackers accordingly
- Runs, per synchronized callback: detection → tracking → camera–LiDAR 3D localization → track-ID attachment → radar fusion (front) → unified `world_objects` → nearest/BEV/count preparation → GNSS/IMU ego motion → metrics → dashboard payload assembly → push to `DashboardBridge` and/or `PerceptionRecorder`

An earlier variant, `unique_objects_localization_ego_coord.py`, implements the same camera–LiDAR fusion idea without radar, tracking, or the dashboard — instead stitching four annotated camera views and a BEV panel into a single OpenCV display window via a `DisplayPipeline` / `ObjectAssociationPipeline` pair, useful as a minimal standalone visual debugger for the 3D localization step.

---

## Software Architecture

```text
Replay Node (or live CARLA bridge)
     │
     ▼
Sensor Synchronization (TimeSynchronizer, 8-way)
     │
     ├──────────────┬──────────────┬─────────────┐
     ▼              ▼              ▼             ▼
DetectionPipeline  ByteTracker  Perception3D   RadarPerception
(per camera)      (per camera)  Pipeline        Pipeline (front)
     │              │              │             │
     └──────────────┴──────┬───────┴─────────────┘
                            ▼
                    PerceptionUtils
          (track IDs · radar fusion · BEV ·
           nearest objects · counting)
                            │
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
      GNSSPipeline   DashboardMetrics  DashboardAdapter
      IMUPipeline
              │             │              │
              └─────────────┴──────┬───────┘
                                    ▼
                          PerceptionStack Node
                                    │
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
              DashboardBridge              PerceptionRecorder
              (WebSocket push)             (CSV, no images)
                      │
                      ▼
            Live Dashboard (HTML/JS)
```

| Module | Responsibility |
|---|---|
| **Replay Node** | Load recorded sensor data and publish ROS2 messages |
| **DatasetSynchronizer** | Align recorded sensor frames using nearest-timestamp matching |
| **DetectionPipeline** | ROS image conversion, hood masking, YOLO TensorRT inference, detection extraction |
| **BYTETracker / TrackingPipeline** | Persistent per-camera object identity across frames |
| **Perception3DPipeline** | LiDAR→camera projection, mask-cloud extraction, per-object distance/position |
| **RadarPerceptionPipeline** | Radar conversion, spherical-to-Cartesian, radar-to-camera transform, projection, association, range/bearing/velocity/motion |
| **GNSSPipeline / IMUPipeline** | Ego position, speed, heading, and discrete motion-state estimation |
| **PerceptionUtils** | Class normalization, track/radar attachment, BEV/nearest/count preparation, temporal motion fallback |
| **DashboardAdapter / DashboardMetrics** | Camera payload prep (boxes, JPEG), FPS/CPU/GPU metrics |
| **PerceptionStack Node** | Subscribe to synchronized 8-sensor input, coordinate every pipeline, assemble dashboard payload |
| **DashboardBridge** | Thread-safe WebSocket broadcast of partial state updates |
| **PerceptionRecorder** | Persist dashboard payload stream to CSV for offline analysis |
| **Dashboard (HTML/JS)** | Render cameras, BEV, nearest objects, ego state, trajectory, and system metrics live |

The architecture keeps computation inside pipeline/util classes while the ROS2 node coordinates synchronized inputs and orchestrates outputs, consistent with the project's existing modular design.

---

## Engineering Outcome

Successfully developed a **unified, real-time, multi-sensor perception stack** that fuses four cameras, LiDAR, radar, IMU, and GNSS inside one time-synchronized ROS2 node, producing a single de-duplicated `world_objects` representation and streaming it — together with ego motion, object counts, and system metrics — to a live web dashboard.

The system estimates, per detected object:

- Class (normalized to `vehicle` / `truck` / `person` / `cyclist`)
- Persistent per-camera track ID
- 3D position and distance (camera–LiDAR)
- Range, bearing, and radial velocity (radar, where available)
- Motion state (radar-derived, or LiDAR-temporal fallback)

And for the ego vehicle:

- Position (local ENU), heading, and speed
- Discrete motion state (accelerating / braking / turning)

This milestone consolidates every prior perception phase — 2D detection, camera–LiDAR 3D localization, four-camera perception, radar motion fusion, GNSS/IMU ego motion — into a single production-shaped node, and adds the observability layer (dashboard + recorder) needed to benchmark, validate, and document the stack before moving beyond perception.

The resulting unified perception output provides a foundation for:

- Global cross-camera object identity (resolving the current camera-local ID limitation)
- Multi-object trajectory prediction
- Time-to-collision estimation
- World-model construction for planning
- Full perception → prediction → planning → control progression

---

## Deliverable

<h3 align="center">Unified Multi-Sensor Perception Stack</h3>

<p align="center">
  <img src="../../assets/gifs/phase12_pipeline.gif" width="350"/>
</p>

<p align="center">
Four tracked camera feeds, BEV, nearest-object ranking, and ego motion — fused from camera, LiDAR, radar, IMU, and GNSS in real time and streamed to a live web dashboard.
</p>

---

## Technologies

- CARLA 0.9.15
- ROS2 Humble
- YOLO26m-seg (TensorRT FP16 engine)
- ByteTrack
- OpenCV
- NumPy / Pandas
- pymap3d
- WebSockets
- psutil / pynvml
- Python
- CV Bridge
- CycloneDDS

---

## External Dependencies

This phase uses the following external software components:

- **Ultralytics YOLO / TensorRT** — accelerated segmentation-detection inference
- **ByteTrack (YOLOX tracker)** — multi-object tracking (`BYTETracker`, `BaseTrack`)
- **message_filters.TimeSynchronizer** — 8-way synchronized sensor message processing
- **sensor_msgs_py.point_cloud2** — radar/LiDAR `PointCloud2` encoding and decoding
- **pymap3d** — geodetic-to-ENU coordinate conversion for GNSS
- **websockets** — dashboard WebSocket transport
- **psutil / pynvml** — CPU / GPU utilization metrics

These dependencies provide detection inference, tracking, message synchronization, and system metrics, while all sensor fusion, world-object construction, and motion-estimation logic remain implemented within the project pipelines.

---

## Scope

Phase 12 focuses on **unifying camera, LiDAR, radar, IMU, and GNSS perception into one synchronized, tracked, dashboard-observable pipeline**.

The following are outside the scope of this milestone:

- Global cross-camera object identity (track IDs remain camera-local)
- Radar fusion on rear/left/right cameras (front camera only)
- Multi-frame trajectory prediction
- Time-to-collision computation
- Behavior prediction
- Planning or control

These capabilities belong to subsequent phases.

---

## Outcome

Phase 12 successfully unified the project's perception stack into a single synchronized ROS2 node that fuses four cameras, LiDAR, radar, IMU, and GNSS into a de-duplicated, tracked `world_objects` representation, with radar fusion providing range/bearing/velocity/motion where available and a LiDAR-temporal estimate as fallback. Ego motion is derived from GNSS/IMU into position, heading, speed, and discrete motion state. The full perception state — camera views, BEV, nearest objects, ego motion, object counts, and system metrics — is streamed live over WebSocket to a web dashboard, and can be captured frame-by-frame to CSV via the recorder for offline benchmarking. With the perception stack now considered feature-complete, the project's next priorities are benchmarking, validation, and documentation, ahead of extending toward world modeling, prediction, planning, and control.