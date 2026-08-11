# 10A — 2D Multi-Object Tracking

## Objective

Extend the perception pipeline from frame-by-frame object detection to **temporal 2D multi-object tracking** by integrating YOLO26m TensorRT FP16 with the official ByteTrack tracker.

This milestone introduces persistent object identities across consecutive frames while maintaining a modular ROS2 architecture and deterministic offline replay workflow.

**Question addressed:**

> **How can detected objects maintain persistent identities across consecutive frames?**

---

## Architecture

```text
              ROS2 Image
                   │
                   ▼
        YOLO26m TensorRT FP16
                   │
                   ▼
          Detection Extraction
        (boxes, scores, classes)
                   │
                   ▼
               ByteTrack
      (Kalman Filter + Association)
                   │
                   ▼
            Tracked Objects
       (track_id + bounding box)
                   │
                   ▼
       TrackingDisplayPipeline
                   │
                   ▼
          Tracking Visualization
```

---

## Project Structure

```text
10A_2d_multi_object_tracking/
├── assets/
│   ├── gifs/
│   └── images/
├── ByteTrack/
│   ├── assets/
│   ├── datasets/
│   ├── deploy/
│   ├── exps/
│   ├── tools/
│   ├── tutorials/
│   ├── videos/
│   ├── yolox/
│   ├── LICENSE
│   ├── README.md
│   ├── requirements.txt
│   ├── setup.cfg
│   └── setup.py
├── config/
├── dev/
│   ├── kalman_filter.py
│   ├── tracking_pipeline.py
│   └── track.py
├── launch/
│   └── tracking.py
├── perception/
│   ├── detection_pipeline.py
│   ├── display_pipeline.py
│   ├── perception_3d_pipeline.py
│   └── tracking_display_pipeline.py
├── replay/
│   └── multi_camera_lidar_replay.py
└── README.md
```

---

## Key Components

### YOLO26m TensorRT Detection

Upgraded the detection stage to YOLO26m using a TensorRT FP16 engine.

Completed:

- YOLO26m detector integration
- TensorRT FP16 inference
- ROS2 image conversion
- Bounding box extraction
- Confidence score extraction
- Class ID extraction

Detection output:

```text
Detection
├── Bounding Box
├── Confidence
└── Class ID
```

---

### ByteTrack Integration

Integrated the official ByteTrack implementation to associate detections across consecutive frames and maintain persistent object identities.

Completed:

- Official ByteTrack integration
- Detection-to-tracker format conversion
- Frame-by-frame tracker updates
- Persistent object ID assignment
- Multi-object tracking

Tracker configuration:

```text
track_thresh = 0.5
track_buffer = 30
match_thresh = 0.8
frame_rate  = 10 FPS
```

NumPy compatibility fixes were applied for NumPy 2.x.

---

### Track Lifecycle Management

Implemented temporal track management through ByteTrack.

Completed:

- Automatic track initialization
- Track updates across consecutive frames
- Track persistence
- Automatic track termination

---

### Tracking Visualization

Implemented a dedicated visualization pipeline focused specifically on temporal tracking.

Displayed:

- Bounding boxes
- Persistent Track IDs

The visualization intentionally excludes:

- Class names
- Confidence scores
- Segmentation masks

This keeps the visualization focused on temporal object identity rather than object detection details.

---

### Deterministic Offline Replay

Validated the tracking pipeline using synchronized offline replay data rather than live CARLA execution.

Completed:

- ROS2 image replay
- Synchronized replay sequences
- Repeatable tracking evaluation
- Frame-by-frame tracker validation

---

## Engineering Outcome

Successfully extended the perception pipeline from frame-level object detection to persistent 2D multi-object tracking using YOLO26m TensorRT FP16 and the official ByteTrack implementation.

The resulting system maintains object identities across consecutive frames with automatic track creation, updating, and termination.

This milestone provides the temporal perception foundation required for:

- 360° multi-object tracking
- 3D object tracking
- Object motion estimation
- Temporal scene understanding

---

## Deliverable

<h3 align="center">2D Multi-Object Tracking</h3>

<p align="center">
  <img src="../../assets/gifs/phase10A_pipeline.gif" width="700"/>
</p>

<p align="center">
Persistent 2D object identities maintained across consecutive frames using YOLO26m TensorRT FP16 and ByteTrack on synchronized offline replay data.
</p>

---

## Technologies

- CARLA 0.9.15
- ROS2 Humble
- YOLO26m
- TensorRT FP16
- ByteTrack
- OpenCV
- NumPy
- Python
- CV Bridge
- CycloneDDS

---

## External Dependency

This phase integrates the official ByteTrack implementation.

The ByteTrack source is included under `ByteTrack/` to preserve the working environment used for the experiments. The original ByteTrack LICENSE and attribution are retained.

---

## Scope

Phase 10A intentionally focuses only on 2D multi-object tracking.

The following are outside the scope of this milestone:

- Velocity estimation
- Trajectory prediction
- Distance smoothing
- Time-To-Collision
- Behavior prediction
- Sensor-fusion improvements
- Planning or control

These capabilities belong to later perception phases.

---

## Outcome

Phase 10A successfully introduced temporal reasoning into the perception stack by integrating YOLO26m TensorRT FP16 with ByteTrack. The system maintains persistent 2D object identities across synchronized offline replay sequences, automatically handling track initialization, updates, and termination. This establishes the temporal perception foundation for subsequent 360° and 3D multi-object tracking.
