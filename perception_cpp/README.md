# Porting Perception Core to C++

## Perception Stack(Python) Flow Diagram:
```
                 CARLA / Dataset Replay
                          │
                          ▼
                   ROS2 Sensor Streams
                          │
                          ▼
                  TimeSynchronizer
                          │
                          ▼
              ┌────────────────────────┐
              │    PYTHON PERCEPTION   │
              │                        │
              │ DetectionPipeline      │
              │ YOLO26m-seg / TRT      │
              │                        │
              │ ByteTrack × 4          │
              │                        │
              │ Perception3DPipeline   │
              │ LiDAR → 3D Objects     │
              │                        │
              │ RadarPerception        │
              │ Radar → Motion         │
              │                        │
              │ GNSS / IMU             │
              │ Ego Motion             │
              │                        │
              │ PerceptionUtils        │
              │ Association / BEV      │
              └───────────┬────────────┘
                          │
                          ▼
                    World Objects
                          │
                          ▼
                  Dashboard Adapter
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
       Dashboard Bridge        Perception Recorder
              │                       │
              ▼                       ▼
        Web Dashboard            CSV / Replay
```

## Desired Porting in C++

```
                 EXISTING PYTHON STACK
                          │
                          │ detections
                          │ tracks
                          │ LiDAR / Radar data
                          │ ego state
                          ▼
                       pybind11
                          │
                          ▼
              ┌────────────────────────┐
              │       C++ CORE         │
              │                        │
              │ Kalman Filter          │
              │ Multi-Object Tracking  │
              │ Data Association       │
              │ Sensor Fusion           │
              │ State Estimation       │
              └───────────┬────────────┘
                          │
                          ▼
                    Fused Tracks
                          │
                          ▼
                 Existing Python
                    Dashboard
```