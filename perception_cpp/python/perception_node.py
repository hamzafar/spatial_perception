import cv2
import rclpy
import numpy as np
import time
import base64

from cv_bridge import CvBridge
from rclpy.node import Node

from message_filters import (
    Subscriber,
    TimeSynchronizer,
)
from sensor_msgs.msg import (
    Image,
    PointCloud2,
    Imu,
    NavSatFix
)
from sensor_msgs_py import point_cloud2

from ultralytics import YOLO

from dashboard_bridge import DashboardBridge
from dashboard_metrics import DashboardMetrics
from yolo_detection_pipeline import DetectionPipeline
from dashboard_adapter import DashboardAdapter
from gnss_pipeline import GNSSPipeline
from imu_pipeline import IMUPipeline
from tracking_pipeline import TrackingPipeline
from perception_3d_pipeline import Perception3DPipeline
from perception_utils import PerceptionUtils
from radar_perception_pipeline import RadarPerceptionPipeline
from recorder_perception import PerceptionRecorder

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent

BYTE_TRACK_ROOT = Path.home() / "ros2_cv_ws/scripts/third_party/ByteTrack"
sys.path.insert(0, str(BYTE_TRACK_ROOT))

from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.basetrack import BaseTrack

CPP_BUILD = ROOT / "build"
sys.path.insert(0, str(CPP_BUILD))

import perception_cpp

class PerceptionStack(Node):

    def __init__(self):

        super().__init__("perception_node")

        self.bridge = CvBridge()

        self.image_width = 640
        self.image_height = 480
        self.fx = 320.0
        self.fy = 320.0
        self.cx = 320.0
        self.cy = 240.0

        self.model = YOLO("/home/hamza/ros2_cv_ws/models/yolo26m-seg.engine")
        # self.model =  YOLO("/home/hamza/ros2_cv_ws/scripts/phase6/engines/yolov8m-seg-trt-int8.engine")
        self.previous_frame = None
        self.trajectory_reset = True


        # Initialize dashboard bridge
        self.dashboard = DashboardBridge()
        self.dashboard.start()

        # Initialize dashboard metrics
        self.metrics = DashboardMetrics()

        # Initialize detection pipeline
        self.pipeline_detection = DetectionPipeline()

        self.pipeline_detection.bridge = self.bridge
        self.pipeline_detection.model = self.model

        self.pipeline_detection.validate()

        # Initialize dashboard adapter
        self.dashboard_adapter = DashboardAdapter(self.pipeline_detection.model.names)

        # Initialize Imu and Gnss pipeline
        self.gnss_pipeline = GNSSPipeline()
        self.imu_pipeline = IMUPipeline()

        # Initialize tracking pipeline
        self.pipeline_tracking = TrackingPipeline()
        
        # Initialize ByteTrack
        self.tracker_args = SimpleNamespace(track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False,)

        self.front_tracker = BYTETracker(self.tracker_args, frame_rate=10)
        self.rear_tracker = BYTETracker(self.tracker_args, frame_rate=10)
        self.left_tracker = BYTETracker(self.tracker_args, frame_rate=10)
        self.right_tracker = BYTETracker(self.tracker_args, frame_rate=10)

        # Initialize cpp 3D perception pipeline
        self.pipeline_3d_cpp = perception_cpp.Perception3DPipeline()

        # Initialize 3D perception pipeline
        self.pipeline_3d = Perception3DPipeline()
        self.pipeline_3d.bridge = self.bridge
        self.pipeline_3d.model = self.model
        self.cameras = {"front": {"yaw": 0}, "rear": {"yaw": 180},"left": {"yaw": -90}, "right": {"yaw": 90},}
        self.pipeline_3d.cameras = self.cameras
        self.pipeline_3d.fx = self.fx
        self.pipeline_3d.fy = self.fy
        self.pipeline_3d.cx = self.cx
        self.pipeline_3d.cy = self.cy
        self.pipeline_3d.validate()

        # Initialize perception utils
        self.perception_utils = PerceptionUtils()

        # Initialize radar perception pipeline
        self.pipeline_radar = RadarPerceptionPipeline()
        self.pipeline_radar.velocity_threshold = 0.5
        self.pipeline_radar.max_assoc_distance_px = 50
        self.pipeline_radar.fx = self.fx
        self.pipeline_radar.fy = self.fy
        self.pipeline_radar.cx = self.cx
        self.pipeline_radar.cy = self.cy
        self.pipeline_radar.width = self.image_width
        self.pipeline_radar.height = self.image_height

        R = np.array([
            [0.0,  1.0,  0.0],
            [0.0,  0.0, -1.0],
            [1.0,  0.0,  0.0],
        ], dtype=np.float32)

        t = np.array([
            0.0,
            1.4,
            3.5,
        ], dtype=np.float32)

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t

        self.pipeline_radar.T_radar_to_cam = T

        # Initialize perception recorder
        self.recorder = PerceptionRecorder(output_root="/home/hamza/ros2_cv_ws/scripts/phase12/perception_recordings")

        # Sensor Subsription
        self.front_sub = Subscriber(self, Image, "/carla/ego_vehicle/rgb_front/image")
        self.rear_sub = Subscriber(self, Image, "/carla/ego_vehicle/rgb_rear/image")
        self.left_sub = Subscriber(self, Image, "/carla/ego_vehicle/rgb_left/image")
        self.right_sub = Subscriber(self, Image, "/carla/ego_vehicle/rgb_right/image")

        self.radar_sub = Subscriber(self, PointCloud2, "/carla/ego_vehicle/radar")
        self.lidar_sub = Subscriber(self, PointCloud2, "/carla/ego_vehicle/lidar")

        self.imu_sub = Subscriber(self, Imu, "/carla/ego_vehicle/imu")
        self.gnss_sub = Subscriber(self, NavSatFix, "/carla/ego_vehicle/gnss")

        self.sync = TimeSynchronizer(
            [
                self.front_sub,
                self.rear_sub,
                self.left_sub,
                self.right_sub,
                self.radar_sub,
                self.lidar_sub,
                self.imu_sub,
                self.gnss_sub
            ],
            queue_size=10
        )

        self.sync.registerCallback(
            self.synchronized_callback
        )

        self.get_logger().info(
            "Perception Stack node started."
        )

    # -------------------------------------------------

    def synchronized_callback(
        self,
        front_msg,
        rear_msg,
        left_msg,
        right_msg,
        radar_msg,
        lidar_msg,
        imu_msg,
        gnss_msg
    ): 
        start_time = time.perf_counter()

        # -------------------------------------------------
        # Detect replay / trajectory reset
        # -------------------------------------------------

        current_frame = int(front_msg.header.frame_id.split("_")[-1])

        self.trajectory_reset = False

        if self.previous_frame is None:
            self.trajectory_reset = True

        elif current_frame < self.previous_frame:
            self.trajectory_reset = True

        if self.trajectory_reset:
            self.reset_trackers()

        self.previous_frame = current_frame

        # Detection
        front = self.pipeline_detection.convert_ros_to_cv(front_msg)
        front, front_results = self.pipeline_detection.detect(front)
        front_boxes, front_scores, front_classes = self.pipeline_detection.extract_detections(front_results)

        rear = self.pipeline_detection.convert_ros_to_cv(rear_msg)
        rear, rear_results = self.pipeline_detection.detect(rear)
        rear_boxes, rear_scores, rear_classes = self.pipeline_detection.extract_detections(rear_results)

        left = self.pipeline_detection.convert_ros_to_cv(left_msg)
        left, left_results = self.pipeline_detection.detect(left)
        left_boxes, left_scores, left_classes = self.pipeline_detection.extract_detections(left_results)

        right = self.pipeline_detection.convert_ros_to_cv(right_msg)
        right, right_results = self.pipeline_detection.detect(right)
        right_boxes, right_scores, right_classes = self.pipeline_detection.extract_detections(right_results)

        # front_camera = self.dashboard_adapter.prepare_camera(front, front_boxes, front_scores, front_classes)
        # rear_camera = self.dashboard_adapter.prepare_camera(rear, rear_boxes, rear_scores, rear_classes)
        # left_camera = self.dashboard_adapter.prepare_camera(left, left_boxes, left_scores, left_classes)
        # right_camera = self.dashboard_adapter.prepare_camera(right, right_boxes, right_scores, right_classes)

        # Tracking
        front_dets = np.concatenate([front_boxes, front_scores.reshape(-1, 1)], axis=1)
        front_targets = self.front_tracker.update(front_dets, (self.image_height, self.image_width), 
            (self.image_height, self.image_width))

        rear_dets = np.concatenate([rear_boxes, rear_scores.reshape(-1, 1)], axis=1)
        rear_targets = self.rear_tracker.update(rear_dets, (self.image_height, self.image_width), 
            (self.image_height, self.image_width))

        left_dets = np.concatenate([left_boxes, left_scores.reshape(-1, 1)], axis=1)
        left_targets = self.left_tracker.update(left_dets, (self.image_height, self.image_width), 
            (self.image_height, self.image_width))

        right_dets = np.concatenate([right_boxes, right_scores.reshape(-1, 1)], axis=1)
        right_targets = self.right_tracker.update(right_dets, (self.image_height, self.image_width), 
            (self.image_height, self.image_width))

        front_camera = self.dashboard_adapter.prepare_tracked_camera(front, front_targets, front_boxes,
            front_scores, front_classes, self.image_width, self.image_height, "F")
        rear_camera = self.dashboard_adapter.prepare_tracked_camera(rear, rear_targets, rear_boxes,
            rear_scores, rear_classes, self.image_width, self.image_height, "R")
        left_camera = self.dashboard_adapter.prepare_tracked_camera(left, left_targets, left_boxes,
            left_scores, left_classes, self.image_width, self.image_height, "L")
        right_camera = self.dashboard_adapter.prepare_tracked_camera(right, right_targets, right_boxes,
            right_scores, right_classes, self.image_width, self.image_height, "RT")

        # BEV(Lidar)
        lidar = self.pipeline_3d.convert_ros_to_numpy(lidar_msg)

        # front, front_u, front_v, front_projected = self.pipeline_3d.project_lidar(front, lidar, "front")
        ## C++ LiDAR projection Front 
        result = self.pipeline_3d_cpp.project_lidar(lidar, "front", front.shape[1], front.shape[0])
        front_u, front_v, front_projected = result.u, result.v, result.ego_points
    
        # rear, rear_u, rear_v, rear_projected = self.pipeline_3d.project_lidar(rear, lidar, "rear")
        ## C++ LiDAR projection Rear
        result = self.pipeline_3d_cpp.project_lidar(lidar, "rear", rear.shape[1], rear.shape[0])
        rear_u, rear_v, rear_projected = result.u, result.v, result.ego_points
    
        # left, left_u, left_v, left_projected = self.pipeline_3d.project_lidar(left, lidar, "left")
        ## C++ LiDAR projection Left
        result = self.pipeline_3d_cpp.project_lidar(lidar, "left", left.shape[1], left.shape[0])
        left_u, left_v, left_projected = result.u, result.v, result.ego_points   

        # right, right_u, right_v, right_projected = self.pipeline_3d.project_lidar(right, lidar, "right")
        ## C++ LiDAR projection Left
        result = self.pipeline_3d_cpp.project_lidar(lidar, "right", right.shape[1], right.shape[0])
        right_u, right_v, right_projected = result.u, result.v, result.ego_points   
        
        # self.validate_cpp_projection(python_u,python_v,python_projected,cpp_u,cpp_v,cpp_projected,tolerance=1e-5)


        # front_clouds = self.pipeline_3d.extract_object_clouds(front, front_u, front_v, front_projected, front_results)
        ## C++ extract object cluouds Front
        if front_results.masks is None:
            front_clouds = []
        else:
            masks = front_results.masks.data.cpu().numpy()
            front_clouds_cpp = self.pipeline_3d_cpp.extract_object_clouds(masks,front_boxes,front_classes,self.model.names,
                front_u,front_v,front_projected,front.shape[1],front.shape[0])
            # front_clouds = self.convert_cloud_cpp_py(front_clouds_cpp)

        # rear_clouds = self.pipeline_3d.extract_object_clouds(rear, rear_u, rear_v, rear_projected, rear_results)
        ## C++ extract object cluouds Rear
        if rear_results.masks is None:
            rear_clouds = []
        else:
            masks = rear_results.masks.data.cpu().numpy()
            rear_clouds_cpp = self.pipeline_3d_cpp.extract_object_clouds(masks,rear_boxes,rear_classes,self.model.names,
                rear_u,rear_v,rear_projected,rear.shape[1],rear.shape[0])
            # rear_clouds = self.convert_cloud_cpp_py(rear_clouds_cpp)

        # left_clouds = self.pipeline_3d.extract_object_clouds(left, left_u, left_v, left_projected, left_results)
        ## C++ extract object cluouds Left
        if left_results.masks is None:
            left_clouds = []
        else:
            masks = left_results.masks.data.cpu().numpy()
            left_clouds_cpp = self.pipeline_3d_cpp.extract_object_clouds(masks,left_boxes,left_classes,self.model.names,
                left_u,left_v,left_projected,left.shape[1],left.shape[0])
            # left_clouds = self.convert_cloud_cpp_py(left_clouds_cpp)

        # right_clouds = self.pipeline_3d.extract_object_clouds(right, right_u, right_v, right_projected, right_results)
        ## C++ extract object cluouds Right
        if right_results.masks is None:
            right_clouds = []
        else:
            masks = right_results.masks.data.cpu().numpy()
            right_clouds_cpp = self.pipeline_3d_cpp.extract_object_clouds(masks,right_boxes,right_classes,self.model.names,
                right_u,right_v,right_projected,right.shape[1],right.shape[0])
            # right_clouds = self.convert_cloud_cpp_py(right_clouds_cpp)

            
        # front, front_objects = self.pipeline_3d.process_object_clouds_and_distance(front, front_clouds, "front")
        ## C++ clouds and distances Front
        front_objects = []
        if front_results.masks is not None:
            front, front_objects_cpp = self.pipeline_3d_cpp.process_object_clouds_and_distance(front, front_clouds_cpp, "front")
            front_objects = self.convert_world_objects_cpp_py(front_objects_cpp)

        # rear, rear_objects = self.pipeline_3d.process_object_clouds_and_distance(rear, rear_clouds, "rear")
        ## C++ clouds and distances rear
        rear_objects = []
        if rear_results.masks is not None:
            rear, rear_objects_cpp = self.pipeline_3d_cpp.process_object_clouds_and_distance(rear, rear_clouds_cpp, "rear")
            rear_objects = self.convert_world_objects_cpp_py(rear_objects_cpp)

        # left, left_objects = self.pipeline_3d.process_object_clouds_and_distance(left, left_clouds, "left")
        ## C++ clouds and distances left
        left_objects = []
        if left_results.masks is not None:
            left, left_objects_cpp = self.pipeline_3d_cpp.process_object_clouds_and_distance(left, left_clouds_cpp, "left")
            left_objects = self.convert_world_objects_cpp_py(left_objects_cpp)

        # right, right_objects = self.pipeline_3d.process_object_clouds_and_distance(right, right_clouds, "right")
        ## C++ clouds and distances right
        right_objects = []
        if right_results.masks is not None:
            right, right_objects_cpp = self.pipeline_3d_cpp.process_object_clouds_and_distance(right, right_clouds_cpp, "right")
            right_objects = self.convert_world_objects_cpp_py(right_objects_cpp)


        front_objects = self.perception_utils.attach_track_ids(front_objects, front_targets, "F")
        rear_objects = self.perception_utils.attach_track_ids(rear_objects, rear_targets, "R")
        left_objects = self.perception_utils.attach_track_ids(left_objects, left_targets, "L")
        right_objects =self.perception_utils.attach_track_ids(right_objects, right_targets, "RT")

        # World Objects Motion (Radar)
        radar_points = self.pipeline_radar.convert_ros_to_numpy(radar_msg)
        front_radar_objects = self.pipeline_radar.process(front_boxes, front_scores, front_classes, radar_points)
        front_objects = self.perception_utils.attach_radar_data(front_objects, front_radar_objects)

        world_objects = (front_objects + rear_objects + left_objects + right_objects)
        world_objects = [obj for obj in world_objects if "id" in obj]

        object_counts = self.perception_utils.count_objects(world_objects)

        timestamp = (gnss_msg.header.stamp.sec + gnss_msg.header.stamp.nanosec * 1e-9)
        nearest_objects = self.perception_utils.prepare_nearest_objects(world_objects, timestamp)

        bev_objects = self.perception_utils.prepare_bev_objects(world_objects)

               
        # Ego Vehicle Motion
        timestamp = (gnss_msg.header.stamp.sec + gnss_msg.header.stamp.nanosec * 1e-9)
        gnss = self.gnss_pipeline.process(gnss_msg.latitude, gnss_msg.longitude, gnss_msg.altitude, timestamp)
        imu = self.imu_pipeline.process(imu_msg, gnss["speed"])
        heading_deg = np.degrees(imu["heading"])


        # (REMEMBER) latency is calcualte just before dashboard.push
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        metrics = self.metrics.get_metrics(
            latency_ms
        )


        dashboard_data = {
            "sensors": {
                "cam": True,
                "radar": True,
                "gnss": True,
                "imu": True,
                "lidar": True
            },

            "frame_idx": front_msg.header.stamp.nanosec,

            "fps": metrics["fps"],
            "latency_ms": metrics["latency_ms"],
            "gpu_pct": metrics["gpu_pct"],
            "cpu_pct": metrics["cpu_pct"],

            "objects_count": object_counts,

            "trajectory_reset": self.trajectory_reset,

            "ego": {
                "heading_deg": float(heading_deg),
                "speed_mps": float(gnss["speed"]),

                "accelerating": "Accelerating" in imu["motion_state"],
                "braking": "Braking" in imu["motion_state"],

                "turning_left": "Turning Left" in imu["motion_state"],
                "turning_right": "Turning Right" in imu["motion_state"],

                "world_x": float(gnss["position"][0]),
                "world_y": float(gnss["position"][1]),
            },

            "cameras": {
                "front": front_camera,
                "left": left_camera,
                "rear": rear_camera,
                "right": right_camera,
            },

            "bev_objects": bev_objects,
            "nearest_objects": nearest_objects
        }

        self.dashboard.push(dashboard_data)
        # print(f"Latency: {latency_ms}: ms")

        # self.recorder.record(dashboard_data)


    def convert_world_objects_cpp_py(self, cpp_world_objects):
        world_objects = []

        for obj in cpp_world_objects:
            world_objects.append(
                {
                    "class": obj.class_name,
                    "camera": obj.camera,
                    "box": obj.box,
                    "position": obj.position,
                    "distance": obj.distance
                }
            )

        return world_objects

    def convert_cloud_cpp_py(self, cpp_cloud):
        py_clouds = [
            {
                "cloud": [
                    {
                        "pixel": (p.u, p.v),
                        "xyz": p.xyz
                    }
                    for p in obj.cloud
                ],
                "box": obj.box,
                "class_name": obj.class_name
            }
            for obj in cpp_cloud
        ]

        return py_clouds



    def reset_trackers(self):
        
        BaseTrack._count = 0

        self.front_tracker = BYTETracker(
            self.tracker_args,
            frame_rate=10
        )

        self.rear_tracker = BYTETracker(
            self.tracker_args,
            frame_rate=10
        )

        self.left_tracker = BYTETracker(
            self.tracker_args,
            frame_rate=10
        )

        self.right_tracker = BYTETracker(
            self.tracker_args,
            frame_rate=10
        )

        self.get_logger().info(
            "ByteTrack trackers reset."
        )



def main():

    rclpy.init()

    node = PerceptionStack()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    cv2.destroyAllWindows()

    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":

    main()