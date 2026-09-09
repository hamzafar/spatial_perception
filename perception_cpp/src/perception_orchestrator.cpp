#include <stdexcept>

#include <pybind11/numpy.h>
#include <algorithm>

#include "perception_orchestrator.hpp"

namespace
{

std::vector<std::vector<std::vector<float>>>
convert_masks(
    const py::object& masks,
    const std::string& camera_name)
{
    if (masks.is_none())
    {
        return {};
    }

    py::array_t<float> masks_array =
        masks.cast<py::array_t<float>>();

    auto buffer = masks_array.request();

    if (buffer.ndim != 3)
    {
        throw std::runtime_error(
            camera_name + " masks must have shape (N,H,W)"
        );
    }

    const ssize_t n = buffer.shape[0];
    const ssize_t h = buffer.shape[1];
    const ssize_t w = buffer.shape[2];

    const float* ptr =
        static_cast<const float*>(buffer.ptr);

    std::vector<std::vector<std::vector<float>>> masks_cpp(
        n,
        std::vector<std::vector<float>>(
            h,
            std::vector<float>(w)
        )
    );

    for (ssize_t i = 0; i < n; ++i)
    {
        for (ssize_t y = 0; y < h; ++y)
        {
            for (ssize_t x = 0; x < w; ++x)
            {
                masks_cpp[i][y][x] =
                    ptr[i * h * w + y * w + x];
            }
        }
    }

    return masks_cpp;
}


std::vector<Eigen::Vector4f>
convert_radar_points(
    const py::object& radar_points)
{
    if (radar_points.is_none())
    {
        return {};
    }

    py::array_t<float> radar_array =
        radar_points.cast<py::array_t<float>>();

    auto buffer = radar_array.request();

    if (buffer.ndim != 2 ||
        buffer.shape[1] != 4)
    {
        throw std::runtime_error(
            "radar_points must have shape (N, 4)"
        );
    }

    const float* ptr =
        static_cast<const float*>(buffer.ptr);

    const ssize_t n =
        buffer.shape[0];

    std::vector<Eigen::Vector4f> radar_points_cpp;

    radar_points_cpp.reserve(
        static_cast<size_t>(n)
    );

    for (ssize_t i = 0; i < n; ++i)
    {
        Eigen::Vector4f target;

        target <<
            ptr[i * 4 + 0],
            ptr[i * 4 + 1],
            ptr[i * 4 + 2],
            ptr[i * 4 + 3];

        radar_points_cpp.push_back(target);
    }

    return radar_points_cpp;
}

} // namespace


PerceptionOrchestrator::PerceptionOrchestrator()
{
}

PerceptionOrchestrator::ProcessResult
PerceptionOrchestrator::process(
    const std::vector<Eigen::Vector3f>& lidar,

    cv::Mat& front,
    const py::object& front_masks,
    const std::vector<Eigen::Vector4f>& front_boxes,
    const std::vector<float>& front_scores,
    const std::vector<int>& front_classes,
    const py::object& front_radar_points,
    const std::vector<PerceptionUtils::TrackTarget>& front_targets,

    cv::Mat& rear,
    const py::object& rear_masks,
    const std::vector<Eigen::Vector4f>& rear_boxes,
    const std::vector<int>& rear_classes,
    const std::vector<PerceptionUtils::TrackTarget>& rear_targets,

    cv::Mat& left,
    const py::object& left_masks,
    const std::vector<Eigen::Vector4f>& left_boxes,
    const std::vector<int>& left_classes,
    const std::vector<PerceptionUtils::TrackTarget>& left_targets,

    cv::Mat& right,
    const py::object& right_masks,
    const std::vector<Eigen::Vector4f>& right_boxes,
    const std::vector<int>& right_classes,
    const std::vector<PerceptionUtils::TrackTarget>& right_targets,

    const std::vector<std::string>& class_names,

    int front_width,
    int front_height,
    int rear_width,
    int rear_height,
    int left_width,
    int left_height,
    int right_width,
    int right_height,

    double timestamp,

    double gnss_latitude,
    double gnss_longitude,
    double gnss_altitude,

    float acceleration_x,
    float acceleration_y,
    float acceleration_z,
    float angular_velocity_z,
    float orientation_z,
    float orientation_w
)
{
    // ============================================================
    // 1. LiDAR projection
    // ============================================================

    auto front_result =
        pipeline_3d.project_lidar(
            lidar,
            "front",
            front_width,
            front_height
        );

    auto rear_result =
        pipeline_3d.project_lidar(
            lidar,
            "rear",
            rear_width,
            rear_height
        );

    auto left_result =
        pipeline_3d.project_lidar(
            lidar,
            "left",
            left_width,
            left_height
        );

    auto right_result =
        pipeline_3d.project_lidar(
            lidar,
            "right",
            right_width,
            right_height
        );


    // ============================================================
    // 2. Extract object clouds
    // ============================================================

    std::vector<Perception3DPipeline::ObjectCloud> front_clouds;
    std::vector<Perception3DPipeline::ObjectCloud> rear_clouds;
    std::vector<Perception3DPipeline::ObjectCloud> left_clouds;
    std::vector<Perception3DPipeline::ObjectCloud> right_clouds;

    // Camera-level WorldObject results.
    // These are built internally from each camera and combined later.
    std::vector<Perception3DPipeline::WorldObject> front_objects_cpp;
    std::vector<Perception3DPipeline::WorldObject> rear_objects_cpp;
    std::vector<Perception3DPipeline::WorldObject> left_objects_cpp;
    std::vector<Perception3DPipeline::WorldObject> right_objects_cpp;


    // ============================================================
    // 3. Process objects, distance, track IDs, and radar
    // ============================================================

    // ------------------------------------------------------------
    // Front
    // ------------------------------------------------------------

    if (!front_masks.is_none())
    {
        auto masks_cpp =
            convert_masks(front_masks, "front");

        front_clouds =
            pipeline_3d.extract_object_clouds(
                masks_cpp,
                front_boxes,
                front_classes,
                class_names,
                front_result.u,
                front_result.v,
                front_result.ego_points,
                front_width,
                front_height
            );

        auto front_processed_result =
            pipeline_3d.process_object_clouds_and_distance(
                front,
                front_clouds,
                "front"
            );

        front = front_processed_result.image;

        front_objects_cpp =
            front_processed_result.world_objects;


        // --------------------------------------------------------
        // Track IDs
        // --------------------------------------------------------

        front_objects_cpp =
            perception_utils.attach_track_ids(
                front_objects_cpp,
                front_targets,
                "F"
            );


        // --------------------------------------------------------
        // Radar
        // --------------------------------------------------------

        if (!front_radar_points.is_none())
        {
            auto radar_points_cpp =
                convert_radar_points(
                    front_radar_points
                );

            auto front_radar_objects =
                pipeline_radar.process(
                    front_boxes,
                    front_scores,
                    front_classes,
                    radar_points_cpp
                );

            if (!front_objects_cpp.empty() &&
                !front_radar_objects.empty())
            {
                front_objects_cpp =
                    perception_utils.attach_radar_data(
                        front_objects_cpp,
                        front_radar_objects,
                        0.3f
                    );
            }
        }
    }


    // ------------------------------------------------------------
    // Rear
    // ------------------------------------------------------------

    if (!rear_masks.is_none())
    {
        auto masks_cpp =
            convert_masks(rear_masks, "rear");

        rear_clouds =
            pipeline_3d.extract_object_clouds(
                masks_cpp,
                rear_boxes,
                rear_classes,
                class_names,
                rear_result.u,
                rear_result.v,
                rear_result.ego_points,
                rear_width,
                rear_height
            );

        auto rear_processed_result =
            pipeline_3d.process_object_clouds_and_distance(
                rear,
                rear_clouds,
                "rear"
            );

        rear = rear_processed_result.image;

        rear_objects_cpp =
            rear_processed_result.world_objects;

        rear_objects_cpp =
            perception_utils.attach_track_ids(
                rear_objects_cpp,
                rear_targets,
                "R"
            );
    }


    // ------------------------------------------------------------
    // Left
    // ------------------------------------------------------------

    if (!left_masks.is_none())
    {
        auto masks_cpp =
            convert_masks(left_masks, "left");

        left_clouds =
            pipeline_3d.extract_object_clouds(
                masks_cpp,
                left_boxes,
                left_classes,
                class_names,
                left_result.u,
                left_result.v,
                left_result.ego_points,
                left_width,
                left_height
            );

        auto left_processed_result =
            pipeline_3d.process_object_clouds_and_distance(
                left,
                left_clouds,
                "left"
            );

        left = left_processed_result.image;

        left_objects_cpp =
            left_processed_result.world_objects;

        left_objects_cpp =
            perception_utils.attach_track_ids(
                left_objects_cpp,
                left_targets,
                "L"
            );
    }


    // ------------------------------------------------------------
    // Right
    // ------------------------------------------------------------

    if (!right_masks.is_none())
    {
        auto masks_cpp =
            convert_masks(right_masks, "right");

        right_clouds =
            pipeline_3d.extract_object_clouds(
                masks_cpp,
                right_boxes,
                right_classes,
                class_names,
                right_result.u,
                right_result.v,
                right_result.ego_points,
                right_width,
                right_height
            );

        auto right_processed_result =
            pipeline_3d.process_object_clouds_and_distance(
                right,
                right_clouds,
                "right"
            );

        right = right_processed_result.image;

        right_objects_cpp =
            right_processed_result.world_objects;

        right_objects_cpp =
            perception_utils.attach_track_ids(
                right_objects_cpp,
                right_targets,
                "RT"
            );
    }


    // --------------------------------------------------
    // Combine all camera world objects
    // --------------------------------------------------

    std::vector<Perception3DPipeline::WorldObject> world_objects;

    world_objects.insert(
        world_objects.end(),
        front_objects_cpp.begin(),
        front_objects_cpp.end()
    );

    world_objects.insert(
        world_objects.end(),
        rear_objects_cpp.begin(),
        rear_objects_cpp.end()
    );

    world_objects.insert(
        world_objects.end(),
        left_objects_cpp.begin(),
        left_objects_cpp.end()
    );

    world_objects.insert(
        world_objects.end(),
        right_objects_cpp.begin(),
        right_objects_cpp.end()
    );


    // --------------------------------------------------
    // Keep only objects with track IDs
    // --------------------------------------------------

    world_objects.erase(
        std::remove_if(
            world_objects.begin(),
            world_objects.end(),
            [](const Perception3DPipeline::WorldObject& obj)
            {
                return obj.id.empty();
            }
        ),
        world_objects.end()
    );

    auto object_counts =
    perception_utils.count_objects(world_objects);

    auto nearest_objects =
    perception_utils.prepare_nearest_objects(
        world_objects,
        timestamp);

    auto bev_objects =
    perception_utils.prepare_bev_objects(
        world_objects);

    auto gnss =
    pipeline_gnss.process(
        gnss_latitude,
        gnss_longitude,
        gnss_altitude,
        timestamp
    );

    auto imu =
    pipeline_imu.process(
        acceleration_x,
        acceleration_y,
        acceleration_z,
        angular_velocity_z,
        orientation_z,
        orientation_w,
        gnss.speed
    );

    constexpr float RAD_TO_DEG =
    180.0f / 3.14159265358979323846f;

    float heading_deg =
        imu.heading * RAD_TO_DEG;

    return {
        front,
        rear,
        left,
        right,

        object_counts,
        nearest_objects,
        bev_objects,

        gnss,
        imu,

        heading_deg
    };
}