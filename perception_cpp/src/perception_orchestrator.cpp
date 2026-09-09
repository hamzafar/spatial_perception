#include <stdexcept>

#include <pybind11/numpy.h>

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

} // namespace


PerceptionOrchestrator::PerceptionOrchestrator()
{
}


void PerceptionOrchestrator::process(
    const std::vector<Eigen::Vector3f>& lidar,

    cv::Mat& front,
    const py::object& front_masks,
    const std::vector<Eigen::Vector4f>& front_boxes,
    const std::vector<int>& front_classes,
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
    int right_height
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


    // ============================================================
    // 3. Process objects, distance, and track IDs
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

        auto front_objects_cpp =
            front_processed_result.world_objects;

        front_objects_cpp =
            perception_utils.attach_track_ids(
                front_objects_cpp,
                front_targets,
                "F"
            );
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

        auto rear_objects_cpp =
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

        auto left_objects_cpp =
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

        auto right_objects_cpp =
            right_processed_result.world_objects;

        right_objects_cpp =
            perception_utils.attach_track_ids(
                right_objects_cpp,
                right_targets,
                "RT"
            );
    }


    // ============================================================
    // End of current call
    //
    // World objects now have tracker IDs attached inside C++.
    // ============================================================
}