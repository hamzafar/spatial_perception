#include "perception_pipeline_cpp.hpp"


PerceptionPipelineCpp::PerceptionPipelineCpp()
{
}


// ---------------------------------------------------------
// LiDAR projection
// ---------------------------------------------------------

Perception3DPipeline::ProjectionResult
PerceptionPipelineCpp::project_lidar(
    const std::vector<Eigen::Vector3f>& lidar,
    const std::string& camera_name,
    int image_width,
    int image_height
)
{
    return perception_3d_.project_lidar(
        lidar,
        camera_name,
        image_width,
        image_height
    );
}


// ---------------------------------------------------------
// Object cloud extraction
// ---------------------------------------------------------

std::vector<Perception3DPipeline::ObjectCloud>
PerceptionPipelineCpp::extract_object_clouds(
    const std::vector<std::vector<std::vector<float>>>& masks,
    const std::vector<Eigen::Vector4f>& boxes,
    const std::vector<int>& classes,
    const std::vector<std::string>& class_names,
    const std::vector<int>& u,
    const std::vector<int>& v,
    const std::vector<Eigen::Vector3f>& projected_points,
    int image_width,
    int image_height
)
{
    return perception_3d_.extract_object_clouds(
        masks,
        boxes,
        classes,
        class_names,
        u,
        v,
        projected_points,
        image_width,
        image_height
    );
}


// ---------------------------------------------------------
// Object processing, position and distance
// ---------------------------------------------------------

Perception3DPipeline::ProcessedObjectsResult
PerceptionPipelineCpp::process_object_clouds_and_distance(
    cv::Mat& image,
    const std::vector<Perception3DPipeline::ObjectCloud>& object_clouds,
    const std::string& camera_name
)
{
    return perception_3d_.process_object_clouds_and_distance(
        image,
        object_clouds,
        camera_name
    );
}


// ---------------------------------------------------------
// Track ID attachment
// ---------------------------------------------------------

std::vector<Perception3DPipeline::WorldObject>
PerceptionPipelineCpp::attach_track_ids(
    std::vector<Perception3DPipeline::WorldObject>& world_objects,
    const std::vector<PerceptionUtils::TrackTarget>& online_targets,
    const std::string& camera_prefix
)
{
    return perception_utils_.attach_track_ids(
        world_objects,
        online_targets,
        camera_prefix
    );
}