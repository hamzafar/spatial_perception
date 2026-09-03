#pragma once

#include <string>
#include <vector>

#include "perception_3d_pipeline.hpp"
#include "perception_utils.hpp"

class PerceptionPipelineCpp
{
public:

    PerceptionPipelineCpp();

    // LiDAR projection
    Perception3DPipeline::ProjectionResult project_lidar(
        const std::vector<Eigen::Vector3f>& lidar,
        const std::string& camera_name,
        int image_width,
        int image_height
    );

    // Object cloud extraction
    std::vector<Perception3DPipeline::ObjectCloud> extract_object_clouds(
        const std::vector<std::vector<std::vector<float>>>& masks,
        const std::vector<Eigen::Vector4f>& boxes,
        const std::vector<int>& classes,
        const std::vector<std::string>& class_names,
        const std::vector<int>& u,
        const std::vector<int>& v,
        const std::vector<Eigen::Vector3f>& projected_points,
        int image_width,
        int image_height
    );

    // Object processing, position and distance
    Perception3DPipeline::ProcessedObjectsResult
    process_object_clouds_and_distance(
        cv::Mat& image,
        const std::vector<Perception3DPipeline::ObjectCloud>& object_clouds,
        const std::string& camera_name
    );

    // Track ID attachment
    std::vector<Perception3DPipeline::WorldObject> attach_track_ids(
        std::vector<Perception3DPipeline::WorldObject>& world_objects,
        const std::vector<PerceptionUtils::TrackTarget>& online_targets,
        const std::string& camera_prefix
    );

private:

    Perception3DPipeline perception_3d_;
    PerceptionUtils perception_utils_;
};