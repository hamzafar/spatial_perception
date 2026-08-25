#include "perception_3d_pipeline.hpp"

#include <cmath>
#include <stdexcept>

Perception3DPipeline::Perception3DPipeline()
    : fx(320.0f),
      fy(320.0f),
      cx(320.0f),
      cy(240.0f)
{
    camera_yaw["front"] = 0.0f;
    camera_yaw["rear"]  = 180.0f;
    camera_yaw["left"]  = -90.0f;
    camera_yaw["right"] = 90.0f;
}

Perception3DPipeline::ProjectionResult
Perception3DPipeline::project_lidar(
    const std::vector<Eigen::Vector3f>& lidar,
    const std::string& camera_name,
    int image_width,
    int image_height
)
{
    ProjectionResult result;    
    

    // Camera configuration
    float yaw_deg = camera_yaw.at(camera_name);

    float yaw =
        -yaw_deg * static_cast<float>(M_PI) / 180.0f;

    // Rotation matrix
    Eigen::Matrix3f R;

    R << std::cos(yaw), -std::sin(yaw), 0.0f,
         std::sin(yaw),  std::cos(yaw), 0.0f,
         0.0f,           0.0f,          1.0f;

    // Process each LiDAR point
    for (const auto& ego_point : lidar)
    {
        // LiDAR -> camera frame
        Eigen::Vector3f point = R * ego_point;

        // LiDAR -> camera translation
        point.x() -= 1.5f;

        // CARLA axes -> camera axes
        float X = point.y();
        float Y = -point.z();
        float Z = point.x();

        // Keep points in front
        if (Z <= 0.0f)
            continue;

        // Project to image
        float u = fx * X / Z + cx;
        float v = fy * Y / Z + cy;

        // Keep points inside image
        if (u < 0.0f ||
            u >= static_cast<float>(image_width) ||
            v < 0.0f ||
            v >= static_cast<float>(image_height))
        {
            continue;
        }

        result.u.push_back(static_cast<int>(u));
        result.v.push_back(static_cast<int>(v));

        // Keep original ego-frame point
        result.ego_points.push_back(ego_point);
    }

    return result;
}