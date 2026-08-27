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
std::vector<Perception3DPipeline::ObjectCloud>
Perception3DPipeline::extract_object_clouds(
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
    std::vector<ObjectCloud> object_clouds;

    for (size_t object_id = 0; object_id < masks.size(); ++object_id)
    {
        // Resize mask to image resolution
        const auto& mask_data = masks[object_id];

        if (mask_data.empty() || mask_data[0].empty())
        {
            continue;
        }

        int mask_height = static_cast<int>(mask_data.size());
        int mask_width = static_cast<int>(mask_data[0].size());

        cv::Mat mask(
            mask_height,
            mask_width,
            CV_32FC1
        );

        for (int y = 0; y < mask_height; ++y)
        {
            for (int x = 0; x < mask_width; ++x)
            {
                mask.at<float>(y, x) = mask_data[y][x];
            }
        }

        cv::Mat mask_uint8;
        mask.convertTo(mask_uint8, CV_8U);

        cv::Mat resized_mask;

        cv::resize(
            mask_uint8,
            resized_mask,
            cv::Size(image_width, image_height),
            0.0,
            0.0,
            cv::INTER_NEAREST
        );

        // Class
        int class_id = classes[object_id];

        std::string class_name = "unknown";

        if (class_id >= 0 &&
            class_id < static_cast<int>(class_names.size()))
        {
            class_name = class_names[class_id];
        }

        // Create object cloud
        ObjectCloud object;

        for (size_t i = 0; i < u.size(); ++i)
        {
            int x = u[i];
            int y = v[i];

            if (x < 0 || x >= image_width ||
                y < 0 || y >= image_height)
            {
                continue;
            }

            if (resized_mask.at<uint8_t>(y, x) > 0)
            {
                ObjectPoint point;

                point.u = x;
                point.v = y;
                point.xyz = projected_points[i];

                object.cloud.push_back(point);
            }
        }

        // Object metadata
        object.box = boxes[object_id];
        object.class_name = class_name;

        object_clouds.push_back(object);
    }

    return object_clouds;
}