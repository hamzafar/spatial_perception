#include "perception_3d_pipeline.hpp"

#include <cmath>
#include <stdexcept>
#include <algorithm>

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
    object_clouds.reserve(masks.size());

    for (size_t object_id = 0; object_id < masks.size(); ++object_id)
    {
        const auto& mask_data = masks[object_id];

        if (mask_data.empty() || mask_data[0].empty())
        {
            continue;
        }

        int mask_height = static_cast<int>(mask_data.size());
        int mask_width = static_cast<int>(mask_data[0].size());

        // Build uint8 mask directly, row by row, via raw pointers
        // (skips the CV_32FC1 intermediate + convertTo pass)
        cv::Mat mask_uint8(mask_height, mask_width, CV_8UC1);

        for (int y = 0; y < mask_height; ++y)
        {
            const float* src = mask_data[y].data();
            uint8_t* dst = mask_uint8.ptr<uint8_t>(y);

            for (int x = 0; x < mask_width; ++x)
            {
                dst[x] = static_cast<uint8_t>(src[x]);
            }
        }

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
        object.cloud.reserve(u.size());

        for (size_t i = 0; i < u.size(); ++i)
        {
            int x = u[i];
            int y = v[i];

            if (x < 0 || x >= image_width ||
                y < 0 || y >= image_height)
            {
                continue;
            }

            if (resized_mask.ptr<uint8_t>(y)[x] > 0)
            {
                ObjectPoint point;
                point.u = x;
                point.v = y;
                point.xyz = projected_points[i];

                object.cloud.push_back(std::move(point));
            }
        }

        object.box = boxes[object_id];
        object.class_name = class_name;

        object_clouds.push_back(std::move(object));
    }

    return object_clouds;
}

Perception3DPipeline::ProcessedObjectsResult
Perception3DPipeline::process_object_clouds_and_distance(
    cv::Mat& image,
    const std::vector<ObjectCloud>& object_clouds,
    const std::string& camera_name
)
{
    ProcessedObjectsResult result;

    result.image = image.clone();

    for (const auto& obj : object_clouds)
    {
        // --------------------------------------------------
        // Draw object cloud
        // --------------------------------------------------

        for (const auto& point : obj.cloud)
        {
            cv::circle(
                result.image,
                cv::Point(point.u, point.v),
                2,
                cv::Scalar(0, 255, 0),
                -1
            );
        }


        // --------------------------------------------------
        // No LiDAR points
        // --------------------------------------------------

        if (obj.cloud.empty())
        {
            continue;
        }


        // --------------------------------------------------
        // Calculate distances
        // --------------------------------------------------

        std::vector<float> distances;

        distances.reserve(obj.cloud.size());

        for (const auto& point : obj.cloud)
        {
            distances.push_back(point.xyz.norm());
        }


        // --------------------------------------------------
        // 10th percentile
        // --------------------------------------------------

        std::vector<float> sorted_distances = distances;

        std::sort(
            sorted_distances.begin(),
            sorted_distances.end()
        );

        size_t percentile_index =
            static_cast<size_t>(
                0.10f * sorted_distances.size()
            );

        if (percentile_index >= sorted_distances.size())
        {
            percentile_index =
                sorted_distances.size() - 1;
        }

        float threshold =
            sorted_distances[percentile_index];

        float distance = threshold;


        // --------------------------------------------------
        // Front-most points
        // --------------------------------------------------

        std::vector<Eigen::Vector3f> front_points;

        for (size_t i = 0; i < obj.cloud.size(); ++i)
        {
            if (distances[i] <= threshold)
            {
                front_points.push_back(
                    obj.cloud[i].xyz
                );
            }
        }


        // --------------------------------------------------
        // Fallback
        // --------------------------------------------------

        if (front_points.empty())
        {
            for (const auto& point : obj.cloud)
            {
                front_points.push_back(point.xyz);
            }
        }


        // --------------------------------------------------
        // Mean position
        // --------------------------------------------------

        Eigen::Vector3f position =
            Eigen::Vector3f::Zero();

        for (const auto& point : front_points)
        {
            position += point;
        }

        position /= static_cast<float>(
            front_points.size()
        );


        // --------------------------------------------------
        // Create world object
        // --------------------------------------------------

        WorldObject world_object;

        world_object.class_name = obj.class_name;
        world_object.camera = camera_name;
        world_object.box = obj.box;
        world_object.position = position;
        world_object.distance = distance;

        result.world_objects.push_back(
            world_object
        );
    }

    return result;
}