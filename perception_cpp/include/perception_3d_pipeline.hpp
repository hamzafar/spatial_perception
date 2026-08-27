#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

class Perception3DPipeline
{
public:

    struct ProjectionResult
    {
        std::vector<int> u;
        std::vector<int> v;
        std::vector<Eigen::Vector3f> ego_points;
    };

    struct ObjectPoint
    {
        int u;
        int v;
        Eigen::Vector3f xyz;
    };

    struct ObjectCloud
    {
        std::vector<ObjectPoint> cloud;
        Eigen::Vector4f box;
        std::string class_name;
    };

    Perception3DPipeline();

    ProjectionResult project_lidar(
        const std::vector<Eigen::Vector3f>& lidar,
        const std::string& camera_name,
        int image_width,
        int image_height
    );

    std::vector<ObjectCloud> extract_object_clouds(
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

private:

    float fx;
    float fy;
    float cx;
    float cy;

    std::unordered_map<std::string, float> camera_yaw;
};