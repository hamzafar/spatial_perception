#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include <Eigen/Dense>

class Perception3DPipeline
{
public:

    Perception3DPipeline();
    
    struct ProjectionResult
    {
        std::vector<int> u;
        std::vector<int> v;
        std::vector<Eigen::Vector3f> ego_points;
    };

    ProjectionResult project_lidar(
        const std::vector<Eigen::Vector3f>& lidar,
        const std::string& camera_name,
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