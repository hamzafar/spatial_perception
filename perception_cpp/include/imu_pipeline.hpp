#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

class IMUPipeline
{
public:

    struct IMUResult
    {
        Eigen::Vector3f acceleration;
        float yaw_rate;
        float heading;
        std::vector<std::string> motion_state;
    };

    IMUPipeline();

    Eigen::Vector3f compute_acceleration(
        float acceleration_x,
        float acceleration_y,
        float acceleration_z
    ) const;

    float compute_yaw_rate(
        float angular_velocity_z
    ) const;

    float compute_heading(
        float orientation_z,
        float orientation_w
    ) const;

    std::vector<std::string> compute_motion_state(
        float speed,
        const Eigen::Vector3f& acceleration,
        float yaw_rate
    ) const;

    IMUResult process(
        float acceleration_x,
        float acceleration_y,
        float acceleration_z,
        float angular_velocity_z,
        float orientation_z,
        float orientation_w,
        float speed
    ) const;

private:

    float speed_threshold;
    float acceleration_threshold;
    float yaw_rate_threshold;
};