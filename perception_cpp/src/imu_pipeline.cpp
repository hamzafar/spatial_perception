#include "imu_pipeline.hpp"

#include <cmath>


IMUPipeline::IMUPipeline()
    : speed_threshold(0.5f),
      acceleration_threshold(0.3f),
      yaw_rate_threshold(0.1f)
{
}


Eigen::Vector3f IMUPipeline::compute_acceleration(
    float acceleration_x,
    float acceleration_y,
    float acceleration_z
) const
{
    return Eigen::Vector3f(
        acceleration_x,
        acceleration_y,
        acceleration_z
    );
}


float IMUPipeline::compute_yaw_rate(
    float angular_velocity_z
) const
{
    return angular_velocity_z;
}


float IMUPipeline::compute_heading(
    float orientation_z,
    float orientation_w
) const
{
    return 2.0f * std::atan2(
        orientation_z,
        orientation_w
    );
}


std::vector<std::string>
IMUPipeline::compute_motion_state(
    float speed,
    const Eigen::Vector3f& acceleration,
    float yaw_rate
) const
{
    std::vector<std::string> state;

    if (speed < speed_threshold)
    {
        state.push_back("Stationary");
    }
    else
    {
        state.push_back("Moving Forward");
    }

    if (acceleration[0] > acceleration_threshold)
    {
        state.push_back("Accelerating");
    }
    else if (acceleration[0] < -acceleration_threshold)
    {
        state.push_back("Braking");
    }

    // CARLA IMU replay:
    // Positive yaw rate -> Turning Right
    // Negative yaw rate -> Turning Left

    if (yaw_rate > yaw_rate_threshold)
    {
        state.push_back("Turning Right");
    }
    else if (yaw_rate < -yaw_rate_threshold)
    {
        state.push_back("Turning Left");
    }

    return state;
}


IMUPipeline::IMUResult IMUPipeline::process(
    float acceleration_x,
    float acceleration_y,
    float acceleration_z,
    float angular_velocity_z,
    float orientation_z,
    float orientation_w,
    float speed
) const
{
    const Eigen::Vector3f acceleration =
        compute_acceleration(
            acceleration_x,
            acceleration_y,
            acceleration_z
        );

    const float yaw_rate =
        compute_yaw_rate(
            angular_velocity_z
        );

    const float heading =
        compute_heading(
            orientation_z,
            orientation_w
        );

    const std::vector<std::string> motion_state =
        compute_motion_state(
            speed,
            acceleration,
            yaw_rate
        );

    return {
        acceleration,
        yaw_rate,
        heading,
        motion_state
    };
}