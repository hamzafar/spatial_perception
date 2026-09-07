#pragma once

#include <Eigen/Dense>

class GNSSPipeline
{
public:

    struct GNSSResult
    {
        Eigen::Vector3f position;
        float speed;
    };

    GNSSPipeline();

    void set_reference(
        double latitude,
        double longitude,
        double altitude
    );

    Eigen::Vector3f compute_position(
        double latitude,
        double longitude,
        double altitude
    );

    float compute_speed(
        const Eigen::Vector3f& position,
        double timestamp
    );

    GNSSResult process(
        double latitude,
        double longitude,
        double altitude,
        double timestamp
    );

private:

    bool reference_set;

    double lat0;
    double lon0;
    double alt0;

    Eigen::Vector3f previous_position;
    double previous_timestamp;
    bool previous_position_set;
};