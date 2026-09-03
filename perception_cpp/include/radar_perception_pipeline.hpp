#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>


class RadarPerceptionPipeline
{
public:

    struct ProjectedRadarTarget
    {
        float u;
        float v;
        Eigen::Vector4f radar_target;
    };

    struct RadarAssociation
    {
        Eigen::Vector4f box;
        float confidence;
        int class_id;

        Eigen::Vector4f radar_target;
    };

    struct RadarObject
    {
        Eigen::Vector4f box;

        float confidence;
        int class_id;

        float range;
        float bearing;
        float velocity;

        std::string motion;
    };


    RadarPerceptionPipeline();


    // --------------------------------------------------
    // Configuration
    // --------------------------------------------------

    void set_velocity_threshold(float threshold);

    void set_max_assoc_distance_px(float distance);

    void set_camera_intrinsics(
        float fx,
        float fy,
        float cx,
        float cy,
        int width,
        int height
    );

    void set_radar_to_camera_transform(
        const Eigen::Matrix4f& transform
    );


    // --------------------------------------------------
    // Radar geometry
    // --------------------------------------------------

    Eigen::Vector4f spherical_to_cartesian(
        const Eigen::Vector4f& radar_target
    ) const;


    bool project_to_image(
        const Eigen::Vector4f& point_radar_frame,
        float& u,
        float& v,
        float& depth_cam
    ) const;


    // --------------------------------------------------
    // Association
    // --------------------------------------------------

    std::vector<RadarAssociation> associate_targets(
        const std::vector<Eigen::Vector4f>& boxes,
        const std::vector<float>& scores,
        const std::vector<int>& classes,
        const std::vector<Eigen::Vector4f>& radar_targets
    );


    // --------------------------------------------------
    // Radar measurements
    // --------------------------------------------------

    float compute_range(
        const Eigen::Vector4f& radar_target
    ) const;


    float compute_bearing(
        const Eigen::Vector4f& radar_target
    ) const;


    float compute_radial_velocity(
        const Eigen::Vector4f& radar_target
    ) const;


    std::string estimate_motion_state(
        float velocity
    ) const;


    // --------------------------------------------------
    // Full radar processing
    // --------------------------------------------------

    std::vector<RadarObject> process(
        const std::vector<Eigen::Vector4f>& boxes,
        const std::vector<float>& scores,
        const std::vector<int>& classes,
        const std::vector<Eigen::Vector4f>& radar_targets
    );


private:

    float velocity_threshold;
    float max_assoc_distance_px;

    float fx;
    float fy;
    float cx;
    float cy;

    int width;
    int height;

    Eigen::Matrix4f T_radar_to_cam;

    std::vector<ProjectedRadarTarget> projected_points;
};