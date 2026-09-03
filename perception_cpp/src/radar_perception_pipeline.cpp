#include "radar_perception_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>


RadarPerceptionPipeline::RadarPerceptionPipeline()
    : velocity_threshold(0.5f),
      max_assoc_distance_px(50.0f),
      fx(320.0f),
      fy(320.0f),
      cx(320.0f),
      cy(240.0f),
      width(640),
      height(480)
{
    T_radar_to_cam.setIdentity();
}


// --------------------------------------------------
// Configuration
// --------------------------------------------------

void RadarPerceptionPipeline::set_velocity_threshold(
    float threshold
)
{
    velocity_threshold = threshold;
}


void RadarPerceptionPipeline::set_max_assoc_distance_px(
    float distance
)
{
    max_assoc_distance_px = distance;
}


void RadarPerceptionPipeline::set_camera_intrinsics(
    float fx_,
    float fy_,
    float cx_,
    float cy_,
    int width_,
    int height_
)
{
    fx = fx_;
    fy = fy_;
    cx = cx_;
    cy = cy_;
    width = width_;
    height = height_;
}


void RadarPerceptionPipeline::set_radar_to_camera_transform(
    const Eigen::Matrix4f& transform
)
{
    T_radar_to_cam = transform;
}


// --------------------------------------------------
// Spherical -> Cartesian
// --------------------------------------------------

Eigen::Vector4f
RadarPerceptionPipeline::spherical_to_cartesian(
    const Eigen::Vector4f& radar_target
) const
{
    /*
        CARLA radar convention:

        radar_target:
            [0] depth
            [1] azimuth
            [2] altitude
            [3] velocity

        Radar frame:
            X = forward
            Y = right
            Z = up
    */

    const float depth = radar_target[0];
    const float azimuth = radar_target[1];
    const float altitude = radar_target[2];

    const float x =
        depth *
        std::cos(altitude) *
        std::cos(azimuth);

    const float y =
        depth *
        std::cos(altitude) *
        std::sin(azimuth);

    const float z =
        depth *
        std::sin(altitude);

    return Eigen::Vector4f(
        x,
        y,
        z,
        1.0f
    );
}


// --------------------------------------------------
// Project radar point -> image
// --------------------------------------------------

bool
RadarPerceptionPipeline::project_to_image(
    const Eigen::Vector4f& point_radar_frame,
    float& u,
    float& v,
    float& depth_cam
) const
{
    Eigen::Vector4f point_cam =
        T_radar_to_cam * point_radar_frame;

    const float X = point_cam[0];
    const float Y = point_cam[1];
    const float Z = point_cam[2];

    // Same behavior as Python:
    //
    // if Z <= 0.0:
    //     return None

    if (Z <= 0.0f)
    {
        return false;
    }

    u =
        (fx * X / Z) +
        cx;

    v =
        (fy * Y / Z) +
        cy;

    depth_cam = Z;

    return true;
}


// --------------------------------------------------
// Associate radar targets with detections
// --------------------------------------------------

std::vector<RadarPerceptionPipeline::RadarAssociation>
RadarPerceptionPipeline::associate_targets(
    const std::vector<Eigen::Vector4f>& boxes,
    const std::vector<float>& scores,
    const std::vector<int>& classes,
    const std::vector<Eigen::Vector4f>& radar_targets
)
{
    std::vector<RadarAssociation> associations;

    projected_points.clear();

    // --------------------------------------------------
    // Project all radar points once
    // --------------------------------------------------

    projected_points.reserve(radar_targets.size());

    for (const auto& radar_target : radar_targets)
    {
        Eigen::Vector4f point_radar =
            spherical_to_cartesian(radar_target);

        float u = 0.0f;
        float v = 0.0f;
        float depth_cam = 0.0f;

        bool valid =
            project_to_image(
                point_radar,
                u,
                v,
                depth_cam
            );

        if (!valid)
        {
            continue;
        }

        // Same Python image bounds:
        //
        // if 0 <= u < width and 0 <= v < height

        if (u >= 0.0f &&
            u < static_cast<float>(width) &&
            v >= 0.0f &&
            v < static_cast<float>(height))
        {
            ProjectedRadarTarget projected;

            projected.u = u;
            projected.v = v;
            projected.radar_target = radar_target;

            projected_points.push_back(
                std::move(projected)
            );
        }
    }


    // --------------------------------------------------
    // Match each detection independently
    // --------------------------------------------------

    const size_t num_detections = boxes.size();

    for (size_t i = 0; i < num_detections; ++i)
    {
        const Eigen::Vector4f& box = boxes[i];

        const float x1 = box[0];
        const float y1 = box[1];
        const float x2 = box[2];
        const float y2 = box[3];

        const float box_cx =
            (x1 + x2) / 2.0f;

        const float box_cy =
            (y1 + y2) / 2.0f;

        float best_dist =
            max_assoc_distance_px;

        int best_idx = -1;


        for (size_t j = 0;
             j < projected_points.size();
             ++j)
        {
            const auto& projected =
                projected_points[j];

            const float u = projected.u;
            const float v = projected.v;


            // Radar point must be inside bbox.

            if (!(x1 <= u &&
                  u <= x2 &&
                  y1 <= v &&
                  v <= y2))
            {
                continue;
            }


            // Distance from radar point to bbox center.

            const float dx =
                u - box_cx;

            const float dy =
                v - box_cy;

            const float dist =
                std::sqrt(
                    dx * dx +
                    dy * dy
                );


            // Same Python condition:
            //
            // if dist < best_dist

            if (dist < best_dist)
            {
                best_dist = dist;
                best_idx = static_cast<int>(j);
            }
        }


        if (best_idx >= 0)
        {
            RadarAssociation association;

            association.box = box;
            association.confidence = scores[i];
            association.class_id = classes[i];

            association.radar_target =
                projected_points[best_idx].radar_target;

            associations.push_back(
                std::move(association)
            );
        }
    }

    return associations;
}


// --------------------------------------------------
// Range
// --------------------------------------------------

float
RadarPerceptionPipeline::compute_range(
    const Eigen::Vector4f& radar_target
) const
{
    return radar_target[0];
}


// --------------------------------------------------
// Bearing
// --------------------------------------------------

float
RadarPerceptionPipeline::compute_bearing(
    const Eigen::Vector4f& radar_target
) const
{
    constexpr float RAD_TO_DEG =
        180.0f / static_cast<float>(M_PI);

    return radar_target[1] * RAD_TO_DEG;
}


// --------------------------------------------------
// Radial velocity
// --------------------------------------------------

float
RadarPerceptionPipeline::compute_radial_velocity(
    const Eigen::Vector4f& radar_target
) const
{
    return radar_target[3];
}


// --------------------------------------------------
// Motion state
// --------------------------------------------------

std::string
RadarPerceptionPipeline::estimate_motion_state(
    float velocity
) const
{
    if (velocity < -velocity_threshold)
    {
        return "Approaching";
    }
    else if (velocity > velocity_threshold)
    {
        return "Receding";
    }

    return "Stationary";
}


// --------------------------------------------------
// Full processing
// --------------------------------------------------

std::vector<RadarPerceptionPipeline::RadarObject>
RadarPerceptionPipeline::process(
    const std::vector<Eigen::Vector4f>& boxes,
    const std::vector<float>& scores,
    const std::vector<int>& classes,
    const std::vector<Eigen::Vector4f>& radar_targets
)
{
    std::vector<RadarObject> objects;

    const auto associations =
        associate_targets(
            boxes,
            scores,
            classes,
            radar_targets
        );


    objects.reserve(associations.size());


    for (const auto& association : associations)
    {
        RadarObject object;

        object.box =
            association.box;

        object.confidence =
            association.confidence;

        object.class_id =
            association.class_id;


        object.range =
            compute_range(
                association.radar_target
            );

        object.bearing =
            compute_bearing(
                association.radar_target
            );

        object.velocity =
            compute_radial_velocity(
                association.radar_target
            );

        object.motion =
            estimate_motion_state(
                object.velocity
            );


        objects.push_back(
            std::move(object)
        );
    }


    return objects;
}