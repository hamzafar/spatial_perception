#pragma once

#include <string>
#include <vector>

#include "perception_3d_pipeline.hpp"

class PerceptionUtils
{
public:

    struct TrackTarget
    {
        int track_id;

        float x;
        float y;
        float width;
        float height;
    };

    PerceptionUtils();

    std::vector<Perception3DPipeline::WorldObject> attach_track_ids(
        std::vector<Perception3DPipeline::WorldObject>& world_objects,
        const std::vector<TrackTarget>& online_targets,
        const std::string& camera_prefix
    );
};