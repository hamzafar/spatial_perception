#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#include "perception_3d_pipeline.hpp"
#include "radar_perception_pipeline.hpp"

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

    std::vector<Perception3DPipeline::WorldObject> attach_radar_data(
        std::vector<Perception3DPipeline::WorldObject>& world_objects,
        const std::vector<RadarPerceptionPipeline::RadarObject>& radar_objects,
        float iou_threshold = 0.3f
    );    

    std::unordered_map<std::string, int> count_objects(
        const std::vector<Perception3DPipeline::WorldObject>& world_objects
    );
    
    std::string normalize_bev_class(
        const std::string& cls
    );
};