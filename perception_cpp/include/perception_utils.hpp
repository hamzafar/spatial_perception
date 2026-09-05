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

    struct TrackMotion
    {
        float speed_mps;
        std::string motion;
    };

    TrackMotion estimate_track_motion(
        const Perception3DPipeline::WorldObject& obj,
        double timestamp
    );

    struct NearestObject
    {
        std::string id;
        std::string cls;
        std::string label;
        float dist_m;
        float speed_mps;
        std::string motion;
    };

    std::vector<NearestObject> prepare_nearest_objects(
        const std::vector<Perception3DPipeline::WorldObject>& world_objects,
        double timestamp
    );

private:

    struct TrackHistory
    {
        Eigen::Vector3f position;
        double timestamp;
    };

    std::unordered_map<std::string, TrackHistory> track_history;

};