#include "perception_utils.hpp"

#include <algorithm>
#include <cmath>


PerceptionUtils::PerceptionUtils()
{
}


std::vector<Perception3DPipeline::WorldObject>
PerceptionUtils::attach_track_ids(
    std::vector<Perception3DPipeline::WorldObject>& world_objects,
    const std::vector<TrackTarget>& online_targets,
    const std::string& camera_prefix
)
{
    if (online_targets.empty())
    {
        return world_objects;
    }

    // Prepare tracker bounding boxes
    std::vector<float> track_x1;
    std::vector<float> track_y1;
    std::vector<float> track_x2;
    std::vector<float> track_y2;
    std::vector<float> track_areas;
    std::vector<int> track_ids;

    track_x1.reserve(online_targets.size());
    track_y1.reserve(online_targets.size());
    track_x2.reserve(online_targets.size());
    track_y2.reserve(online_targets.size());
    track_areas.reserve(online_targets.size());
    track_ids.reserve(online_targets.size());

    for (const auto& target : online_targets)
    {
        float x1 = target.x;
        float y1 = target.y;
        float width = target.width;
        float height = target.height;

        track_x1.push_back(x1);
        track_y1.push_back(y1);
        track_x2.push_back(x1 + width);
        track_y2.push_back(y1 + height);

        track_areas.push_back(width * height);
        track_ids.push_back(target.track_id);
    }


    // Match each world object with the best tracker
    for (auto& obj : world_objects)
    {
        float ox1 = obj.box[0];
        float oy1 = obj.box[1];
        float ox2 = obj.box[2];
        float oy2 = obj.box[3];

        float object_width = ox2 - ox1;
        float object_height = oy2 - oy1;

        float object_area =
            object_width * object_height;


        float best_iou = 0.0f;
        int best_idx = -1;


        for (size_t i = 0; i < online_targets.size(); ++i)
        {
            float ix1 = std::max(ox1, track_x1[i]);
            float iy1 = std::max(oy1, track_y1[i]);

            float ix2 = std::min(ox2, track_x2[i]);
            float iy2 = std::min(oy2, track_y2[i]);


            float intersection_width =
                std::max(0.0f, ix2 - ix1);

            float intersection_height =
                std::max(0.0f, iy2 - iy1);

            float intersection =
                intersection_width *
                intersection_height;


            float union_area =
                object_area +
                track_areas[i] -
                intersection;


            float iou = 0.0f;

            if (union_area > 0.0f)
            {
                iou = intersection / union_area;
            }


            if (iou > best_iou)
            {
                best_iou = iou;
                best_idx = static_cast<int>(i);
            }
        }


        // Same threshold as Python
        if (best_idx >= 0 && best_iou >= 0.3f)
        {
            obj.id =
                camera_prefix +
                "_" +
                std::to_string(track_ids[best_idx]);
        }
    }


    return world_objects;
}

std::vector<Perception3DPipeline::WorldObject>
PerceptionUtils::attach_radar_data(
    std::vector<Perception3DPipeline::WorldObject>& world_objects,
    const std::vector<RadarPerceptionPipeline::RadarObject>& radar_objects,
    float iou_threshold
)
{
    if (world_objects.empty() || radar_objects.empty())
    {
        return world_objects;
    }

    for (auto& obj : world_objects)
    {
        float ox1 = obj.box[0];
        float oy1 = obj.box[1];
        float ox2 = obj.box[2];
        float oy2 = obj.box[3];

        float object_area =
            (ox2 - ox1) * (oy2 - oy1);

        float best_iou = 0.0f;
        int best_idx = -1;

        for (size_t i = 0; i < radar_objects.size(); ++i)
        {
            const auto& radar = radar_objects[i];

            float rx1 = radar.box[0];
            float ry1 = radar.box[1];
            float rx2 = radar.box[2];
            float ry2 = radar.box[3];

            float radar_area =
                (rx2 - rx1) * (ry2 - ry1);

            float ix1 = std::max(ox1, rx1);
            float iy1 = std::max(oy1, ry1);
            float ix2 = std::min(ox2, rx2);
            float iy2 = std::min(oy2, ry2);

            float intersection =
                std::max(0.0f, ix2 - ix1) *
                std::max(0.0f, iy2 - iy1);

            float union_area =
                object_area +
                radar_area -
                intersection;

            float iou = 0.0f;

            if (union_area > 0.0f)
            {
                iou = intersection / union_area;
            }

            if (iou > best_iou)
            {
                best_iou = iou;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx >= 0 &&
            best_iou > 0.0f &&
            best_iou >= iou_threshold)
        {
            const auto& radar = radar_objects[best_idx];

            obj.has_radar = true;
            obj.radar_range = radar.range;
            obj.radar_bearing = radar.bearing;
            obj.radar_velocity = radar.velocity;
            obj.radar_motion = radar.motion;
        }
    }

    return world_objects;
}

std::string PerceptionUtils::normalize_bev_class(
    const std::string& cls
)
{
    if (cls == "person")
    {
        return "person";
    }

    if (cls == "car")
    {
        return "vehicle";
    }

    if (cls == "truck" || cls == "bus")
    {
        return "truck";
    }

    if (cls == "bicycle" || cls == "motorcycle")
    {
        return "cyclist";
    }

    return "";
}

std::unordered_map<std::string, int>
PerceptionUtils::count_objects(
    const std::vector<Perception3DPipeline::WorldObject>& world_objects
)
{
    std::unordered_map<std::string, int> counts = {
        {"vehicle", 0},
        {"person", 0},
        {"cyclist", 0}
    };

    for (const auto& obj : world_objects)
    {
        std::string cls = normalize_bev_class(obj.class_name);

        if (cls == "vehicle" || cls == "truck")
        {
            counts["vehicle"]++;
        }
        else if (cls == "person")
        {
            counts["person"]++;
        }
        else if (cls == "cyclist")
        {
            counts["cyclist"]++;
        }
    }

    return counts;
}