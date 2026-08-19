#include "track.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

class TrackingPipeline
{
public:

    TrackingPipeline()
        : next_track_id_(0)
    {
    }

    std::vector<Track*> update(
        const std::vector<Detection>& detections
    )
    {
        const float iou_threshold = 0.3f;

        // Predict all tracks once
        for (auto& track : tracks_)
        {
            track->kalman.predict();
        }

        for (const auto& detection : detections)
        {
            Track* best_track = nullptr;
            float best_iou = 0.0f;

            // Find the best matching existing track
            for (auto& track : tracks_)
            {
                float iou = compute_iou(
                    track->bbox,
                    detection.bbox
                );

                if (iou > best_iou)
                {
                    best_iou = iou;
                    best_track = track.get();
                }
            }

            // Update existing track
            if (best_track != nullptr &&
                best_iou >= iou_threshold)
            {
                best_track->bbox = detection.bbox;
                best_track->class_id = detection.class_id;
                best_track->confidence = detection.confidence;
            }
            else
            {
                // Create new track
                auto track = std::make_unique<Track>(
                    next_track_id_,
                    detection.bbox,
                    detection.class_id,
                    detection.confidence
                );

                tracks_.push_back(std::move(track));

                ++next_track_id_;
            }
        }

        // Return tracks
        std::vector<Track*> result;

        for (auto& track : tracks_)
        {
            result.push_back(track.get());
        }

        return result;
    }

private:

    float compute_iou(
        const std::array<float, 4>& box1,
        const std::array<float, 4>& box2
    ) const
    {
        const float x_left =
            std::max(box1[0], box2[0]);

        const float y_top =
            std::max(box1[1], box2[1]);

        const float x_right =
            std::min(box1[2], box2[2]);

        const float y_bottom =
            std::min(box1[3], box2[3]);

        if (x_right <= x_left ||
            y_bottom <= y_top)
        {
            return 0.0f;
        }

        const float intersection =
            (x_right - x_left) *
            (y_bottom - y_top);

        const float area1 =
            (box1[2] - box1[0]) *
            (box1[3] - box1[1]);

        const float area2 =
            (box2[2] - box2[0]) *
            (box2[3] - box2[1]);

        const float union_area =
            area1 + area2 - intersection;

        if (union_area <= 0.0f)
        {
            return 0.0f;
        }

        return intersection / union_area;
    }

private:

    std::vector<std::unique_ptr<Track>> tracks_;

    int next_track_id_;
};