#include "track.hpp"

Track::Track(
    int track_id,
    const std::array<float, 4>& bbox,
    int class_id,
    float confidence
)
    : track_id(track_id),
      bbox(bbox),
      class_id(class_id),
      confidence(confidence),
      kalman()
{
}