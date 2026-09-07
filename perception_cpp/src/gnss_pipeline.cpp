#include "gnss_pipeline.hpp"

#include <cmath>

namespace
{
    constexpr double WGS84_A = 6378137.0;
    constexpr double WGS84_E = 8.1819190842622e-2;

    constexpr double DEG_TO_RAD =
        3.14159265358979323846 / 180.0;

    struct ECEF
    {
        double x;
        double y;
        double z;
    };

    ECEF geodetic_to_ecef(
        double latitude,
        double longitude,
        double altitude
    )
    {
        const double lat = latitude * DEG_TO_RAD;
        const double lon = longitude * DEG_TO_RAD;

        const double sin_lat = std::sin(lat);
        const double cos_lat = std::cos(lat);
        const double sin_lon = std::sin(lon);
        const double cos_lon = std::cos(lon);

        const double N =
            WGS84_A /
            std::sqrt(
                1.0 -
                WGS84_E * WGS84_E *
                sin_lat * sin_lat
            );

        ECEF result;

        result.x =
            (N + altitude) *
            cos_lat *
            cos_lon;

        result.y =
            (N + altitude) *
            cos_lat *
            sin_lon;

        result.z =
            (
                N * (1.0 - WGS84_E * WGS84_E) +
                altitude
            ) *
            sin_lat;

        return result;
    }
}


GNSSPipeline::GNSSPipeline()
    : reference_set(false),
      lat0(0.0),
      lon0(0.0),
      alt0(0.0),
      previous_position(Eigen::Vector3f::Zero()),
      previous_timestamp(0.0),
      previous_position_set(false)
{
}


void GNSSPipeline::set_reference(
    double latitude,
    double longitude,
    double altitude
)
{
    lat0 = latitude;
    lon0 = longitude;
    alt0 = altitude;

    reference_set = true;
}


Eigen::Vector3f GNSSPipeline::compute_position(
    double latitude,
    double longitude,
    double altitude
)
{
    if (!reference_set)
    {
        set_reference(
            latitude,
            longitude,
            altitude
        );
    }

    const ECEF reference =
        geodetic_to_ecef(
            lat0,
            lon0,
            alt0
        );

    const ECEF current =
        geodetic_to_ecef(
            latitude,
            longitude,
            altitude
        );

    const double lat =
        lat0 * DEG_TO_RAD;

    const double lon =
        lon0 * DEG_TO_RAD;

    const double sin_lat = std::sin(lat);
    const double cos_lat = std::cos(lat);

    const double sin_lon = std::sin(lon);
    const double cos_lon = std::cos(lon);

    const double dx =
        current.x - reference.x;

    const double dy =
        current.y - reference.y;

    const double dz =
        current.z - reference.z;

    // ECEF -> ENU
    const double east =
        -sin_lon * dx +
        cos_lon * dy;

    const double north =
        -sin_lat * cos_lon * dx
        -sin_lat * sin_lon * dy
        +cos_lat * dz;

    const double up =
        cos_lat * cos_lon * dx
        +cos_lat * sin_lon * dy
        +sin_lat * dz;

    return Eigen::Vector3f(
        static_cast<float>(east),
        static_cast<float>(north),
        static_cast<float>(up)
    );
}


float GNSSPipeline::compute_speed(
    const Eigen::Vector3f& position,
    double timestamp
)
{
    if (!previous_position_set)
    {
        previous_position = position;
        previous_timestamp = timestamp;
        previous_position_set = true;

        return 0.0f;
    }

    const double dt =
        timestamp - previous_timestamp;

    if (dt <= 0.0)
    {
        return 0.0f;
    }

    const float distance =
        (position - previous_position).norm();

    const float speed =
        distance /
        static_cast<float>(dt);

    previous_position = position;
    previous_timestamp = timestamp;

    return speed;
}


GNSSPipeline::GNSSResult GNSSPipeline::process(
    double latitude,
    double longitude,
    double altitude,
    double timestamp
)
{
    const Eigen::Vector3f position =
        compute_position(
            latitude,
            longitude,
            altitude
        );

    const float speed =
        compute_speed(
            position,
            timestamp
        );

    return {
        position,
        speed
    };
}