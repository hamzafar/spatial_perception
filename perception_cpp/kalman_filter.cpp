#include "kalman_filter.hpp"

#include <Eigen/Dense>

KalmanFilter::KalmanFilter()
    : state(Eigen::VectorXf::Zero(8)),
      covariance(Eigen::MatrixXf::Identity(8, 8))
{
}

void KalmanFilter::initiate(
    const std::array<float, 4>& bbox
)
{
    Eigen::Vector4f measurement = xyxy_to_xyah(bbox);

    state.setZero();
    state.head<4>() = measurement;

    covariance.setIdentity();
}

void KalmanFilter::predict()
{
    // Not implemented yet — same as Python version.
}

void KalmanFilter::update(
    const std::array<float, 4>& bbox
)
{
    // Not implemented yet — same as Python version.
}

Eigen::Vector4f KalmanFilter::xyxy_to_xyah(
    const std::array<float, 4>& bbox
) const
{
    const float x1 = bbox[0];
    const float y1 = bbox[1];
    const float x2 = bbox[2];
    const float y2 = bbox[3];

    const float width = x2 - x1;
    const float height = y2 - y1;

    const float center_x = x1 + width / 2.0f;
    const float center_y = y1 + height / 2.0f;

    const float aspect_ratio = width / height;

    return Eigen::Vector4f(
        center_x,
        center_y,
        aspect_ratio,
        height
    );
}

std::array<float, 4> KalmanFilter::xyah_to_xyxy(
    const Eigen::VectorXf& state
) const
{
    const float center_x = state[0];
    const float center_y = state[1];
    const float aspect_ratio = state[2];
    const float height = state[3];

    const float width = aspect_ratio * height;

    const float x1 = center_x - width / 2.0f;
    const float y1 = center_y - height / 2.0f;
    const float x2 = center_x + width / 2.0f;
    const float y2 = center_y + height / 2.0f;

    return {
        x1,
        y1,
        x2,
        y2
    };
}