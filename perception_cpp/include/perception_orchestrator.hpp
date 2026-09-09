#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>
#include <pybind11/pybind11.h>

#include "perception_3d_pipeline.hpp"
#include "perception_utils.hpp"

namespace py = pybind11;

class PerceptionOrchestrator
{
public:
    PerceptionOrchestrator();

    void process(
        const std::vector<Eigen::Vector3f>& lidar,

        cv::Mat& front,
        const py::object& front_masks,
        const std::vector<Eigen::Vector4f>& front_boxes,
        const std::vector<int>& front_classes,
        const std::vector<PerceptionUtils::TrackTarget>& front_targets,

        cv::Mat& rear,
        const py::object& rear_masks,
        const std::vector<Eigen::Vector4f>& rear_boxes,
        const std::vector<int>& rear_classes,
        const std::vector<PerceptionUtils::TrackTarget>& rear_targets,

        cv::Mat& left,
        const py::object& left_masks,
        const std::vector<Eigen::Vector4f>& left_boxes,
        const std::vector<int>& left_classes,
        const std::vector<PerceptionUtils::TrackTarget>& left_targets,

        cv::Mat& right,
        const py::object& right_masks,
        const std::vector<Eigen::Vector4f>& right_boxes,
        const std::vector<int>& right_classes,
        const std::vector<PerceptionUtils::TrackTarget>& right_targets,

        const std::vector<std::string>& class_names,

        int front_width,
        int front_height,
        int rear_width,
        int rear_height,
        int left_width,
        int left_height,
        int right_width,
        int right_height
    );

private:
    Perception3DPipeline pipeline_3d;
    PerceptionUtils perception_utils;
};