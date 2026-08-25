#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "perception_3d_pipeline.hpp"

namespace py = pybind11;

PYBIND11_MODULE(perception_cpp, m)
{
    py::class_<Perception3DPipeline::ProjectionResult>(
        m,
        "ProjectionResult"
    )
        .def_readonly(
            "u",
            &Perception3DPipeline::ProjectionResult::u
        )
        .def_readonly(
            "v",
            &Perception3DPipeline::ProjectionResult::v
        )
        .def_readonly(
            "ego_points",
            &Perception3DPipeline::ProjectionResult::ego_points
        );

    py::class_<Perception3DPipeline>(m, "Perception3DPipeline")
        .def(py::init<>())

        .def(
            "project_lidar",
            &Perception3DPipeline::project_lidar,
            py::arg("lidar"),
            py::arg("camera_name"),
            py::arg("image_width"),
            py::arg("image_height")
        );
}