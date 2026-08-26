#include <pybind11/pybind11.h>
#include <pybind11/stl.h>      // <-- restore: needed for std::vector<Eigen::Vector3f> input param
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "perception_3d_pipeline.hpp"

namespace py = pybind11;

PYBIND11_MODULE(perception_cpp, m)
{
    py::class_<Perception3DPipeline::ProjectionResult>(
        m,
        "ProjectionResult"
    )
        .def_property_readonly(
            "u",
            [](const Perception3DPipeline::ProjectionResult &r) {
                py::array_t<int> arr(r.u.size());
                std::memcpy(
                    arr.mutable_data(),
                    r.u.data(),
                    r.u.size() * sizeof(int)
                );
                return arr;
            }
        )

        .def_property_readonly(
            "v",
            [](const Perception3DPipeline::ProjectionResult &r) {
                py::array_t<int> arr(r.v.size());
                std::memcpy(
                    arr.mutable_data(),
                    r.v.data(),
                    r.v.size() * sizeof(int)
                );
                return arr;
            }
        )

        .def_property_readonly(
            "ego_points",
            [](const Perception3DPipeline::ProjectionResult &r) {
                size_t n = r.ego_points.size();
                py::array_t<float> arr({n, static_cast<size_t>(3)});
                std::memcpy(
                    arr.mutable_data(),
                    r.ego_points.data(),
                    n * 3 * sizeof(float)
                );
                return arr;
            }
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