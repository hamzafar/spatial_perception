#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "perception_3d_pipeline.hpp"

namespace py = pybind11;

PYBIND11_MODULE(perception_cpp, m)
{
    // --------------------------------------------------
    // ObjectPoint
    // --------------------------------------------------

    py::class_<Perception3DPipeline::ObjectPoint>(
        m,
        "ObjectPoint"
    )
        .def_readonly(
            "u",
            &Perception3DPipeline::ObjectPoint::u
        )
        .def_readonly(
            "v",
            &Perception3DPipeline::ObjectPoint::v
        )
        .def_readonly(
            "xyz",
            &Perception3DPipeline::ObjectPoint::xyz
        );


    // --------------------------------------------------
    // ObjectCloud
    // --------------------------------------------------

    py::class_<Perception3DPipeline::ObjectCloud>(
        m,
        "ObjectCloud"
    )
        .def_readonly(
            "cloud",
            &Perception3DPipeline::ObjectCloud::cloud
        )
        .def_readonly(
            "box",
            &Perception3DPipeline::ObjectCloud::box
        )
        .def_readonly(
            "class_name",
            &Perception3DPipeline::ObjectCloud::class_name
        );


    // --------------------------------------------------
    // ProjectionResult
    // --------------------------------------------------

    py::class_<Perception3DPipeline::ProjectionResult>(
        m,
        "ProjectionResult"
    )
        .def_property_readonly(
            "u",
            [](const Perception3DPipeline::ProjectionResult& r) {

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
            [](const Perception3DPipeline::ProjectionResult& r) {

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
            [](const Perception3DPipeline::ProjectionResult& r) {

                size_t n = r.ego_points.size();

                py::array_t<float> arr({
                    n,
                    static_cast<size_t>(3)
                });

                std::memcpy(
                    arr.mutable_data(),
                    r.ego_points.data(),
                    n * 3 * sizeof(float)
                );

                return arr;
            }
        );


    // --------------------------------------------------
    // Perception3DPipeline
    // --------------------------------------------------

    py::class_<Perception3DPipeline>(
        m,
        "Perception3DPipeline"
    )
        .def(py::init<>())

        // LiDAR projection
        .def(
            "project_lidar",
            &Perception3DPipeline::project_lidar,
            py::arg("lidar"),
            py::arg("camera_name"),
            py::arg("image_width"),
            py::arg("image_height")
        )

        // Object cloud extraction
        .def(
            "extract_object_clouds",
            &Perception3DPipeline::extract_object_clouds,
            py::arg("masks"),
            py::arg("boxes"),
            py::arg("classes"),
            py::arg("class_names"),
            py::arg("u"),
            py::arg("v"),
            py::arg("projected_points"),
            py::arg("image_width"),
            py::arg("image_height")
        );
}