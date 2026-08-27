#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

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
            [](Perception3DPipeline& self,
            py::array_t<uint8_t, py::array::c_style | py::array::forcecast> masks,
            py::array_t<float, py::array::c_style | py::array::forcecast> boxes,
            py::array_t<float, py::array::c_style | py::array::forcecast> classes,
            py::dict class_names,
            py::array_t<int, py::array::c_style | py::array::forcecast> u,
            py::array_t<int, py::array::c_style | py::array::forcecast> v,
            py::array_t<float, py::array::c_style | py::array::forcecast> projected_points,
            int image_width,
            int image_height)
            {
                // --------------------------------------------------
                // masks: NumPy (N, H, W)
                // -> vector<vector<vector<float>>>
                // --------------------------------------------------

                auto masks_buf = masks.request();

                if (masks_buf.ndim != 3)
                {
                    throw std::runtime_error(
                        "masks must have shape (N, H, W)"
                    );
                }

                const int num_masks =
                    static_cast<int>(masks_buf.shape[0]);

                const int mask_height =
                    static_cast<int>(masks_buf.shape[1]);

                const int mask_width =
                    static_cast<int>(masks_buf.shape[2]);

                const uint8_t* masks_ptr =
                    static_cast<const uint8_t*>(masks_buf.ptr);

                std::vector<std::vector<std::vector<float>>> masks_cpp(
                    num_masks,
                    std::vector<std::vector<float>>(
                        mask_height,
                        std::vector<float>(mask_width)
                    )
                );

                for (int n = 0; n < num_masks; ++n)
                {
                    for (int y = 0; y < mask_height; ++y)
                    {
                        for (int x = 0; x < mask_width; ++x)
                        {
                            const size_t index =
                                static_cast<size_t>(n) *
                                    mask_height * mask_width +
                                static_cast<size_t>(y) *
                                    mask_width +
                                static_cast<size_t>(x);

                            masks_cpp[n][y][x] =
                                static_cast<float>(masks_ptr[index]);
                        }
                    }
                }


                // --------------------------------------------------
                // boxes: NumPy (N, 4)
                // -> vector<Eigen::Vector4f>
                // --------------------------------------------------

                auto boxes_buf = boxes.request();

                if (boxes_buf.ndim != 2 ||
                    boxes_buf.shape[1] != 4)
                {
                    throw std::runtime_error(
                        "boxes must have shape (N, 4)"
                    );
                }

                const float* boxes_ptr =
                    static_cast<const float*>(boxes_buf.ptr);

                std::vector<Eigen::Vector4f> boxes_cpp;

                boxes_cpp.reserve(
                    static_cast<size_t>(boxes_buf.shape[0])
                );

                for (ssize_t i = 0; i < boxes_buf.shape[0]; ++i)
                {
                    Eigen::Vector4f box;

                    box <<
                        boxes_ptr[i * 4 + 0],
                        boxes_ptr[i * 4 + 1],
                        boxes_ptr[i * 4 + 2],
                        boxes_ptr[i * 4 + 3];

                    boxes_cpp.push_back(box);
                }


                // --------------------------------------------------
                // classes: NumPy (N,)
                // -> vector<int>
                // --------------------------------------------------

                auto classes_buf = classes.request();

                if (classes_buf.ndim != 1)
                {
                    throw std::runtime_error(
                        "classes must be a 1D NumPy array"
                    );
                }

                const float* classes_ptr =
                    static_cast<const float*>(classes_buf.ptr);

                std::vector<int> classes_cpp;

                classes_cpp.reserve(
                    static_cast<size_t>(classes_buf.shape[0])
                );

                for (ssize_t i = 0; i < classes_buf.shape[0]; ++i)
                {
                    classes_cpp.push_back(
                        static_cast<int>(classes_ptr[i])
                    );
                }


                // --------------------------------------------------
                // class_names: Python dict
                // -> vector<string>
                // --------------------------------------------------

                std::vector<std::string> class_names_cpp(80, "unknown");

                for (auto item : class_names)
                {
                    int class_id = item.first.cast<int>();

                    if (class_id >= 0 &&
                        class_id < static_cast<int>(class_names_cpp.size()))
                    {
                        class_names_cpp[class_id] =
                            item.second.cast<std::string>();
                    }
                }


                // --------------------------------------------------
                // u: NumPy (N,)
                // -> vector<int>
                // --------------------------------------------------

                auto u_buf = u.request();

                if (u_buf.ndim != 1)
                {
                    throw std::runtime_error(
                        "u must be a 1D NumPy array"
                    );
                }

                const int* u_ptr =
                    static_cast<const int*>(u_buf.ptr);

                std::vector<int> u_cpp(
                    u_ptr,
                    u_ptr + u_buf.shape[0]
                );


                // --------------------------------------------------
                // v: NumPy (N,)
                // -> vector<int>
                // --------------------------------------------------

                auto v_buf = v.request();

                if (v_buf.ndim != 1)
                {
                    throw std::runtime_error(
                        "v must be a 1D NumPy array"
                    );
                }

                const int* v_ptr =
                    static_cast<const int*>(v_buf.ptr);

                std::vector<int> v_cpp(
                    v_ptr,
                    v_ptr + v_buf.shape[0]
                );


                // --------------------------------------------------
                // projected_points: NumPy (N, 3)
                // -> vector<Eigen::Vector3f>
                // --------------------------------------------------

                auto points_buf = projected_points.request();

                if (points_buf.ndim != 2 ||
                    points_buf.shape[1] != 3)
                {
                    throw std::runtime_error(
                        "projected_points must have shape (N, 3)"
                    );
                }

                const float* points_ptr =
                    static_cast<const float*>(points_buf.ptr);

                std::vector<Eigen::Vector3f> points_cpp;

                points_cpp.reserve(
                    static_cast<size_t>(points_buf.shape[0])
                );

                for (ssize_t i = 0; i < points_buf.shape[0]; ++i)
                {
                    Eigen::Vector3f point;

                    point <<
                        points_ptr[i * 3 + 0],
                        points_ptr[i * 3 + 1],
                        points_ptr[i * 3 + 2];

                    points_cpp.push_back(point);
                }


                // --------------------------------------------------
                // Call actual C++ implementation
                // --------------------------------------------------

                return self.extract_object_clouds(
                    masks_cpp,
                    boxes_cpp,
                    classes_cpp,
                    class_names_cpp,
                    u_cpp,
                    v_cpp,
                    points_cpp,
                    image_width,
                    image_height
                );
            },
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