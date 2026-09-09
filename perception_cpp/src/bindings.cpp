#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "perception_orchestrator.hpp"

namespace py = pybind11;


PYBIND11_MODULE(perception_cpp, m)
{
    // --------------------------------------------------
    // PerceptionOrchestrator
    // --------------------------------------------------

    py::class_<PerceptionOrchestrator>(
        m,
        "PerceptionOrchestrator"
    )
        .def(py::init<>())

        .def(
            "process",
            [](PerceptionOrchestrator& self,

               // --------------------------------------------------
               // LiDAR
               // --------------------------------------------------

               py::array_t<float,
                   py::array::c_style |
                   py::array::forcecast> lidar,

               // --------------------------------------------------
               // Front
               // --------------------------------------------------

               py::array_t<uint8_t,
                   py::array::c_style |
                   py::array::forcecast> front,

               py::object front_masks,

               py::array_t<float,
                   py::array::c_style |
                   py::array::forcecast> front_boxes,

               py::array_t<float,
                   py::array::c_style |
                   py::array::forcecast> front_classes,

               py::list front_targets,

               // --------------------------------------------------
               // Rear
               // --------------------------------------------------

               py::array_t<uint8_t,
                   py::array::c_style |
                   py::array::forcecast> rear,

               py::object rear_masks,

               py::array_t<float,
                   py::array::c_style |
                   py::array::forcecast> rear_boxes,

               py::array_t<float,
                   py::array::c_style |
                   py::array::forcecast> rear_classes,

               py::list rear_targets,

               // --------------------------------------------------
               // Left
               // --------------------------------------------------

               py::array_t<uint8_t,
                   py::array::c_style |
                   py::array::forcecast> left,

               py::object left_masks,

               py::array_t<float,
                   py::array::c_style |
                   py::array::forcecast> left_boxes,

               py::array_t<float,
                   py::array::c_style |
                   py::array::forcecast> left_classes,

               py::list left_targets,

               // --------------------------------------------------
               // Right
               // --------------------------------------------------

               py::array_t<uint8_t,
                   py::array::c_style |
                   py::array::forcecast> right,

               py::object right_masks,

               py::array_t<float,
                   py::array::c_style |
                   py::array::forcecast> right_boxes,

               py::array_t<float,
                   py::array::c_style |
                   py::array::forcecast> right_classes,

               py::list right_targets,

               // --------------------------------------------------
               // Class names
               // --------------------------------------------------

               py::dict class_names,

               // --------------------------------------------------
               // Image dimensions
               // --------------------------------------------------

               int front_width,
               int front_height,

               int rear_width,
               int rear_height,

               int left_width,
               int left_height,

               int right_width,
               int right_height)
            {
                // --------------------------------------------------
                // LiDAR: NumPy -> C++
                // --------------------------------------------------

                auto lidar_buf = lidar.request();

                if (lidar_buf.ndim != 2 ||
                    lidar_buf.shape[1] != 3)
                {
                    throw std::runtime_error(
                        "lidar must have shape (N, 3)"
                    );
                }

                const float* lidar_ptr =
                    static_cast<const float*>(lidar_buf.ptr);

                std::vector<Eigen::Vector3f> lidar_cpp;

                lidar_cpp.reserve(
                    static_cast<size_t>(lidar_buf.shape[0])
                );

                for (ssize_t i = 0;
                     i < lidar_buf.shape[0];
                     ++i)
                {
                    Eigen::Vector3f point(
                        lidar_ptr[i * 3 + 0],
                        lidar_ptr[i * 3 + 1],
                        lidar_ptr[i * 3 + 2]
                    );

                    lidar_cpp.push_back(point);
                }


                // --------------------------------------------------
                // Front image: NumPy -> cv::Mat
                // --------------------------------------------------

                auto front_buf = front.request();

                if (front_buf.ndim != 3 ||
                    front_buf.shape[2] != 3)
                {
                    throw std::runtime_error(
                        "front image must have shape (H, W, 3)"
                    );
                }

                cv::Mat front_cpp(
                    static_cast<int>(front_buf.shape[0]),
                    static_cast<int>(front_buf.shape[1]),
                    CV_8UC3,
                    front_buf.ptr
                );


                // --------------------------------------------------
                // Rear image: NumPy -> cv::Mat
                // --------------------------------------------------

                auto rear_buf = rear.request();

                if (rear_buf.ndim != 3 ||
                    rear_buf.shape[2] != 3)
                {
                    throw std::runtime_error(
                        "rear image must have shape (H, W, 3)"
                    );
                }

                cv::Mat rear_cpp(
                    static_cast<int>(rear_buf.shape[0]),
                    static_cast<int>(rear_buf.shape[1]),
                    CV_8UC3,
                    rear_buf.ptr
                );


                // --------------------------------------------------
                // Left image: NumPy -> cv::Mat
                // --------------------------------------------------

                auto left_buf = left.request();

                if (left_buf.ndim != 3 ||
                    left_buf.shape[2] != 3)
                {
                    throw std::runtime_error(
                        "left image must have shape (H, W, 3)"
                    );
                }

                cv::Mat left_cpp(
                    static_cast<int>(left_buf.shape[0]),
                    static_cast<int>(left_buf.shape[1]),
                    CV_8UC3,
                    left_buf.ptr
                );


                // --------------------------------------------------
                // Right image: NumPy -> cv::Mat
                // --------------------------------------------------

                auto right_buf = right.request();

                if (right_buf.ndim != 3 ||
                    right_buf.shape[2] != 3)
                {
                    throw std::runtime_error(
                        "right image must have shape (H, W, 3)"
                    );
                }

                cv::Mat right_cpp(
                    static_cast<int>(right_buf.shape[0]),
                    static_cast<int>(right_buf.shape[1]),
                    CV_8UC3,
                    right_buf.ptr
                );


                // --------------------------------------------------
                // Boxes: NumPy -> C++
                // --------------------------------------------------

                auto convert_boxes =
                    [](const py::array_t<float>& boxes)
                    {
                        auto buf = boxes.request();

                        if (buf.ndim != 2 ||
                            buf.shape[1] != 4)
                        {
                            throw std::runtime_error(
                                "boxes must have shape (N, 4)"
                            );
                        }

                        const float* ptr =
                            static_cast<const float*>(buf.ptr);

                        std::vector<Eigen::Vector4f> boxes_cpp;

                        boxes_cpp.reserve(
                            static_cast<size_t>(buf.shape[0])
                        );

                        for (ssize_t i = 0;
                             i < buf.shape[0];
                             ++i)
                        {
                            Eigen::Vector4f box(
                                ptr[i * 4 + 0],
                                ptr[i * 4 + 1],
                                ptr[i * 4 + 2],
                                ptr[i * 4 + 3]
                            );

                            boxes_cpp.push_back(box);
                        }

                        return boxes_cpp;
                    };


                auto front_boxes_cpp =
                    convert_boxes(front_boxes);

                auto rear_boxes_cpp =
                    convert_boxes(rear_boxes);

                auto left_boxes_cpp =
                    convert_boxes(left_boxes);

                auto right_boxes_cpp =
                    convert_boxes(right_boxes);


                // --------------------------------------------------
                // Classes: NumPy -> C++
                // --------------------------------------------------

                auto convert_classes =
                    [](const py::array_t<float>& classes)
                    {
                        auto buf = classes.request();

                        if (buf.ndim != 1)
                        {
                            throw std::runtime_error(
                                "classes must be a 1D NumPy array"
                            );
                        }

                        const float* ptr =
                            static_cast<const float*>(buf.ptr);

                        std::vector<int> classes_cpp;

                        classes_cpp.reserve(
                            static_cast<size_t>(buf.shape[0])
                        );

                        for (ssize_t i = 0;
                             i < buf.shape[0];
                             ++i)
                        {
                            classes_cpp.push_back(
                                static_cast<int>(ptr[i])
                            );
                        }

                        return classes_cpp;
                    };


                auto front_classes_cpp =
                    convert_classes(front_classes);

                auto rear_classes_cpp =
                    convert_classes(rear_classes);

                auto left_classes_cpp =
                    convert_classes(left_classes);

                auto right_classes_cpp =
                    convert_classes(right_classes);


                // --------------------------------------------------
                // Tracker targets: Python -> C++
                // --------------------------------------------------

                auto convert_targets =
                    [](const py::list& online_targets)
                    {
                        std::vector<
                            PerceptionUtils::TrackTarget
                        > targets_cpp;

                        targets_cpp.reserve(
                            online_targets.size()
                        );

                        for (auto target : online_targets)
                        {
                            PerceptionUtils::TrackTarget t;

                            t.track_id =
                                target.attr(
                                    "track_id"
                                ).cast<int>();

                            auto tlwh =
                                target.attr(
                                    "tlwh"
                                ).cast<py::sequence>();

                            t.x =
                                tlwh[0].cast<float>();

                            t.y =
                                tlwh[1].cast<float>();

                            t.width =
                                tlwh[2].cast<float>();

                            t.height =
                                tlwh[3].cast<float>();

                            targets_cpp.push_back(t);
                        }

                        return targets_cpp;
                    };


                auto front_targets_cpp =
                    convert_targets(front_targets);

                auto rear_targets_cpp =
                    convert_targets(rear_targets);

                auto left_targets_cpp =
                    convert_targets(left_targets);

                auto right_targets_cpp =
                    convert_targets(right_targets);


                // --------------------------------------------------
                // Class names: Python dict -> C++
                // --------------------------------------------------

                std::vector<std::string> class_names_cpp(
                    80,
                    "unknown"
                );

                for (auto item : class_names)
                {
                    int class_id =
                        item.first.cast<int>();

                    if (class_id >= 0 &&
                        class_id < static_cast<int>(
                            class_names_cpp.size()
                        ))
                    {
                        class_names_cpp[class_id] =
                            item.second.cast<std::string>();
                    }
                }


                // --------------------------------------------------
                // Call C++ orchestrator
                // --------------------------------------------------

                self.process(
                    lidar_cpp,

                    front_cpp,
                    front_masks,
                    front_boxes_cpp,
                    front_classes_cpp,
                    front_targets_cpp,

                    rear_cpp,
                    rear_masks,
                    rear_boxes_cpp,
                    rear_classes_cpp,
                    rear_targets_cpp,

                    left_cpp,
                    left_masks,
                    left_boxes_cpp,
                    left_classes_cpp,
                    left_targets_cpp,

                    right_cpp,
                    right_masks,
                    right_boxes_cpp,
                    right_classes_cpp,
                    right_targets_cpp,

                    class_names_cpp,

                    front_width,
                    front_height,

                    rear_width,
                    rear_height,

                    left_width,
                    left_height,

                    right_width,
                    right_height
                );
            },

            py::arg("lidar"),

            py::arg("front"),
            py::arg("front_masks"),
            py::arg("front_boxes"),
            py::arg("front_classes"),
            py::arg("front_targets"),

            py::arg("rear"),
            py::arg("rear_masks"),
            py::arg("rear_boxes"),
            py::arg("rear_classes"),
            py::arg("rear_targets"),

            py::arg("left"),
            py::arg("left_masks"),
            py::arg("left_boxes"),
            py::arg("left_classes"),
            py::arg("left_targets"),

            py::arg("right"),
            py::arg("right_masks"),
            py::arg("right_boxes"),
            py::arg("right_classes"),
            py::arg("right_targets"),

            py::arg("class_names"),

            py::arg("front_width"),
            py::arg("front_height"),

            py::arg("rear_width"),
            py::arg("rear_height"),

            py::arg("left_width"),
            py::arg("left_height"),

            py::arg("right_width"),
            py::arg("right_height")
        );
}