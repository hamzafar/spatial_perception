#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "perception_orchestrator.hpp"

namespace py = pybind11;


// ============================================================
// Helper: NumPy image -> cv::Mat
// ============================================================

static cv::Mat numpy_to_cvmat(
    const py::array_t<uint8_t>& image)
{
    auto buf = image.request();

    if (buf.ndim != 3 ||
        buf.shape[2] != 3)
    {
        throw std::runtime_error(
            "Image must have shape (H, W, 3)");
    }

    cv::Mat mat(
        static_cast<int>(buf.shape[0]),
        static_cast<int>(buf.shape[1]),
        CV_8UC3,
        buf.ptr
    );

    return mat.clone();
}


// ============================================================
// Helper: NumPy LiDAR -> vector<Eigen::Vector3f>
// ============================================================

static std::vector<Eigen::Vector3f> numpy_to_lidar(
    const py::array_t<float>& points)
{
    auto buf = points.request();

    if (buf.ndim != 2 ||
        buf.shape[1] != 3)
    {
        throw std::runtime_error(
            "LiDAR points must have shape (N, 3)");
    }

    const int n =
        static_cast<int>(buf.shape[0]);

    const float* ptr =
        static_cast<float*>(buf.ptr);

    std::vector<Eigen::Vector3f> result;
    result.reserve(n);

    for (int i = 0; i < n; ++i)
    {
        result.emplace_back(
            ptr[i * 3 + 0],
            ptr[i * 3 + 1],
            ptr[i * 3 + 2]
        );
    }

    return result;
}


// ============================================================
// Helper: NumPy radar -> vector<Eigen::Vector4f>
// [depth, azimuth, altitude, velocity]
// ============================================================

static std::vector<Eigen::Vector4f> numpy_to_radar(
    const py::array_t<float>& points)
{
    auto buf = points.request();

    if (buf.ndim != 2 ||
        buf.shape[1] != 4)
    {
        throw std::runtime_error(
            "Radar points must have shape (N, 4)");
    }

    const int n =
        static_cast<int>(buf.shape[0]);

    const float* ptr =
        static_cast<float*>(buf.ptr);

    std::vector<Eigen::Vector4f> result;
    result.reserve(n);

    for (int i = 0; i < n; ++i)
    {
        result.emplace_back(
            ptr[i * 4 + 0],
            ptr[i * 4 + 1],
            ptr[i * 4 + 2],
            ptr[i * 4 + 3]
        );
    }

    return result;
}


// ============================================================
// Helper: Python CameraInput -> C++ CameraInput
// ============================================================

static PerceptionOrchestrator::CameraInput
convert_camera_input(
    const py::dict& input)
{
    PerceptionOrchestrator::CameraInput camera;

    // --------------------------------------------------------
    // Image
    // --------------------------------------------------------

    if (input.contains("image"))
    {
        auto image =
            input["image"].cast<py::array_t<uint8_t>>();

        camera.image =
            numpy_to_cvmat(image);
    }


    // --------------------------------------------------------
    // Masks
    // Expected shape: (N, H, W)
    // --------------------------------------------------------

    if (input.contains("masks") &&
        !input["masks"].is_none())
    {
        auto masks =
            input["masks"].cast<
                py::array_t<float>>();

        auto buf = masks.request();

        if (buf.ndim != 3)
        {
            throw std::runtime_error(
                "Masks must have shape (N, H, W)");
        }

        const int n =
            static_cast<int>(buf.shape[0]);

        const int h =
            static_cast<int>(buf.shape[1]);

        const int w =
            static_cast<int>(buf.shape[2]);

        const float* ptr =
            static_cast<float*>(buf.ptr);

        camera.masks.resize(n);

        for (int i = 0; i < n; ++i)
        {
            camera.masks[i].resize(h);

            for (int y = 0; y < h; ++y)
            {
                camera.masks[i][y].resize(w);

                for (int x = 0; x < w; ++x)
                {
                    const size_t index =
                        static_cast<size_t>(i) * h * w +
                        static_cast<size_t>(y) * w +
                        x;

                    camera.masks[i][y][x] =
                        ptr[index];
                }
            }
        }
    }


    // --------------------------------------------------------
    // Boxes
    // Expected shape: (N, 4)
    // --------------------------------------------------------

    if (input.contains("boxes"))
    {
        auto boxes =
            input["boxes"].cast<
                py::array_t<float>>();

        auto buf = boxes.request();

        if (buf.ndim != 2 ||
            buf.shape[1] != 4)
        {
            throw std::runtime_error(
                "Boxes must have shape (N, 4)");
        }

        const int n =
            static_cast<int>(buf.shape[0]);

        const float* ptr =
            static_cast<float*>(buf.ptr);

        camera.boxes.reserve(n);

        for (int i = 0; i < n; ++i)
        {
            camera.boxes.emplace_back(
                ptr[i * 4 + 0],
                ptr[i * 4 + 1],
                ptr[i * 4 + 2],
                ptr[i * 4 + 3]
            );
        }
    }


    // --------------------------------------------------------
    // Scores
    // --------------------------------------------------------

    if (input.contains("scores"))
    {
        auto scores =
            input["scores"].cast<
                py::array_t<float>>();

        auto buf = scores.request();

        if (buf.ndim != 1)
        {
            throw std::runtime_error(
                "Scores must have shape (N)");
        }

        const int n =
            static_cast<int>(buf.shape[0]);

        const float* ptr =
            static_cast<float*>(buf.ptr);

        camera.scores.assign(
            ptr,
            ptr + n
        );
    }


    // --------------------------------------------------------
    // Classes
    // --------------------------------------------------------

    if (input.contains("classes"))
    {
        auto classes =
            input["classes"].cast<
                py::array_t<int>>();

        auto buf = classes.request();

        if (buf.ndim != 1)
        {
            throw std::runtime_error(
                "Classes must have shape (N)");
        }

        const int n =
            static_cast<int>(buf.shape[0]);

        const int* ptr =
            static_cast<int*>(buf.ptr);

        camera.classes.assign(
            ptr,
            ptr + n
        );
    }


    // --------------------------------------------------------
    // Track IDs
    // --------------------------------------------------------

    if (input.contains("track_ids"))
    {
        auto ids =
            input["track_ids"].cast<
                py::array_t<int>>();

        auto buf = ids.request();

        if (buf.ndim != 1)
        {
            throw std::runtime_error(
                "track_ids must have shape (N)");
        }

        const int n =
            static_cast<int>(buf.shape[0]);

        const int* ptr =
            static_cast<int*>(buf.ptr);

        camera.track_ids.assign(
            ptr,
            ptr + n
        );
    }


    // --------------------------------------------------------
    // Track boxes
    // Expected shape: (N, 4)
    // --------------------------------------------------------

    if (input.contains("track_boxes"))
    {
        auto boxes =
            input["track_boxes"].cast<
                py::array_t<float>>();

        auto buf = boxes.request();

        if (buf.ndim != 2 ||
            buf.shape[1] != 4)
        {
            throw std::runtime_error(
                "track_boxes must have shape (N, 4)");
        }

        const int n =
            static_cast<int>(buf.shape[0]);

        const float* ptr =
            static_cast<float*>(buf.ptr);

        camera.track_boxes.reserve(n);

        for (int i = 0; i < n; ++i)
        {
            camera.track_boxes.emplace_back(
                ptr[i * 4 + 0],
                ptr[i * 4 + 1],
                ptr[i * 4 + 2],
                ptr[i * 4 + 3]
            );
        }
    }


    // --------------------------------------------------------
    // Camera metadata
    // --------------------------------------------------------

    if (input.contains("camera_name"))
    {
        camera.camera_name =
            input["camera_name"].cast<std::string>();
    }

    if (input.contains("camera_prefix"))
    {
        camera.camera_prefix =
            input["camera_prefix"].cast<std::string>();
    }


    return camera;
}


// ============================================================
// Helper: WorldObject -> Python dict
// ============================================================

static py::dict world_object_to_dict(
    const Perception3DPipeline::WorldObject& obj)
{
    py::dict d;

    d["class"] = obj.class_name;
    d["camera"] = obj.camera;
    d["box"] = py::make_tuple(
        obj.box[0],
        obj.box[1],
        obj.box[2],
        obj.box[3]
    );

    d["position"] = py::make_tuple(
        obj.position[0],
        obj.position[1],
        obj.position[2]
    );

    d["distance"] = obj.distance;
    d["id"] = obj.id;

    return d;
}


// ============================================================
// Binding
// ============================================================

void bind_perception_orchestrator(
    py::module_& m)
{
    py::class_<PerceptionOrchestrator>(
        m,
        "PerceptionOrchestrator"
    )
        .def(
            py::init<>()
        )

        .def(
            "set_camera_parameters",
            &PerceptionOrchestrator::set_camera_parameters,
            py::arg("fx"),
            py::arg("fy"),
            py::arg("cx"),
            py::arg("cy")
        )

        .def(
            "set_image_size",
            &PerceptionOrchestrator::set_image_size,
            py::arg("width"),
            py::arg("height")
        )

        .def(
            "set_radar_transform",
            &PerceptionOrchestrator::set_radar_transform,
            py::arg("transform")
        )

        .def(
            "process",
            [](
                PerceptionOrchestrator& self,
                const py::dict& front,
                const py::dict& rear,
                const py::dict& left,
                const py::dict& right,
                const py::array_t<float>& lidar_points,
                const py::array_t<float>& radar_points,
                double gnss_latitude,
                double gnss_longitude,
                double gnss_altitude,
                float imu_acceleration_x,
                float imu_acceleration_y,
                float imu_acceleration_z,
                float imu_angular_velocity_z,
                float imu_orientation_z,
                float imu_orientation_w,
                double timestamp
            )
            {
                auto front_cpp =
                    convert_camera_input(front);

                auto rear_cpp =
                    convert_camera_input(rear);

                auto left_cpp =
                    convert_camera_input(left);

                auto right_cpp =
                    convert_camera_input(right);

                auto lidar_cpp =
                    numpy_to_lidar(lidar_points);

                auto radar_cpp =
                    numpy_to_radar(radar_points);

                auto result =
                    self.process(
                        front_cpp,
                        rear_cpp,
                        left_cpp,
                        right_cpp,
                        lidar_cpp,
                        radar_cpp,
                        gnss_latitude,
                        gnss_longitude,
                        gnss_altitude,
                        imu_acceleration_x,
                        imu_acceleration_y,
                        imu_acceleration_z,
                        imu_angular_velocity_z,
                        imu_orientation_z,
                        imu_orientation_w,
                        timestamp
                    );

                py::dict output;

                // ------------------------------------------------
                // Camera results
                // ------------------------------------------------

                auto camera_to_dict =
                    [](const PerceptionOrchestrator::CameraResult& camera)
                    {
                        py::dict d;

                        d["image"] =
                            py::array_t<uint8_t>(
                                {
                                    camera.image.rows,
                                    camera.image.cols,
                                    3
                                },
                                {
                                    camera.image.step[0],
                                    camera.image.step[1],
                                    sizeof(uint8_t)
                                },
                                camera.image.data
                            );

                        py::list objects;

                        for (const auto& obj :
                             camera.world_objects)
                        {
                            objects.append(
                                world_object_to_dict(obj)
                            );
                        }

                        d["world_objects"] = objects;

                        return d;
                    };

                output["front"] =
                    camera_to_dict(result.front);

                output["rear"] =
                    camera_to_dict(result.rear);

                output["left"] =
                    camera_to_dict(result.left);

                output["right"] =
                    camera_to_dict(result.right);


                // ------------------------------------------------
                // World objects
                // ------------------------------------------------

                py::list world_objects;

                for (const auto& obj :
                     result.world_objects)
                {
                    world_objects.append(
                        world_object_to_dict(obj)
                    );
                }

                output["world_objects"] =
                    world_objects;


                // ------------------------------------------------
                // Object counts
                // ------------------------------------------------

                py::dict counts;

                for (const auto& item :
                     result.object_counts)
                {
                    counts[
                        py::str(item.first)
                    ] = item.second;
                }

                output["object_counts"] =
                    counts;


                // ------------------------------------------------
                // Nearest objects
                // ------------------------------------------------

                py::list nearest;

                for (const auto& obj :
                     result.nearest_objects)
                {
                    py::dict d;

                    d["id"] =
                        obj.id;

                    d["cls"] =
                        obj.cls;

                    d["label"] =
                        obj.label;

                    d["dist_m"] =
                        obj.dist_m;

                    d["speed_mps"] =
                        obj.speed_mps;

                    d["motion"] =
                        obj.motion;

                    nearest.append(d);
                }

                output["nearest_objects"] =
                    nearest;


                // ------------------------------------------------
                // BEV objects
                // ------------------------------------------------

                py::list bev;

                for (const auto& obj :
                     result.bev_objects)
                {
                    py::dict d;

                    d["id"] =
                        obj.id;

                    d["cls"] =
                        obj.cls;

                    d["x"] =
                        obj.x;

                    d["y"] =
                        obj.y;

                    d["distance"] =
                        obj.distance;

                    bev.append(d);
                }

                output["bev_objects"] =
                    bev;


                // ------------------------------------------------
                // GNSS
                // ------------------------------------------------

                py::dict gnss;

                gnss["position"] =
                    result.ego.gnss.position;

                gnss["speed"] =
                    result.ego.gnss.speed;

                output["gnss"] =
                    gnss;


                // ------------------------------------------------
                // IMU
                // ------------------------------------------------

                py::dict imu;

                imu["acceleration"] =
                    result.ego.imu.acceleration;

                imu["yaw_rate"] =
                    result.ego.imu.yaw_rate;

                imu["heading"] =
                    result.ego.imu.heading;

                imu["motion_state"] =
                    result.ego.imu.motion_state;

                output["imu"] =
                    imu;

                return output;
            },

            py::arg("front"),
            py::arg("rear"),
            py::arg("left"),
            py::arg("right"),
            py::arg("lidar_points"),
            py::arg("radar_points"),
            py::arg("gnss_latitude"),
            py::arg("gnss_longitude"),
            py::arg("gnss_altitude"),
            py::arg("imu_acceleration_x"),
            py::arg("imu_acceleration_y"),
            py::arg("imu_acceleration_z"),
            py::arg("imu_angular_velocity_z"),
            py::arg("imu_orientation_z"),
            py::arg("imu_orientation_w"),
            py::arg("timestamp")
        );
}