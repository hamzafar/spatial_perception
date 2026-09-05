#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "perception_3d_pipeline.hpp"
#include "perception_utils.hpp"
#include "radar_perception_pipeline.hpp"

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
    // WorldObject
    // --------------------------------------------------

    py::class_<Perception3DPipeline::WorldObject>(
        m,
        "WorldObject"
    )
        .def_readonly(
            "class_name",
            &Perception3DPipeline::WorldObject::class_name
        )
        .def_readonly(
            "camera",
            &Perception3DPipeline::WorldObject::camera
        )
        .def_readonly(
            "box",
            &Perception3DPipeline::WorldObject::box
        )
        .def_readonly(
            "position",
            &Perception3DPipeline::WorldObject::position
        )
        .def_readonly(
            "distance",
            &Perception3DPipeline::WorldObject::distance
        )
        .def_readonly("id",
              &Perception3DPipeline::WorldObject::id)
        .def_readonly(
            "has_radar",
            &Perception3DPipeline::WorldObject::has_radar
        )
        .def_readonly(
            "radar_range",
            &Perception3DPipeline::WorldObject::radar_range
        )
        .def_readonly(
            "radar_bearing",
            &Perception3DPipeline::WorldObject::radar_bearing
        )
        .def_readonly(
            "radar_velocity",
            &Perception3DPipeline::WorldObject::radar_velocity
        )
        .def_readonly(
            "radar_motion",
            &Perception3DPipeline::WorldObject::radar_motion
        );


    py::class_<PerceptionUtils>(
        m,
        "PerceptionUtils"
    )
        .def(
            py::init<>()
        )
        
    // --------------------------------------------------
    // PrepareNearestObjects
    // --------------------------------------------------

        .def(
            "prepare_nearest_objects",
            [](PerceptionUtils& self,
            py::list world_objects,
            double timestamp)
            {
                std::vector<Perception3DPipeline::WorldObject>
                    world_objects_cpp;

                world_objects_cpp.reserve(
                    world_objects.size()
                );

                for (auto item : world_objects)
                {
                    py::dict py_obj = item.cast<py::dict>();

                    Perception3DPipeline::WorldObject obj;

                    obj.class_name =
                        py_obj["class"].cast<std::string>();

                    obj.camera =
                        py_obj["camera"].cast<std::string>();

                    obj.box =
                        py_obj["box"].cast<
                            Eigen::Vector4f
                        >();

                    obj.position =
                        py_obj["position"].cast<
                            Eigen::Vector3f
                        >();

                    obj.distance =
                        py_obj["distance"].cast<float>();

                    if (py_obj.contains("id"))
                    {
                        obj.id =
                            py_obj["id"].cast<std::string>();
                    }

                    if (py_obj.contains("radar"))
                    {
                        py::dict radar =
                            py_obj["radar"].cast<py::dict>();

                        obj.has_radar = true;

                        obj.radar_range =
                            radar["range"].cast<float>();

                        obj.radar_bearing =
                            radar["bearing"].cast<float>();

                        obj.radar_velocity =
                            radar["velocity"].cast<float>();

                        obj.radar_motion =
                            radar["motion"].cast<std::string>();
                    }

                    world_objects_cpp.push_back(obj);
                }

                auto result =
                    self.prepare_nearest_objects(
                        world_objects_cpp,
                        timestamp
                    );

                py::list output;

                for (const auto& obj : result)
                {
                    py::dict item;

                    item["id"] = obj.id;
                    item["cls"] = obj.cls;
                    item["label"] = obj.label;
                    item["dist_m"] = obj.dist_m;
                    item["speed_mps"] = obj.speed_mps;
                    item["motion"] = obj.motion;

                    output.append(item);
                }

                return output;
            },
            py::arg("world_objects"),
            py::arg("timestamp")
        )

    // --------------------------------------------------
    // AttachTrackIds
    // --------------------------------------------------
        .def(
            "attach_track_ids",
            [](PerceptionUtils& self,
            std::vector<Perception3DPipeline::WorldObject>& world_objects,
            py::list online_targets,
            const std::string& camera_prefix)
            {
                std::vector<PerceptionUtils::TrackTarget> targets;

                targets.reserve(online_targets.size());

                for (auto target : online_targets)
                {
                    PerceptionUtils::TrackTarget t;

                    t.track_id =
                        target.attr("track_id").cast<int>();

                    auto tlwh =
                        target.attr("tlwh").cast<py::sequence>();

                    t.x =
                        tlwh[0].cast<float>();

                    t.y =
                        tlwh[1].cast<float>();

                    t.width =
                        tlwh[2].cast<float>();

                    t.height =
                        tlwh[3].cast<float>();

                    targets.push_back(t);
                }

                return self.attach_track_ids(
                    world_objects,
                    targets,
                    camera_prefix
                );
            },
            py::arg("world_objects"),
            py::arg("online_targets"),
            py::arg("camera_prefix")
        )
    // --------------------------------------------------
    // AttachRadarData
    // --------------------------------------------------
        .def(
            "attach_radar_data",
            [](PerceptionUtils& self,
            std::vector<Perception3DPipeline::WorldObject>& world_objects,
            py::list radar_objects,
            float iou_threshold)
            {
                std::vector<RadarPerceptionPipeline::RadarObject> radar_objects_cpp;
                radar_objects_cpp.reserve(radar_objects.size());

                for (auto item : radar_objects)
                {
                    py::dict radar = item.cast<py::dict>();

                    RadarPerceptionPipeline::RadarObject obj;

                    auto bbox = radar["bbox"].cast<py::sequence>();

                    obj.box <<
                        bbox[0].cast<float>(),
                        bbox[1].cast<float>(),
                        bbox[2].cast<float>(),
                        bbox[3].cast<float>();

                    obj.confidence =
                        radar["confidence"].cast<float>();

                    obj.class_id =
                        radar["class_id"].cast<int>();

                    obj.range =
                        radar["range"].cast<float>();

                    obj.bearing =
                        radar["bearing"].cast<float>();

                    obj.velocity =
                        radar["velocity"].cast<float>();

                    obj.motion =
                        radar["motion"].cast<std::string>();

                    radar_objects_cpp.push_back(obj);
                }

                return self.attach_radar_data(
                    world_objects,
                    radar_objects_cpp,
                    iou_threshold
                );
            },
            py::arg("world_objects"),
            py::arg("radar_objects"),
            py::arg("iou_threshold") = 0.3f
        )
    // --------------------------------------------------
    // CountObjects
    // --------------------------------------------------       
        .def(
            "count_objects",
            [](PerceptionUtils& self, py::list world_objects)
            {
                std::vector<Perception3DPipeline::WorldObject> world_objects_cpp;
                world_objects_cpp.reserve(world_objects.size());

                for (auto item : world_objects)
                {
                    py::dict py_obj = item.cast<py::dict>();

                    Perception3DPipeline::WorldObject obj;

                    obj.class_name =
                        py_obj["class"].cast<std::string>();

                    world_objects_cpp.push_back(obj);
                }

                return self.count_objects(world_objects_cpp);
            },
            py::arg("world_objects")
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
    // RadarPerceptionPipeline
    // --------------------------------------------------

    py::class_<RadarPerceptionPipeline>(
        m,
        "RadarPerceptionPipeline"
    )
        .def(
            py::init<>()
        )

        .def(
            "set_velocity_threshold",
            &RadarPerceptionPipeline::set_velocity_threshold,
            py::arg("threshold")
        )

        .def(
            "set_max_assoc_distance_px",
            &RadarPerceptionPipeline::set_max_assoc_distance_px,
            py::arg("distance")
        )

        .def(
            "set_camera_intrinsics",
            &RadarPerceptionPipeline::set_camera_intrinsics,
            py::arg("fx"),
            py::arg("fy"),
            py::arg("cx"),
            py::arg("cy"),
            py::arg("width"),
            py::arg("height")
        )

        .def(
            "set_radar_to_camera_transform",
            &RadarPerceptionPipeline::set_radar_to_camera_transform,
            py::arg("transform")
        )

        .def(
            "process",
            [](RadarPerceptionPipeline& self,

               py::array_t<float,
                   py::array::c_style |
                   py::array::forcecast> boxes,

               py::array_t<float,
                   py::array::c_style |
                   py::array::forcecast> scores,

               py::array_t<float,
                   py::array::c_style |
                   py::array::forcecast> classes,

               py::array_t<float,
                   py::array::c_style |
                   py::array::forcecast> radar_targets)
            {
                // --------------------------------------------------
                // Validate boxes
                // NumPy shape: (N, 4)
                // --------------------------------------------------

                auto boxes_buf = boxes.request();

                if (boxes_buf.ndim != 2 ||
                    boxes_buf.shape[1] != 4)
                {
                    throw std::runtime_error(
                        "boxes must have shape (N, 4)"
                    );
                }


                // --------------------------------------------------
                // Validate scores
                // NumPy shape: (N,)
                // --------------------------------------------------

                auto scores_buf = scores.request();

                if (scores_buf.ndim != 1)
                {
                    throw std::runtime_error(
                        "scores must be a 1D NumPy array"
                    );
                }


                // --------------------------------------------------
                // Validate classes
                // NumPy shape: (N,)
                // --------------------------------------------------

                auto classes_buf = classes.request();

                if (classes_buf.ndim != 1)
                {
                    throw std::runtime_error(
                        "classes must be a 1D NumPy array"
                    );
                }


                // --------------------------------------------------
                // Validate radar targets
                // NumPy shape: (M, 4)
                //
                // [depth, azimuth, altitude, velocity]
                // --------------------------------------------------

                auto radar_buf = radar_targets.request();

                if (radar_buf.ndim != 2 ||
                    radar_buf.shape[1] != 4)
                {
                    throw std::runtime_error(
                        "radar_targets must have shape (N, 4)"
                    );
                }


                // --------------------------------------------------
                // Validate detection array sizes
                // --------------------------------------------------

                const ssize_t num_detections =
                    boxes_buf.shape[0];

                if (scores_buf.shape[0] != num_detections ||
                    classes_buf.shape[0] != num_detections)
                {
                    throw std::runtime_error(
                        "boxes, scores, and classes must have "
                        "the same number of detections"
                    );
                }


                // --------------------------------------------------
                // NumPy -> C++ boxes
                // --------------------------------------------------

                const float* boxes_ptr =
                    static_cast<const float*>(
                        boxes_buf.ptr
                    );

                std::vector<Eigen::Vector4f> boxes_cpp;

                boxes_cpp.reserve(
                    static_cast<size_t>(num_detections)
                );

                for (ssize_t i = 0;
                     i < num_detections;
                     ++i)
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
                // NumPy -> C++ scores
                // --------------------------------------------------

                const float* scores_ptr =
                    static_cast<const float*>(
                        scores_buf.ptr
                    );

                std::vector<float> scores_cpp(
                    scores_ptr,
                    scores_ptr + scores_buf.shape[0]
                );


                // --------------------------------------------------
                // NumPy -> C++ classes
                // --------------------------------------------------

                const float* classes_ptr =
                    static_cast<const float*>(
                        classes_buf.ptr
                    );

                std::vector<int> classes_cpp;

                classes_cpp.reserve(
                    static_cast<size_t>(
                        classes_buf.shape[0]
                    )
                );

                for (ssize_t i = 0;
                     i < classes_buf.shape[0];
                     ++i)
                {
                    classes_cpp.push_back(
                        static_cast<int>(
                            classes_ptr[i]
                        )
                    );
                }


                // --------------------------------------------------
                // NumPy -> C++ radar targets
                //
                // [depth, azimuth, altitude, velocity]
                // --------------------------------------------------

                const float* radar_ptr =
                    static_cast<const float*>(
                        radar_buf.ptr
                    );

                const ssize_t num_radar_targets =
                    radar_buf.shape[0];

                std::vector<Eigen::Vector4f>
                    radar_targets_cpp;

                radar_targets_cpp.reserve(
                    static_cast<size_t>(
                        num_radar_targets
                    )
                );

                for (ssize_t i = 0;
                     i < num_radar_targets;
                     ++i)
                {
                    Eigen::Vector4f target;

                    target <<
                        radar_ptr[i * 4 + 0],
                        radar_ptr[i * 4 + 1],
                        radar_ptr[i * 4 + 2],
                        radar_ptr[i * 4 + 3];

                    radar_targets_cpp.push_back(
                        target
                    );
                }


                // --------------------------------------------------
                // Call C++ implementation
                // --------------------------------------------------

                auto objects =
                    self.process(
                        boxes_cpp,
                        scores_cpp,
                        classes_cpp,
                        radar_targets_cpp
                    );


                // --------------------------------------------------
                // C++ -> Python
                //
                // Return exactly the structure expected from
                // Python RadarPerceptionPipeline.process()
                // --------------------------------------------------

                py::list result;

                for (const auto& object : objects)
                {
                    py::dict obj;

                    obj["bbox"] = py::make_tuple(
                        static_cast<int>(object.box[0]),
                        static_cast<int>(object.box[1]),
                        static_cast<int>(object.box[2]),
                        static_cast<int>(object.box[3])
                    );

                    obj["confidence"] =
                        object.confidence;

                    obj["class_id"] =
                        object.class_id;

                    obj["range"] =
                        object.range;

                    obj["bearing"] =
                        object.bearing;

                    obj["velocity"] =
                        object.velocity;

                    obj["motion"] =
                        object.motion;

                    result.append(obj);
                }

                return result;
            },

            py::arg("boxes"),
            py::arg("scores"),
            py::arg("classes"),
            py::arg("radar_targets")
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
        )

        // --------------------------------------------------
        // Process object clouds and distance
        // --------------------------------------------------

        .def(
            "process_object_clouds_and_distance",
            [](Perception3DPipeline& self,
            py::array_t<uint8_t, py::array::c_style | py::array::forcecast> image,
            const std::vector<Perception3DPipeline::ObjectCloud>& object_clouds,
            const std::string& camera_name)
            {
                auto image_buf = image.request();

                if (image_buf.ndim != 3 ||
                    image_buf.shape[2] != 3)
                {
                    throw std::runtime_error(
                        "image must have shape (H, W, 3)"
                    );
                }

                cv::Mat image_cpp(
                    static_cast<int>(image_buf.shape[0]),
                    static_cast<int>(image_buf.shape[1]),
                    CV_8UC3,
                    image_buf.ptr
                );

                auto result =
                    self.process_object_clouds_and_distance(
                        image_cpp,
                        object_clouds,
                        camera_name
                    );

                py::array_t<uint8_t> output_image({
                    static_cast<size_t>(result.image.rows),
                    static_cast<size_t>(result.image.cols),
                    static_cast<size_t>(3)
                });

                std::memcpy(
                    output_image.mutable_data(),
                    result.image.data,
                    result.image.total() *
                    result.image.elemSize()
                );

                return py::make_tuple(
                    output_image,
                    result.world_objects
                );
            },
            py::arg("image"),
            py::arg("object_clouds"),
            py::arg("camera_name")
        );

        
}