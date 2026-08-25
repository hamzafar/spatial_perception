#include "perception_3d_pipeline.hpp"

#include <iostream>

int main()
{
    Perception3DPipeline pipeline;

    std::vector<Eigen::Vector3f> lidar = {
        {10.0f, 0.0f, 1.5f},
        {10.0f, 2.0f, 1.5f},
        {10.0f, -2.0f, 1.5f},
        {-5.0f, 0.0f, 1.5f}
    };

    auto result = pipeline.project_lidar(
        lidar,
        "front",
        640,
        480
    );

    std::cout << "Projected points: "
              << result.u.size() << "\n";

    for (size_t i = 0; i < result.u.size(); ++i)
    {
        std::cout
            << "u=" << result.u[i]
            << ", v=" << result.v[i]
            << ", xyz="
            << result.ego_points[i].transpose()
            << "\n";
    }

    return 0;
}