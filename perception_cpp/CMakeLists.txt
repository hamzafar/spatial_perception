cmake_minimum_required(VERSION 3.16)

project(perception_cpp)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

pybind11_add_module(perception_cpp
    src/bindings.cpp
    src/perception_3d_pipeline.cpp
)

target_include_directories(perception_cpp
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(perception_cpp
    PRIVATE
    Eigen3::Eigen
)