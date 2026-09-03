cmake_minimum_required(VERSION 3.16)

project(perception_cpp)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

pybind11_add_module(perception_cpp
    src/bindings.cpp
    src/perception_3d_pipeline.cpp
    src/perception_utils.cpp
    src/radar_perception_pipeline.cpp
)

target_compile_options(perception_cpp
    PRIVATE
    -O3
    -DNDEBUG
    -march=native
)

target_include_directories(perception_cpp
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(perception_cpp
    PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
)