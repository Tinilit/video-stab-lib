cmake_minimum_required(VERSION 3.10)
project(video_stab_lib)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

add_library(video_stab_lib SHARED
    src/StabilizerWrapper.cpp
    src/exports.cpp
)

target_include_directories(video_stab_lib PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(video_stab_lib ${OpenCV_LIBS})