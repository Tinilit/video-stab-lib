cmake_minimum_required(VERSION 3.10)
project(video_stab_lib)

set(OpenCV_DIR "D:/Stabilization/NativeModules/opencv_install")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SRC_FILES src/*.cpp)
file(GLOB HEADER_FILES src/*.h)

add_library(video_stab_lib SHARED
${SRC_FILES} ${HEADER_FILES}
)

target_include_directories(video_stab_lib PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(video_stab_lib PRIVATE ${OpenCV_LIBS})