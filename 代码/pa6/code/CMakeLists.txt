cmake_minimum_required(VERSION 2.8)
project(part_five)
add_compile_options(-std=c++11)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

find_package(Pangolin REQUIRED)

find_package(OpenCV 4.4.0 REQUIRED
    PATHS /usr/local/opencv/lib/cmake/opencv4/
    NO_DEFAULT_PATH
)

find_package(Eigen3
    PATHS /usr/local/share/eigen3/cmake
    NO_DEFAULT_PATH
)
find_package(Sophus
    PATHS /usr/local/share/sophus/cmake/
    NO_DEFAULT_PATH
)
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

message("opencv lib" ${OpenCV_LIBS})
message("eigen path:" ${EIGEN3_INCLUDE_DIR})
message("pangolin path" ${Pangolin_INCLUDE_DIR})
message("Sophus_INCLUDE_DIRS" ${SOPHUS_INCLUDE_DIRS})

add_executable(optical_flow optical_flow.cpp)
target_link_libraries(optical_flow 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${OpenCV_LIBS}
)
add_executable(test_addressing_time test_addressing_time.cpp)
target_link_libraries(test_addressing_time 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${OpenCV_LIBS}
)

add_executable(direct_method direct_method.cpp)
target_link_libraries(direct_method 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${OpenCV_LIBS}
)
add_executable(culculate_disparity culculate_disparity.cpp)
target_link_libraries(culculate_disparity 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${OpenCV_LIBS}
)