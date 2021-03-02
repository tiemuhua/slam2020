cmake_minimum_required(VERSION 2.8)
project(part_four)
add_compile_options(-std=c++11)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

find_package(Pangolin REQUIRED)

find_package(OpenCV 4.4.0 REQUIRED
    PATHS /usr/local/opencv
    NO_DEFAULT_PATH
)

find_package(Eigen3 3.3.0 REQUIRED
             PATHS ~/3rd-party/eigen-3.3.90/install/ # look here
             NO_DEFAULT_PATH)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

message("opencv lib" ${OpenCV_LIBS_DIRS})

add_executable(disparity_exe disparity.cpp)
target_link_libraries(disparity_exe 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${OpenCV_LIBS}
)

add_executable(un_distort_image undistort_image.cpp)
target_link_libraries(un_distort_image 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${OpenCV_LIBS}
)

add_executable(gaussnewton gaussnewton.cpp)
target_link_libraries(gaussnewton 
${EIGEN3_LIBS}
${OpenCV_LIBS}
)

add_executable(test test.cpp)
target_link_libraries(test 
${EIGEN3_LIBS}
)