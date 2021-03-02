cmake_minimum_required(VERSION 2.8)
project(draw_traj)
add_compile_options(-std=c++11)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

find_package(Eigen3 3.3.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Sophus REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
)

message("Sophus_INCLUDE_DIRS" ${SOPHUS_INCLUDE_DIRS})
message("EIGEN3_INCLUDE_DIR" ${EIGEN3_INCLUDE_DIR})

add_executable(draw_traj_exe draw_trajectory.cpp)
target_link_libraries(draw_traj_exe 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
)

add_executable(culculate_error_exe culculate_error.cpp)
target_link_libraries(culculate_error_exe 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
)