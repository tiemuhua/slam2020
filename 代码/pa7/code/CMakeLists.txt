cmake_minimum_required(VERSION 2.8)
project(part_five)
add_compile_options(-std=c++11)
set(CMAKE_CXX_STANDARD 11)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(Pangolin REQUIRED)

find_package(OpenCV 4.4.0 REQUIRED
    PATHS /usr/local/opencv/lib/cmake/opencv4/
    NO_DEFAULT_PATH
)

find_package(Eigen3
    PATHS /usr/local/share/eigen3/cmake
    NO_DEFAULT_PATH
)

find_package(g2o REQUIRED
    PATHS /usr/local/lib/cmake/g2o
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
    ${CSPARSE_INCLUDE_DIR}
)

message("opencv lib" ${OpenCV_LIBS})
message("eigen path:" ${EIGEN3_INCLUDE_DIR})
message("pangolin path" ${Pangolin_INCLUDE_DIR})
message("g2o:\t" ${CSPARSE_INCLUDE_DIR})
message("g2o:\t" ${CHOLMOD_INCLUDE_DIR})

add_executable(my_direct_ba my_direct_ba.cpp)
target_link_libraries(my_direct_ba 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${OpenCV_LIBS}
/usr/local/lib/libg2o_solver_structure_only.so 
/usr/local/lib/libg2o_solver_slam2d_linear.so 
/usr/local/lib/libg2o_cli.so
/usr/local/lib/libg2o_stuff.so
/usr/local/lib/libg2o_core.so
/usr/local/lib/libg2o_types_data.so
/usr/local/lib/libg2o_csparse_extension.so   
/usr/local/lib/libg2o_types_icp.so           
/usr/local/lib/libg2o_ext_csparse.so         
/usr/local/lib/libg2o_types_sba.so           
/usr/local/lib/libg2o_ext_freeglut_minimal.so
/usr/local/lib/libg2o_types_sclam2d.so       
/usr/local/lib/libg2o_hierarchical.so        
/usr/local/lib/libg2o_types_sim3.so          
/usr/local/lib/libg2o_interface.so           
/usr/local/lib/libg2o_types_slam2d_addons.so 
/usr/local/lib/libg2o_opengl_helper.so       
/usr/local/lib/libg2o_types_slam2d.so
/usr/local/lib/libg2o_parser.so
/usr/local/lib/libg2o_types_slam3d_addons.so
/usr/local/lib/libg2o_simulator.so
/usr/local/lib/libg2o_types_slam3d.so
/usr/local/lib/libg2o_solver_csparse.so
/usr/local/lib/libg2o_solver_dense.so                    
/usr/local/lib/libg2o_solver_eigen.so                    
/usr/local/lib/libg2o_solver_pcg.so  
)