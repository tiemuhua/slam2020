# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zhuchen/vslam/pa5/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhuchen/vslam/pa5/code/build

# Include any dependencies generated for this target.
include CMakeFiles/E2Rt.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/E2Rt.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/E2Rt.dir/flags.make

CMakeFiles/E2Rt.dir/E2Rt.cpp.o: CMakeFiles/E2Rt.dir/flags.make
CMakeFiles/E2Rt.dir/E2Rt.cpp.o: ../E2Rt.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhuchen/vslam/pa5/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/E2Rt.dir/E2Rt.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/E2Rt.dir/E2Rt.cpp.o -c /home/zhuchen/vslam/pa5/code/E2Rt.cpp

CMakeFiles/E2Rt.dir/E2Rt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/E2Rt.dir/E2Rt.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhuchen/vslam/pa5/code/E2Rt.cpp > CMakeFiles/E2Rt.dir/E2Rt.cpp.i

CMakeFiles/E2Rt.dir/E2Rt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/E2Rt.dir/E2Rt.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhuchen/vslam/pa5/code/E2Rt.cpp -o CMakeFiles/E2Rt.dir/E2Rt.cpp.s

CMakeFiles/E2Rt.dir/E2Rt.cpp.o.requires:

.PHONY : CMakeFiles/E2Rt.dir/E2Rt.cpp.o.requires

CMakeFiles/E2Rt.dir/E2Rt.cpp.o.provides: CMakeFiles/E2Rt.dir/E2Rt.cpp.o.requires
	$(MAKE) -f CMakeFiles/E2Rt.dir/build.make CMakeFiles/E2Rt.dir/E2Rt.cpp.o.provides.build
.PHONY : CMakeFiles/E2Rt.dir/E2Rt.cpp.o.provides

CMakeFiles/E2Rt.dir/E2Rt.cpp.o.provides.build: CMakeFiles/E2Rt.dir/E2Rt.cpp.o


# Object files for target E2Rt
E2Rt_OBJECTS = \
"CMakeFiles/E2Rt.dir/E2Rt.cpp.o"

# External object files for target E2Rt
E2Rt_EXTERNAL_OBJECTS =

E2Rt: CMakeFiles/E2Rt.dir/E2Rt.cpp.o
E2Rt: CMakeFiles/E2Rt.dir/build.make
E2Rt: /usr/local/lib/libpangolin.so
E2Rt: /usr/local/opencv/lib/libopencv_video.so.4.5.0
E2Rt: /usr/local/opencv/lib/libopencv_ml.so.4.5.0
E2Rt: /usr/local/opencv/lib/libopencv_stitching.so.4.5.0
E2Rt: /usr/local/opencv/lib/libopencv_highgui.so.4.5.0
E2Rt: /usr/local/opencv/lib/libopencv_dnn.so.4.5.0
E2Rt: /usr/local/opencv/lib/libopencv_photo.so.4.5.0
E2Rt: /usr/local/opencv/lib/libopencv_objdetect.so.4.5.0
E2Rt: /usr/local/opencv/lib/libopencv_videoio.so.4.5.0
E2Rt: /usr/lib/x86_64-linux-gnu/libGLU.so
E2Rt: /usr/lib/x86_64-linux-gnu/libGL.so
E2Rt: /usr/lib/x86_64-linux-gnu/libGLEW.so
E2Rt: /usr/lib/x86_64-linux-gnu/libSM.so
E2Rt: /usr/lib/x86_64-linux-gnu/libICE.so
E2Rt: /usr/lib/x86_64-linux-gnu/libX11.so
E2Rt: /usr/lib/x86_64-linux-gnu/libXext.so
E2Rt: /usr/lib/x86_64-linux-gnu/libGLU.so
E2Rt: /usr/lib/x86_64-linux-gnu/libGL.so
E2Rt: /usr/lib/x86_64-linux-gnu/libGLEW.so
E2Rt: /usr/lib/x86_64-linux-gnu/libSM.so
E2Rt: /usr/lib/x86_64-linux-gnu/libICE.so
E2Rt: /usr/lib/x86_64-linux-gnu/libX11.so
E2Rt: /usr/lib/x86_64-linux-gnu/libXext.so
E2Rt: /usr/lib/x86_64-linux-gnu/libdc1394.so
E2Rt: /usr/lib/x86_64-linux-gnu/libavcodec.so
E2Rt: /usr/lib/x86_64-linux-gnu/libavformat.so
E2Rt: /usr/lib/x86_64-linux-gnu/libavutil.so
E2Rt: /usr/lib/x86_64-linux-gnu/libswscale.so
E2Rt: /usr/lib/x86_64-linux-gnu/libavdevice.so
E2Rt: /usr/lib/libOpenNI.so
E2Rt: /usr/lib/x86_64-linux-gnu/libpng.so
E2Rt: /usr/lib/x86_64-linux-gnu/libz.so
E2Rt: /usr/lib/x86_64-linux-gnu/libjpeg.so
E2Rt: /usr/lib/x86_64-linux-gnu/libtiff.so
E2Rt: /usr/lib/x86_64-linux-gnu/libIlmImf.so
E2Rt: /usr/lib/x86_64-linux-gnu/liblz4.so
E2Rt: /usr/local/opencv/lib/libopencv_imgcodecs.so.4.5.0
E2Rt: /usr/local/opencv/lib/libopencv_calib3d.so.4.5.0
E2Rt: /usr/local/opencv/lib/libopencv_features2d.so.4.5.0
E2Rt: /usr/local/opencv/lib/libopencv_imgproc.so.4.5.0
E2Rt: /usr/local/opencv/lib/libopencv_flann.so.4.5.0
E2Rt: /usr/local/opencv/lib/libopencv_core.so.4.5.0
E2Rt: CMakeFiles/E2Rt.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhuchen/vslam/pa5/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable E2Rt"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/E2Rt.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/E2Rt.dir/build: E2Rt

.PHONY : CMakeFiles/E2Rt.dir/build

CMakeFiles/E2Rt.dir/requires: CMakeFiles/E2Rt.dir/E2Rt.cpp.o.requires

.PHONY : CMakeFiles/E2Rt.dir/requires

CMakeFiles/E2Rt.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/E2Rt.dir/cmake_clean.cmake
.PHONY : CMakeFiles/E2Rt.dir/clean

CMakeFiles/E2Rt.dir/depend:
	cd /home/zhuchen/vslam/pa5/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhuchen/vslam/pa5/code /home/zhuchen/vslam/pa5/code /home/zhuchen/vslam/pa5/code/build /home/zhuchen/vslam/pa5/code/build /home/zhuchen/vslam/pa5/code/build/CMakeFiles/E2Rt.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/E2Rt.dir/depend

