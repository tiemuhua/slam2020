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
CMAKE_SOURCE_DIR = /home/zhuchen/shenlan-slam/pa3/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zhuchen/shenlan-slam/pa3/code/build

# Include any dependencies generated for this target.
include CMakeFiles/draw_traj_exe.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/draw_traj_exe.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/draw_traj_exe.dir/flags.make

CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.o: CMakeFiles/draw_traj_exe.dir/flags.make
CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.o: ../draw_trajectory.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/zhuchen/shenlan-slam/pa3/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.o -c /home/zhuchen/shenlan-slam/pa3/code/draw_trajectory.cpp

CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/zhuchen/shenlan-slam/pa3/code/draw_trajectory.cpp > CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.i

CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/zhuchen/shenlan-slam/pa3/code/draw_trajectory.cpp -o CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.s

CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.o.requires:

.PHONY : CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.o.requires

CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.o.provides: CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.o.requires
	$(MAKE) -f CMakeFiles/draw_traj_exe.dir/build.make CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.o.provides.build
.PHONY : CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.o.provides

CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.o.provides.build: CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.o


# Object files for target draw_traj_exe
draw_traj_exe_OBJECTS = \
"CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.o"

# External object files for target draw_traj_exe
draw_traj_exe_EXTERNAL_OBJECTS =

draw_traj_exe: CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.o
draw_traj_exe: CMakeFiles/draw_traj_exe.dir/build.make
draw_traj_exe: /home/zhuchen/3rd-party/pangolin/install/lib/libpangolin.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libGLU.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libGL.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libGLEW.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libSM.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libICE.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libX11.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libXext.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libGLU.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libGL.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libGLEW.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libSM.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libICE.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libX11.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libXext.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libdc1394.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libavcodec.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libavformat.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libavutil.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libswscale.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libavdevice.so
draw_traj_exe: /usr/lib/libOpenNI.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libpng.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libz.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libjpeg.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libtiff.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/libIlmImf.so
draw_traj_exe: /usr/lib/x86_64-linux-gnu/liblz4.so
draw_traj_exe: CMakeFiles/draw_traj_exe.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/zhuchen/shenlan-slam/pa3/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable draw_traj_exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/draw_traj_exe.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/draw_traj_exe.dir/build: draw_traj_exe

.PHONY : CMakeFiles/draw_traj_exe.dir/build

CMakeFiles/draw_traj_exe.dir/requires: CMakeFiles/draw_traj_exe.dir/draw_trajectory.cpp.o.requires

.PHONY : CMakeFiles/draw_traj_exe.dir/requires

CMakeFiles/draw_traj_exe.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/draw_traj_exe.dir/cmake_clean.cmake
.PHONY : CMakeFiles/draw_traj_exe.dir/clean

CMakeFiles/draw_traj_exe.dir/depend:
	cd /home/zhuchen/shenlan-slam/pa3/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zhuchen/shenlan-slam/pa3/code /home/zhuchen/shenlan-slam/pa3/code /home/zhuchen/shenlan-slam/pa3/code/build /home/zhuchen/shenlan-slam/pa3/code/build /home/zhuchen/shenlan-slam/pa3/code/build/CMakeFiles/draw_traj_exe.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/draw_traj_exe.dir/depend

