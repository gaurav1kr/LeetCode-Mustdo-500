# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/build

# Include any dependencies generated for this target.
include CMakeFiles/FDP.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/FDP.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/FDP.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FDP.dir/flags.make

CMakeFiles/FDP.dir/ToyFactory.cpp.o: CMakeFiles/FDP.dir/flags.make
CMakeFiles/FDP.dir/ToyFactory.cpp.o: ../ToyFactory.cpp
CMakeFiles/FDP.dir/ToyFactory.cpp.o: CMakeFiles/FDP.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FDP.dir/ToyFactory.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/FDP.dir/ToyFactory.cpp.o -MF CMakeFiles/FDP.dir/ToyFactory.cpp.o.d -o CMakeFiles/FDP.dir/ToyFactory.cpp.o -c /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/ToyFactory.cpp

CMakeFiles/FDP.dir/ToyFactory.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FDP.dir/ToyFactory.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/ToyFactory.cpp > CMakeFiles/FDP.dir/ToyFactory.cpp.i

CMakeFiles/FDP.dir/ToyFactory.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FDP.dir/ToyFactory.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/ToyFactory.cpp -o CMakeFiles/FDP.dir/ToyFactory.cpp.s

CMakeFiles/FDP.dir/Object.cpp.o: CMakeFiles/FDP.dir/flags.make
CMakeFiles/FDP.dir/Object.cpp.o: ../Object.cpp
CMakeFiles/FDP.dir/Object.cpp.o: CMakeFiles/FDP.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/FDP.dir/Object.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/FDP.dir/Object.cpp.o -MF CMakeFiles/FDP.dir/Object.cpp.o.d -o CMakeFiles/FDP.dir/Object.cpp.o -c /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/Object.cpp

CMakeFiles/FDP.dir/Object.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FDP.dir/Object.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/Object.cpp > CMakeFiles/FDP.dir/Object.cpp.i

CMakeFiles/FDP.dir/Object.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FDP.dir/Object.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/Object.cpp -o CMakeFiles/FDP.dir/Object.cpp.s

CMakeFiles/FDP.dir/Client.cpp.o: CMakeFiles/FDP.dir/flags.make
CMakeFiles/FDP.dir/Client.cpp.o: ../Client.cpp
CMakeFiles/FDP.dir/Client.cpp.o: CMakeFiles/FDP.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/FDP.dir/Client.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/FDP.dir/Client.cpp.o -MF CMakeFiles/FDP.dir/Client.cpp.o.d -o CMakeFiles/FDP.dir/Client.cpp.o -c /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/Client.cpp

CMakeFiles/FDP.dir/Client.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FDP.dir/Client.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/Client.cpp > CMakeFiles/FDP.dir/Client.cpp.i

CMakeFiles/FDP.dir/Client.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FDP.dir/Client.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/Client.cpp -o CMakeFiles/FDP.dir/Client.cpp.s

# Object files for target FDP
FDP_OBJECTS = \
"CMakeFiles/FDP.dir/ToyFactory.cpp.o" \
"CMakeFiles/FDP.dir/Object.cpp.o" \
"CMakeFiles/FDP.dir/Client.cpp.o"

# External object files for target FDP
FDP_EXTERNAL_OBJECTS =

../bin/FDP: CMakeFiles/FDP.dir/ToyFactory.cpp.o
../bin/FDP: CMakeFiles/FDP.dir/Object.cpp.o
../bin/FDP: CMakeFiles/FDP.dir/Client.cpp.o
../bin/FDP: CMakeFiles/FDP.dir/build.make
../bin/FDP: CMakeFiles/FDP.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ../bin/FDP"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FDP.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FDP.dir/build: ../bin/FDP
.PHONY : CMakeFiles/FDP.dir/build

CMakeFiles/FDP.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FDP.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FDP.dir/clean

CMakeFiles/FDP.dir/depend:
	cd /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/build /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/build /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/build/CMakeFiles/FDP.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FDP.dir/depend

