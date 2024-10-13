cmake_minimum_required(VERSION 3.10)

# Project name and language
project(MyCppProject LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the output directory for binaries (e.g., bin folder)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/bin)

# Add executable (no need to specify output path here, just executable name)
add_executable(FDP
	    /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/ToyFactory.cpp
	    /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/Object.cpp
	    /home/gakumar/LeetCode-Mustdo-500/DESIGN-PATTERNS/Factory/code/Client.cpp
	    )

