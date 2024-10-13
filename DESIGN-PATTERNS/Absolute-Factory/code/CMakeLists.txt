cmake_minimum_required(VERSION 3.10)

# Project name and language
project(MyCppProject LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the output directory for binaries (e.g., bin folder)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Add preprocessor definition (LUXURY_CAR)
# In case if simple car , change it to SIMPLE_CAR
add_definitions(-DLUXURY_CAR)

# Add source files
set(SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/CarFactory.cpp
    ${CMAKE_SOURCE_DIR}/Car.cpp
    ${CMAKE_SOURCE_DIR}/client.cpp
)

# Add executable target
add_executable(AFDP ${SOURCE_FILES})

