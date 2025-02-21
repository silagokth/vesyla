cmake_minimum_required(VERSION 3.10)

# Set the project name
project(timingModel)

# Add the executable
add_library(timingModel timingModel.cpp)
set_property(TARGET timingModel PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(timingModel PRIVATE .)