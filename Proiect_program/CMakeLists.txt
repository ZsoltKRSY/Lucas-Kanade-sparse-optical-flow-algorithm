cmake_minimum_required(VERSION 3.24)
project(Proiect_program)

set(CMAKE_CXX_STANDARD 17)

# --- Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Proiect_program main.cpp preprocessing.h preprocessing.cpp opticalflow.h opticalflow.cpp utility.h utility.cpp cornerpoints.h cornerpoints.cpp)

target_link_libraries(Proiect_program ${OpenCV_LIBS})
