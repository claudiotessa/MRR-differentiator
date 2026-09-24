cmake_minimum_required(VERSION 3.16)

project(MRRDifferentiator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

add_executable(MRR main.cpp MRR.cpp MRR.hpp MRRCascade.hpp MRRCascade.cpp Simulation.cpp Simulation.hpp Plotter.hpp Plotter.cpp MonteCarlo.hpp MonteCarlo.cpp)

target_include_directories(MRR PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(MRR PRIVATE
    Python3::Python
    Python3::NumPy
)

# Python 3.14 ha deprecato le vecchie funzioni C-API
target_compile_options(MRR PRIVATE -Wall -Wextra -Wno-deprecated-declarations)
