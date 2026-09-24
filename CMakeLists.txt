cmake_minimum_required(VERSION 3.16)

project(MRRDifferentiator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

set(MODEL_SOURCES MRR.cpp MRR.hpp Fabrication.hpp MRRCascade.hpp MRRCascade.cpp Simulation.cpp Simulation.hpp Plotter.hpp Plotter.cpp MonteCarlo.hpp MonteCarlo.cpp)

# MRR: the interactive study. figures: every figure of docs/main.tex, to PDF.
add_executable(MRR main.cpp ${MODEL_SOURCES})
add_executable(figures figures.cpp ${MODEL_SOURCES})

foreach(target MRR figures)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
    )

    target_link_libraries(${target} PRIVATE
        Python3::Python
        Python3::NumPy
    )

    # Python 3.14 deprecated the old C-API functions
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wno-deprecated-declarations)
endforeach()
