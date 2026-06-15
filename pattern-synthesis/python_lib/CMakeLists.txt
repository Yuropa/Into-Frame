cmake_minimum_required(VERSION 3.16)
project(pattern_synthesis_py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(pybind11 CONFIG REQUIRED)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  libigl
  GIT_REPOSITORY https://github.com/libigl/libigl.git
  GIT_TAG v2.5.0
)
set(LIBIGL_GLFW OFF)
set(LIBIGL_OPENGL OFF)
set(LIBIGL_USE_STATIC_LIBRARY OFF)
FetchContent_MakeAvailable(libigl)

find_package(CGAL REQUIRED)

# Apple Clang needs explicit libomp hints.
if(APPLE)
  find_path(LIBOMP_INCLUDE_DIR
    NAMES omp.h
    HINTS
      /opt/homebrew/opt/libomp/include
      /usr/local/opt/libomp/include)
  find_library(LIBOMP_LIBRARY
    NAMES omp
    HINTS
      /opt/homebrew/opt/libomp/lib
      /usr/local/opt/libomp/lib)
  if(LIBOMP_INCLUDE_DIR AND LIBOMP_LIBRARY)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_INCLUDE_DIR}")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_INCLUDE_DIR}")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_LIBRARY}")
  endif()
endif()
find_package(OpenMP)

# PATTERN_SYNTHESIS_SRC_DIR: path to pattern-synthesis/src.
# Defaults to ../src relative to this CMakeLists.txt.
# Override with -DPATTERN_SYNTHESIS_SRC_DIR=/absolute/path when the relative
# path doesn't resolve correctly (e.g. when building via pip from setup.sh).
if(NOT DEFINED PATTERN_SYNTHESIS_SRC_DIR)
  get_filename_component(PATTERN_SYNTHESIS_SRC_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../src" REALPATH)
endif()

if(NOT EXISTS "${PATTERN_SYNTHESIS_SRC_DIR}/sampling.cpp")
  message(FATAL_ERROR
    "Cannot find sampling.cpp in PATTERN_SYNTHESIS_SRC_DIR='${PATTERN_SYNTHESIS_SRC_DIR}'.\n"
    "Pass -DPATTERN_SYNTHESIS_SRC_DIR=/path/to/pattern-synthesis/src to cmake, "
    "or set it via pip: --config-settings=cmake.define.PATTERN_SYNTHESIS_SRC_DIR=...")
endif()

message(STATUS "pattern-synthesis src: ${PATTERN_SYNTHESIS_SRC_DIR}")

pybind11_add_module(_pattern_synthesis
  bindings.cpp
  synthesis_core.cpp
  ${PATTERN_SYNTHESIS_SRC_DIR}/sampling.cpp
  ${PATTERN_SYNTHESIS_SRC_DIR}/lloyd_relaxation.cpp
)

target_include_directories(_pattern_synthesis PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PATTERN_SYNTHESIS_SRC_DIR}
)

target_link_libraries(_pattern_synthesis PRIVATE igl::core CGAL::CGAL)

if(OpenMP_CXX_FOUND)
  target_link_libraries(_pattern_synthesis PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP enabled for pattern_synthesis")
endif()

# Install the C extension inside the pattern_synthesis Python package directory.
install(TARGETS _pattern_synthesis
  LIBRARY DESTINATION pattern_synthesis
  RUNTIME DESTINATION pattern_synthesis
)
