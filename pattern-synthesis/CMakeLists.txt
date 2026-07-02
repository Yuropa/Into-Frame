cmake_minimum_required(VERSION 3.16)
project(pattern_synthesis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# libigl
FetchContent_Declare(
  libigl
  GIT_REPOSITORY https://github.com/libigl/libigl.git
  GIT_TAG v2.5.0
)
set(LIBIGL_GLFW OFF)
set(LIBIGL_OPENGL OFF)
set(LIBIGL_USE_STATIC_LIBRARY ON)
FetchContent_MakeAvailable(libigl)

# Polyscope
FetchContent_Declare(
  polyscope
  GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
  GIT_TAG v2.2.1
)
set(POLYSCOPE_BACKEND_OPENGL3_GLFW ON)
set(POLYSCOPE_BACKEND_OPENGL3_EGL OFF)
set(POLYSCOPE_BACKEND_OPENGL3_WGL OFF)
FetchContent_MakeAvailable(polyscope)

add_executable(pattern_synthesis
  src/main.cpp
  src/sampling.cpp
  src/parameterization.cpp
  src/lloyd_relaxation.cpp
  src/interaction.cpp
  src/voronoi-pcf.cpp
)

add_executable(pcf_cli
  src/pcf_cli.cpp
  src/lloyd_relaxation.cpp
  src/sampling.cpp
)

add_executable(synthesize_cli
  src/synthesize_cli.cpp
  src/lloyd_relaxation.cpp
  src/sampling.cpp
  python_lib/synthesis_core.cpp
)
# synthesis_core.h (python_lib/) includes lloyd_relaxation.h (src/) — both dirs
# must be on the include path, same as python_lib/CMakeLists.txt's pybind target.
target_include_directories(synthesize_cli PRIVATE src python_lib)

find_package(CGAL REQUIRED)

# Apple Clang often needs explicit libomp hints.
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

find_package(OpenMP REQUIRED)

target_link_libraries(pattern_synthesis PRIVATE igl::core polyscope CGAL::CGAL)
target_link_libraries(pattern_synthesis PRIVATE OpenMP::OpenMP_CXX)
message(STATUS "OpenMP enabled")

target_link_libraries(pcf_cli PRIVATE igl::core CGAL::CGAL)
if(OpenMP_CXX_FOUND)
  target_link_libraries(pcf_cli PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(synthesize_cli PRIVATE igl::core CGAL::CGAL)
if(OpenMP_CXX_FOUND)
  target_link_libraries(synthesize_cli PRIVATE OpenMP::OpenMP_CXX)
endif()
