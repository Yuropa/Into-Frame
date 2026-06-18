# Pattern Synthesis Guidelines

Project guidelines for Copilot and AI agents.

## Architecture
- **Tech Stack:** C++17, CMake, Eigen, CGAL, libigl, Polyscope, ImGui.
- **Core Concept:** Interactive pattern generation system synthesizing output point distributions to match input pair-wise connectivity frequency (PCF) histograms while accounting for surface distortion.
- **Distance Metric:** Multiplicative penalty formula `d_eff = r_hop × (1 + weight × D_path_normalized)`.
- **Optimization:** Random point movement algorithm attempting to minimize L2 distance to the target PCF.
- **Structure:** Modular compilation units (`sampling`, `parameterization`, `lloyd_relaxation`, `interaction`, `voronoi-pcf`).
- **State Management:** Encapsulated in global state structs (`SamplingState`, `LloydState`, `TestingState`, `InteractionState`).
- **Algorithm Design:** Delaunay-based traversal graph (`DelaunayTraversalHelper`), spatial indexing (`UniformGridTriangleFinder`), and barycentric point representations.

## Build and Run
- **Build:** `cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo` then `cmake --build build -j` (or use the CMake Tools extension).
- **Run:** Execute `./build/pattern_synthesis [optional_model.obj]` or use the provided VS Code task "Run pattern_synthesis".
- **Testing:** No formal automated test framework. Manual testing is performed via the ImGui UI panel (evaluate metrics between two placed points). Optimization progress is logged to stdout.

## Conventions
- **Naming:** `snake_case` for functions and variables, `PascalCase` for classes and structs.
- **File Structure:** `.h` files with `#pragma once` for declarations; `.cpp` files for implementations.
- **Data Types:** Heavy use of Eigen matrices/vectors. Pass by `const &` where appropriate.
- **Resource Management:** Prefer `std::unique_ptr` for complex stateful objects.
- **UI:** Expose properties cleanly through ImGui controls within the central event loop. 
