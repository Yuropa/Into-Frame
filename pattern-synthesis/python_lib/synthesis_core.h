#pragma once

#include <vector>
#include <Eigen/Core>

#include "lloyd_relaxation.h"

struct SynthesisResult {
    std::vector<Eigen::Vector2d> output_uv;  // final output point positions
    std::vector<float> exemplar_pcf;          // exemplar avg PCF histogram
    std::vector<float> output_pcf;            // output avg PCF histogram
    double final_energy;
    int iterations_ran;
};

// Exemplar-based point pattern synthesis via PCF matching.
//
// Places n_points inside output_boundary_uv such that their pair-correlation
// function (measured via Delaunay hop distances) matches that of exemplar_uv
// points measured inside input_boundary_uv.
//
// delaunay must already be built from the full mesh UV domain.
// n_points <= 0 infers target count from exemplar density.
SynthesisResult synthesize_pattern(
    const DelaunayTraversalHelper& delaunay,
    const std::vector<Eigen::Vector2d>& exemplar_uv,
    const Eigen::MatrixXd& input_boundary_uv,
    const Eigen::MatrixXd& output_boundary_uv,
    int n_points,
    int bin_count,
    int max_iters,
    int seed
);
