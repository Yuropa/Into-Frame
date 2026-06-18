#pragma once

#include <vector>
#include <Eigen/Core>

// Struct to store sampled points with barycentric coordinates
struct DartSample {
  Eigen::Vector2d uv;
  int tri_idx;
  Eigen::Vector3d bary;
};

// Uniformly sample n random points inside the mesh triangles using barycentric coords
// Returns empty vector if inputs are invalid
std::vector<DartSample> sample_random_points(
  const Eigen::MatrixXd &UV,
  const Eigen::MatrixXi &F,
  int n,
  int seed = 42);

// Uniformly sample n random points over the UV domain by weighting triangles
// by their UV area. Returns empty vector if inputs are invalid.
std::vector<DartSample> sample_random_points_uniform_uv(
  const Eigen::MatrixXd &UV,
  const Eigen::MatrixXi &F,
  int n,
  int seed = 42);

// Project sampled points to 3D mesh using barycentric coordinates
// Returns empty matrix if inputs are invalid
Eigen::MatrixXd project_samples_to_3d(
  const std::vector<DartSample> &samples,
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXi &F);

// Convert UV sample list to 3D points (z=0)
// Returns empty matrix if inputs are invalid or sample count is zero
Eigen::MatrixXd samples_to_points_3d(
  const std::vector<DartSample> &samples,
  const Eigen::MatrixXd &UV);
