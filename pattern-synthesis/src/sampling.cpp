#include "sampling.h"
#include "uv_display.h"
#include <cmath>
#include <random>
#include <iostream>

std::vector<DartSample> sample_random_points(
  const Eigen::MatrixXd &UV,
  const Eigen::MatrixXi &F,
  int n,
  int seed) {
  std::vector<DartSample> samples;
  
  // Validate inputs
  if (n <= 0) {
    std::cerr << "Warning: sample_random_points - n must be positive, got " << n << std::endl;
    return samples;
  }
  if (F.rows() == 0) {
    std::cerr << "Warning: sample_random_points - mesh has no faces" << std::endl;
    return samples;
  }
  if (UV.rows() == 0 || UV.cols() < 2) {
    std::cerr << "Warning: sample_random_points - invalid UV matrix dimensions" << std::endl;
    return samples;
  }
  
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> uni(0, 1);

  samples.reserve(n);

  for (int i = 0; i < n; ++i) {
    // Pick triangle uniformly at random
    int tri = std::uniform_int_distribution<>(0, F.rows() - 1)(gen);

    // Sample barycentric coordinates uniformly in triangle
    double u = uni(gen);
    double v = uni(gen);
    if (u + v > 1.0) {
      u = 1.0 - u;
      v = 1.0 - v;
    }
    double w = 1.0 - u - v;
    Eigen::Vector3d bary(u, v, w);

    // Compute UV position using barycentric interpolation
    Eigen::Vector2d uv_pos = bary.x() * UV.row(F(tri, 0)).head<2>() +
                             bary.y() * UV.row(F(tri, 1)).head<2>() +
                             bary.z() * UV.row(F(tri, 2)).head<2>();

    samples.push_back({uv_pos, tri, bary});
  }
  return samples;
}

std::vector<DartSample> sample_random_points_uniform_uv(
  const Eigen::MatrixXd &UV,
  const Eigen::MatrixXi &F,
  int n,
  int seed) {
  std::vector<DartSample> samples;

  if (n <= 0) {
    std::cerr << "Warning: sample_random_points_uniform_uv - n must be positive, got " << n << std::endl;
    return samples;
  }
  if (F.rows() == 0) {
    std::cerr << "Warning: sample_random_points_uniform_uv - mesh has no faces" << std::endl;
    return samples;
  }
  if (UV.rows() == 0 || UV.cols() < 2) {
    std::cerr << "Warning: sample_random_points_uniform_uv - invalid UV matrix dimensions" << std::endl;
    return samples;
  }

  std::vector<double> triangle_weights;
  triangle_weights.reserve(F.rows());

  for (int tri = 0; tri < F.rows(); ++tri) {
    const int i0 = F(tri, 0);
    const int i1 = F(tri, 1);
    const int i2 = F(tri, 2);
    if (i0 < 0 || i1 < 0 || i2 < 0 ||
        i0 >= UV.rows() || i1 >= UV.rows() || i2 >= UV.rows()) {
      triangle_weights.push_back(0.0);
      continue;
    }

    const Eigen::Vector2d u0 = UV.row(i0).head<2>();
    const Eigen::Vector2d u1 = UV.row(i1).head<2>();
    const Eigen::Vector2d u2 = UV.row(i2).head<2>();
    const double area = 0.5 * std::abs(
      (u1.x() - u0.x()) * (u2.y() - u0.y()) -
      (u1.y() - u0.y()) * (u2.x() - u0.x()));
    triangle_weights.push_back(std::isfinite(area) ? area : 0.0);
  }

  std::discrete_distribution<> triangle_dist(triangle_weights.begin(), triangle_weights.end());
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> uni(0.0, 1.0);

  samples.reserve(n);

  for (int i = 0; i < n; ++i) {
    const int tri = triangle_dist(gen);

    double u = uni(gen);
    double v = uni(gen);
    if (u + v > 1.0) {
      u = 1.0 - u;
      v = 1.0 - v;
    }
    const double w = 1.0 - u - v;
    const Eigen::Vector3d bary(u, v, w);

    const Eigen::Vector2d uv_pos = bary.x() * UV.row(F(tri, 0)).head<2>() +
                                   bary.y() * UV.row(F(tri, 1)).head<2>() +
                                   bary.z() * UV.row(F(tri, 2)).head<2>();

    samples.push_back({uv_pos, tri, bary});
  }

  return samples;
}

Eigen::MatrixXd project_samples_to_3d(
  const std::vector<DartSample> &samples,
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXi &F) {
  Eigen::MatrixXd points3d;
  
  // Validate inputs
  if (samples.empty()) {
    std::cerr << "Warning: project_samples_to_3d - samples vector is empty" << std::endl;
    return points3d;
  }
  if (V.rows() == 0 || V.cols() != 3) {
    std::cerr << "Warning: project_samples_to_3d - invalid vertex matrix" << std::endl;
    return points3d;
  }
  if (F.rows() == 0) {
    std::cerr << "Warning: project_samples_to_3d - mesh has no faces" << std::endl;
    return points3d;
  }
  
  points3d.resize(samples.size(), 3);
  for (size_t i = 0; i < samples.size(); ++i) {
    int tri = samples[i].tri_idx;
    const Eigen::Vector3d &bary = samples[i].bary;
    points3d.row(i) = bary.x() * V.row(F(tri, 0)) +
                      bary.y() * V.row(F(tri, 1)) +
                      bary.z() * V.row(F(tri, 2));
  }
  return points3d;
}

Eigen::MatrixXd samples_to_points_3d(
  const std::vector<DartSample> &samples,
  const Eigen::MatrixXd &UV) {
  Eigen::MatrixXd points;
  
  // Validate inputs
  if (samples.empty()) {
    std::cerr << "Warning: samples_to_points_3d - samples vector is empty" << std::endl;
    return points;
  }
  if (UV.rows() == 0 || UV.cols() < 2) {
    std::cerr << "Warning: samples_to_points_3d - invalid UV matrix" << std::endl;
    return points;
  }
  points.resize(samples.size(), 3);
  const Eigen::Vector2d uv_display_offset = uv_display_offset_from_vertices(UV);

  for (size_t i = 0; i < samples.size(); ++i) {
    points.row(static_cast<int>(i)) =
      uv_to_display_3d(samples[i].uv, uv_display_offset).transpose();
  }
  return points;
}
