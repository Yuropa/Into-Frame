#pragma once

#include <array>
#include <vector>
#include <Eigen/Core>

#include "sampling.h"

class UniformGridTriangleFinder {
private:
  int grid_size;
  std::vector<std::vector<int>> grid;
  Eigen::Vector2d uv_min, uv_max;
  const Eigen::MatrixXd &UV;
  const Eigen::MatrixXi &F;

public:
  UniformGridTriangleFinder(
    const Eigen::MatrixXd &UV_,
    const Eigen::MatrixXi &F_,
    int grid_size_ = 64);

  int find_containing_triangle(const Eigen::Vector2d &uv) const;

  Eigen::Vector3d compute_barycentric_coords_uv(
    const Eigen::Vector2d &p,
    int tri_idx) const;

private:
  bool is_point_in_triangle_uv(const Eigen::Vector2d &p, int tri_idx) const;
};

Eigen::MatrixXd dart_samples_to_uv_points(const std::vector<DartSample> &samples);

bool build_sample_from_uv(
  const Eigen::Vector2d &uv,
  const UniformGridTriangleFinder &finder,
  DartSample &out_sample);

// Build constrained Delaunay adjacency for UV sample points.
// If boundary_uv has fewer than 3 vertices, unconstrained triangulation is used.
std::vector<std::vector<int>> build_constrained_delaunay_adjacency(
  const Eigen::MatrixXd& points_uv,
  const Eigen::MatrixXd& boundary_uv);

class DelaunayTraversalHelper {
public:
  bool build(
    const Eigen::MatrixXd& points_uv,
    const Eigen::MatrixXd& boundary_uv);

  bool is_ready() const;
  int triangle_count() const;
  bool triangle_center(int triangle_idx, Eigen::Vector2d& out_center_uv) const;
  void triangle_adjacency_edges(std::vector<Eigen::Vector2i>& out_edges) const;

  // Get neighbor triangle indices for a given triangle
  // Returns up to 3 neighbors (valid neighbors have index >= 0)
  void get_triangle_neighbors(int triangle_idx, std::array<int, 3>& out_neighbors) const;

  // Returns triangle index and corresponding sample vertex ids if found.
  bool find_containing_triangle(
    const Eigen::Vector2d& uv,
    int& out_triangle_idx,
    Eigen::Vector3i& out_triangle_vertices) const;

  // Returns number of crossed Delaunay triangle boundaries (edge crossings) by
  // the segment [start_uv, end_uv].
  // - same containing triangle => 0
  // - invalid/unresolved traversal => -1
  int count_triangles_crossed(
    const Eigen::Vector2d& start_uv,
    const Eigen::Vector2d& end_uv) const;

private:
  struct FaceData {
    Eigen::Vector3i vertex_indices = Eigen::Vector3i(-1, -1, -1);
    Eigen::Matrix<double, 3, 2> tri_points = Eigen::Matrix<double, 3, 2>::Zero();
    Eigen::Vector2d p0 = Eigen::Vector2d::Zero();
    Eigen::Matrix2d inv_basis = Eigen::Matrix2d::Zero();
    bool has_inv_basis = false;
    std::array<int, 3> neighbors = { -1, -1, -1 };
  };

  int locate_triangle_index(const Eigen::Vector2d& uv) const;
  bool point_in_face(const FaceData& face, const Eigen::Vector2d& uv) const;

  std::vector<FaceData> faces_;

  int grid_res_ = 0;
  Eigen::Vector2d bbox_min_ = Eigen::Vector2d::Zero();
  Eigen::Vector2d bbox_max_ = Eigen::Vector2d::Zero();
  std::vector<std::vector<int>> grid_faces_;
};

Eigen::MatrixXd lloyd_relaxation(
  const Eigen::MatrixXd &points_uv,
  const Eigen::MatrixXd &UV,
  const Eigen::MatrixXi &F,
  const Eigen::VectorXd &vertex_distortion,
  int max_iters,
  double convergence_threshold,
  int grid_size,
  const UniformGridTriangleFinder *finder,
  double *out_last_max_displacement,
  double *out_last_min_displacement,
  double *out_last_avg_displacement,
  int *out_iters_run);
