#pragma once

#include <Eigen/Core>

class DelaunayTraversalHelper;
struct InteractionState;

void reset_voronoi_pcf(InteractionState& state);
void compute_voronoi_pcf_histogram(
  InteractionState& state,
  const DelaunayTraversalHelper* delaunay_helper);
void draw_voronoi_pcf_ui(
  InteractionState& state,
  const DelaunayTraversalHelper* delaunay_helper,
  const Eigen::MatrixXd& points_3d,
  const Eigen::MatrixXd& points_uv);
