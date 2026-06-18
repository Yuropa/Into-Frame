#pragma once

#include <string>
#include <Eigen/Core>

// Compute area distortion between 3D mesh and UV parameterization
Eigen::VectorXd compute_triangle_area_distortion(
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXd &UV,
  const Eigen::MatrixXi &F);

// Matches the Polyscope distortion display pipeline:
// face averaging, log transform, and normalization to [0, 1].
Eigen::VectorXd compute_visualized_face_distortion(
  const Eigen::VectorXd& vertex_distortion,
  const Eigen::MatrixXi& F);

// Matches the Polyscope distortion display remap without converting to faces:
// per-vertex log transform and normalization to [0, 1].
Eigen::VectorXd compute_visualized_vertex_distortion(
  const Eigen::VectorXd& vertex_distortion);

// Perform LSCM parameterization and visualization
// Returns the computed UV matrix
Eigen::MatrixXd parameterisation(
  const std::string& meshName,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F);
