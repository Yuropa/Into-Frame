#pragma once

#include <Eigen/Core>

inline Eigen::Vector2d uv_display_offset_from_vertices(const Eigen::MatrixXd& UV) {
  if (UV.rows() == 0 || UV.cols() < 2) {
    return Eigen::Vector2d::Zero();
  }
  return UV.leftCols<2>().colwise().mean().transpose();
}

inline Eigen::Vector2d uv_to_display_2d(
  const Eigen::Vector2d& uv,
  const Eigen::Vector2d& offset) {
  return uv - offset;
}

inline Eigen::Vector3d uv_to_display_3d(
  const Eigen::Vector2d& uv,
  const Eigen::Vector2d& offset) {
  const Eigen::Vector2d display_uv = uv_to_display_2d(uv, offset);
  return Eigen::Vector3d(display_uv.x(), display_uv.y(), 0.0);
}

inline Eigen::MatrixXd uv_matrix_to_display_2d(
  const Eigen::MatrixXd& uv,
  const Eigen::Vector2d& offset) {
  if (uv.rows() == 0 || uv.cols() < 2) {
    return Eigen::MatrixXd(0, 2);
  }
  Eigen::MatrixXd display = uv.leftCols<2>();
  display.col(0).array() -= offset.x();
  display.col(1).array() -= offset.y();
  return display;
}

inline Eigen::MatrixXd uv_matrix_to_display_3d(
  const Eigen::MatrixXd& uv,
  const Eigen::Vector2d& offset) {
  if (uv.rows() == 0 || uv.cols() < 2) {
    return Eigen::MatrixXd(0, 3);
  }
  Eigen::MatrixXd display(uv.rows(), 3);
  display.leftCols<2>() = uv_matrix_to_display_2d(uv, offset);
  display.col(2).setZero();
  return display;
}
