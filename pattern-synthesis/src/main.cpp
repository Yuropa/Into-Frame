#include <algorithm>
#include <cmath>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <igl/read_triangle_mesh.h>
#include <igl/lscm.h>
#include <igl/boundary_loop.h>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>
#include <polyscope/curve_network.h>
#include <polyscope/view.h>
#include <imgui.h>

#include "sampling.h"
#include "parameterization.h"
#include "lloyd_relaxation.h"
#include "interaction.h"
#include "voronoi-pcf.h"

// Global state for samples and parameters
struct SamplingState {
  std::vector<DartSample> samples_3d;
  std::vector<DartSample> samples_uv;
  Eigen::MatrixXd points_uv;
  Eigen::MatrixXd points_3d;
  Eigen::MatrixXd points_uv_3d;
  int n_samples = 5000;
  double point_size = 0.0005;
  bool show_samples = false;
  bool show_delaunay_graph = false;
  int random_seed = 42;
  std::vector<std::vector<int>> delaunay_adjacency;
  int delaunay_adjacency_edges = 0;
  bool has_delaunay_adjacency = false;
  std::unique_ptr<DelaunayTraversalHelper> delaunay_helper;
  int delaunay_triangle_count = 0;
  bool has_delaunay_helper = false;
  char samples_io_path[512] = "../data/10000_saved_samples.txt";
  std::string samples_io_status;
  bool samples_io_status_error = false;
} g_sampling;

struct LloydState {
  int max_iters = 50;
  double convergence_threshold = 1e-4;
  int grid_size = 64;
  double last_max_displacement = 0.0;
  double last_min_displacement = 0.0;
  double last_avg_displacement = 0.0;
  int last_iters_run = 0;
  bool has_stats = false;
  std::unique_ptr<UniformGridTriangleFinder> finder;
  int finder_grid_size = 0;
  int finder_uv_rows = 0;
  int finder_f_rows = 0;
} g_lloyd;

InteractionState g_interaction;

struct ModelLoadState {
  char model_path[512] = "";
  std::string status;
  bool status_error = false;
} g_model_load;

static double polygon_area_abs_2d(const Eigen::MatrixXd& poly);


static Eigen::Matrix2d sanitize_metric_tensor_main(const Eigen::Matrix2d& input) {
  Eigen::Matrix2d g = 0.5 * (input + input.transpose());
  if (!g.allFinite()) {
    return Eigen::Matrix2d::Identity();
  }
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(g);
  if (eig.info() != Eigen::Success) {
    return Eigen::Matrix2d::Identity();
  }
  Eigen::Vector2d vals = eig.eigenvalues();
  vals[0] = std::clamp(vals[0], 1e-10, 1e10);
  vals[1] = std::clamp(vals[1], 1e-10, 1e10);
  return eig.eigenvectors() * vals.asDiagonal() * eig.eigenvectors().transpose();
}

static double median_of_values(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  const size_t n = values.size();
  const size_t mid = n / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
  if ((n % 2) == 1) {
    return values[mid];
  }
  const double upper = values[mid];
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid - 1), values.end());
  const double lower = values[mid - 1];
  return 0.5 * (lower + upper);
}

static double compute_mode_nn_median(
  const Eigen::MatrixXd& points_uv,
  const std::vector<int>& center_indices,
  const std::vector<Eigen::Matrix2d>& sample_metrics,
  bool use_anisotropic_metric) {

  const int n_points = static_cast<int>(points_uv.rows());
  if (n_points < 2 || points_uv.cols() < 2) {
    return 0.0;
  }

  std::vector<int> centers;
  centers.reserve(center_indices.size());
  for (int idx : center_indices) {
    if (idx >= 0 && idx < n_points) {
      centers.push_back(idx);
    }
  }
  if (centers.empty()) {
    centers.resize(static_cast<size_t>(n_points));
    for (int i = 0; i < n_points; ++i) {
      centers[static_cast<size_t>(i)] = i;
    }
  }

  std::vector<double> min_dists;
  min_dists.reserve(centers.size());
  for (int i : centers) {
    const Eigen::Vector2d pi = points_uv.row(i).head<2>().transpose();
    Eigen::Matrix2d g_i = Eigen::Matrix2d::Identity();
    if (use_anisotropic_metric && i >= 0 &&
        i < static_cast<int>(sample_metrics.size())) {
      g_i = sanitize_metric_tensor_main(sample_metrics[static_cast<size_t>(i)]);
    }

    double min_d = std::numeric_limits<double>::infinity();
    for (int j = 0; j < n_points; ++j) {
      if (j == i) {
        continue;
      }
      const Eigen::Vector2d pj = points_uv.row(j).head<2>().transpose();
      const Eigen::Vector2d du = pj - pi;
      const double d = use_anisotropic_metric
        ? std::sqrt(std::max(0.0, du.dot(g_i * du)))
        : du.norm();
      if (!std::isfinite(d) || d <= 1e-14) {
        continue;
      }
      min_d = std::min(min_d, d);
    }
    if (std::isfinite(min_d) && min_d > 1e-14) {
      min_dists.push_back(min_d);
    }
  }

  return median_of_values(std::move(min_dists));
}

static bool point_in_polygon_uv_main(const Eigen::Vector2d& p, const Eigen::MatrixXd& poly) {
  if (poly.rows() < 3 || poly.cols() < 2) {
    return false;
  }

  const Eigen::Vector2d bb_min = poly.leftCols<2>().colwise().minCoeff().transpose();
  const Eigen::Vector2d bb_max = poly.leftCols<2>().colwise().maxCoeff().transpose();
  const double scale = std::max(1.0, (bb_max - bb_min).norm());
  const double eps = 1e-10 * scale;

  if (p.x() < bb_min.x() - eps || p.x() > bb_max.x() + eps ||
      p.y() < bb_min.y() - eps || p.y() > bb_max.y() + eps) {
    return false;
  }

  const auto orient2d = [](const Eigen::Vector2d& a, const Eigen::Vector2d& b, const Eigen::Vector2d& c) {
    const Eigen::Vector2d ab = b - a;
    const Eigen::Vector2d ac = c - a;
    return ab.x() * ac.y() - ab.y() * ac.x();
  };
  const auto point_on_segment = [eps](const Eigen::Vector2d& q, const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    const Eigen::Vector2d ab = b - a;
    const double ab2 = ab.squaredNorm();
    if (ab2 <= eps * eps) {
      return (q - a).squaredNorm() <= eps * eps;
    }
    const double t = std::clamp((q - a).dot(ab) / ab2, 0.0, 1.0);
    const Eigen::Vector2d proj = a + t * ab;
    return (q - proj).squaredNorm() <= eps * eps;
  };

  int winding_number = 0;
  for (int i = 0; i < poly.rows(); ++i) {
    const int j = (i + 1) % poly.rows();
    const Eigen::Vector2d a = poly.row(i).head<2>().transpose();
    const Eigen::Vector2d b = poly.row(j).head<2>().transpose();

    if (point_on_segment(p, a, b)) {
      return false;
    }

    if (a.y() <= p.y()) {
      if (b.y() > p.y() && orient2d(a, b, p) > eps) {
        ++winding_number;
      }
    } else {
      if (b.y() <= p.y() && orient2d(a, b, p) < -eps) {
        --winding_number;
      }
    }
  }
  return winding_number != 0;
}

static double bounding_box_area_2d(const Eigen::MatrixXd& points_uv) {
  if (points_uv.rows() < 1 || points_uv.cols() < 2) {
    return 0.0;
  }
  const double min_x = points_uv.col(0).minCoeff();
  const double max_x = points_uv.col(0).maxCoeff();
  const double min_y = points_uv.col(1).minCoeff();
  const double max_y = points_uv.col(1).maxCoeff();
  return std::max(0.0, max_x - min_x) * std::max(0.0, max_y - min_y);
}

static glm::vec3 fallback_front_dir_for_up(const glm::vec3& up_dir) {
  const glm::vec3 candidates[] = {
    glm::vec3(0.f, 0.f, 1.f),
    glm::vec3(1.f, 0.f, 0.f),
    glm::vec3(0.f, 1.f, 0.f)
  };

  for (const glm::vec3& candidate : candidates) {
    if (std::abs(glm::dot(candidate, up_dir)) < 0.95f) {
      return candidate;
    }
  }

  return glm::vec3(1.f, 0.f, 0.f);
}

static float sanitized_structure_length_scale(
  polyscope::Structure* structure,
  const glm::vec3& bbox_min,
  const glm::vec3& bbox_max) {
  if (!structure || !structure->hasExtents()) {
    return 1.0f;
  }

  float length_scale = structure->lengthScale();
  const float bbox_diagonal = glm::length(bbox_max - bbox_min);
  if (!std::isfinite(length_scale) || length_scale <= 1e-6f) {
    length_scale = bbox_diagonal;
  }
  if (!std::isfinite(length_scale) || length_scale <= 1e-6f) {
    length_scale = 1.0f;
  }
  return length_scale;
}

static void sync_viewer_scene_to_structure(polyscope::Structure* structure) {
  if (!structure || !structure->hasExtents()) {
    return;
  }

  const auto bbox = structure->boundingBox();
  const glm::vec3 bbox_min = std::get<0>(bbox);
  const glm::vec3 bbox_max = std::get<1>(bbox);
  polyscope::state::boundingBox = bbox;
  polyscope::state::lengthScale =
    sanitized_structure_length_scale(structure, bbox_min, bbox_max);
}

static void frame_view_to_structure(polyscope::Structure* structure, bool fly_to = false) {
  if (!structure || !structure->hasExtents()) {
    return;
  }

  const auto bbox = structure->boundingBox();
  const glm::vec3 bbox_min = std::get<0>(bbox);
  const glm::vec3 bbox_max = std::get<1>(bbox);
  const glm::vec3 target = 0.5f * (bbox_min + bbox_max);
  sync_viewer_scene_to_structure(structure);

  glm::vec3 up_dir = polyscope::view::getUpVec();
  if (!std::isfinite(up_dir.x) || !std::isfinite(up_dir.y) || !std::isfinite(up_dir.z) ||
      glm::length2(up_dir) <= 1e-12f) {
    up_dir = glm::vec3(0.f, 1.f, 0.f);
  } else {
    up_dir = glm::normalize(up_dir);
  }

  glm::vec3 front_dir = polyscope::view::getFrontVec();
  if (!std::isfinite(front_dir.x) || !std::isfinite(front_dir.y) || !std::isfinite(front_dir.z) ||
      glm::length2(front_dir) <= 1e-12f) {
    front_dir = fallback_front_dir_for_up(up_dir);
  } else {
    front_dir = glm::normalize(front_dir);
    if (std::abs(glm::dot(front_dir, up_dir)) > 0.95f) {
      front_dir = fallback_front_dir_for_up(up_dir);
    }
  }

  const float length_scale =
    sanitized_structure_length_scale(structure, bbox_min, bbox_max);

  polyscope::view::fov = polyscope::view::defaultFov;
  polyscope::view::nearClipRatio = polyscope::view::defaultNearClipRatio;
  polyscope::view::farClipRatio = polyscope::view::defaultFarClipRatio;

  const glm::vec3 camera_location =
    target + 0.1f * length_scale * up_dir + 1.5f * length_scale * front_dir;
  polyscope::view::lookAt(camera_location, target, up_dir, fly_to);
}

static double viewer_overlay_radius(polyscope::Structure* structure, double relative_size) {
  if (!structure || !structure->hasExtents()) {
    return std::max(relative_size, 1e-6);
  }

  double length_scale = static_cast<double>(structure->lengthScale());
  if (!std::isfinite(length_scale) || length_scale <= 1e-6) {
    length_scale = 1.0;
  }

  return std::max(relative_size * length_scale, 1e-6);
}

static double preserved_point_radius(polyscope::PointCloud* point_cloud, double fallback) {
  if (!point_cloud) {
    return fallback;
  }

  const double radius = point_cloud->getPointRadius();
  if (!std::isfinite(radius) || radius <= 0.0) {
    return fallback;
  }
  return radius;
}

static float preserved_curve_radius(polyscope::CurveNetwork* curve, double fallback) {
  if (!curve) {
    return static_cast<float>(fallback);
  }

  const float radius = curve->getRadius();
  if (!std::isfinite(radius) || radius <= 0.f) {
    return static_cast<float>(fallback);
  }
  return radius;
}

static polyscope::PointCloud* register_or_refresh_point_cloud(
  const std::string& name,
  const Eigen::MatrixXd& points,
  double default_radius) {
  double radius = default_radius;
  if (polyscope::hasPointCloud(name)) {
    auto* point_cloud = polyscope::getPointCloud(name);
    if (point_cloud) {
      radius = preserved_point_radius(point_cloud, default_radius);
      if (point_cloud->nPoints() == static_cast<size_t>(points.rows())) {
        point_cloud->updatePointPositions(points);
        return point_cloud;
      }
    }
    polyscope::removePointCloud(name);
  }

  auto* point_cloud = polyscope::registerPointCloud(name, points);
  if (point_cloud) {
    point_cloud->setPointRadius(radius, false);
  }
  return point_cloud;
}

static polyscope::CurveNetwork* register_or_refresh_curve_network(
  const std::string& name,
  const Eigen::MatrixXd& nodes,
  const Eigen::MatrixXi& edges,
  double default_radius) {
  float radius = static_cast<float>(default_radius);
  if (polyscope::hasCurveNetwork(name)) {
    auto* curve = polyscope::getCurveNetwork(name);
    if (curve) {
      radius = preserved_curve_radius(curve, default_radius);
    }
    polyscope::removeCurveNetwork(name);
  }

  auto* curve = polyscope::registerCurveNetwork(name, nodes, edges);
  if (curve) {
    curve->setRadius(radius, false);
  }
  return curve;
}

static polyscope::Structure* active_view_structure(
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh) {
  if (surfaceMesh && surfaceMesh->isEnabled()) {
    return surfaceMesh;
  }
  if (uvMesh && uvMesh->isEnabled()) {
    return uvMesh;
  }
  if (surfaceMesh) {
    return surfaceMesh;
  }
  return uvMesh;
}

static Eigen::VectorXd compute_face_area_distortion_asurf_over_auv(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F) {

  Eigen::VectorXd out = Eigen::VectorXd::Ones(F.rows());
  if (V.rows() == 0 || V.cols() < 3 || UV.rows() == 0 || UV.cols() < 2 ||
      F.rows() == 0 || F.cols() < 3) {
    return out;
  }

  for (int f = 0; f < F.rows(); ++f) {
    const int i0 = F(f, 0);
    const int i1 = F(f, 1);
    const int i2 = F(f, 2);
    if (i0 < 0 || i1 < 0 || i2 < 0 ||
        i0 >= V.rows() || i1 >= V.rows() || i2 >= V.rows() ||
        i0 >= UV.rows() || i1 >= UV.rows() || i2 >= UV.rows()) {
      out[f] = 1.0;
      continue;
    }

    const Eigen::Vector3d p0 = V.row(i0).head<3>();
    const Eigen::Vector3d p1 = V.row(i1).head<3>();
    const Eigen::Vector3d p2 = V.row(i2).head<3>();
    const double area3d = 0.5 * ((p1 - p0).cross(p2 - p0)).norm();

    const Eigen::Vector2d u0 = UV.row(i0).head<2>();
    const Eigen::Vector2d u1 = UV.row(i1).head<2>();
    const Eigen::Vector2d u2 = UV.row(i2).head<2>();
    const double area2 = (u1.x() - u0.x()) * (u2.y() - u0.y()) -
                         (u1.y() - u0.y()) * (u2.x() - u0.x());
    const double area_uv = 0.5 * std::abs(area2);

    double d = 1.0;
    if (area_uv > 1e-14) {
      d = area3d / area_uv;
    }
    if (!std::isfinite(d) || d <= 1e-12) {
      d = 1e-12;
    }
    out[f] = d;
  }

  return out;
}

static std::vector<Eigen::Matrix2d> compute_face_metric_tensors_uv_to_surface(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F) {

  std::vector<Eigen::Matrix2d> out(static_cast<size_t>(F.rows()), Eigen::Matrix2d::Identity());
  if (V.rows() == 0 || V.cols() < 3 || UV.rows() == 0 || UV.cols() < 2 ||
      F.rows() == 0 || F.cols() < 3) {
    return out;
  }

  for (int f = 0; f < F.rows(); ++f) {
    const int i0 = F(f, 0);
    const int i1 = F(f, 1);
    const int i2 = F(f, 2);
    if (i0 < 0 || i1 < 0 || i2 < 0 ||
        i0 >= V.rows() || i1 >= V.rows() || i2 >= V.rows() ||
        i0 >= UV.rows() || i1 >= UV.rows() || i2 >= UV.rows()) {
      continue;
    }

    const Eigen::Vector3d p0 = V.row(i0).head<3>();
    const Eigen::Vector3d p1 = V.row(i1).head<3>();
    const Eigen::Vector3d p2 = V.row(i2).head<3>();
    Eigen::Matrix<double, 3, 2> x;
    x.col(0) = p1 - p0;
    x.col(1) = p2 - p0;

    const Eigen::Vector2d u0 = UV.row(i0).head<2>();
    const Eigen::Vector2d u1 = UV.row(i1).head<2>();
    const Eigen::Vector2d u2 = UV.row(i2).head<2>();
    Eigen::Matrix2d u;
    u.col(0) = u1 - u0;
    u.col(1) = u2 - u0;

    const double det_u = u.determinant();
    if (!std::isfinite(det_u) || std::abs(det_u) <= 1e-14) {
      continue;
    }

    const Eigen::Matrix<double, 3, 2> j = x * u.inverse();
    Eigen::Matrix2d g = j.transpose() * j;
    g = 0.5 * (g + g.transpose());

    if (!g.allFinite()) {
      continue;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(g);
    if (eig.info() != Eigen::Success) {
      continue;
    }
    Eigen::Vector2d vals = eig.eigenvalues();
    vals[0] = std::clamp(vals[0], 1e-10, 1e10);
    vals[1] = std::clamp(vals[1], 1e-10, 1e10);
    out[static_cast<size_t>(f)] = eig.eigenvectors() * vals.asDiagonal() * eig.eigenvectors().transpose();
  }

  return out;
}

static std::vector<Eigen::Matrix2d> sample_metric_tensors_uv_to_surface(
  const std::vector<DartSample>& samples_uv,
  const std::vector<Eigen::Matrix2d>& face_metric_tensors_uv_to_surface) {

  std::vector<Eigen::Matrix2d> out(samples_uv.size(), Eigen::Matrix2d::Identity());
  if (samples_uv.empty() || face_metric_tensors_uv_to_surface.empty()) {
    return out;
  }

  for (size_t i = 0; i < samples_uv.size(); ++i) {
    const int tri = samples_uv[i].tri_idx;
    if (tri < 0 || tri >= static_cast<int>(face_metric_tensors_uv_to_surface.size())) {
      continue;
    }
    out[i] = face_metric_tensors_uv_to_surface[static_cast<size_t>(tri)];
  }
  return out;
}

static std::vector<double> sample_distortion_values_asurf_over_auv(
  const std::vector<DartSample>& samples_uv,
  const Eigen::VectorXd& face_distortion_asurf_over_auv) {

  std::vector<double> out(samples_uv.size(), 1.0);
  if (samples_uv.empty() || face_distortion_asurf_over_auv.size() == 0) {
    return out;
  }

  for (size_t i = 0; i < samples_uv.size(); ++i) {
    const DartSample& s = samples_uv[i];
    const int tri = s.tri_idx;
    if (tri < 0 || tri >= face_distortion_asurf_over_auv.size()) {
      continue;
    }

    double d = face_distortion_asurf_over_auv[tri];
    if (!std::isfinite(d) || d <= 1e-12) {
      d = 1e-12;
    }
    out[i] = d;
  }

  return out;
}

static double polygon_area_abs_2d(const Eigen::MatrixXd& poly) {
  if (poly.rows() < 3 || poly.cols() < 2) {
    return 0.0;
  }

  double area2 = 0.0;
  for (int i = 0, j = poly.rows() - 1; i < poly.rows(); j = i++) {
    area2 += poly(j, 0) * poly(i, 1) - poly(i, 0) * poly(j, 1);
  }
  return 0.5 * std::abs(area2);
}

static double compute_total_surface_area_3d(
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F) {
  if (V.rows() == 0 || V.cols() < 3 || F.rows() == 0 || F.cols() < 3) {
    return 0.0;
  }

  double total_area = 0.0;
  for (int f = 0; f < F.rows(); ++f) {
    const int i0 = F(f, 0);
    const int i1 = F(f, 1);
    const int i2 = F(f, 2);
    if (i0 < 0 || i1 < 0 || i2 < 0 ||
        i0 >= V.rows() || i1 >= V.rows() || i2 >= V.rows()) {
      continue;
    }
    const Eigen::Vector3d p0 = V.row(i0).head<3>();
    const Eigen::Vector3d p1 = V.row(i1).head<3>();
    const Eigen::Vector3d p2 = V.row(i2).head<3>();
    total_area += 0.5 * ((p1 - p0).cross(p2 - p0)).norm();
  }
  return total_area;
}

static double compute_total_uv_area(
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F) {
  if (UV.rows() == 0 || UV.cols() < 2 || F.rows() == 0 || F.cols() < 3) {
    return 0.0;
  }

  double total_area = 0.0;
  for (int f = 0; f < F.rows(); ++f) {
    const int i0 = F(f, 0);
    const int i1 = F(f, 1);
    const int i2 = F(f, 2);
    if (i0 < 0 || i1 < 0 || i2 < 0 ||
        i0 >= UV.rows() || i1 >= UV.rows() || i2 >= UV.rows()) {
      continue;
    }
    const Eigen::Vector2d u0 = UV.row(i0).head<2>();
    const Eigen::Vector2d u1 = UV.row(i1).head<2>();
    const Eigen::Vector2d u2 = UV.row(i2).head<2>();
    const double area2 = (u1.x() - u0.x()) * (u2.y() - u0.y()) -
                         (u1.y() - u0.y()) * (u2.x() - u0.x());
    total_area += 0.5 * std::abs(area2);
  }
  return total_area;
}

static void update_output_points_clouds(
  const std::vector<int>& points_indices,
  const Eigen::MatrixXd& points_3d,
  const Eigen::MatrixXd& points_uv_3d,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh) {
  Eigen::MatrixXd out_3d;
  Eigen::MatrixXd out_uv_3d;

  if (!points_indices.empty() && points_3d.rows() > 0 && points_uv_3d.rows() > 0) {
    out_3d.resize(points_indices.size(), 3);
    out_uv_3d.resize(points_indices.size(), 3);

    size_t out_idx = 0;
    const int max_rows = std::min(points_3d.rows(), points_uv_3d.rows());
    for (int idx : points_indices) {
      if (idx >= 0 && idx < max_rows) {
        out_3d.row(out_idx) = points_3d.row(idx);
        out_uv_3d.row(out_idx) = points_uv_3d.row(idx);
        out_idx++;
      }
    }
    out_3d.conservativeResize(out_idx, 3);
    out_uv_3d.conservativeResize(out_idx, 3);
  }

  const std::string kOutputPoints3D = "output_points_3d";
  const std::string kOutputPointsUV = "output_points_uv";

  if (out_3d.rows() == 0) {
    if (polyscope::hasPointCloud(kOutputPoints3D)) {
      polyscope::removePointCloud(kOutputPoints3D);
    }
    if (polyscope::hasPointCloud(kOutputPointsUV)) {
      polyscope::removePointCloud(kOutputPointsUV);
    }
    return;
  }

  polyscope::PointCloud* pc_3d = register_or_refresh_point_cloud(
    kOutputPoints3D,
    out_3d,
    viewer_overlay_radius(surfaceMesh, g_sampling.point_size * 1.4));
  polyscope::PointCloud* pc_uv = register_or_refresh_point_cloud(
    kOutputPointsUV,
    out_uv_3d,
    viewer_overlay_radius(uvMesh, g_sampling.point_size * 1.4));

  if (pc_3d) {
    pc_3d->setPointColor(glm::vec3(1.0f, 0.6f, 0.1f));
    if (surfaceMesh) {
      pc_3d->setEnabled(surfaceMesh->isEnabled());
    }
  }
  if (pc_uv) {
    pc_uv->setPointColor(glm::vec3(1.0f, 0.6f, 0.1f));
    if (uvMesh) {
      pc_uv->setEnabled(uvMesh->isEnabled());
    }
  }
}

static void update_sampling_point_clouds(
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh) {
  polyscope::PointCloud* pc_3d = register_or_refresh_point_cloud(
    "samples_3d",
    g_sampling.points_3d,
    viewer_overlay_radius(surfaceMesh, g_sampling.point_size));
  polyscope::PointCloud* pc_uv = register_or_refresh_point_cloud(
    "samples_uv",
    g_sampling.points_uv_3d,
    viewer_overlay_radius(uvMesh, g_sampling.point_size));

  if (pc_3d) {
    pc_3d->setEnabled(surfaceMesh && surfaceMesh->isEnabled() && g_sampling.show_samples);
  }
  if (pc_uv) {
    pc_uv->setEnabled(uvMesh && uvMesh->isEnabled() && g_sampling.show_samples);
  }
}

static void update_delaunay_graph_curve_networks(
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh) {
  const std::string kDelaunayGraph3D = "delaunay_graph_3d";
  const std::string kDelaunayGraphUV = "delaunay_graph_uv";

  auto disable_existing = [&]() {
    if (polyscope::hasCurveNetwork(kDelaunayGraph3D)) {
      auto* curve = polyscope::getCurveNetwork(kDelaunayGraph3D);
      if (curve) {
        curve->setEnabled(false);
      }
    }
    if (polyscope::hasCurveNetwork(kDelaunayGraphUV)) {
      auto* curve = polyscope::getCurveNetwork(kDelaunayGraphUV);
      if (curve) {
        curve->setEnabled(false);
      }
    }
  };

  if (!g_sampling.show_delaunay_graph ||
      !g_sampling.delaunay_helper ||
      !g_sampling.delaunay_helper->is_ready()) {
    disable_existing();
    return;
  }

  if (!g_sampling.has_delaunay_adjacency || g_sampling.delaunay_adjacency.empty()) {
    disable_existing();
    return;
  }

  const int n3 = g_sampling.points_3d.rows();
  const int nu = g_sampling.points_uv_3d.rows();
  const int n_nodes = std::min(n3, nu);
  if (n_nodes < 2 ||
      g_sampling.points_3d.cols() < 3 ||
      g_sampling.points_uv_3d.cols() < 3) {
    disable_existing();
    return;
  }

  std::vector<Eigen::Vector2i> edges_list;
  edges_list.reserve(static_cast<size_t>(std::max(0, g_sampling.delaunay_adjacency_edges)));
  for (int i = 0; i < n_nodes; ++i) {
    if (i >= static_cast<int>(g_sampling.delaunay_adjacency.size())) {
      break;
    }
    for (int j : g_sampling.delaunay_adjacency[static_cast<size_t>(i)]) {
      if (j <= i || j < 0 || j >= n_nodes) {
        continue;
      }
      edges_list.emplace_back(i, j);
    }
  }
  if (edges_list.empty()) {
    disable_existing();
    return;
  }

  Eigen::MatrixXi edges(static_cast<int>(edges_list.size()), 2);
  for (int e = 0; e < static_cast<int>(edges_list.size()); ++e) {
    edges.row(e) = edges_list[static_cast<size_t>(e)];
  }

  Eigen::MatrixXd nodes_3d = g_sampling.points_3d.topRows(n_nodes);
  Eigen::MatrixXd nodes_uv = g_sampling.points_uv_3d.topRows(n_nodes);

  auto* curve_3d = register_or_refresh_curve_network(
    kDelaunayGraph3D,
    nodes_3d,
    edges,
    viewer_overlay_radius(surfaceMesh, g_sampling.point_size * 0.7));
  if (curve_3d) {
    curve_3d->setColor(glm::vec3(0.95f, 0.85f, 0.25f));
    curve_3d->setEnabled(surfaceMesh && surfaceMesh->isEnabled());
  }

  auto* curve_uv = register_or_refresh_curve_network(
    kDelaunayGraphUV,
    nodes_uv,
    edges,
    viewer_overlay_radius(uvMesh, g_sampling.point_size * 0.7));
  if (curve_uv) {
    curve_uv->setColor(glm::vec3(0.95f, 0.85f, 0.25f));
    curve_uv->setEnabled(uvMesh && uvMesh->isEnabled());
  }
}

static bool save_samples_to_file(
  const std::string& path,
  const std::vector<DartSample>& samples,
  std::string& out_error) {
  if (samples.empty()) {
    out_error = "No samples to save.";
    return false;
  }

  std::ofstream out(path, std::ios::trunc);
  if (!out.is_open()) {
    out_error = "Failed to open file for writing.";
    return false;
  }

  out << "PATTERN_SYNTHESIS_SAMPLES_V1\n";
  out << samples.size() << "\n";
  out << std::setprecision(17);

  for (const DartSample& s : samples) {
    if (!s.uv.allFinite() || !s.bary.allFinite()) {
      out_error = "Encountered non-finite sample values.";
      return false;
    }
    out << s.tri_idx << " "
        << s.uv.x() << " " << s.uv.y() << " "
        << s.bary.x() << " " << s.bary.y() << " " << s.bary.z() << "\n";
  }

  if (!out.good()) {
    out_error = "Write error while saving samples.";
    return false;
  }

  return true;
}

static bool load_samples_from_file(
  const std::string& path,
  int face_count,
  std::vector<DartSample>& out_samples,
  std::string& out_error) {
  out_samples.clear();

  std::ifstream in(path);
  if (!in.is_open()) {
    out_error = "Failed to open file for reading.";
    return false;
  }

  std::string header;
  if (!std::getline(in, header)) {
    out_error = "Empty file.";
    return false;
  }
  if (header != "PATTERN_SYNTHESIS_SAMPLES_V1") {
    out_error = "Unsupported sample file format.";
    return false;
  }

  size_t sample_count = 0;
  if (!(in >> sample_count)) {
    out_error = "Failed to read sample count.";
    return false;
  }

  if (sample_count == 0) {
    out_error = "Sample file has zero samples.";
    return false;
  }

  out_samples.reserve(sample_count);
  for (size_t i = 0; i < sample_count; ++i) {
    int tri_idx = -1;
    double uv_x = 0.0;
    double uv_y = 0.0;
    double b0 = 0.0;
    double b1 = 0.0;
    double b2 = 0.0;

    if (!(in >> tri_idx >> uv_x >> uv_y >> b0 >> b1 >> b2)) {
      out_error = "Unexpected end of file while reading samples.";
      return false;
    }

    if (tri_idx < 0 || tri_idx >= face_count) {
      out_error = "Sample file contains triangle indices outside current mesh.";
      return false;
    }

    const Eigen::Vector2d uv(uv_x, uv_y);
    const Eigen::Vector3d bary(b0, b1, b2);
    if (!uv.allFinite() || !bary.allFinite()) {
      out_error = "Sample file contains non-finite values.";
      return false;
    }

    out_samples.push_back({uv, tri_idx, bary});
  }

  return true;
}

static int count_undirected_edges(const std::vector<std::vector<int>>& adjacency) {
  long long directed_count = 0;
  for (const auto& nbrs : adjacency) {
    directed_count += static_cast<long long>(nbrs.size());
  }
  return static_cast<int>(directed_count / 2LL);
}

static bool rebuild_loaded_samples_adjacency(
  const Eigen::MatrixXd& boundary_uv,
  std::string& out_error) {
  g_sampling.delaunay_adjacency.clear();
  g_sampling.delaunay_adjacency_edges = 0;
  g_sampling.has_delaunay_adjacency = false;
  g_sampling.delaunay_helper.reset();
  g_sampling.delaunay_triangle_count = 0;
  g_sampling.has_delaunay_helper = false;

  if (g_sampling.points_uv.rows() < 2 || g_sampling.points_uv.cols() < 2) {
    out_error = "Not enough UV points for adjacency.";
    return false;
  }

  g_sampling.delaunay_adjacency = build_constrained_delaunay_adjacency(
    g_sampling.points_uv,
    boundary_uv);
  if (static_cast<int>(g_sampling.delaunay_adjacency.size()) != g_sampling.points_uv.rows()) {
    out_error = "Adjacency build returned invalid size.";
    g_sampling.delaunay_adjacency.clear();
    g_sampling.delaunay_adjacency_edges = 0;
    g_sampling.has_delaunay_adjacency = false;
    return false;
  }

  g_sampling.delaunay_adjacency_edges =
    count_undirected_edges(g_sampling.delaunay_adjacency);
  g_sampling.has_delaunay_adjacency = true;

  g_sampling.delaunay_helper = std::make_unique<DelaunayTraversalHelper>();
  if (!g_sampling.delaunay_helper->build(g_sampling.points_uv, boundary_uv)) {
    g_sampling.delaunay_helper.reset();
    out_error = "Adjacency built, but Delaunay triangle helper build failed.";
    return false;
  }
  g_sampling.delaunay_triangle_count = g_sampling.delaunay_helper->triangle_count();
  g_sampling.has_delaunay_helper = true;

  return true;
}

int main(int argc, char** argv) {
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;

  std::vector<std::string> mesh_candidates;
  if (argc > 1) {
    mesh_candidates.emplace_back(argv[1]);
  } else {
    mesh_candidates = {
      "data/bunnyhead.obj",
      "../data/bunnyhead.obj",
      "data/camelhead.off",
      "../data/camelhead.off",
    };
  }

  std::string meshName;
  for (const std::string& candidate : mesh_candidates) {
    if (igl::read_triangle_mesh(candidate, V, F)) {
      meshName = candidate;
      break;
    }
  }
  if (meshName.empty()) {
    std::cerr << "Failed to load mesh. Provide a mesh path as the first argument." << std::endl;
    return 1;
  }

  // Get UV from parameterization (now returns the computed UV matrix)
  Eigen::MatrixXd UV = parameterisation(meshName, V, F);
  Eigen::VectorXd vertex_distortion = compute_triangle_area_distortion(V, UV, F);
  Eigen::VectorXd face_distortion_asurf_over_auv = compute_face_area_distortion_asurf_over_auv(V, UV, F);
  std::vector<Eigen::Matrix2d> face_metric_tensors_uv_to_surface =
    compute_face_metric_tensors_uv_to_surface(V, UV, F);
  double total_uv_area = compute_total_uv_area(UV, F);
  double total_surface_area_3d = compute_total_surface_area_3d(V, F);

  Eigen::VectorXi mesh_boundary_indices;
  igl::boundary_loop(F, mesh_boundary_indices);
  Eigen::MatrixXd mesh_boundary_uv;
  if (mesh_boundary_indices.size() >= 3 && UV.rows() > 0 && UV.cols() >= 2) {
    mesh_boundary_uv.resize(mesh_boundary_indices.size(), 2);
    for (int i = 0; i < mesh_boundary_indices.size(); ++i) {
      const int vid = mesh_boundary_indices[i];
      if (vid >= 0 && vid < UV.rows()) {
        mesh_boundary_uv.row(i) = UV.row(vid).head<2>();
      } else {
        mesh_boundary_uv.row(i).setZero();
      }
    }
  } else {
    mesh_boundary_uv.resize(0, 2);
  }

  // Get mesh references - these should exist after parameterisation()
  polyscope::SurfaceMesh* surfaceMesh = nullptr;
  polyscope::SurfaceMesh* uvMesh = nullptr;
  
  for (auto& p : polyscope::state::structures) {
    for (auto& s : p.second) {
      auto* mesh = dynamic_cast<polyscope::SurfaceMesh*>(s.second.get());
      if (mesh) {
        if (s.first == "uv_layout") {
          uvMesh = mesh;
        } else if (s.first == meshName) {
          surfaceMesh = mesh;
        }
      }
    }
  }

  if (!surfaceMesh || !uvMesh) {
    std::cerr << "Error: Could not find mesh structures" << std::endl;
    return 1;
  }

  init_interaction(g_interaction, V, F);
  std::snprintf(
    g_model_load.model_path,
    sizeof(g_model_load.model_path),
    "%s",
    meshName.c_str());

  // Hide UV mesh initially
  if (uvMesh) uvMesh->setEnabled(false);
  frame_view_to_structure(surfaceMesh);

  // Toggle view lambda
  auto toggleView = [&surfaceMesh, &uvMesh]() {
    if (surfaceMesh && uvMesh) {
      bool meshEnabled = surfaceMesh->isEnabled();
      surfaceMesh->setEnabled(!meshEnabled);
      uvMesh->setEnabled(meshEnabled);

      if (polyscope::hasPointCloud("samples_3d")) {
        auto* pc_3d = polyscope::getPointCloud("samples_3d");
        if (pc_3d) pc_3d->setEnabled(!meshEnabled && g_sampling.show_samples);
      }
      if (polyscope::hasPointCloud("samples_uv")) {
        auto* pc_uv = polyscope::getPointCloud("samples_uv");
        if (pc_uv) pc_uv->setEnabled(meshEnabled && g_sampling.show_samples);
      }
      if (polyscope::hasCurveNetwork("delaunay_graph_3d")) {
        auto* graph_3d = polyscope::getCurveNetwork("delaunay_graph_3d");
        if (graph_3d) graph_3d->setEnabled(!meshEnabled && g_sampling.show_delaunay_graph);
      }
      if (polyscope::hasCurveNetwork("delaunay_graph_uv")) {
        auto* graph_uv = polyscope::getCurveNetwork("delaunay_graph_uv");
        if (graph_uv) graph_uv->setEnabled(meshEnabled && g_sampling.show_delaunay_graph);
      }

      polyscope::Structure* target_structure =
        meshEnabled ? static_cast<polyscope::Structure*>(uvMesh)
                    : static_cast<polyscope::Structure*>(surfaceMesh);
      frame_view_to_structure(target_structure);

      std::cout << (meshEnabled ? "Switched to UV view" : "Switched to 3D mesh view") << std::endl;
    }
  };

  const auto clear_sampling_visuals = []() {
    if (polyscope::hasPointCloud("samples_3d")) {
      polyscope::removePointCloud("samples_3d");
    }
    if (polyscope::hasPointCloud("samples_uv")) {
      polyscope::removePointCloud("samples_uv");
    }
    if (polyscope::hasCurveNetwork("delaunay_graph_3d")) {
      polyscope::removeCurveNetwork("delaunay_graph_3d");
    }
    if (polyscope::hasCurveNetwork("delaunay_graph_uv")) {
      polyscope::removeCurveNetwork("delaunay_graph_uv");
    }
  };

  const auto recompute_mesh_boundary_uv = [&]() {
    Eigen::VectorXi boundary_indices;
    igl::boundary_loop(F, boundary_indices);
    if (boundary_indices.size() >= 3 && UV.rows() > 0 && UV.cols() >= 2) {
      mesh_boundary_uv.resize(boundary_indices.size(), 2);
      for (int i = 0; i < boundary_indices.size(); ++i) {
        const int vid = boundary_indices[i];
        if (vid >= 0 && vid < UV.rows()) {
          mesh_boundary_uv.row(i) = UV.row(vid).head<2>();
        } else {
          mesh_boundary_uv.row(i).setZero();
        }
      }
    } else {
      mesh_boundary_uv.resize(0, 2);
    }
  };

  const auto load_model_from_path = [&](const std::string& requested_path) {
    if (requested_path.empty()) {
      g_model_load.status = "Please provide a model path.";
      g_model_load.status_error = true;
      return;
    }

    Eigen::MatrixXd loaded_V;
    Eigen::MatrixXi loaded_F;
    if (!igl::read_triangle_mesh(requested_path, loaded_V, loaded_F)) {
      g_model_load.status = "Failed to load model: " + requested_path;
      g_model_load.status_error = true;
      return;
    }

    Eigen::VectorXi loaded_boundary;
    igl::boundary_loop(loaded_F, loaded_boundary);
    if (loaded_boundary.size() == 0) {
      g_model_load.status = "Loaded mesh has no boundary for LSCM parameterization.";
      g_model_load.status_error = true;
      return;
    }

    Eigen::MatrixXd loaded_UV;
    if (!igl::lscm(loaded_V, loaded_F, loaded_UV) ||
        loaded_UV.rows() != loaded_V.rows() ||
        loaded_UV.cols() < 2) {
      g_model_load.status = "Failed to parameterize model: " + requested_path;
      g_model_load.status_error = true;
      return;
    }

    const std::string old_mesh_name = meshName;
    reset_interaction_for_new_model(g_interaction, surfaceMesh, uvMesh);
    clear_sampling_visuals();
    if (!old_mesh_name.empty() && polyscope::hasSurfaceMesh(old_mesh_name)) {
      polyscope::removeSurfaceMesh(old_mesh_name, false);
    }
    if (polyscope::hasSurfaceMesh("uv_layout")) {
      polyscope::removeSurfaceMesh("uv_layout", false);
    }

    V = std::move(loaded_V);
    F = std::move(loaded_F);
    meshName = requested_path;
    UV = parameterisation(meshName, V, F);
    if (UV.rows() != V.rows() || UV.cols() < 2) {
      g_model_load.status = "Model loaded, but parameterization result was invalid.";
      g_model_load.status_error = true;
      surfaceMesh = nullptr;
      uvMesh = nullptr;
      return;
    }

    surfaceMesh = polyscope::hasSurfaceMesh(meshName)
      ? polyscope::getSurfaceMesh(meshName)
      : nullptr;
    uvMesh = polyscope::hasSurfaceMesh("uv_layout")
      ? polyscope::getSurfaceMesh("uv_layout")
      : nullptr;
    if (!surfaceMesh || !uvMesh) {
      g_model_load.status = "Model loaded, but Polyscope mesh registration failed.";
      g_model_load.status_error = true;
      return;
    }

    vertex_distortion = compute_triangle_area_distortion(V, UV, F);
    face_distortion_asurf_over_auv = compute_face_area_distortion_asurf_over_auv(V, UV, F);
    face_metric_tensors_uv_to_surface =
      compute_face_metric_tensors_uv_to_surface(V, UV, F);
    total_uv_area = compute_total_uv_area(UV, F);
    total_surface_area_3d = compute_total_surface_area_3d(V, F);
    recompute_mesh_boundary_uv();

    g_sampling = SamplingState{};
    g_lloyd = LloydState{};
    init_interaction(g_interaction, V, F);
    surfaceMesh->setEnabled(true);
    uvMesh->setEnabled(false);
    frame_view_to_structure(surfaceMesh);

    g_model_load.status = "Loaded model: " + requested_path;
    g_model_load.status_error = false;
    std::cout << g_model_load.status << std::endl;
  };

  std::cout << "\n=== Controls ===" << std::endl;
  std::cout << "SPACE - Toggle between 3D mesh view and UV layout view" << std::endl;
  std::cout << "=== Initial View ===" << std::endl;
  std::cout << "Showing: 3D mesh view" << std::endl;

  // UI callback
  polyscope::state::userCallback = [&]() {
    sync_viewer_scene_to_structure(active_view_structure(surfaceMesh, uvMesh));

    // Handle space key press
    if (ImGui::IsKeyPressed(ImGuiKey_Space, false)) {
      toggleView();
    }

    const auto reset_regions_for_sampling_change = [&]() {
      const int saved_region_index = g_interaction.active_region_index;
      for (int region_index = 0; region_index < region_count(g_interaction); ++region_index) {
        g_interaction.active_region_index = region_index;
        PatternRegionState& region = active_region(g_interaction);
        region.pattern_points_3d.clear();
        region.pattern_points_uv.clear();
        region.pattern_processing_uv.clear();
        region.pattern_points_delaunay_triangle.clear();
        region.pattern_points_delaunay_vertices.clear();
        region.pattern_point_class_ids.clear();
        region.last_crossed_triangle_count = -1;
        reset_voronoi_pcf(g_interaction);
        region.selected_sample_indices.clear();
        region.input_reference_indices.clear();
        region.selected_dirty = true;
        region.input_boundary_dirty = true;
        region.output_boundary_dirty = true;
      }
      g_interaction.active_region_index = saved_region_index;
    };

    // UI Panel - positioned on the right side
    {
      ImGuiIO& io = ImGui::GetIO();
      ImVec2 rightPos = ImVec2(io.DisplaySize.x - 380, 10);
      ImGui::SetNextWindowPos(rightPos, ImGuiCond_FirstUseEver);
      ImGui::SetNextWindowSize(ImVec2(370, 600), ImGuiCond_FirstUseEver);
      ImGui::Begin("Pattern Synthesis Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
      {
        ImGui::TextUnformatted("Model");
        ImGui::SetNextItemWidth(260.0f);
        ImGui::InputText(
          "Path##model_path",
          g_model_load.model_path,
          IM_ARRAYSIZE(g_model_load.model_path));
        if (ImGui::Button("Load Model", ImVec2(-1, 0))) {
          load_model_from_path(std::string(g_model_load.model_path));
        }
        if (!g_model_load.status.empty()) {
          const ImVec4 color = g_model_load.status_error
            ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f)
            : ImVec4(0.5f, 0.9f, 0.5f, 1.0f);
          ImGui::TextColored(color, "%s", g_model_load.status.c_str());
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // View Controls
        if (ImGui::CollapsingHeader("View Controls", ImGuiTreeNodeFlags_DefaultOpen)) {
          ImGui::Indent();
          ImGui::BulletText("Press SPACE to toggle 3D/UV view");
          ImGui::Text(
            "3D Mesh: %s",
            (surfaceMesh && surfaceMesh->isEnabled()) ? "VISIBLE" : "hidden");
          ImGui::Text(
            "UV Layout: %s",
            (uvMesh && uvMesh->isEnabled()) ? "VISIBLE" : "hidden");
          ImGui::Unindent();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        // Sampling Controls
        if (ImGui::CollapsingHeader("Random Point Sampling", ImGuiTreeNodeFlags_DefaultOpen)) {
          ImGui::Indent();

          ImGui::Text("Current samples: %d", g_sampling.n_samples);
          ImGui::Spacing();

          int n_new = g_sampling.n_samples;
          ImGui::SliderInt("Number of Samples##slider", &n_new, 100, 50000, "%d");
          g_sampling.n_samples = n_new;

          ImGui::Spacing();

          ImGui::Spacing();

          ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
          ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.8f, 0.3f, 1.0f));
          if (ImGui::Button("Generate Samples", ImVec2(-1, 0))) {
            const int n_face_uniform = g_sampling.n_samples / 1.5;
            const int n_uv_uniform = g_sampling.n_samples - n_face_uniform;
            g_sampling.samples_3d = sample_random_points(UV, F, n_face_uniform, g_sampling.random_seed);
            std::vector<DartSample> uv_uniform_samples =
              sample_random_points_uniform_uv(UV, F, n_uv_uniform, g_sampling.random_seed + 1);
            g_sampling.samples_3d.reserve(static_cast<size_t>(g_sampling.n_samples));
            g_sampling.samples_3d.insert(
              g_sampling.samples_3d.end(),
              uv_uniform_samples.begin(),
              uv_uniform_samples.end());
            g_sampling.samples_uv = g_sampling.samples_3d;
            g_sampling.points_uv = dart_samples_to_uv_points(g_sampling.samples_uv);
            g_sampling.points_3d = project_samples_to_3d(g_sampling.samples_3d, V, F);
            g_sampling.points_uv_3d = samples_to_points_3d(g_sampling.samples_uv, UV);
            reset_regions_for_sampling_change();

            update_sampling_point_clouds(surfaceMesh, uvMesh);
            std::string adjacency_error;
            if (rebuild_loaded_samples_adjacency(mesh_boundary_uv, adjacency_error)) {
              g_sampling.samples_io_status =
                "Generated " + std::to_string(g_sampling.n_samples) +
                " samples, Delaunay triangles: " + std::to_string(g_sampling.delaunay_triangle_count) +
                ", edges: " + std::to_string(g_sampling.delaunay_adjacency_edges) + ".";
              g_sampling.samples_io_status_error = false;
            } else {
              g_sampling.samples_io_status =
                "Generated " + std::to_string(g_sampling.n_samples) +
                " samples, but Delaunay build failed: " + adjacency_error;
              g_sampling.samples_io_status_error = true;
            }
            update_delaunay_graph_curve_networks(surfaceMesh, uvMesh);
            std::cout << g_sampling.samples_io_status << std::endl;
          }
          ImGui::PopStyleColor(2);

          ImGui::Spacing();

          ImGui::InputText("Samples File", g_sampling.samples_io_path, IM_ARRAYSIZE(g_sampling.samples_io_path));
          if (ImGui::Button("Save Current Samples", ImVec2(-1, 0))) {
            std::string io_error;
            const std::string path(g_sampling.samples_io_path);
            if (path.empty()) {
              g_sampling.samples_io_status = "Please provide a file path.";
              g_sampling.samples_io_status_error = true;
            } else if (save_samples_to_file(path, g_sampling.samples_uv, io_error)) {
              g_sampling.samples_io_status = "Saved " + std::to_string(g_sampling.samples_uv.size()) + " samples to " + path;
              g_sampling.samples_io_status_error = false;
            } else {
              g_sampling.samples_io_status = "Save failed: " + io_error;
              g_sampling.samples_io_status_error = true;
            }
            std::cout << g_sampling.samples_io_status << std::endl;
          }

          if (ImGui::Button("Load Samples", ImVec2(-1, 0))) {
            std::string io_error;
            std::vector<DartSample> loaded_samples;
            const std::string path(g_sampling.samples_io_path);
            if (path.empty()) {
              g_sampling.samples_io_status = "Please provide a file path.";
              g_sampling.samples_io_status_error = true;
            } else if (load_samples_from_file(path, F.rows(), loaded_samples, io_error)) {
              g_sampling.samples_uv = loaded_samples;
              g_sampling.samples_3d = loaded_samples;
              g_sampling.n_samples = static_cast<int>(loaded_samples.size());
              g_sampling.points_uv = dart_samples_to_uv_points(g_sampling.samples_uv);
              g_sampling.points_3d = project_samples_to_3d(g_sampling.samples_3d, V, F);
              g_sampling.points_uv_3d = samples_to_points_3d(g_sampling.samples_uv, UV);

              reset_regions_for_sampling_change();

              update_sampling_point_clouds(surfaceMesh, uvMesh);
              std::string adjacency_error;
              if (rebuild_loaded_samples_adjacency(mesh_boundary_uv, adjacency_error)) {
                g_sampling.samples_io_status =
                  "Loaded " + std::to_string(g_sampling.samples_uv.size()) +
                  " samples, Delaunay triangles: " + std::to_string(g_sampling.delaunay_triangle_count) +
                  ", edges: " + std::to_string(g_sampling.delaunay_adjacency_edges) + ".";
                g_sampling.samples_io_status_error = false;
              } else {
                g_sampling.samples_io_status =
                  "Loaded " + std::to_string(g_sampling.samples_uv.size()) +
                  " samples, but Delaunay build failed: " + adjacency_error;
                g_sampling.samples_io_status_error = true;
              }
              update_delaunay_graph_curve_networks(surfaceMesh, uvMesh);
            } else {
              g_sampling.samples_io_status = "Load failed: " + io_error;
              g_sampling.samples_io_status_error = true;
            }
            std::cout << g_sampling.samples_io_status << std::endl;
          }

          if (!g_sampling.samples_io_status.empty()) {
            const ImVec4 okColor(0.3f, 0.85f, 0.35f, 1.0f);
            const ImVec4 errColor(1.0f, 0.45f, 0.45f, 1.0f);
            ImGui::TextColored(
              g_sampling.samples_io_status_error ? errColor : okColor,
              "%s",
              g_sampling.samples_io_status.c_str());
          }

          ImGui::Spacing();
          ImGui::Separator();
          ImGui::Spacing();

          if (ImGui::Checkbox("Show samples##checkbox", &g_sampling.show_samples)) {
            auto* pc_3d = polyscope::hasPointCloud("samples_3d")
              ? polyscope::getPointCloud("samples_3d")
              : nullptr;
            auto* pc_uv = polyscope::hasPointCloud("samples_uv")
              ? polyscope::getPointCloud("samples_uv")
              : nullptr;
            if (pc_3d) pc_3d->setEnabled(g_sampling.show_samples && surfaceMesh->isEnabled());
            if (pc_uv) pc_uv->setEnabled(g_sampling.show_samples && uvMesh->isEnabled());
            std::cout << (g_sampling.show_samples ? "Showing samples" : "Hiding samples") << std::endl;
          }
          if (ImGui::Checkbox("Show Delaunay graph##checkbox", &g_sampling.show_delaunay_graph)) {
            update_delaunay_graph_curve_networks(surfaceMesh, uvMesh);
            std::cout << (g_sampling.show_delaunay_graph ? "Showing Delaunay graph" : "Hiding Delaunay graph") << std::endl;
          }
          ImGui::TextUnformatted("Graph shows Delaunay triangulation edges between sample points.");

          ImGui::Spacing();
          ImGui::Text("Points info:");
          ImGui::BulletText("3D points: %zu", g_sampling.points_3d.rows());
          ImGui::BulletText("UV points: %zu", g_sampling.points_uv_3d.rows());

          ImGui::Unindent();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        if (ImGui::CollapsingHeader("Lloyd Relaxation", ImGuiTreeNodeFlags_DefaultOpen)) {
          ImGui::Indent();

          ImGui::SliderInt("Max Iters##lloyd_iters", &g_lloyd.max_iters, 1, 500);
          float convergence = static_cast<float>(g_lloyd.convergence_threshold);
          if (ImGui::SliderFloat("Convergence##lloyd_conv", &convergence, 1e-6f, 1e-2f, "%.6f")) {
            g_lloyd.convergence_threshold = static_cast<double>(convergence);
          }

          ImGui::Spacing();

          ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.4f, 0.8f, 1.0f));
          ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.5f, 0.9f, 1.0f));
          if (ImGui::Button("Run Lloyd Relaxation", ImVec2(-1, 0))) {
            if (g_sampling.samples_uv.empty()) {
              std::cerr << "No samples available. Generate samples first." << std::endl;
            } else if (UV.rows() == 0 || UV.cols() < 2 || F.rows() == 0) {
              std::cerr << "UV/F data unavailable. Cannot run Lloyd relaxation." << std::endl;
            } else {
              if (g_sampling.points_uv.rows() == 0 || g_sampling.points_uv.cols() < 2) {
                g_sampling.points_uv = dart_samples_to_uv_points(g_sampling.samples_uv);
              }

              if (!g_lloyd.finder ||
                  g_lloyd.finder_grid_size != g_lloyd.grid_size ||
                  g_lloyd.finder_uv_rows != UV.rows() ||
                  g_lloyd.finder_f_rows != F.rows()) {
                g_lloyd.finder = std::make_unique<UniformGridTriangleFinder>(UV, F, g_lloyd.grid_size);
                g_lloyd.finder_grid_size = g_lloyd.grid_size;
                g_lloyd.finder_uv_rows = UV.rows();
                g_lloyd.finder_f_rows = F.rows();
              }

              Eigen::MatrixXd relaxed_uv = lloyd_relaxation(
                g_sampling.points_uv,
                UV,
                F,
                vertex_distortion,
                g_lloyd.max_iters,
                g_lloyd.convergence_threshold,
                g_lloyd.grid_size,
                g_lloyd.finder.get(),
                &g_lloyd.last_max_displacement,
                &g_lloyd.last_min_displacement,
                &g_lloyd.last_avg_displacement,
                &g_lloyd.last_iters_run);

              g_sampling.points_uv = relaxed_uv;
              std::vector<DartSample> new_samples = g_sampling.samples_uv;

              for (int i = 0; i < relaxed_uv.rows(); ++i) {
                DartSample updated;
                Eigen::Vector2d uv = relaxed_uv.row(i).head<2>();
                if (build_sample_from_uv(uv, *g_lloyd.finder, updated)) {
                  if (i < static_cast<int>(new_samples.size())) {
                    new_samples[i] = updated;
                  }
                }
              }

              g_sampling.samples_uv = new_samples;
              g_sampling.samples_3d = new_samples;
              g_sampling.points_3d = project_samples_to_3d(g_sampling.samples_3d, V, F);
              g_sampling.points_uv_3d = samples_to_points_3d(g_sampling.samples_uv, UV);
              reset_regions_for_sampling_change();

              std::string adjacency_error;
              if (!rebuild_loaded_samples_adjacency(mesh_boundary_uv, adjacency_error)) {
                std::cerr << "Adjacency build after Lloyd failed: " << adjacency_error << std::endl;
              }
              update_delaunay_graph_curve_networks(surfaceMesh, uvMesh);
              update_sampling_point_clouds(surfaceMesh, uvMesh);

              g_lloyd.has_stats = true;
              std::cout << "Lloyd relaxation complete (iters: " << g_lloyd.last_iters_run << ")" << std::endl;
            }
          }
          ImGui::PopStyleColor(2);

          ImGui::Unindent();
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

        build_interaction_ui(
          g_interaction,
          surfaceMesh,
          uvMesh,
          g_sampling.samples_uv,
          V,
          g_sampling.points_3d,
          g_sampling.points_uv,
          UV,
          F,
          g_sampling.delaunay_helper.get(),
          &face_distortion_asurf_over_auv);

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

      }
      ImGui::End();
    }

    handle_interaction_input(
      g_interaction,
      surfaceMesh,
      uvMesh,
      V,
      UV,
      F,
      g_sampling.points_3d,
      g_sampling.points_uv,
      g_sampling.delaunay_helper.get());

    sync_viewer_scene_to_structure(active_view_structure(surfaceMesh, uvMesh));
  };

  polyscope::show();

  return 0;
}
