#include "parameterization.h"
#include "uv_display.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <vector>
#include <igl/boundary_loop.h>
#include <igl/lscm.h>
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>

namespace {

Eigen::VectorXd log_normalize_distortion_values(const Eigen::VectorXd& values) {
  if (values.size() == 0) {
    return Eigen::VectorXd();
  }

  Eigen::VectorXd safe_values(values.size());
  for (int i = 0; i < values.size(); ++i) {
    const double value = values(i);
    safe_values(i) = std::isfinite(value) ? std::max(value, 1e-7) : 1e-7;
  }

  Eigen::VectorXd processed = safe_values.array().log().matrix();
  const double min_value = processed.minCoeff();
  const double max_value = processed.maxCoeff();
  const double span = max_value - min_value;

  if (!std::isfinite(min_value) || !std::isfinite(max_value) || span <= 1e-12) {
    return Eigen::VectorXd::Zero(values.size());
  }

  Eigen::VectorXd normalized = ((processed.array() - min_value) / span).matrix();
  return normalized.array().max(0.0).min(1.0).matrix();
}

struct UvQuality {
  bool valid = false;
  int degenerate_face_count = std::numeric_limits<int>::max();
  double min_positive_face_area = 0.0;
  double total_face_area = 0.0;
};

UvQuality evaluate_uv_quality(
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F) {
  UvQuality quality;
  if (UV.rows() == 0 || UV.cols() < 2 || F.rows() == 0 || F.cols() < 3) {
    return quality;
  }

  quality.valid = true;
  quality.degenerate_face_count = 0;
  double min_positive_face_area = std::numeric_limits<double>::infinity();

  for (int f = 0; f < F.rows(); ++f) {
    const int i0 = F(f, 0);
    const int i1 = F(f, 1);
    const int i2 = F(f, 2);
    if (i0 < 0 || i1 < 0 || i2 < 0 ||
        i0 >= UV.rows() || i1 >= UV.rows() || i2 >= UV.rows()) {
      quality.valid = false;
      return quality;
    }

    const Eigen::Vector2d u0 = UV.row(i0).head<2>();
    const Eigen::Vector2d u1 = UV.row(i1).head<2>();
    const Eigen::Vector2d u2 = UV.row(i2).head<2>();
    if (!u0.allFinite() || !u1.allFinite() || !u2.allFinite()) {
      quality.valid = false;
      return quality;
    }

    const double area = 0.5 * std::abs(
      (u1.x() - u0.x()) * (u2.y() - u0.y()) -
      (u1.y() - u0.y()) * (u2.x() - u0.x()));
    if (!std::isfinite(area)) {
      quality.valid = false;
      return quality;
    }

    quality.total_face_area += area;
    if (area < 1e-12) {
      ++quality.degenerate_face_count;
    } else if (area < min_positive_face_area) {
      min_positive_face_area = area;
    }
  }

  quality.min_positive_face_area =
    std::isfinite(min_positive_face_area) ? min_positive_face_area : 0.0;
  return quality;
}

bool uv_quality_better(const UvQuality& lhs, const UvQuality& rhs) {
  if (!lhs.valid) {
    return false;
  }
  if (!rhs.valid) {
    return true;
  }
  if (lhs.degenerate_face_count != rhs.degenerate_face_count) {
    return lhs.degenerate_face_count < rhs.degenerate_face_count;
  }
  if (lhs.min_positive_face_area != rhs.min_positive_face_area) {
    return lhs.min_positive_face_area > rhs.min_positive_face_area;
  }
  return lhs.total_face_area > rhs.total_face_area;
}

bool build_lscm_boundary_constraints(
  const Eigen::MatrixXd& V,
  int vertex_a,
  int vertex_b,
  Eigen::VectorXi& out_b,
  Eigen::MatrixXd& out_bc) {
  if (vertex_a < 0 || vertex_b < 0 || vertex_a == vertex_b ||
      vertex_a >= V.rows() || vertex_b >= V.rows()) {
    return false;
  }

  double pin_spacing = 1.0;
  if (V.cols() >= 3) {
    const Eigen::Vector3d p0 = V.row(vertex_a).head<3>();
    const Eigen::Vector3d p1 = V.row(vertex_b).head<3>();
    pin_spacing = (p1 - p0).norm();
  }
  if (!std::isfinite(pin_spacing) || pin_spacing <= 1e-8) {
    pin_spacing = 1.0;
  }

  out_b.resize(2);
  out_b << vertex_a, vertex_b;
  out_bc.resize(2, 2);
  out_bc << 0.0, 0.0,
            pin_spacing, 0.0;
  return true;
}

std::vector<std::pair<int, int>> build_lscm_candidate_pairs(
  const Eigen::MatrixXd& V,
  const Eigen::VectorXi& boundary_loop) {
  std::vector<std::pair<int, int>> candidate_pairs;
  std::set<std::pair<int, int>> seen_pairs;
  if (boundary_loop.size() < 2) {
    return candidate_pairs;
  }

  auto add_candidate = [&](int vertex_a, int vertex_b) {
    if (vertex_a < 0 || vertex_b < 0 || vertex_a == vertex_b) {
      return;
    }
    const std::pair<int, int> key(
      std::min(vertex_a, vertex_b),
      std::max(vertex_a, vertex_b));
    if (!seen_pairs.insert(key).second) {
      return;
    }
    candidate_pairs.emplace_back(vertex_a, vertex_b);
  };

  const int loop_size = static_cast<int>(boundary_loop.size());
  for (int k = 0; k < 8; ++k) {
    const int offset_a = (k * loop_size) / 8;
    const int offset_b = (offset_a + loop_size / 2) % loop_size;
    add_candidate(boundary_loop[offset_a], boundary_loop[offset_b]);
  }

  double best_sq_dist = -1.0;
  int farthest_a = -1;
  int farthest_b = -1;
  if (V.cols() >= 3) {
    for (int i = 0; i < loop_size; ++i) {
      const int vertex_i = boundary_loop[i];
      if (vertex_i < 0 || vertex_i >= V.rows()) {
        continue;
      }
      const Eigen::Vector3d pi = V.row(vertex_i).head<3>();
      for (int j = i + 1; j < loop_size; ++j) {
        const int vertex_j = boundary_loop[j];
        if (vertex_j < 0 || vertex_j >= V.rows()) {
          continue;
        }
        const Eigen::Vector3d pj = V.row(vertex_j).head<3>();
        const double sq_dist = (pj - pi).squaredNorm();
        if (sq_dist > best_sq_dist) {
          best_sq_dist = sq_dist;
          farthest_a = vertex_i;
          farthest_b = vertex_j;
        }
      }
    }
  }
  add_candidate(farthest_a, farthest_b);

  return candidate_pairs;
}

} // namespace

Eigen::VectorXd compute_triangle_area_distortion(
  const Eigen::MatrixXd &V,
  const Eigen::MatrixXd &UV,
  const Eigen::MatrixXi &F)
{
  int nV = static_cast<int>(V.rows());
  int nF = static_cast<int>(F.rows());

  // Prepare accumulators for EACH VERTEX
  Eigen::VectorXd area3DAcc = Eigen::VectorXd::Zero(nV);
  Eigen::VectorXd areaUVAcc = Eigen::VectorXd::Zero(nV);

  // Loop over faces
  for (int f = 0; f < nF; ++f)
  {
    // Face's three vertices
    int i0 = F(f, 0);
    int i1 = F(f, 1);
    int i2 = F(f, 2);

    // ========== 3D AREA CALCULATION ==========
    Eigen::Vector3d p0 = V.row(i0);
    Eigen::Vector3d p1 = V.row(i1);
    Eigen::Vector3d p2 = V.row(i2);
    double faceArea3D = 0.5 * ((p1 - p0).cross(p2 - p0)).norm();

    // ========== UV AREA CALCULATION ==========
    Eigen::Vector2d u0 = UV.row(i0).head<2>();
    Eigen::Vector2d u1 = UV.row(i1).head<2>();
    Eigen::Vector2d u2 = UV.row(i2).head<2>();

    // 2D cross product = determinant for signed area
    double faceAreaUV = 0.5 * std::abs(
                    (u1 - u0).x() * (u2 - u0).y() -
                    (u1 - u0).y() * (u2 - u0).x());

    // ========== ACCUMULATE TO VERTICES ==========
    area3DAcc[i0] += faceArea3D;
    area3DAcc[i1] += faceArea3D;
    area3DAcc[i2] += faceArea3D;

    areaUVAcc[i0] += faceAreaUV;
    areaUVAcc[i1] += faceAreaUV;
    areaUVAcc[i2] += faceAreaUV;
  }

  // Now compute ratio per vertex
  Eigen::VectorXd s(nV);
  s.setZero();

  for (int v = 0; v < nV; ++v)
  {
    if (areaUVAcc[v] > 1e-14)
      s[v] = area3DAcc[v] / areaUVAcc[v];
    else
      s[v] = 0.0;
  }

  return s;
}

Eigen::VectorXd compute_visualized_face_distortion(
  const Eigen::VectorXd& vertex_distortion,
  const Eigen::MatrixXi& F) {
  if (vertex_distortion.size() == 0 || F.rows() == 0 || F.cols() == 0) {
    return Eigen::VectorXd();
  }

  Eigen::VectorXd face_distortion(F.rows());
  for (int f = 0; f < F.rows(); ++f) {
    double face_stretch_sum = 0.0;
    int valid_corners = 0;

    for (int c = 0; c < F.cols(); ++c) {
      const int v_idx = F(f, c);
      if (v_idx >= 0 && v_idx < vertex_distortion.size() &&
          std::isfinite(vertex_distortion(v_idx))) {
        face_stretch_sum += vertex_distortion(v_idx);
        valid_corners++;
      }
    }

    face_distortion(f) = (valid_corners > 0)
      ? (face_stretch_sum / static_cast<double>(valid_corners))
      : 0.0;
  }

  return log_normalize_distortion_values(face_distortion);
}

Eigen::VectorXd compute_visualized_vertex_distortion(
  const Eigen::VectorXd& vertex_distortion) {
  return log_normalize_distortion_values(vertex_distortion);
}

Eigen::MatrixXd parameterisation(
  const std::string& meshName,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F) {
  Eigen::MatrixXd UV;
  Eigen::VectorXi boundaryLoops;
  igl::boundary_loop(F, boundaryLoops);

  if (boundaryLoops.size() == 0) {
    std::cerr << "No boundary detected.\n";
    return UV;
  } else {
    std::cout << "Boundary detected with " << boundaryLoops.size() << " vertices.\n";
  }

  bool lscm_success = false;
  UvQuality best_quality;
  std::pair<int, int> best_pin_pair(-1, -1);
  const std::vector<std::pair<int, int>> candidate_pairs =
    build_lscm_candidate_pairs(V, boundaryLoops);
  for (const auto& candidate_pair : candidate_pairs) {
    Eigen::VectorXi pinned_vertices;
    Eigen::MatrixXd pinned_uv;
    if (!build_lscm_boundary_constraints(
          V,
          candidate_pair.first,
          candidate_pair.second,
          pinned_vertices,
          pinned_uv)) {
      continue;
    }

    Eigen::MatrixXd candidate_uv;
    if (!igl::lscm(V, F, pinned_vertices, pinned_uv, candidate_uv)) {
      continue;
    }

    const UvQuality candidate_quality = evaluate_uv_quality(candidate_uv, F);
    if (uv_quality_better(candidate_quality, best_quality)) {
      UV = candidate_uv;
      best_quality = candidate_quality;
      best_pin_pair = candidate_pair;
      lscm_success = true;
    }
  }

  if (!lscm_success) {
    lscm_success = igl::lscm(V, F, UV);
    best_quality = evaluate_uv_quality(UV, F);
  }

  if (!lscm_success) {
    std::cerr << "LSCM failed.\n";
  } else {
    if (best_pin_pair.first >= 0 && best_pin_pair.second >= 0) {
      std::cout
        << "LSCM pinned boundary vertices: "
        << best_pin_pair.first << " and " << best_pin_pair.second << "\n";
    }
    if (best_quality.valid && best_quality.degenerate_face_count > 0) {
      std::cerr
        << "LSCM produced " << best_quality.degenerate_face_count
        << " degenerate UV faces.\n";
    }
  }

  polyscope::init();
  auto* surfaceMesh = polyscope::registerSurfaceMesh(meshName, V, F);

  // Create UV mesh (3D projection with Z=0), centered for display only.
  Eigen::MatrixXd UV_3d;
  if (UV.size() > 0 && UV.rows() == V.rows() && UV.cols() == 2) {
    UV_3d = uv_matrix_to_display_3d(UV, uv_display_offset_from_vertices(UV));
  }

  auto* uvMesh = polyscope::registerSurfaceMesh("uv_layout", UV_3d, F);

  if (UV.size() > 0 && UV.rows() == V.rows() && UV.cols() == 2) {
    Eigen::VectorXd distortion_raw = compute_triangle_area_distortion(V, UV, F);

    if (distortion_raw.size() == V.rows()) {
      Eigen::VectorXd S_normalized = compute_visualized_face_distortion(distortion_raw, F);

      auto* distortion3D = surfaceMesh->addFaceScalarQuantity("distortion", S_normalized);
      if (distortion3D) {
        distortion3D->setColorMap("jet");
        distortion3D->setEnabled(true);
      }

      if (uvMesh) {
        auto* distortionUV = uvMesh->addFaceScalarQuantity("distortion", S_normalized);
        if (distortionUV) {
          distortionUV->setColorMap("jet");
          distortionUV->setEnabled(true);
        }
      }
    }
  }
  return UV;
}
