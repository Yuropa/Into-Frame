#include "interaction.h"
#include "lloyd_relaxation.h"
#include "uv_display.h"
#include "voronoi-pcf.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cmath>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include <Eigen/Dense>
#include <imgui.h>
#include <glm/glm.hpp>
#include <polyscope/curve_network.h>
#include <polyscope/pick.h>
#include <polyscope/point_cloud.h>
#include <polyscope/surface_color_quantity.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/view.h>

namespace {

const std::string kFaceCentersName = "face_centers";
const std::string kSelectedSamples3DBaseName = "selected_samples_3d";
const std::string kSelectedSamplesUVBaseName = "selected_samples_uv";
const std::string kOutputPattern3DBaseName = "output_pattern_3d";
const std::string kOutputPatternUVBaseName = "output_pattern_uv";
const std::string kInputReference3DBaseName = "input_reference_3d";
const std::string kInputReferenceUVBaseName = "input_reference_uv";
const std::string kInputBoundaryBaseName = "input_boundary";
const std::string kInputBoundary3DBaseName = "input_boundary_3d";
const std::string kInputBoundaryFaceBaseName = "input_boundary_faces";
const std::string kOutputBoundaryBaseName = "output_boundary";
const std::string kOutputBoundaryCurve3DBaseName = "output_boundary_curve_3d";
const std::string kOutputBoundaryCurveUVBaseName = "output_boundary_curve_uv";
const std::string kPaintBrushPreviewCurve3DBaseName = "paint_brush_preview_curve_3d";
const std::string kPaintBrushPreviewCurveUVBaseName = "paint_brush_preview_curve_uv";
const std::string kWholeModelPatchPreviewCurve3DBaseName = "whole_model_patch_preview_curve_3d";
const std::string kWholeModelPatchPreviewCurveUVBaseName = "whole_model_patch_preview_curve_uv";
const std::string kWholeModelPatchPreviewFaces3DName = "whole_model_patch_preview_faces_3d";
const std::string kWholeModelPatchPreviewFacesUVName = "whole_model_patch_preview_faces_uv";
const std::string kPatternFileHeader = "PATTERN_SYNTHESIS_INPUT_PATTERN_V1";
const std::string kOutputPatternFileHeader = "PATTERN_SYNTHESIS_OUTPUT_PATTERN_V1";
const std::string kClass1ObjectSuffix = "_class_1";

const glm::vec3 kClass0Color(0.18f, 0.42f, 1.0f);
const glm::vec3 kClass1Color(0.95f, 0.25f, 0.62f);
const glm::vec3 kInputPaintPreviewColor(0.30f, 0.92f, 0.56f);
const glm::vec3 kOutputPaintPreviewColor(1.0f, 0.76f, 0.20f);
const glm::vec3 kWholeModelPatchPreviewGapColor(0.16f, 0.16f, 0.16f);
constexpr double kPatchSeedCompactnessWeight = 0.15;
constexpr double kPatchDetourCompactnessWeight = 0.35;

struct Point2D {
  double x;
  double y;
  int idx;
};

std::string region_object_name(const std::string& base_name, int region_id) {
  return base_name + "_region_" + std::to_string(region_id);
}

std::string whole_model_patch_preview_name(const std::string& base_name, int patch_index) {
  return base_name + "_" + std::to_string(patch_index);
}

std::uint64_t undirected_edge_key(int a, int b) {
  const std::uint32_t lo = static_cast<std::uint32_t>(std::min(a, b));
  const std::uint32_t hi = static_cast<std::uint32_t>(std::max(a, b));
  return (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
}

glm::vec3 whole_model_patch_preview_color(int patch_index) {
  static const std::array<glm::vec3, 8> kPatchPreviewPalette = {
    glm::vec3(0.92f, 0.32f, 0.28f),
    glm::vec3(0.20f, 0.62f, 0.98f),
    glm::vec3(0.23f, 0.78f, 0.46f),
    glm::vec3(0.93f, 0.70f, 0.18f),
    glm::vec3(0.66f, 0.40f, 0.95f),
    glm::vec3(0.96f, 0.46f, 0.72f),
    glm::vec3(0.20f, 0.82f, 0.82f),
    glm::vec3(0.95f, 0.58f, 0.22f)
  };
  return kPatchPreviewPalette[static_cast<size_t>(patch_index) % kPatchPreviewPalette.size()];
}

void build_boundary_from_painted_faces(
  const std::vector<int>& painted_face_indices,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F,
  Eigen::MatrixXd& out_boundary_3d,
  Eigen::MatrixXd& out_boundary_uv);

void set_region_default_name(PatternRegionState& region, int default_index) {
  const std::string label = default_pattern_region_label(default_index);
  std::snprintf(region.display_name, sizeof(region.display_name), "%s", label.c_str());
}

void ensure_region_identity(InteractionState& root_state, PatternRegionState& region, int default_index) {
  if (region.region_id < 0) {
    region.region_id = root_state.next_region_id++;
  }
  if (region.display_name[0] == '\0') {
    set_region_default_name(region, default_index);
  }
}

bool is_valid_transition_source(
  const InteractionState& root_state,
  int self_region_id,
  int candidate_region_id,
  int other_source_region_id = -1) {
  if (candidate_region_id < 0 || candidate_region_id == self_region_id) {
    return false;
  }
  if (other_source_region_id >= 0 && candidate_region_id == other_source_region_id) {
    return false;
  }
  const PatternRegionState* candidate = find_region_by_id(root_state, candidate_region_id);
  return candidate && region_is_exemplar(*candidate);
}

int first_valid_transition_source_id(
  const InteractionState& root_state,
  int self_region_id,
  int other_source_region_id = -1) {
  for (int region_index = 0; region_index < region_count(root_state); ++region_index) {
    const PatternRegionState& candidate = region_state(root_state, region_index);
    if (is_valid_transition_source(
          root_state,
          self_region_id,
          candidate.region_id,
          other_source_region_id)) {
      return candidate.region_id;
    }
  }
  return -1;
}

template <typename StructureT>
double viewer_overlay_radius(StructureT* structure, double relative_size) {
  if (!std::isfinite(relative_size) || relative_size <= 1e-6) {
    return 1e-6;
  }
  if (!structure) {
    return std::max(relative_size, 1e-6);
  }

  double length_scale = static_cast<double>(structure->lengthScale());
  if (!std::isfinite(length_scale) || length_scale <= 1e-6) {
    length_scale = 1.0;
  }

  return std::max(relative_size * length_scale, 1e-6);
}

double preserved_point_cloud_radius(polyscope::PointCloud* point_cloud, double fallback) {
  if (!point_cloud) {
    return fallback;
  }

  const double radius = point_cloud->getPointRadius();
  if (!std::isfinite(radius) || radius <= 0.0) {
    return fallback;
  }
  return radius;
}

float preserved_curve_network_radius(polyscope::CurveNetwork* curve, double fallback) {
  if (!curve) {
    return static_cast<float>(fallback);
  }

  const float radius = curve->getRadius();
  if (!std::isfinite(radius) || radius <= 0.f) {
    return static_cast<float>(fallback);
  }
  return radius;
}

polyscope::PointCloud* ensure_point_cloud_with_preserved_radius(
  polyscope::PointCloud*& point_cloud,
  const std::string& name,
  const Eigen::MatrixXd& data,
  double default_radius) {
  if (!polyscope::hasPointCloud(name)) {
    point_cloud = nullptr;
  } else if (!point_cloud) {
    point_cloud = polyscope::getPointCloud(name);
  }

  double radius = default_radius;
  if (point_cloud) {
    radius = preserved_point_cloud_radius(point_cloud, default_radius);
    if (point_cloud->nPoints() == static_cast<size_t>(data.rows())) {
      point_cloud->updatePointPositions(data);
      return point_cloud;
    }
    polyscope::removePointCloud(name);
  }

  point_cloud = polyscope::registerPointCloud(name, data);
  if (point_cloud) {
    point_cloud->setPointRadius(radius, false);
  }
  return point_cloud;
}

polyscope::CurveNetwork* ensure_curve_network_loop_with_preserved_radius(
  polyscope::CurveNetwork*& curve,
  const std::string& name,
  const Eigen::MatrixXd& nodes,
  double default_radius) {
  if (!polyscope::hasCurveNetwork(name)) {
    curve = nullptr;
  } else if (!curve) {
    curve = polyscope::getCurveNetwork(name);
  }

  float radius = static_cast<float>(default_radius);
  if (curve) {
    radius = preserved_curve_network_radius(curve, default_radius);
    if (curve->nNodes() == static_cast<size_t>(nodes.rows())) {
      curve->updateNodePositions(nodes);
      return curve;
    }
    polyscope::removeCurveNetwork(name);
  }

  curve = polyscope::registerCurveNetworkLoop(name, nodes);
  if (curve) {
    curve->setRadius(radius, false);
  }
  return curve;
}

polyscope::CurveNetwork* ensure_curve_network_loop_2d_with_preserved_radius(
  polyscope::CurveNetwork*& curve,
  const std::string& name,
  const Eigen::MatrixXd& nodes,
  double default_radius) {
  if (!polyscope::hasCurveNetwork(name)) {
    curve = nullptr;
  } else if (!curve) {
    curve = polyscope::getCurveNetwork(name);
  }

  float radius = static_cast<float>(default_radius);
  if (curve) {
    radius = preserved_curve_network_radius(curve, default_radius);
    if (curve->nNodes() == static_cast<size_t>(nodes.rows())) {
      curve->updateNodePositions2D(nodes);
      return curve;
    }
    polyscope::removeCurveNetwork(name);
  }

  curve = polyscope::registerCurveNetworkLoop2D(name, nodes);
  if (curve) {
    curve->setRadius(radius, false);
  }
  return curve;
}

void sanitize_transition_sources(InteractionState& root_state, PatternRegionState& region) {
  if (is_valid_transition_source(
        root_state,
        region.region_id,
        region.transition_source_a_region_id,
        region.transition_source_b_region_id)) {
    // Source A is already valid.
  } else {
    region.transition_source_a_region_id =
      first_valid_transition_source_id(root_state, region.region_id);
  }

  if (is_valid_transition_source(
        root_state,
        region.region_id,
        region.transition_source_b_region_id,
        region.transition_source_a_region_id)) {
    // Source B is already valid.
  } else {
    region.transition_source_b_region_id =
      first_valid_transition_source_id(
        root_state,
        region.region_id,
        region.transition_source_a_region_id);
  }
}

void ensure_region_metadata(InteractionState& root_state) {
  if (root_state.regions.empty()) {
    root_state.regions.emplace_back();
  }
  root_state.active_region_index = clamp_region_index(root_state, root_state.active_region_index);
  for (int region_index = 0; region_index < region_count(root_state); ++region_index) {
    PatternRegionState& region = region_state(root_state, region_index);
    ensure_region_identity(root_state, region, region_index);
    region.region_mode = static_cast<int>(PatternRegionMode::Exemplar);
    region.transition_source_a_region_id = -1;
    region.transition_source_b_region_id = -1;
    region.active_pattern_class_id = 0;
    std::fill(
      region.pattern_point_class_ids.begin(),
      region.pattern_point_class_ids.end(),
      0);
    std::fill(
      region.output_pattern_class_ids.begin(),
      region.output_pattern_class_ids.end(),
      0);
  }
}

void invalidate_transition_regions(
  InteractionState& root_state,
  int source_region_id = -1) {
  const int saved_active_region_index = root_state.active_region_index;
  for (int region_index = 0; region_index < region_count(root_state); ++region_index) {
    PatternRegionState& region = region_state(root_state, region_index);
    if (!region_is_transition(region)) {
      continue;
    }
    if (source_region_id >= 0 &&
        region.transition_source_a_region_id != source_region_id &&
        region.transition_source_b_region_id != source_region_id) {
      continue;
    }
    root_state.active_region_index = region_index;
    reset_voronoi_pcf(root_state);
  }
  root_state.active_region_index = saved_active_region_index;
}

void clear_whole_model_patch_preview(
  InteractionState& root_state,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh) {
  const int preview_count =
    static_cast<int>(root_state.whole_model_patch_preview_face_partitions.size());
  for (int patch_index = 0; patch_index < preview_count; ++patch_index) {
    const std::string preview_curve_3d_name =
      whole_model_patch_preview_name(kWholeModelPatchPreviewCurve3DBaseName, patch_index);
    const std::string preview_curve_uv_name =
      whole_model_patch_preview_name(kWholeModelPatchPreviewCurveUVBaseName, patch_index);
    if (polyscope::hasCurveNetwork(preview_curve_3d_name)) {
      polyscope::removeCurveNetwork(preview_curve_3d_name);
    }
    if (polyscope::hasCurveNetwork(preview_curve_uv_name)) {
      polyscope::removeCurveNetwork(preview_curve_uv_name);
    }
  }
  if (surfaceMesh) {
    surfaceMesh->removeQuantity(kWholeModelPatchPreviewFaces3DName, false);
  }
  if (uvMesh) {
    uvMesh->removeQuantity(kWholeModelPatchPreviewFacesUVName, false);
  }
  root_state.whole_model_patch_preview_face_partitions.clear();
  root_state.whole_model_patch_preview_active = false;
  root_state.whole_model_patch_preview_source_region_id = -1;
  root_state.whole_model_patch_preview_surface = nullptr;
  root_state.whole_model_patch_preview_uv = nullptr;
}

void remove_region_visuals(
  const PatternRegionState& region,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh) {
  const std::string selected_samples_3d_name =
    region_object_name(kSelectedSamples3DBaseName, region.region_id);
  const std::string selected_samples_uv_name =
    region_object_name(kSelectedSamplesUVBaseName, region.region_id);
  const std::string selected_samples_3d_class_1_name =
    selected_samples_3d_name + kClass1ObjectSuffix;
  const std::string selected_samples_uv_class_1_name =
    selected_samples_uv_name + kClass1ObjectSuffix;
  const std::string output_pattern_3d_name =
    region_object_name(kOutputPattern3DBaseName, region.region_id);
  const std::string output_pattern_uv_name =
    region_object_name(kOutputPatternUVBaseName, region.region_id);
  const std::string output_pattern_3d_class_1_name =
    output_pattern_3d_name + kClass1ObjectSuffix;
  const std::string output_pattern_uv_class_1_name =
    output_pattern_uv_name + kClass1ObjectSuffix;
  const std::string input_reference_3d_name =
    region_object_name(kInputReference3DBaseName, region.region_id);
  const std::string input_reference_uv_name =
    region_object_name(kInputReferenceUVBaseName, region.region_id);
  const std::string input_boundary_name =
    region_object_name(kInputBoundaryBaseName, region.region_id);
  const std::string input_boundary_3d_name =
    region_object_name(kInputBoundary3DBaseName, region.region_id);
  const std::string output_boundary_curve_3d_name =
    region_object_name(kOutputBoundaryCurve3DBaseName, region.region_id);
  const std::string output_boundary_curve_uv_name =
    region_object_name(kOutputBoundaryCurveUVBaseName, region.region_id);
  const std::string paint_brush_preview_curve_3d_name =
    region_object_name(kPaintBrushPreviewCurve3DBaseName, region.region_id);
  const std::string paint_brush_preview_curve_uv_name =
    region_object_name(kPaintBrushPreviewCurveUVBaseName, region.region_id);
  const std::string output_boundary_name =
    region_object_name(kOutputBoundaryBaseName, region.region_id);
  const std::string input_boundary_face_name =
    region_object_name(kInputBoundaryFaceBaseName, region.region_id);

  if (polyscope::hasPointCloud(selected_samples_3d_name)) {
    polyscope::removePointCloud(selected_samples_3d_name);
  }
  if (polyscope::hasPointCloud(selected_samples_uv_name)) {
    polyscope::removePointCloud(selected_samples_uv_name);
  }
  if (polyscope::hasPointCloud(selected_samples_3d_class_1_name)) {
    polyscope::removePointCloud(selected_samples_3d_class_1_name);
  }
  if (polyscope::hasPointCloud(selected_samples_uv_class_1_name)) {
    polyscope::removePointCloud(selected_samples_uv_class_1_name);
  }
  if (polyscope::hasPointCloud(output_pattern_3d_name)) {
    polyscope::removePointCloud(output_pattern_3d_name);
  }
  if (polyscope::hasPointCloud(output_pattern_uv_name)) {
    polyscope::removePointCloud(output_pattern_uv_name);
  }
  if (polyscope::hasPointCloud(output_pattern_3d_class_1_name)) {
    polyscope::removePointCloud(output_pattern_3d_class_1_name);
  }
  if (polyscope::hasPointCloud(output_pattern_uv_class_1_name)) {
    polyscope::removePointCloud(output_pattern_uv_class_1_name);
  }
  if (polyscope::hasPointCloud(input_reference_3d_name)) {
    polyscope::removePointCloud(input_reference_3d_name);
  }
  if (polyscope::hasPointCloud(input_reference_uv_name)) {
    polyscope::removePointCloud(input_reference_uv_name);
  }
  if (polyscope::hasCurveNetwork(input_boundary_name)) {
    polyscope::removeCurveNetwork(input_boundary_name);
  }
  if (polyscope::hasCurveNetwork(input_boundary_3d_name)) {
    polyscope::removeCurveNetwork(input_boundary_3d_name);
  }
  if (polyscope::hasCurveNetwork(output_boundary_curve_3d_name)) {
    polyscope::removeCurveNetwork(output_boundary_curve_3d_name);
  }
  if (polyscope::hasCurveNetwork(output_boundary_curve_uv_name)) {
    polyscope::removeCurveNetwork(output_boundary_curve_uv_name);
  }
  if (polyscope::hasCurveNetwork(paint_brush_preview_curve_3d_name)) {
    polyscope::removeCurveNetwork(paint_brush_preview_curve_3d_name);
  }
  if (polyscope::hasCurveNetwork(paint_brush_preview_curve_uv_name)) {
    polyscope::removeCurveNetwork(paint_brush_preview_curve_uv_name);
  }
  if (surfaceMesh) {
    surfaceMesh->removeQuantity(output_boundary_name, false);
    surfaceMesh->removeQuantity(input_boundary_face_name, false);
  }
  if (uvMesh) {
    uvMesh->removeQuantity(output_boundary_name, false);
    uvMesh->removeQuantity(input_boundary_face_name, false);
  }
}

double cross(const Point2D& O, const Point2D& A, const Point2D& B) {
  return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

double orient2d(
  const Eigen::Vector2d& a,
  const Eigen::Vector2d& b,
  const Eigen::Vector2d& c) {
  const Eigen::Vector2d ab = b - a;
  const Eigen::Vector2d ac = c - a;
  return ab.x() * ac.y() - ab.y() * ac.x();
}

bool point_on_segment_2d(
  const Eigen::Vector2d& p,
  const Eigen::Vector2d& a,
  const Eigen::Vector2d& b,
  double eps) {
  const Eigen::Vector2d ab = b - a;
  const double ab2 = ab.squaredNorm();
  if (ab2 <= eps * eps) {
    return (p - a).squaredNorm() <= eps * eps;
  }
  const double t = std::clamp((p - a).dot(ab) / ab2, 0.0, 1.0);
  const Eigen::Vector2d proj = a + t * ab;
  return (p - proj).squaredNorm() <= eps * eps;
}

bool point_in_polygon(const Eigen::Vector2d& p, const Eigen::MatrixXd& poly) {
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

  int winding_number = 0;
  for (int i = 0; i < poly.rows(); ++i) {
    const int j = (i + 1) % poly.rows();
    const Eigen::Vector2d a = poly.row(i).head<2>().transpose();
    const Eigen::Vector2d b = poly.row(j).head<2>().transpose();

    // Treat boundary points as outside to avoid false-positive inclusions.
    if (point_on_segment_2d(p, a, b, eps)) {
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

std::vector<int> convex_hull_indices(const std::vector<Eigen::Vector2d>& pts) {
  if (pts.size() < 3) {
    return {};
  }

  std::vector<Point2D> p;
  p.reserve(pts.size());
  for (size_t i = 0; i < pts.size(); ++i) {
    p.push_back({pts[i].x(), pts[i].y(), static_cast<int>(i)});
  }

  std::sort(p.begin(), p.end(), [](const Point2D& a, const Point2D& b) {
    if (a.x != b.x) return a.x < b.x;
    return a.y < b.y;
  });

  std::vector<Point2D> H;
  H.reserve(p.size() * 2);

  for (const auto& pt : p) {
    while (H.size() >= 2 && cross(H[H.size() - 2], H[H.size() - 1], pt) <= 0.0) {
      H.pop_back();
    }
    H.push_back(pt);
  }

  size_t lower_size = H.size();
  for (int i = static_cast<int>(p.size()) - 2; i >= 0; --i) {
    const auto& pt = p[static_cast<size_t>(i)];
    while (H.size() > lower_size && cross(H[H.size() - 2], H[H.size() - 1], pt) <= 0.0) {
      H.pop_back();
    }
    H.push_back(pt);
  }

  if (H.size() < 4) {
    return {};
  }

  H.pop_back();

  std::vector<int> hull;
  hull.reserve(H.size());
  for (const auto& pt : H) {
    hull.push_back(pt.idx);
  }
  return hull;
}

Eigen::MatrixXd compute_face_centers(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
  Eigen::MatrixXd centers(F.rows(), 3);
  for (int f = 0; f < F.rows(); ++f) {
    Eigen::Vector3d c = Eigen::Vector3d::Zero();
    for (int k = 0; k < F.cols(); ++k) {
      c += V.row(F(f, k));
    }
    c /= static_cast<double>(F.cols());
    centers.row(f) = c.transpose();
  }
  return centers;
}

Eigen::MatrixXd compute_face_centers_uv(const Eigen::MatrixXd& UV, const Eigen::MatrixXi& F) {
  Eigen::MatrixXd centers(F.rows(), 2);
  for (int f = 0; f < F.rows(); ++f) {
    Eigen::Vector2d c = Eigen::Vector2d::Zero();
    for (int k = 0; k < F.cols(); ++k) {
      c += UV.row(F(f, k)).head<2>();
    }
    c /= static_cast<double>(F.cols());
    centers.row(f) = c.transpose();
  }
  return centers;
}

void clear_region_visual_handles(PatternRegionState& region) {
  region.face_centers = nullptr;
  region.selected_samples_3d = nullptr;
  region.selected_samples_uv = nullptr;
  region.selected_samples_3d_class_1 = nullptr;
  region.selected_samples_uv_class_1 = nullptr;
  region.output_pattern_3d = nullptr;
  region.output_pattern_uv = nullptr;
  region.output_pattern_3d_class_1 = nullptr;
  region.output_pattern_uv_class_1 = nullptr;
  region.input_reference_3d = nullptr;
  region.input_reference_uv = nullptr;
  region.input_boundary_curve_3d = nullptr;
  region.input_boundary_curve = nullptr;
  region.output_boundary_curve_3d = nullptr;
  region.output_boundary_curve_uv = nullptr;
  region.output_boundary_surface = nullptr;
  region.output_boundary_uv = nullptr;
  region.input_boundary_surface = nullptr;
  region.input_boundary_uv_faces = nullptr;
}

void clear_region_output_state(PatternRegionState& region) {
  region.output_pattern_sample_indices.clear();
  region.output_pattern_points_3d.clear();
  region.output_pattern_points_uv.clear();
  region.output_pattern_class_ids.clear();
  region.output_voronoi_pcf_hist_counts.clear();
  region.output_voronoi_pcf_hist_plot.clear();
  for (auto& hist : region.two_class_output_voronoi_pcf_hist_counts) {
    hist.clear();
  }
  for (auto& plot : region.two_class_output_voronoi_pcf_hist_plot) {
    plot.clear();
  }
  region.two_class_output_counts = {0, 0};
  region.two_class_output_voronoi_pcf_pair_count = {0, 0, 0};
  region.output_voronoi_pcf_max_k = 0;
  region.output_voronoi_pcf_pair_count = 0;
  region.output_voronoi_pcf_ready = false;
  region.output_voronoi_pcf_energy = 0.0;
  region.output_voronoi_objective_energy = 0.0;
  region.optimizer_improvements = 0;
  region.optimizer_iterations_ran = 0;
  region.output_pattern_dirty = false;
  region.output_support_denominator_cache_valid = false;
  region.output_support_denominator_cache_bin_count = 0;
  region.output_support_denominator_cache_triangle_count = -1;
  region.output_support_denominator_cache_boundary_uv.resize(0, 0);
  region.output_support_uv_cache.clear();
  region.output_support_tri_indices_cache.clear();
  region.output_support_k_denominator_cache.clear();
  region.output_support_pairwise_cache_valid = false;
  region.output_support_pairwise_distances.clear();
  region.output_boundary_uv_poly.resize(0, 2);
  region.output_boundary_3d_poly.resize(0, 3);
  region.output_boundary_preview_uv_poly.resize(0, 2);
  region.output_boundary_preview_3d_poly.resize(0, 3);
  region.last_painted_face = -1;
  region.output_boundary_pending_edits = false;
}

PatternRegionState clone_patch_region_template(const PatternRegionState& source) {
  PatternRegionState clone = source;
  clear_region_visual_handles(clone);
  clear_region_output_state(clone);
  clone.region_id = -1;
  clone.region_mode = static_cast<int>(PatternRegionMode::Exemplar);
  clone.transition_source_a_region_id = -1;
  clone.transition_source_b_region_id = -1;
  clone.generated_patch_family_id = -1;
  clone.generated_patch_source_region_id = -1;
  clone.generated_patch_index = -1;
  clone.generated_patch_support_gap_steps = 0;
  clone.generated_patch_batch_optimize_requested = false;
  clone.enable_input_selection = false;
  clone.enable_input_paint = false;
  clone.enable_output_paint = false;
  clone.selected_dirty = false;
  clone.input_boundary_dirty = false;
  clone.output_boundary_dirty = false;
  clone.output_boundary_pending_edits = false;
  clone.painted_face_indices.clear();
  clone.generated_patch_interface_segment_uv_starts.clear();
  clone.generated_patch_interface_segment_uv_ends.clear();
  clone.last_painted_input_face = -1;
  clone.plot_export_status.clear();
  clone.plot_export_status_is_error = false;
  return clone;
}

struct FacePatchBlock {
  std::vector<int> face_indices;
};

struct PatchInterfaceSegments {
  std::vector<Eigen::Vector2d> segment_uv_starts;
  std::vector<Eigen::Vector2d> segment_uv_ends;
};

struct PatchFrontierEntry {
  double priority_distance = 0.0;
  double path_distance = 0.0;
  double seed_distance = 0.0;
  int face_index = -1;
};

struct PatchFrontierEntryGreater {
  bool operator()(const PatchFrontierEntry& lhs, const PatchFrontierEntry& rhs) const {
    if (lhs.priority_distance != rhs.priority_distance) {
      return lhs.priority_distance > rhs.priority_distance;
    }
    if (lhs.seed_distance != rhs.seed_distance) {
      return lhs.seed_distance > rhs.seed_distance;
    }
    if (lhs.path_distance != rhs.path_distance) {
      return lhs.path_distance > rhs.path_distance;
    }
    return lhs.face_index > rhs.face_index;
  }
};

struct PatchGrowthState {
  std::vector<int> face_indices;
  double assigned_weight = 0.0;
  int seed_face_index = -1;
  Eigen::Vector2d seed_uv = Eigen::Vector2d::Zero();
  std::priority_queue<
    PatchFrontierEntry,
    std::vector<PatchFrontierEntry>,
    PatchFrontierEntryGreater> frontier;
};

double face_patch_weight(
  const Eigen::VectorXd& face_weights,
  int face_index) {
  if (face_index < 0 || face_index >= face_weights.size()) {
    return 1.0;
  }
  return std::max(0.0, face_weights(face_index));
}

Eigen::Vector2d face_center_uv_at(
  const Eigen::MatrixXd& face_centers_uv,
  int face_index) {
  if (face_index < 0 || face_index >= face_centers_uv.rows() || face_centers_uv.cols() < 2) {
    return Eigen::Vector2d::Zero();
  }
  return face_centers_uv.row(face_index).head<2>().transpose();
}

double triangle_area_uv(
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F,
  int face_index) {
  if (face_index < 0 || face_index >= F.rows() || F.cols() < 3 || UV.rows() <= 0 || UV.cols() < 2) {
    return 0.0;
  }

  const int v0 = F(face_index, 0);
  const int v1 = F(face_index, 1);
  const int v2 = F(face_index, 2);
  if (v0 < 0 || v1 < 0 || v2 < 0 ||
      v0 >= UV.rows() || v1 >= UV.rows() || v2 >= UV.rows()) {
    return 0.0;
  }

  const Eigen::Vector2d u0 = UV.row(v0).head<2>().transpose();
  const Eigen::Vector2d u1 = UV.row(v1).head<2>().transpose();
  const Eigen::Vector2d u2 = UV.row(v2).head<2>().transpose();
  const double area2 =
    (u1.x() - u0.x()) * (u2.y() - u0.y()) -
    (u1.y() - u0.y()) * (u2.x() - u0.x());
  return 0.5 * std::abs(area2);
}

Eigen::VectorXd compute_face_areas_uv(
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F) {
  Eigen::VectorXd face_areas = Eigen::VectorXd::Zero(F.rows());
  for (int face_index = 0; face_index < F.rows(); ++face_index) {
    face_areas(face_index) = triangle_area_uv(UV, F, face_index);
  }
  return face_areas;
}

Eigen::Vector2d face_patch_block_centroid(
  const FacePatchBlock& block,
  const Eigen::MatrixXd& face_centers_uv) {
  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  if (block.face_indices.empty()) {
    return centroid;
  }
  for (int face_index : block.face_indices) {
    centroid += face_center_uv_at(face_centers_uv, face_index);
  }
  centroid /= static_cast<double>(block.face_indices.size());
  return centroid;
}

std::vector<std::vector<int>> build_face_adjacency(
  const Eigen::MatrixXi& F) {
  const int face_count = std::max(0, static_cast<int>(F.rows()));
  std::vector<std::vector<int>> adjacency(static_cast<size_t>(face_count));
  if (face_count <= 0 || F.cols() < 3) {
    return adjacency;
  }

  std::unordered_map<std::uint64_t, std::vector<int>> edge_to_faces;
  edge_to_faces.reserve(static_cast<size_t>(F.rows()) * 3);
  for (int face_index = 0; face_index < F.rows(); ++face_index) {
    for (int edge_index = 0; edge_index < 3; ++edge_index) {
      const int v0 = F(face_index, edge_index);
      const int v1 = F(face_index, (edge_index + 1) % 3);
      if (v0 < 0 || v1 < 0) {
        continue;
      }
      edge_to_faces[undirected_edge_key(v0, v1)].push_back(face_index);
    }
  }

  for (const auto& entry : edge_to_faces) {
    const std::vector<int>& edge_faces = entry.second;
    for (size_t i = 0; i < edge_faces.size(); ++i) {
      for (size_t j = i + 1; j < edge_faces.size(); ++j) {
        const int face_a = edge_faces[i];
        const int face_b = edge_faces[j];
        if (face_a < 0 || face_b < 0 || face_a == face_b) {
          continue;
        }
        adjacency[static_cast<size_t>(face_a)].push_back(face_b);
        adjacency[static_cast<size_t>(face_b)].push_back(face_a);
      }
    }
  }

  for (std::vector<int>& neighbors : adjacency) {
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
  }
  return adjacency;
}

std::vector<PatchInterfaceSegments> build_patch_interface_segments_uv(
  const std::vector<std::vector<int>>& face_partitions,
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F) {
  std::vector<PatchInterfaceSegments> patch_segments(face_partitions.size());
  if (face_partitions.empty() || F.rows() <= 0 || F.cols() < 3 || UV.rows() <= 0 || UV.cols() < 2) {
    return patch_segments;
  }

  std::vector<int> face_owner(static_cast<size_t>(F.rows()), -1);
  for (int patch_index = 0; patch_index < static_cast<int>(face_partitions.size()); ++patch_index) {
    for (int face_index : face_partitions[static_cast<size_t>(patch_index)]) {
      if (face_index >= 0 && face_index < F.rows()) {
        face_owner[static_cast<size_t>(face_index)] = patch_index;
      }
    }
  }

  std::unordered_map<std::uint64_t, std::vector<int>> edge_to_faces;
  edge_to_faces.reserve(static_cast<size_t>(F.rows()) * 3);
  for (int face_index = 0; face_index < F.rows(); ++face_index) {
    for (int edge_index = 0; edge_index < 3; ++edge_index) {
      const int v0 = F(face_index, edge_index);
      const int v1 = F(face_index, (edge_index + 1) % 3);
      if (v0 < 0 || v1 < 0) {
        continue;
      }
      edge_to_faces[undirected_edge_key(v0, v1)].push_back(face_index);
    }
  }

  for (int patch_index = 0; patch_index < static_cast<int>(face_partitions.size()); ++patch_index) {
    PatchInterfaceSegments& segments = patch_segments[static_cast<size_t>(patch_index)];
    std::unordered_set<std::uint64_t> seen_interface_edges;
    seen_interface_edges.reserve(face_partitions[static_cast<size_t>(patch_index)].size() * 2);
    for (int face_index : face_partitions[static_cast<size_t>(patch_index)]) {
      if (face_index < 0 || face_index >= F.rows()) {
        continue;
      }
      for (int edge_index = 0; edge_index < 3; ++edge_index) {
        const int v0 = F(face_index, edge_index);
        const int v1 = F(face_index, (edge_index + 1) % 3);
        if (v0 < 0 || v1 < 0 || v0 >= UV.rows() || v1 >= UV.rows()) {
          continue;
        }
        const std::uint64_t edge_key = undirected_edge_key(v0, v1);
        auto edge_it = edge_to_faces.find(edge_key);
        if (edge_it == edge_to_faces.end()) {
          continue;
        }
        bool is_patch_interface_edge = false;
        for (int other_face_index : edge_it->second) {
          if (other_face_index == face_index || other_face_index < 0 || other_face_index >= F.rows()) {
            continue;
          }
          const int other_owner = face_owner[static_cast<size_t>(other_face_index)];
          if (other_owner >= 0 && other_owner != patch_index) {
            is_patch_interface_edge = true;
            break;
          }
        }
        if (!is_patch_interface_edge || !seen_interface_edges.insert(edge_key).second) {
          continue;
        }
        segments.segment_uv_starts.push_back(UV.row(v0).head<2>().transpose());
        segments.segment_uv_ends.push_back(UV.row(v1).head<2>().transpose());
      }
    }
  }

  return patch_segments;
}

std::vector<int> choose_farthest_patch_seed_faces(
  const Eigen::MatrixXd& face_centers_uv,
  const Eigen::VectorXd& face_weights,
  int seed_count) {
  std::vector<int> seeds;
  if (seed_count <= 0 || face_centers_uv.rows() <= 0 || face_centers_uv.cols() < 2) {
    return seeds;
  }

  const int clamped_seed_count = std::min(seed_count, static_cast<int>(face_centers_uv.rows()));
  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  double centroid_weight_sum = 0.0;
  for (int face_index = 0; face_index < face_centers_uv.rows(); ++face_index) {
    const Eigen::Vector2d uv = face_center_uv_at(face_centers_uv, face_index);
    if (!std::isfinite(uv.x()) || !std::isfinite(uv.y())) {
      continue;
    }
    const double weight = face_patch_weight(face_weights, face_index);
    if (weight > 0.0) {
      centroid += weight * uv;
      centroid_weight_sum += weight;
    }
  }

  if (centroid_weight_sum > 1e-14) {
    centroid /= centroid_weight_sum;
  } else {
    int finite_count = 0;
    for (int face_index = 0; face_index < face_centers_uv.rows(); ++face_index) {
      const Eigen::Vector2d uv = face_center_uv_at(face_centers_uv, face_index);
      if (!std::isfinite(uv.x()) || !std::isfinite(uv.y())) {
        continue;
      }
      centroid += uv;
      ++finite_count;
    }
    if (finite_count <= 0) {
      return seeds;
    }
    centroid /= static_cast<double>(finite_count);
  }

  int first_seed = -1;
  double first_seed_distance = -std::numeric_limits<double>::infinity();
  double first_seed_weight = -std::numeric_limits<double>::infinity();
  for (int face_index = 0; face_index < face_centers_uv.rows(); ++face_index) {
    const Eigen::Vector2d uv = face_center_uv_at(face_centers_uv, face_index);
    const double distance = (uv - centroid).squaredNorm();
    const double weight = face_patch_weight(face_weights, face_index);
    if (distance > first_seed_distance + 1e-12) {
      first_seed = face_index;
      first_seed_distance = distance;
      first_seed_weight = weight;
      continue;
    }
    if (std::abs(distance - first_seed_distance) <= 1e-12 &&
        weight > first_seed_weight + 1e-12) {
      first_seed = face_index;
      first_seed_weight = weight;
    }
  }

  if (first_seed < 0) {
    return seeds;
  }

  std::vector<char> is_seed(static_cast<size_t>(face_centers_uv.rows()), 0);
  std::vector<double> min_seed_squared_distance(
    static_cast<size_t>(face_centers_uv.rows()),
    std::numeric_limits<double>::infinity());
  const auto add_seed = [&](int seed_face_index) {
    if (seed_face_index < 0 || seed_face_index >= face_centers_uv.rows() ||
        is_seed[static_cast<size_t>(seed_face_index)] != 0) {
      return;
    }
    seeds.push_back(seed_face_index);
    is_seed[static_cast<size_t>(seed_face_index)] = 1;
    const Eigen::Vector2d seed_uv = face_center_uv_at(face_centers_uv, seed_face_index);
    for (int face_index = 0; face_index < face_centers_uv.rows(); ++face_index) {
      const Eigen::Vector2d uv = face_center_uv_at(face_centers_uv, face_index);
      const double distance = (uv - seed_uv).squaredNorm();
      double& best_distance = min_seed_squared_distance[static_cast<size_t>(face_index)];
      best_distance = std::min(best_distance, distance);
    }
  };

  add_seed(first_seed);
  while (static_cast<int>(seeds.size()) < clamped_seed_count) {
    int next_seed = -1;
    double next_seed_distance = -std::numeric_limits<double>::infinity();
    double next_seed_weight = -std::numeric_limits<double>::infinity();
    for (int face_index = 0; face_index < face_centers_uv.rows(); ++face_index) {
      if (is_seed[static_cast<size_t>(face_index)] != 0) {
        continue;
      }
      const double distance =
        min_seed_squared_distance[static_cast<size_t>(face_index)];
      const double weight = face_patch_weight(face_weights, face_index);
      if (distance > next_seed_distance + 1e-12) {
        next_seed = face_index;
        next_seed_distance = distance;
        next_seed_weight = weight;
        continue;
      }
      if (std::abs(distance - next_seed_distance) <= 1e-12 &&
          weight > next_seed_weight + 1e-12) {
        next_seed = face_index;
        next_seed_weight = weight;
      }
    }
    if (next_seed < 0) {
      break;
    }
    add_seed(next_seed);
  }

  return seeds;
}

void apply_face_partition_gap_rings(
  const std::vector<std::vector<int>>& face_adjacency,
  int gap_ring_count,
  std::vector<FacePatchBlock>& blocks) {
  if (gap_ring_count <= 0 || blocks.empty()) {
    return;
  }

  const int face_count = static_cast<int>(face_adjacency.size());
  std::vector<int> face_owner(static_cast<size_t>(face_count), -1);
  for (int patch_index = 0; patch_index < static_cast<int>(blocks.size()); ++patch_index) {
    for (int face_index : blocks[static_cast<size_t>(patch_index)].face_indices) {
      if (face_index >= 0 && face_index < face_count) {
        face_owner[static_cast<size_t>(face_index)] = patch_index;
      }
    }
  }

  for (int patch_index = 0; patch_index < static_cast<int>(blocks.size()); ++patch_index) {
    FacePatchBlock& block = blocks[static_cast<size_t>(patch_index)];
    std::vector<int> depth(static_cast<size_t>(face_count), -1);
    std::queue<int> boundary_queue;
    for (int face_index : block.face_indices) {
      if (face_index < 0 || face_index >= face_count) {
        continue;
      }
      bool is_interface_face = false;
      for (int neighbor_face : face_adjacency[static_cast<size_t>(face_index)]) {
        if (neighbor_face < 0 || neighbor_face >= face_count) {
          continue;
        }
        const int neighbor_owner = face_owner[static_cast<size_t>(neighbor_face)];
        if (neighbor_owner >= 0 && neighbor_owner != patch_index) {
          is_interface_face = true;
          break;
        }
      }
      if (!is_interface_face) {
        continue;
      }
      depth[static_cast<size_t>(face_index)] = 0;
      boundary_queue.push(face_index);
    }

    while (!boundary_queue.empty()) {
      const int face_index = boundary_queue.front();
      boundary_queue.pop();
      const int face_depth = depth[static_cast<size_t>(face_index)];
      if (face_depth + 1 >= gap_ring_count) {
        continue;
      }
      for (int neighbor_face : face_adjacency[static_cast<size_t>(face_index)]) {
        if (neighbor_face < 0 || neighbor_face >= face_count) {
          continue;
        }
        if (face_owner[static_cast<size_t>(neighbor_face)] != patch_index) {
          continue;
        }
        int& neighbor_depth = depth[static_cast<size_t>(neighbor_face)];
        if (neighbor_depth >= 0) {
          continue;
        }
        neighbor_depth = face_depth + 1;
        boundary_queue.push(neighbor_face);
      }
    }

    std::vector<int> kept_faces;
    kept_faces.reserve(block.face_indices.size());
    for (int face_index : block.face_indices) {
      if (face_index < 0 || face_index >= face_count) {
        continue;
      }
      if (depth[static_cast<size_t>(face_index)] >= 0 &&
          depth[static_cast<size_t>(face_index)] < gap_ring_count) {
        continue;
      }
      kept_faces.push_back(face_index);
    }
    block.face_indices = std::move(kept_faces);
  }
}

std::vector<std::vector<int>> partition_face_indices_into_patches(
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& face_centers_uv,
  const Eigen::VectorXd& face_weights,
  int patch_count) {
  std::vector<std::vector<int>> partitions;
  if (patch_count <= 0 || face_centers_uv.rows() <= 0 || face_centers_uv.cols() < 2) {
    return partitions;
  }

  const int face_count = static_cast<int>(face_centers_uv.rows());
  const int clamped_patch_count = std::min(patch_count, face_count);
  const std::vector<std::vector<int>> face_adjacency = build_face_adjacency(F);
  std::vector<int> seed_faces = choose_farthest_patch_seed_faces(
    face_centers_uv,
    face_weights,
    clamped_patch_count);
  if (seed_faces.empty()) {
    return partitions;
  }

  std::vector<PatchGrowthState> patch_states(seed_faces.size());
  for (int patch_index = 0; patch_index < static_cast<int>(seed_faces.size()); ++patch_index) {
    PatchGrowthState& patch_state = patch_states[static_cast<size_t>(patch_index)];
    patch_state.seed_face_index = seed_faces[static_cast<size_t>(patch_index)];
    patch_state.seed_uv = face_center_uv_at(
      face_centers_uv,
      seed_faces[static_cast<size_t>(patch_index)]);
  }
  std::vector<int> face_owner(static_cast<size_t>(face_count), -1);
  std::vector<std::vector<double>> best_frontier_distance(
    seed_faces.size(),
    std::vector<double>(static_cast<size_t>(face_count), std::numeric_limits<double>::infinity()));
  const auto push_patch_neighbors = [&](int patch_index, int face_index, double base_distance) {
    if (patch_index < 0 || patch_index >= static_cast<int>(patch_states.size()) ||
        face_index < 0 || face_index >= face_count) {
      return;
    }
    for (int neighbor_face : face_adjacency[static_cast<size_t>(face_index)]) {
      if (neighbor_face < 0 || neighbor_face >= face_count) {
        continue;
      }
      if (face_owner[static_cast<size_t>(neighbor_face)] >= 0) {
        continue;
      }
      const double edge_distance = std::max(
        1e-8,
        (face_center_uv_at(face_centers_uv, face_index) -
         face_center_uv_at(face_centers_uv, neighbor_face)).norm());
      const double new_distance = base_distance + edge_distance;
      double& best_distance =
        best_frontier_distance[static_cast<size_t>(patch_index)][static_cast<size_t>(neighbor_face)];
      if (new_distance + 1e-12 >= best_distance) {
        continue;
      }
      best_distance = new_distance;
      const Eigen::Vector2d neighbor_uv = face_center_uv_at(face_centers_uv, neighbor_face);
      const double seed_distance =
        (neighbor_uv - patch_states[static_cast<size_t>(patch_index)].seed_uv).norm();
      const double detour_distance = std::max(0.0, new_distance - seed_distance);
      const double priority_distance =
        new_distance +
        kPatchSeedCompactnessWeight * seed_distance +
        kPatchDetourCompactnessWeight * detour_distance;
      patch_states[static_cast<size_t>(patch_index)].frontier.push(
        PatchFrontierEntry{priority_distance, new_distance, seed_distance, neighbor_face});
    }
  };

  const auto claim_face = [&](int patch_index, int face_index, double path_distance) {
    if (patch_index < 0 || patch_index >= static_cast<int>(patch_states.size()) ||
        face_index < 0 || face_index >= face_count ||
        face_owner[static_cast<size_t>(face_index)] >= 0) {
      return false;
    }
    face_owner[static_cast<size_t>(face_index)] = patch_index;
    PatchGrowthState& patch_state = patch_states[static_cast<size_t>(patch_index)];
    patch_state.face_indices.push_back(face_index);
    patch_state.assigned_weight += face_patch_weight(face_weights, face_index);
    push_patch_neighbors(patch_index, face_index, path_distance);
    return true;
  };

  int assigned_face_count = 0;
  for (int patch_index = 0; patch_index < static_cast<int>(seed_faces.size()); ++patch_index) {
    if (claim_face(patch_index, seed_faces[static_cast<size_t>(patch_index)], 0.0)) {
      ++assigned_face_count;
    }
  }

  double total_weight = 0.0;
  for (int face_index = 0; face_index < face_count; ++face_index) {
    total_weight += face_patch_weight(face_weights, face_index);
  }
  const bool use_weight_balance = total_weight > 1e-14;
  const double target_patch_measure =
    use_weight_balance
      ? (total_weight / static_cast<double>(std::max<size_t>(1, seed_faces.size())))
      : (static_cast<double>(face_count) /
         static_cast<double>(std::max<size_t>(1, seed_faces.size())));
  const auto patch_measure = [&](const PatchGrowthState& patch_state) {
    return use_weight_balance
      ? patch_state.assigned_weight
      : static_cast<double>(patch_state.face_indices.size());
  };

  while (assigned_face_count < face_count) {
    int selected_patch = -1;
    double selected_fill_ratio = std::numeric_limits<double>::infinity();
    double selected_frontier_priority = std::numeric_limits<double>::infinity();
    double selected_frontier_seed_distance = std::numeric_limits<double>::infinity();
    double selected_frontier_path_distance = std::numeric_limits<double>::infinity();
    double selected_patch_measure = std::numeric_limits<double>::infinity();
    for (int patch_index = 0; patch_index < static_cast<int>(patch_states.size()); ++patch_index) {
      PatchGrowthState& patch_state = patch_states[static_cast<size_t>(patch_index)];
      while (!patch_state.frontier.empty()) {
        const PatchFrontierEntry& entry = patch_state.frontier.top();
        if (entry.face_index >= 0 && entry.face_index < face_count &&
            face_owner[static_cast<size_t>(entry.face_index)] < 0) {
          break;
        }
        patch_state.frontier.pop();
      }
      if (patch_state.frontier.empty()) {
        continue;
      }

      const double current_patch_measure = patch_measure(patch_state);
      const double fill_ratio =
        (target_patch_measure > 1e-14)
          ? (current_patch_measure / target_patch_measure)
          : current_patch_measure;
      const PatchFrontierEntry& frontier_entry = patch_state.frontier.top();
      const double frontier_priority = frontier_entry.priority_distance;
      const double frontier_seed_distance = frontier_entry.seed_distance;
      const double frontier_path_distance = frontier_entry.path_distance;
      if (fill_ratio < selected_fill_ratio - 1e-12 ||
          (std::abs(fill_ratio - selected_fill_ratio) <= 1e-12 &&
           frontier_priority < selected_frontier_priority - 1e-12) ||
          (std::abs(fill_ratio - selected_fill_ratio) <= 1e-12 &&
           std::abs(frontier_priority - selected_frontier_priority) <= 1e-12 &&
           frontier_seed_distance < selected_frontier_seed_distance - 1e-12) ||
          (std::abs(fill_ratio - selected_fill_ratio) <= 1e-12 &&
           std::abs(frontier_priority - selected_frontier_priority) <= 1e-12 &&
           std::abs(frontier_seed_distance - selected_frontier_seed_distance) <= 1e-12 &&
           frontier_path_distance < selected_frontier_path_distance - 1e-12) ||
          (std::abs(fill_ratio - selected_fill_ratio) <= 1e-12 &&
           std::abs(frontier_priority - selected_frontier_priority) <= 1e-12 &&
           std::abs(frontier_seed_distance - selected_frontier_seed_distance) <= 1e-12 &&
           std::abs(frontier_path_distance - selected_frontier_path_distance) <= 1e-12 &&
           current_patch_measure < selected_patch_measure - 1e-12) ||
          (std::abs(fill_ratio - selected_fill_ratio) <= 1e-12 &&
           std::abs(frontier_priority - selected_frontier_priority) <= 1e-12 &&
           std::abs(frontier_seed_distance - selected_frontier_seed_distance) <= 1e-12 &&
           std::abs(frontier_path_distance - selected_frontier_path_distance) <= 1e-12 &&
           std::abs(current_patch_measure - selected_patch_measure) <= 1e-12 &&
           patch_index < selected_patch)) {
        selected_patch = patch_index;
        selected_fill_ratio = fill_ratio;
        selected_frontier_priority = frontier_priority;
        selected_frontier_seed_distance = frontier_seed_distance;
        selected_frontier_path_distance = frontier_path_distance;
        selected_patch_measure = current_patch_measure;
      }
    }

    if (selected_patch < 0) {
      selected_patch = 0;
      selected_fill_ratio = std::numeric_limits<double>::infinity();
      selected_patch_measure = std::numeric_limits<double>::infinity();
      for (int patch_index = 0; patch_index < static_cast<int>(patch_states.size()); ++patch_index) {
        const double current_patch_measure =
          patch_measure(patch_states[static_cast<size_t>(patch_index)]);
        const double fill_ratio =
          (target_patch_measure > 1e-14)
            ? (current_patch_measure / target_patch_measure)
            : current_patch_measure;
        if (fill_ratio < selected_fill_ratio - 1e-12 ||
            (std::abs(fill_ratio - selected_fill_ratio) <= 1e-12 &&
             current_patch_measure < selected_patch_measure - 1e-12) ||
            (std::abs(fill_ratio - selected_fill_ratio) <= 1e-12 &&
             std::abs(current_patch_measure - selected_patch_measure) <= 1e-12 &&
             patch_index < selected_patch)) {
          selected_patch = patch_index;
          selected_fill_ratio = fill_ratio;
          selected_patch_measure = current_patch_measure;
        }
      }

      int fallback_face = -1;
      double fallback_distance = std::numeric_limits<double>::infinity();
      const Eigen::Vector2d fallback_seed_uv =
        patch_states[static_cast<size_t>(selected_patch)].seed_uv;
      for (int face_index = 0; face_index < face_count; ++face_index) {
        if (face_owner[static_cast<size_t>(face_index)] >= 0) {
          continue;
        }
        const double distance =
          (face_center_uv_at(face_centers_uv, face_index) - fallback_seed_uv).squaredNorm();
        if (distance < fallback_distance - 1e-12) {
          fallback_face = face_index;
          fallback_distance = distance;
        }
      }
      if (fallback_face < 0) {
        break;
      }

      if (claim_face(selected_patch, fallback_face, 0.0)) {
        ++assigned_face_count;
      }
      continue;
    }

    PatchGrowthState& patch_state = patch_states[static_cast<size_t>(selected_patch)];
    const PatchFrontierEntry entry = patch_state.frontier.top();
    patch_state.frontier.pop();
    if (claim_face(selected_patch, entry.face_index, entry.path_distance)) {
      ++assigned_face_count;
    }
  }

  std::vector<FacePatchBlock> blocks;
  blocks.reserve(patch_states.size());
  for (PatchGrowthState& patch_state : patch_states) {
    if (patch_state.face_indices.empty()) {
      continue;
    }
    std::sort(patch_state.face_indices.begin(), patch_state.face_indices.end());
    blocks.push_back(FacePatchBlock{std::move(patch_state.face_indices)});
  }

  std::sort(blocks.begin(), blocks.end(), [&](const FacePatchBlock& lhs, const FacePatchBlock& rhs) {
    const Eigen::Vector2d lhs_centroid = face_patch_block_centroid(lhs, face_centers_uv);
    const Eigen::Vector2d rhs_centroid = face_patch_block_centroid(rhs, face_centers_uv);
    if (lhs_centroid.x() != rhs_centroid.x()) {
      return lhs_centroid.x() < rhs_centroid.x();
    }
    return lhs_centroid.y() < rhs_centroid.y();
  });

  partitions.reserve(blocks.size());
  for (FacePatchBlock& block : blocks) {
    if (!block.face_indices.empty()) {
      partitions.push_back(std::move(block.face_indices));
    }
  }
  return partitions;
}

void update_whole_model_patch_preview(
  InteractionState& root_state,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F,
  const std::vector<std::vector<int>>& face_partitions) {
  const int old_preview_count =
    static_cast<int>(root_state.whole_model_patch_preview_face_partitions.size());
  const int new_preview_count = static_cast<int>(face_partitions.size());

  for (int patch_index = new_preview_count; patch_index < old_preview_count; ++patch_index) {
    const std::string preview_curve_3d_name =
      whole_model_patch_preview_name(kWholeModelPatchPreviewCurve3DBaseName, patch_index);
    const std::string preview_curve_uv_name =
      whole_model_patch_preview_name(kWholeModelPatchPreviewCurveUVBaseName, patch_index);
    if (polyscope::hasCurveNetwork(preview_curve_3d_name)) {
      polyscope::removeCurveNetwork(preview_curve_3d_name);
    }
    if (polyscope::hasCurveNetwork(preview_curve_uv_name)) {
      polyscope::removeCurveNetwork(preview_curve_uv_name);
    }
  }

  root_state.whole_model_patch_preview_face_partitions = face_partitions;
  root_state.whole_model_patch_preview_active = !face_partitions.empty();

  if (face_partitions.empty()) {
    if (surfaceMesh) {
      surfaceMesh->removeQuantity(kWholeModelPatchPreviewFaces3DName, false);
    }
    if (uvMesh) {
      uvMesh->removeQuantity(kWholeModelPatchPreviewFacesUVName, false);
    }
    root_state.whole_model_patch_preview_surface = nullptr;
    root_state.whole_model_patch_preview_uv = nullptr;
    return;
  }

  std::vector<glm::vec3> preview_face_colors(
    static_cast<size_t>(F.rows()),
    kWholeModelPatchPreviewGapColor);
  for (int patch_index = 0; patch_index < new_preview_count; ++patch_index) {
    const glm::vec3 patch_color = whole_model_patch_preview_color(patch_index);
    for (int face_index : face_partitions[static_cast<size_t>(patch_index)]) {
      if (face_index >= 0 && face_index < F.rows()) {
        preview_face_colors[static_cast<size_t>(face_index)] = patch_color;
      }
    }
  }

  if (surfaceMesh) {
    if (!root_state.whole_model_patch_preview_surface) {
      root_state.whole_model_patch_preview_surface =
        surfaceMesh->addFaceColorQuantity(kWholeModelPatchPreviewFaces3DName, preview_face_colors);
    } else {
      root_state.whole_model_patch_preview_surface->updateData(preview_face_colors);
    }
    if (root_state.whole_model_patch_preview_surface) {
      root_state.whole_model_patch_preview_surface->setEnabled(surfaceMesh->isEnabled());
    }
  }

  if (uvMesh) {
    if (!root_state.whole_model_patch_preview_uv) {
      root_state.whole_model_patch_preview_uv =
        uvMesh->addFaceColorQuantity(kWholeModelPatchPreviewFacesUVName, preview_face_colors);
    } else {
      root_state.whole_model_patch_preview_uv->updateData(preview_face_colors);
    }
    if (root_state.whole_model_patch_preview_uv) {
      root_state.whole_model_patch_preview_uv->setEnabled(uvMesh->isEnabled());
    }
  }

  for (int patch_index = 0; patch_index < new_preview_count; ++patch_index) {
    Eigen::MatrixXd boundary_3d;
    Eigen::MatrixXd boundary_uv;
    build_boundary_from_painted_faces(
      face_partitions[static_cast<size_t>(patch_index)],
      V,
      UV,
      F,
      boundary_3d,
      boundary_uv);

    const std::string preview_curve_3d_name =
      whole_model_patch_preview_name(kWholeModelPatchPreviewCurve3DBaseName, patch_index);
    const std::string preview_curve_uv_name =
      whole_model_patch_preview_name(kWholeModelPatchPreviewCurveUVBaseName, patch_index);
    const glm::vec3 patch_color = whole_model_patch_preview_color(patch_index);

    if (boundary_3d.rows() < 3) {
      if (polyscope::hasCurveNetwork(preview_curve_3d_name)) {
        polyscope::removeCurveNetwork(preview_curve_3d_name);
      }
    } else {
      polyscope::CurveNetwork* preview_curve_3d = nullptr;
      ensure_curve_network_loop_with_preserved_radius(
        preview_curve_3d,
        preview_curve_3d_name,
        boundary_3d,
        viewer_overlay_radius(surfaceMesh, 0.0045));
      if (preview_curve_3d) {
        preview_curve_3d->setColor(patch_color);
        preview_curve_3d->setEnabled(surfaceMesh && surfaceMesh->isEnabled());
      }
    }

    if (boundary_uv.rows() < 3) {
      if (polyscope::hasCurveNetwork(preview_curve_uv_name)) {
        polyscope::removeCurveNetwork(preview_curve_uv_name);
      }
    } else {
      const Eigen::MatrixXd display_uv =
        uv_matrix_to_display_2d(boundary_uv, Eigen::Vector2d::Zero());
      polyscope::CurveNetwork* preview_curve_uv = nullptr;
      ensure_curve_network_loop_2d_with_preserved_radius(
        preview_curve_uv,
        preview_curve_uv_name,
        display_uv,
        viewer_overlay_radius(uvMesh, 0.0045));
      if (preview_curve_uv) {
        preview_curve_uv->setColor(patch_color);
        preview_curve_uv->setEnabled(uvMesh && uvMesh->isEnabled());
      }
    }
  }
}

void build_boundary_from_painted_faces(
  const std::vector<int>& painted_face_indices,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F,
  Eigen::MatrixXd& out_boundary_3d,
  Eigen::MatrixXd& out_boundary_uv) {

  out_boundary_3d.resize(0, 3);
  out_boundary_uv.resize(0, 2);
  if (painted_face_indices.empty() || F.rows() <= 0 || F.cols() < 2 || UV.cols() < 2 || V.cols() < 3) {
    return;
  }

  std::vector<char> is_selected(static_cast<size_t>(F.rows()), 0);
  std::vector<int> selected_faces;
  selected_faces.reserve(painted_face_indices.size());
  for (int face_index : painted_face_indices) {
    if (face_index < 0 || face_index >= F.rows()) {
      continue;
    }
    if (is_selected[static_cast<size_t>(face_index)] != 0) {
      continue;
    }
    is_selected[static_cast<size_t>(face_index)] = 1;
    selected_faces.push_back(face_index);
  }
  if (selected_faces.empty()) {
    return;
  }

  std::unordered_map<std::uint64_t, int> edge_use_count;
  edge_use_count.reserve(selected_faces.size() * static_cast<size_t>(F.cols()));
  for (int face_index : selected_faces) {
    for (int edge_offset = 0; edge_offset < F.cols(); ++edge_offset) {
      const int a = F(face_index, edge_offset);
      const int b = F(face_index, (edge_offset + 1) % F.cols());
      ++edge_use_count[undirected_edge_key(a, b)];
    }
  }

  struct BoundaryEdge {
    int start_vertex = -1;
    int end_vertex = -1;
  };

  std::vector<BoundaryEdge> boundary_edges;
  boundary_edges.reserve(selected_faces.size() * static_cast<size_t>(F.cols()));
  for (int face_index : selected_faces) {
    for (int edge_offset = 0; edge_offset < F.cols(); ++edge_offset) {
      const int a = F(face_index, edge_offset);
      const int b = F(face_index, (edge_offset + 1) % F.cols());
      const auto count_it = edge_use_count.find(undirected_edge_key(a, b));
      if (count_it != edge_use_count.end() && count_it->second == 1) {
        boundary_edges.push_back({a, b});
      }
    }
  }
  if (boundary_edges.size() < 3) {
    return;
  }

  std::unordered_map<int, std::vector<int>> boundary_edge_indices_by_start;
  boundary_edge_indices_by_start.reserve(boundary_edges.size());
  for (int edge_index = 0; edge_index < static_cast<int>(boundary_edges.size()); ++edge_index) {
    boundary_edge_indices_by_start[boundary_edges[static_cast<size_t>(edge_index)].start_vertex].push_back(edge_index);
  }

  std::vector<char> boundary_edge_used(boundary_edges.size(), 0);
  std::vector<int> best_loop_vertices;
  double best_loop_area = -1.0;

  const auto loop_area_abs = [&](const std::vector<int>& loop_vertices) {
    if (loop_vertices.size() < 3) {
      return 0.0;
    }
    double signed_area = 0.0;
    for (size_t i = 0; i < loop_vertices.size(); ++i) {
      const int curr_vertex = loop_vertices[i];
      const int next_vertex = loop_vertices[(i + 1) % loop_vertices.size()];
      if (curr_vertex < 0 || curr_vertex >= UV.rows() || next_vertex < 0 || next_vertex >= UV.rows()) {
        continue;
      }
      const Eigen::Vector2d curr_uv = UV.row(curr_vertex).head<2>().transpose();
      const Eigen::Vector2d next_uv = UV.row(next_vertex).head<2>().transpose();
      signed_area += curr_uv.x() * next_uv.y() - next_uv.x() * curr_uv.y();
    }
    return std::abs(0.5 * signed_area);
  };

  for (int seed_edge_index = 0; seed_edge_index < static_cast<int>(boundary_edges.size()); ++seed_edge_index) {
    if (boundary_edge_used[static_cast<size_t>(seed_edge_index)] != 0) {
      continue;
    }

    std::vector<int> loop_vertices;
    int current_edge_index = seed_edge_index;
    const int start_vertex = boundary_edges[static_cast<size_t>(seed_edge_index)].start_vertex;
    bool loop_closed = false;

    for (size_t step = 0; step < boundary_edges.size(); ++step) {
      if (current_edge_index < 0 || current_edge_index >= static_cast<int>(boundary_edges.size())) {
        break;
      }
      if (boundary_edge_used[static_cast<size_t>(current_edge_index)] != 0) {
        break;
      }

      boundary_edge_used[static_cast<size_t>(current_edge_index)] = 1;
      const BoundaryEdge& edge = boundary_edges[static_cast<size_t>(current_edge_index)];
      loop_vertices.push_back(edge.start_vertex);

      const int next_vertex = edge.end_vertex;
      if (next_vertex == start_vertex) {
        loop_closed = true;
        break;
      }

      current_edge_index = -1;
      auto next_edges_it = boundary_edge_indices_by_start.find(next_vertex);
      if (next_edges_it == boundary_edge_indices_by_start.end()) {
        break;
      }
      for (int candidate_edge_index : next_edges_it->second) {
        if (boundary_edge_used[static_cast<size_t>(candidate_edge_index)] == 0) {
          current_edge_index = candidate_edge_index;
          break;
        }
      }
      if (current_edge_index < 0) {
        break;
      }
    }

    if (!loop_closed || loop_vertices.size() < 3) {
      continue;
    }

    const double loop_area = loop_area_abs(loop_vertices);
    if (loop_area > best_loop_area) {
      best_loop_area = loop_area;
      best_loop_vertices = std::move(loop_vertices);
    }
  }

  if (best_loop_vertices.size() < 3) {
    return;
  }

  out_boundary_3d.resize(static_cast<int>(best_loop_vertices.size()), 3);
  out_boundary_uv.resize(static_cast<int>(best_loop_vertices.size()), 2);
  for (int row = 0; row < static_cast<int>(best_loop_vertices.size()); ++row) {
    const int vertex_index = best_loop_vertices[static_cast<size_t>(row)];
    if (vertex_index >= 0 && vertex_index < V.rows() && vertex_index < UV.rows()) {
      out_boundary_3d.row(row) = V.row(vertex_index).head<3>();
      out_boundary_uv.row(row) = UV.row(vertex_index).head<2>();
    } else {
      out_boundary_3d.row(row).setZero();
      out_boundary_uv.row(row).setZero();
    }
  }

}

Eigen::Vector3d normalize_barycentric(const Eigen::Vector3d& bary) {
  Eigen::Vector3d clamped = bary.cwiseMax(0.0);
  const double sum = clamped.sum();
  if (sum <= 1e-14) {
    return Eigen::Vector3d::Constant(1.0 / 3.0);
  }
  return clamped / sum;
}

Eigen::Vector3d barycentric_from_3d(
  const Eigen::Vector3d& p,
  const Eigen::Vector3d& a,
  const Eigen::Vector3d& b,
  const Eigen::Vector3d& c) {
  const Eigen::Vector3d v0 = b - a;
  const Eigen::Vector3d v1 = c - a;
  const Eigen::Vector3d v2 = p - a;

  const double d00 = v0.dot(v0);
  const double d01 = v0.dot(v1);
  const double d11 = v1.dot(v1);
  const double d20 = v2.dot(v0);
  const double d21 = v2.dot(v1);
  const double denom = d00 * d11 - d01 * d01;

  if (std::abs(denom) <= 1e-14) {
    return Eigen::Vector3d::Constant(1.0 / 3.0);
  }

  const double w1 = (d11 * d20 - d01 * d21) / denom;
  const double w2 = (d00 * d21 - d01 * d20) / denom;
  const double w0 = 1.0 - w1 - w2;
  return normalize_barycentric(Eigen::Vector3d(w0, w1, w2));
}

Eigen::Vector3d barycentric_from_2d(
  const Eigen::Vector2d& p,
  const Eigen::Vector2d& a,
  const Eigen::Vector2d& b,
  const Eigen::Vector2d& c) {
  Eigen::Matrix2d basis;
  basis.col(0) = b - a;
  basis.col(1) = c - a;
  const double det = basis.determinant();
  if (std::abs(det) <= 1e-14) {
    return Eigen::Vector3d::Constant(1.0 / 3.0);
  }

  const Eigen::Vector2d w12 = basis.inverse() * (p - a);
  const double w1 = w12.x();
  const double w2 = w12.y();
  const double w0 = 1.0 - w1 - w2;
  return normalize_barycentric(Eigen::Vector3d(w0, w1, w2));
}

int nearest_face_center_3d(
  const Eigen::Vector3d& p,
  const Eigen::MatrixXd& centers_3d) {
  if (centers_3d.rows() == 0 || centers_3d.cols() < 3) {
    return -1;
  }

  int best_idx = -1;
  double best_d2 = std::numeric_limits<double>::infinity();
  for (int i = 0; i < centers_3d.rows(); ++i) {
    const Eigen::Vector3d c = centers_3d.row(i).head<3>().transpose();
    const double d2 = (c - p).squaredNorm();
    if (d2 < best_d2) {
      best_d2 = d2;
      best_idx = i;
    }
  }
  return best_idx;
}

int nearest_face_center_uv(
  const Eigen::Vector2d& p,
  const Eigen::MatrixXd& centers_uv) {
  if (centers_uv.rows() == 0 || centers_uv.cols() < 2) {
    return -1;
  }

  int best_idx = -1;
  double best_d2 = std::numeric_limits<double>::infinity();
  for (int i = 0; i < centers_uv.rows(); ++i) {
    const Eigen::Vector2d c = centers_uv.row(i).head<2>().transpose();
    const double d2 = (c - p).squaredNorm();
    if (d2 < best_d2) {
      best_d2 = d2;
      best_idx = i;
    }
  }
  return best_idx;
}

void erase_pattern_point_at(InteractionState& root_state, size_t idx) {
  PatternRegionState& state = active_region(root_state);
  auto erase_if_valid = [idx](auto& vec) {
    if (idx < vec.size()) {
      vec.erase(vec.begin() + static_cast<std::ptrdiff_t>(idx));
    }
  };
  erase_if_valid(state.pattern_points_3d);
  erase_if_valid(state.pattern_points_uv);
  erase_if_valid(state.pattern_processing_uv);
  erase_if_valid(state.pattern_points_delaunay_triangle);
  erase_if_valid(state.pattern_points_delaunay_vertices);
  erase_if_valid(state.pattern_point_class_ids);
  erase_if_valid(state.selected_sample_indices);
  if (state.selected_sample_indices.size() > state.pattern_points_uv.size()) {
    state.selected_sample_indices.resize(state.pattern_points_uv.size());
  }
  if (state.pattern_point_class_ids.size() > state.pattern_points_uv.size()) {
    state.pattern_point_class_ids.resize(state.pattern_points_uv.size());
  }
}

void recompute_last_crossed_triangle_count(
  InteractionState& root_state,
  const DelaunayTraversalHelper* delaunay_helper) {
  PatternRegionState& state = active_region(root_state);
  state.last_crossed_triangle_count = -1;
  if (!delaunay_helper || !delaunay_helper->is_ready()) {
    return;
  }
  if (state.pattern_processing_uv.size() >= 2) {
    const size_t n_pts = state.pattern_processing_uv.size();
    state.last_crossed_triangle_count = delaunay_helper->count_triangles_crossed(
      state.pattern_processing_uv[n_pts - 2],
      state.pattern_processing_uv[n_pts - 1]);
  }
}

void update_selected_samples_clouds(
  InteractionState& root_state,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh) {
  PatternRegionState& state = active_region(root_state);
  const std::string selected_samples_3d_name =
    region_object_name(kSelectedSamples3DBaseName, state.region_id);
  const std::string selected_samples_uv_name =
    region_object_name(kSelectedSamplesUVBaseName, state.region_id);
  const std::string selected_samples_3d_class_1_name =
    selected_samples_3d_name + kClass1ObjectSuffix;
  const std::string selected_samples_uv_class_1_name =
    selected_samples_uv_name + kClass1ObjectSuffix;

  const auto update_cloud_or_disable =
    [&](polyscope::PointCloud*& pc,
        const std::string& name,
        const Eigen::MatrixXd& data,
        const glm::vec3& color,
        polyscope::SurfaceMesh* owner_mesh) {
      if (data.rows() <= 0) {
        if (pc) {
          pc->setEnabled(false);
        }
        return;
      }
      ensure_point_cloud_with_preserved_radius(
        pc,
        name,
        data,
        viewer_overlay_radius(owner_mesh, 0.006));
      if (pc) {
        pc->setPointColor(color);
        pc->setEnabled(owner_mesh && owner_mesh->isEnabled());
      }
    };

  if (state.pattern_points_3d.empty() || state.pattern_points_uv.empty()) {
    if (state.selected_samples_3d) state.selected_samples_3d->setEnabled(false);
    if (state.selected_samples_uv) state.selected_samples_uv->setEnabled(false);
    if (state.selected_samples_3d_class_1) state.selected_samples_3d_class_1->setEnabled(false);
    if (state.selected_samples_uv_class_1) state.selected_samples_uv_class_1->setEnabled(false);
    return;
  }

  const size_t n_points = std::min(state.pattern_points_3d.size(), state.pattern_points_uv.size());
  if (region_is_two_class(state)) {
    std::array<int, kPatternClassCount> class_counts = {0, 0};
    for (size_t i = 0; i < n_points; ++i) {
      const int class_id =
        (i < state.pattern_point_class_ids.size())
          ? sanitize_pattern_class_id(state.pattern_point_class_ids[i])
          : 0;
      ++class_counts[static_cast<size_t>(class_id)];
    }

    std::array<Eigen::MatrixXd, kPatternClassCount> selected_3d_by_class;
    std::array<Eigen::MatrixXd, kPatternClassCount> selected_uv_by_class;
    for (int class_id = 0; class_id < kPatternClassCount; ++class_id) {
      selected_3d_by_class[static_cast<size_t>(class_id)].resize(
        class_counts[static_cast<size_t>(class_id)], 3);
      selected_uv_by_class[static_cast<size_t>(class_id)].resize(
        class_counts[static_cast<size_t>(class_id)], 3);
    }

    std::array<int, kPatternClassCount> offsets = {0, 0};
    for (size_t i = 0; i < n_points; ++i) {
      const int class_id =
        (i < state.pattern_point_class_ids.size())
          ? sanitize_pattern_class_id(state.pattern_point_class_ids[i])
          : 0;
      const int out_row = offsets[static_cast<size_t>(class_id)]++;
      selected_3d_by_class[static_cast<size_t>(class_id)].row(out_row) =
        state.pattern_points_3d[i].transpose();
      selected_uv_by_class[static_cast<size_t>(class_id)].row(out_row) =
        uv_to_display_3d(state.pattern_points_uv[i], state.uv_display_offset).transpose();
    }

    update_cloud_or_disable(
      state.selected_samples_3d,
      selected_samples_3d_name,
      selected_3d_by_class[0],
      kClass0Color,
      surfaceMesh);
    update_cloud_or_disable(
      state.selected_samples_uv,
      selected_samples_uv_name,
      selected_uv_by_class[0],
      kClass0Color,
      uvMesh);
    update_cloud_or_disable(
      state.selected_samples_3d_class_1,
      selected_samples_3d_class_1_name,
      selected_3d_by_class[1],
      kClass1Color,
      surfaceMesh);
    update_cloud_or_disable(
      state.selected_samples_uv_class_1,
      selected_samples_uv_class_1_name,
      selected_uv_by_class[1],
      kClass1Color,
      uvMesh);
    return;
  }

  if (state.selected_samples_3d_class_1) state.selected_samples_3d_class_1->setEnabled(false);
  if (state.selected_samples_uv_class_1) state.selected_samples_uv_class_1->setEnabled(false);

  Eigen::MatrixXd selected_3d(static_cast<int>(n_points), 3);
  Eigen::MatrixXd selected_uv_3d(static_cast<int>(n_points), 3);
  for (size_t i = 0; i < n_points; ++i) {
    selected_3d.row(static_cast<int>(i)) = state.pattern_points_3d[i].transpose();
    selected_uv_3d.row(static_cast<int>(i)) =
      uv_to_display_3d(state.pattern_points_uv[i], state.uv_display_offset).transpose();
  }

  ensure_point_cloud_with_preserved_radius(
    state.selected_samples_3d,
    selected_samples_3d_name,
    selected_3d,
    viewer_overlay_radius(surfaceMesh, 0.006));
  ensure_point_cloud_with_preserved_radius(
    state.selected_samples_uv,
    selected_samples_uv_name,
    selected_uv_3d,
    viewer_overlay_radius(uvMesh, 0.006));

  if (state.selected_samples_3d) {
    state.selected_samples_3d->setPointColor(glm::vec3(0.2f, 0.4f, 1.0f));
  }
  if (state.selected_samples_uv) {
    state.selected_samples_uv->setPointColor(glm::vec3(0.2f, 0.4f, 1.0f));
  }

  if (surfaceMesh && state.selected_samples_3d) {
    state.selected_samples_3d->setEnabled(surfaceMesh->isEnabled());
  }
  if (uvMesh && state.selected_samples_uv) {
    state.selected_samples_uv->setEnabled(uvMesh->isEnabled());
  }
}

void update_output_pattern_clouds(
  InteractionState& root_state,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh) {
  PatternRegionState& state = active_region(root_state);
  const std::string output_pattern_3d_name =
    region_object_name(kOutputPattern3DBaseName, state.region_id);
  const std::string output_pattern_uv_name =
    region_object_name(kOutputPatternUVBaseName, state.region_id);
  const std::string output_pattern_3d_class_1_name =
    output_pattern_3d_name + kClass1ObjectSuffix;
  const std::string output_pattern_uv_class_1_name =
    output_pattern_uv_name + kClass1ObjectSuffix;

  const auto update_cloud_or_disable =
    [&](polyscope::PointCloud*& pc,
        const std::string& name,
        const Eigen::MatrixXd& data,
        const glm::vec3& color,
        polyscope::SurfaceMesh* owner_mesh) {
      if (data.rows() <= 0) {
        if (pc) {
          pc->setEnabled(false);
        }
        return;
      }
      ensure_point_cloud_with_preserved_radius(
        pc,
        name,
        data,
        viewer_overlay_radius(owner_mesh, 0.0065));
      if (pc) {
        pc->setPointColor(color);
        pc->setEnabled(owner_mesh && owner_mesh->isEnabled());
      }
    };

  if (state.output_pattern_points_3d.empty() || state.output_pattern_points_uv.empty()) {
    if (state.output_pattern_3d) state.output_pattern_3d->setEnabled(false);
    if (state.output_pattern_uv) state.output_pattern_uv->setEnabled(false);
    if (state.output_pattern_3d_class_1) state.output_pattern_3d_class_1->setEnabled(false);
    if (state.output_pattern_uv_class_1) state.output_pattern_uv_class_1->setEnabled(false);
    return;
  }

  const size_t n_points =
    std::min(state.output_pattern_points_3d.size(), state.output_pattern_points_uv.size());
  if (n_points == 0) {
    if (state.output_pattern_3d) state.output_pattern_3d->setEnabled(false);
    if (state.output_pattern_uv) state.output_pattern_uv->setEnabled(false);
    if (state.output_pattern_3d_class_1) state.output_pattern_3d_class_1->setEnabled(false);
    if (state.output_pattern_uv_class_1) state.output_pattern_uv_class_1->setEnabled(false);
    return;
  }

  if (region_is_two_class(state)) {
    std::array<int, kPatternClassCount> class_counts = {0, 0};
    for (size_t i = 0; i < n_points; ++i) {
      const int class_id =
        (i < state.output_pattern_class_ids.size())
          ? sanitize_pattern_class_id(state.output_pattern_class_ids[i])
          : 0;
      ++class_counts[static_cast<size_t>(class_id)];
    }

    std::array<Eigen::MatrixXd, kPatternClassCount> out_3d_by_class;
    std::array<Eigen::MatrixXd, kPatternClassCount> out_uv_by_class;
    for (int class_id = 0; class_id < kPatternClassCount; ++class_id) {
      out_3d_by_class[static_cast<size_t>(class_id)].resize(
        class_counts[static_cast<size_t>(class_id)], 3);
      out_uv_by_class[static_cast<size_t>(class_id)].resize(
        class_counts[static_cast<size_t>(class_id)], 3);
    }

    std::array<int, kPatternClassCount> offsets = {0, 0};
    for (size_t i = 0; i < n_points; ++i) {
      const int class_id =
        (i < state.output_pattern_class_ids.size())
          ? sanitize_pattern_class_id(state.output_pattern_class_ids[i])
          : 0;
      const int out_row = offsets[static_cast<size_t>(class_id)]++;
      out_3d_by_class[static_cast<size_t>(class_id)].row(out_row) =
        state.output_pattern_points_3d[i].transpose();
      out_uv_by_class[static_cast<size_t>(class_id)].row(out_row) =
        uv_to_display_3d(state.output_pattern_points_uv[i], state.uv_display_offset).transpose();
    }

    update_cloud_or_disable(
      state.output_pattern_3d,
      output_pattern_3d_name,
      out_3d_by_class[0],
      kClass0Color,
      surfaceMesh);
    update_cloud_or_disable(
      state.output_pattern_uv,
      output_pattern_uv_name,
      out_uv_by_class[0],
      kClass0Color,
      uvMesh);
    update_cloud_or_disable(
      state.output_pattern_3d_class_1,
      output_pattern_3d_class_1_name,
      out_3d_by_class[1],
      kClass1Color,
      surfaceMesh);
    update_cloud_or_disable(
      state.output_pattern_uv_class_1,
      output_pattern_uv_class_1_name,
      out_uv_by_class[1],
      kClass1Color,
      uvMesh);
    return;
  }

  if (state.output_pattern_3d_class_1) state.output_pattern_3d_class_1->setEnabled(false);
  if (state.output_pattern_uv_class_1) state.output_pattern_uv_class_1->setEnabled(false);

  Eigen::MatrixXd out_3d(static_cast<int>(n_points), 3);
  Eigen::MatrixXd out_uv_3d(static_cast<int>(n_points), 3);
  for (size_t i = 0; i < n_points; ++i) {
    out_3d.row(static_cast<int>(i)) = state.output_pattern_points_3d[i].transpose();
    out_uv_3d.row(static_cast<int>(i)) =
      uv_to_display_3d(state.output_pattern_points_uv[i], state.uv_display_offset).transpose();
  }

  ensure_point_cloud_with_preserved_radius(
    state.output_pattern_3d,
    output_pattern_3d_name,
    out_3d,
    viewer_overlay_radius(surfaceMesh, 0.0065));
  ensure_point_cloud_with_preserved_radius(
    state.output_pattern_uv,
    output_pattern_uv_name,
    out_uv_3d,
    viewer_overlay_radius(uvMesh, 0.0065));

  if (state.output_pattern_3d) {
    state.output_pattern_3d->setPointColor(glm::vec3(1.0f, 0.35f, 0.1f));
  }
  if (state.output_pattern_uv) {
    state.output_pattern_uv->setPointColor(glm::vec3(1.0f, 0.35f, 0.1f));
  }

  if (surfaceMesh && state.output_pattern_3d) {
    state.output_pattern_3d->setEnabled(surfaceMesh->isEnabled());
  }
  if (uvMesh && state.output_pattern_uv) {
    state.output_pattern_uv->setEnabled(uvMesh->isEnabled());
  }
}

void update_input_reference_clouds(
  InteractionState& root_state,
  const Eigen::MatrixXd& points_3d,
  const Eigen::MatrixXd& points_uv,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh) {
  PatternRegionState& state = active_region(root_state);
  const std::string input_reference_3d_name =
    region_object_name(kInputReference3DBaseName, state.region_id);
  const std::string input_reference_uv_name =
    region_object_name(kInputReferenceUVBaseName, state.region_id);

  if (!state.input_reference_indices.empty()) {
    std::vector<int> valid_indices;
    valid_indices.reserve(state.input_reference_indices.size());
    const int max_rows = std::min(points_3d.rows(), points_uv.rows());
    for (int idx : state.input_reference_indices) {
      if (idx >= 0 && idx < max_rows) {
        valid_indices.push_back(idx);
      }
    }
    if (valid_indices.size() != state.input_reference_indices.size()) {
      state.input_reference_indices.swap(valid_indices);
    }
  }

  Eigen::MatrixXd reference_3d;
  Eigen::MatrixXd reference_uv_3d;

  if (!state.input_reference_indices.empty() &&
      points_3d.rows() > 0 && points_uv.rows() > 0 && points_uv.cols() >= 2) {
    reference_3d.resize(state.input_reference_indices.size(), 3);
    reference_uv_3d.resize(state.input_reference_indices.size(), 3);

    size_t out_idx = 0;
    for (int idx : state.input_reference_indices) {
      if (idx >= 0 && idx < points_3d.rows()) {
        reference_3d.row(out_idx) = points_3d.row(idx);
        reference_uv_3d.row(static_cast<int>(out_idx)) =
          uv_to_display_3d(points_uv.row(idx).head<2>(), state.uv_display_offset).transpose();
        out_idx++;
      }
    }
    reference_3d.conservativeResize(out_idx, 3);
    reference_uv_3d.conservativeResize(out_idx, 3);
  }

  if (reference_3d.rows() == 0) {
    if (state.input_reference_3d) state.input_reference_3d->setEnabled(false);
    if (state.input_reference_uv) state.input_reference_uv->setEnabled(false);
    return;
  }

  ensure_point_cloud_with_preserved_radius(
    state.input_reference_3d,
    input_reference_3d_name,
    reference_3d,
    viewer_overlay_radius(surfaceMesh, 0.004));
  ensure_point_cloud_with_preserved_radius(
    state.input_reference_uv,
    input_reference_uv_name,
    reference_uv_3d,
    viewer_overlay_radius(uvMesh, 0.004));

  if (state.input_reference_3d) {
    state.input_reference_3d->setPointColor(glm::vec3(0.2f, 0.9f, 0.6f));
  }
  if (state.input_reference_uv) {
    state.input_reference_uv->setPointColor(glm::vec3(0.2f, 0.9f, 0.6f));
  }

  if (surfaceMesh && state.input_reference_3d) {
    state.input_reference_3d->setEnabled(surfaceMesh->isEnabled());
  }
  if (uvMesh && state.input_reference_uv) {
    state.input_reference_uv->setEnabled(uvMesh->isEnabled());
  }
}

void update_input_boundary_curve(
  InteractionState& root_state,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh) {
  PatternRegionState& state = active_region(root_state);
  const std::string input_boundary_name =
    region_object_name(kInputBoundaryBaseName, state.region_id);
  const std::string input_boundary_3d_name =
    region_object_name(kInputBoundary3DBaseName, state.region_id);

  if (state.input_boundary_uv.rows() < 3) {
    if (polyscope::hasCurveNetwork(input_boundary_name)) {
      polyscope::removeCurveNetwork(input_boundary_name);
    }
    state.input_boundary_curve = nullptr;
  }

  if (state.input_boundary_3d.rows() < 3) {
    if (polyscope::hasCurveNetwork(input_boundary_3d_name)) {
      polyscope::removeCurveNetwork(input_boundary_3d_name);
    }
    state.input_boundary_curve_3d = nullptr;
  }

  if (state.input_boundary_uv.rows() >= 3) {
    Eigen::MatrixXd display_uv =
      uv_matrix_to_display_2d(state.input_boundary_uv, state.uv_display_offset);
    ensure_curve_network_loop_2d_with_preserved_radius(
      state.input_boundary_curve,
      input_boundary_name,
      display_uv,
      viewer_overlay_radius(uvMesh, 0.003));
    if (state.input_boundary_curve) {
      state.input_boundary_curve->setColor(glm::vec3(0.2f, 0.9f, 0.2f));
    }
  }

  if (state.input_boundary_3d.rows() >= 3) {
    ensure_curve_network_loop_with_preserved_radius(
      state.input_boundary_curve_3d,
      input_boundary_3d_name,
      state.input_boundary_3d,
      viewer_overlay_radius(surfaceMesh, 0.003));
    if (state.input_boundary_curve_3d) {
      state.input_boundary_curve_3d->setColor(glm::vec3(0.2f, 0.9f, 0.2f));
    }
  }

  if (uvMesh && state.input_boundary_curve) {
    state.input_boundary_curve->setEnabled(uvMesh->isEnabled());
  }
  if (surfaceMesh && state.input_boundary_curve_3d) {
    state.input_boundary_curve_3d->setEnabled(surfaceMesh->isEnabled());
  }
}

void update_output_boundary_quantities(
  InteractionState& root_state,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh,
  const Eigen::MatrixXi& F) {
  PatternRegionState& state = active_region(root_state);
  const std::string output_boundary_name =
    region_object_name(kOutputBoundaryBaseName, state.region_id);

  if (!surfaceMesh) return;

  Eigen::VectorXd mask = Eigen::VectorXd::Zero(F.rows());
  for (int idx : state.painted_face_indices) {
    if (idx >= 0 && idx < F.rows()) {
      mask(idx) = 1.0;
    }
  }

  if (!state.output_boundary_surface) {
    state.output_boundary_surface = surfaceMesh->addFaceScalarQuantity(output_boundary_name, mask);
    if (state.output_boundary_surface) {
      state.output_boundary_surface->setColorMap("coolwarm");
      state.output_boundary_surface->setEnabled(false);
    }
  } else {
    state.output_boundary_surface->updateData(mask);
    state.output_boundary_surface->setEnabled(false);
  }

  if (uvMesh) {
    if (!state.output_boundary_uv) {
      state.output_boundary_uv = uvMesh->addFaceScalarQuantity(output_boundary_name, mask);
      if (state.output_boundary_uv) {
        state.output_boundary_uv->setColorMap("coolwarm");
        state.output_boundary_uv->setEnabled(false);
      }
    } else {
      state.output_boundary_uv->updateData(mask);
      state.output_boundary_uv->setEnabled(false);
    }
  }
}

void update_input_boundary_quantities(
  InteractionState& root_state,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh,
  const Eigen::MatrixXi& F) {
  PatternRegionState& state = active_region(root_state);
  const std::string input_boundary_face_name =
    region_object_name(kInputBoundaryFaceBaseName, state.region_id);

  if (!surfaceMesh) return;

  Eigen::VectorXd mask = Eigen::VectorXd::Zero(F.rows());
  for (int idx : state.input_painted_face_indices) {
    if (idx >= 0 && idx < F.rows()) {
      mask(idx) = 1.0;
    }
  }

  if (!state.input_boundary_surface) {
    state.input_boundary_surface = surfaceMesh->addFaceScalarQuantity(input_boundary_face_name, mask);
    if (state.input_boundary_surface) {
      state.input_boundary_surface->setColorMap("coolwarm");
      state.input_boundary_surface->setEnabled(false);
    }
  } else {
    state.input_boundary_surface->updateData(mask);
    state.input_boundary_surface->setEnabled(false);
  }

  if (uvMesh) {
    if (!state.input_boundary_uv_faces) {
      state.input_boundary_uv_faces = uvMesh->addFaceScalarQuantity(input_boundary_face_name, mask);
      if (state.input_boundary_uv_faces) {
        state.input_boundary_uv_faces->setColorMap("coolwarm");
        state.input_boundary_uv_faces->setEnabled(false);
      }
    } else {
      state.input_boundary_uv_faces->updateData(mask);
      state.input_boundary_uv_faces->setEnabled(false);
    }
  }
}

void toggle_index(std::vector<int>& indices, int idx) {
  auto it = std::find(indices.begin(), indices.end(), idx);
  if (it == indices.end()) {
    indices.push_back(idx);
  } else {
    indices.erase(it);
  }
}

void add_index(std::vector<int>& indices, int idx) {
  if (std::find(indices.begin(), indices.end(), idx) == indices.end()) {
    indices.push_back(idx);
  }
}

void remove_index(std::vector<int>& indices, int idx) {
  auto it = std::find(indices.begin(), indices.end(), idx);
  if (it != indices.end()) {
    indices.erase(it);
  }
}

double point_segment_distance_squared_2d(
  const Eigen::Vector2d& p,
  const Eigen::Vector2d& a,
  const Eigen::Vector2d& b) {
  const Eigen::Vector2d ab = b - a;
  const double ab2 = ab.squaredNorm();
  if (ab2 <= 1e-20) {
    return (p - a).squaredNorm();
  }
  const double t = std::clamp((p - a).dot(ab) / ab2, 0.0, 1.0);
  const Eigen::Vector2d proj = a + t * ab;
  return (p - proj).squaredNorm();
}

bool point_in_triangle_inclusive_2d(
  const Eigen::Vector2d& p,
  const Eigen::Vector2d& a,
  const Eigen::Vector2d& b,
  const Eigen::Vector2d& c,
  double eps) {
  if (point_on_segment_2d(p, a, b, eps) ||
      point_on_segment_2d(p, b, c, eps) ||
      point_on_segment_2d(p, c, a, eps)) {
    return true;
  }

  const double o0 = orient2d(a, b, p);
  const double o1 = orient2d(b, c, p);
  const double o2 = orient2d(c, a, p);
  const bool has_neg = (o0 < -eps) || (o1 < -eps) || (o2 < -eps);
  const bool has_pos = (o0 > eps) || (o1 > eps) || (o2 > eps);
  return !(has_neg && has_pos);
}

bool triangle_intersects_circle_2d(
  const Eigen::Vector2d& center,
  double radius,
  const Eigen::Vector2d& a,
  const Eigen::Vector2d& b,
  const Eigen::Vector2d& c) {
  const double clamped_radius = std::max(0.0, radius);
  const double scale = std::max({1.0, clamped_radius, (b - a).norm(), (c - b).norm(), (a - c).norm()});
  const double eps = 1e-10 * scale;
  const double r2 = clamped_radius * clamped_radius + eps;

  if ((center - a).squaredNorm() <= r2 ||
      (center - b).squaredNorm() <= r2 ||
      (center - c).squaredNorm() <= r2) {
    return true;
  }
  if (point_in_triangle_inclusive_2d(center, a, b, c, eps)) {
    return true;
  }
  if (point_segment_distance_squared_2d(center, a, b) <= r2 ||
      point_segment_distance_squared_2d(center, b, c) <= r2 ||
      point_segment_distance_squared_2d(center, c, a) <= r2) {
    return true;
  }
  return false;
}

std::vector<int> collect_face_brush_faces(
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F,
  const Eigen::Vector2d& brush_center_uv,
  double brush_radius) {
  std::vector<int> brush_faces;
  if (UV.rows() <= 0 || UV.cols() < 2 || F.rows() <= 0 || F.cols() < 3) {
    return brush_faces;
  }

  const double clamped_radius = std::max(0.0, brush_radius);
  const int reserve_hint = F.rows() > 32 ? static_cast<int>(F.rows() / 32) : 1;
  brush_faces.reserve(static_cast<size_t>(reserve_hint));
  for (int face_index = 0; face_index < F.rows(); ++face_index) {
    const int v0 = F(face_index, 0);
    const int v1 = F(face_index, 1);
    const int v2 = F(face_index, 2);
    if (v0 < 0 || v1 < 0 || v2 < 0 ||
        v0 >= UV.rows() || v1 >= UV.rows() || v2 >= UV.rows()) {
      continue;
    }

    const Eigen::Vector2d a = UV.row(v0).head<2>().transpose();
    const Eigen::Vector2d b = UV.row(v1).head<2>().transpose();
    const Eigen::Vector2d c = UV.row(v2).head<2>().transpose();
    const Eigen::Vector2d tri_min(
      std::min({a.x(), b.x(), c.x()}),
      std::min({a.y(), b.y(), c.y()}));
    const Eigen::Vector2d tri_max(
      std::max({a.x(), b.x(), c.x()}),
      std::max({a.y(), b.y(), c.y()}));
    if (brush_center_uv.x() < tri_min.x() - clamped_radius ||
        brush_center_uv.x() > tri_max.x() + clamped_radius ||
        brush_center_uv.y() < tri_min.y() - clamped_radius ||
        brush_center_uv.y() > tri_max.y() + clamped_radius) {
      continue;
    }

    if (triangle_intersects_circle_2d(brush_center_uv, clamped_radius, a, b, c)) {
      brush_faces.push_back(face_index);
    }
  }

  return brush_faces;
}

void apply_face_brush(
  std::vector<int>& indices,
  const std::vector<int>& brush_faces,
  bool select_mode,
  bool deselect_mode) {
  if (brush_faces.empty()) {
    return;
  }

  for (int face_index : brush_faces) {
    if (select_mode) {
      add_index(indices, face_index);
    }
    if (deselect_mode) {
      remove_index(indices, face_index);
    }
  }
}

void apply_face_brush(
  std::vector<int>& indices,
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F,
  const Eigen::Vector2d& brush_center_uv,
  double brush_radius,
  bool select_mode,
  bool deselect_mode) {
  const std::vector<int> brush_faces =
    collect_face_brush_faces(UV, F, brush_center_uv, brush_radius);
  apply_face_brush(indices, brush_faces, select_mode, deselect_mode);
}

Eigen::MatrixXd build_uv_circle_preview(
  const Eigen::Vector2d& brush_center_uv,
  double brush_radius,
  int sample_count = 48) {
  if (brush_radius <= 0.0 || sample_count < 3) {
    return Eigen::MatrixXd(0, 2);
  }

  Eigen::MatrixXd circle_uv(sample_count, 2);
  constexpr double kTwoPi = 6.28318530717958647692;
  for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
    const double angle = kTwoPi * static_cast<double>(sample_index) / static_cast<double>(sample_count);
    circle_uv(sample_index, 0) = brush_center_uv.x() + brush_radius * std::cos(angle);
    circle_uv(sample_index, 1) = brush_center_uv.y() + brush_radius * std::sin(angle);
  }
  return circle_uv;
}

void clear_paint_brush_preview(const PatternRegionState& state) {
  const std::string preview_curve_3d_name =
    region_object_name(kPaintBrushPreviewCurve3DBaseName, state.region_id);
  const std::string preview_curve_uv_name =
    region_object_name(kPaintBrushPreviewCurveUVBaseName, state.region_id);
  if (polyscope::hasCurveNetwork(preview_curve_3d_name)) {
    polyscope::removeCurveNetwork(preview_curve_3d_name);
  }
  if (polyscope::hasCurveNetwork(preview_curve_uv_name)) {
    polyscope::removeCurveNetwork(preview_curve_uv_name);
  }
}

void update_paint_brush_preview(
  const PatternRegionState& state,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F,
  const Eigen::Vector2d& brush_center_uv,
  double brush_radius,
  const std::vector<int>& preview_faces,
  const glm::vec3& preview_color) {
  const Eigen::MatrixXd circle_uv = build_uv_circle_preview(brush_center_uv, brush_radius);
  if (preview_faces.empty() && circle_uv.rows() < 3) {
    clear_paint_brush_preview(state);
    return;
  }

  Eigen::MatrixXd boundary_3d;
  if (!preview_faces.empty()) {
    Eigen::MatrixXd boundary_uv;
    build_boundary_from_painted_faces(preview_faces, V, UV, F, boundary_3d, boundary_uv);
  }

  const std::string preview_curve_3d_name =
    region_object_name(kPaintBrushPreviewCurve3DBaseName, state.region_id);
  const std::string preview_curve_uv_name =
    region_object_name(kPaintBrushPreviewCurveUVBaseName, state.region_id);

  if (boundary_3d.rows() < 3) {
    if (polyscope::hasCurveNetwork(preview_curve_3d_name)) {
      polyscope::removeCurveNetwork(preview_curve_3d_name);
    }
  } else {
    polyscope::CurveNetwork* preview_curve_3d = nullptr;
    if (!polyscope::hasCurveNetwork(preview_curve_3d_name)) {
      preview_curve_3d = polyscope::registerCurveNetworkLoop(preview_curve_3d_name, boundary_3d);
    } else {
      preview_curve_3d = polyscope::getCurveNetwork(preview_curve_3d_name);
      if (preview_curve_3d && preview_curve_3d->nNodes() == static_cast<size_t>(boundary_3d.rows())) {
        preview_curve_3d->updateNodePositions(boundary_3d);
      } else {
        polyscope::removeCurveNetwork(preview_curve_3d_name);
        preview_curve_3d = polyscope::registerCurveNetworkLoop(preview_curve_3d_name, boundary_3d);
      }
    }
    if (preview_curve_3d) {
      preview_curve_3d->setColor(preview_color);
      preview_curve_3d->setEnabled(surfaceMesh && surfaceMesh->isEnabled());
    }
  }

  if (circle_uv.rows() < 3) {
    if (polyscope::hasCurveNetwork(preview_curve_uv_name)) {
      polyscope::removeCurveNetwork(preview_curve_uv_name);
    }
  } else {
    const Eigen::MatrixXd display_uv =
      uv_matrix_to_display_2d(circle_uv, state.uv_display_offset);
    polyscope::CurveNetwork* preview_curve_uv = nullptr;
    ensure_curve_network_loop_2d_with_preserved_radius(
      preview_curve_uv,
      preview_curve_uv_name,
      display_uv,
      viewer_overlay_radius(uvMesh, 0.0025));
    if (preview_curve_uv) {
      preview_curve_uv->setColor(preview_color);
      preview_curve_uv->setEnabled(uvMesh && uvMesh->isEnabled());
    }
  }
}

void update_output_boundary_from_faces(
  InteractionState& root_state,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F) {
  PatternRegionState& state = active_region(root_state);

  if (state.painted_face_indices.empty()) {
    state.output_boundary_3d_poly.resize(0, 3);
    state.output_boundary_uv_poly.resize(0, 2);
    return;
  }

  build_boundary_from_painted_faces(
    state.painted_face_indices,
    V,
    UV,
    F,
    state.output_boundary_3d_poly,
    state.output_boundary_uv_poly);
}

void update_output_boundary_preview_from_faces(
  InteractionState& root_state,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F) {
  PatternRegionState& state = active_region(root_state);

  if (state.painted_face_indices.empty()) {
    state.output_boundary_preview_3d_poly.resize(0, 3);
    state.output_boundary_preview_uv_poly.resize(0, 2);
    return;
  }

  build_boundary_from_painted_faces(
    state.painted_face_indices,
    V,
    UV,
    F,
    state.output_boundary_preview_3d_poly,
    state.output_boundary_preview_uv_poly);
}

void commit_output_boundary_preview(PatternRegionState& state) {
  state.output_boundary_3d_poly = state.output_boundary_preview_3d_poly;
  state.output_boundary_uv_poly = state.output_boundary_preview_uv_poly;
  state.output_boundary_preview_3d_poly.resize(0, 3);
  state.output_boundary_preview_uv_poly.resize(0, 2);
}

void update_input_boundary_path_from_faces(
  InteractionState& root_state,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F) {
  PatternRegionState& state = active_region(root_state);

  if (state.input_painted_face_indices.empty()) {
    state.input_boundary_uv.resize(0, 2);
    state.input_boundary_3d.resize(0, 3);
    return;
  }

  build_boundary_from_painted_faces(
    state.input_painted_face_indices,
    V,
    UV,
    F,
    state.input_boundary_3d,
    state.input_boundary_uv);
  if (state.input_boundary_uv.rows() < 3) {
    state.input_boundary_uv.resize(0, 2);
    state.input_boundary_3d.resize(0, 3);
  }
}

void update_output_boundary_curve(
  InteractionState& root_state,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh) {
  PatternRegionState& state = active_region(root_state);
  const Eigen::MatrixXd& active_boundary_3d =
    state.output_boundary_pending_edits ? state.output_boundary_preview_3d_poly : state.output_boundary_3d_poly;
  const Eigen::MatrixXd& active_boundary_uv =
    state.output_boundary_pending_edits ? state.output_boundary_preview_uv_poly : state.output_boundary_uv_poly;
  const std::string output_boundary_curve_3d_name =
    region_object_name(kOutputBoundaryCurve3DBaseName, state.region_id);
  const std::string output_boundary_curve_uv_name =
    region_object_name(kOutputBoundaryCurveUVBaseName, state.region_id);

  if (active_boundary_3d.rows() < 3) {
    if (polyscope::hasCurveNetwork(output_boundary_curve_3d_name)) {
      polyscope::removeCurveNetwork(output_boundary_curve_3d_name);
    }
    state.output_boundary_curve_3d = nullptr;
  }

  if (active_boundary_uv.rows() < 3) {
    if (polyscope::hasCurveNetwork(output_boundary_curve_uv_name)) {
      polyscope::removeCurveNetwork(output_boundary_curve_uv_name);
    }
    state.output_boundary_curve_uv = nullptr;
  }

  if (active_boundary_3d.rows() >= 3) {
    ensure_curve_network_loop_with_preserved_radius(
      state.output_boundary_curve_3d,
      output_boundary_curve_3d_name,
      active_boundary_3d,
      viewer_overlay_radius(surfaceMesh, 0.003));
    if (state.output_boundary_curve_3d) {
      state.output_boundary_curve_3d->setColor(glm::vec3(1.0f, 0.6f, 0.1f));
    }
  }

  Eigen::MatrixXd hull_nodes_uv_display =
    uv_matrix_to_display_2d(active_boundary_uv, state.uv_display_offset);

  if (active_boundary_uv.rows() >= 3) {
    ensure_curve_network_loop_2d_with_preserved_radius(
      state.output_boundary_curve_uv,
      output_boundary_curve_uv_name,
      hull_nodes_uv_display,
      viewer_overlay_radius(uvMesh, 0.003));
    if (state.output_boundary_curve_uv) {
      state.output_boundary_curve_uv->setColor(glm::vec3(1.0f, 0.6f, 0.1f));
    }
  }

  if (surfaceMesh && state.output_boundary_curve_3d) {
    state.output_boundary_curve_3d->setEnabled(surfaceMesh->isEnabled());
  }
  if (uvMesh && state.output_boundary_curve_uv) {
    state.output_boundary_curve_uv->setEnabled(uvMesh->isEnabled());
  }
}

void update_input_boundary_from_faces(
  InteractionState& root_state,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F,
  const std::vector<DartSample>& samples_uv) {
  PatternRegionState& state = active_region(root_state);

  update_input_boundary_path_from_faces(root_state, V, UV, F);
  if (state.input_boundary_uv.rows() < 3) {
    state.input_reference_indices.clear();
    return;
  }

  state.input_reference_indices.clear();
  for (int i = 0; i < static_cast<int>(samples_uv.size()); ++i) {
    if (point_in_polygon(samples_uv[i].uv, state.input_boundary_uv)) {
      state.input_reference_indices.push_back(i);
    }
  }
}

std::string sanitize_path_component(const std::string& value) {
  std::string sanitized;
  sanitized.reserve(value.size());
  for (char ch : value) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isalnum(uch) || ch == '-' || ch == '_') {
      sanitized.push_back(ch);
    } else if (ch == ' ' || ch == '.') {
      sanitized.push_back('_');
    }
  }
  if (sanitized.empty()) {
    return "region";
  }
  return sanitized;
}

bool save_output_pattern_region_to_file(
  const std::filesystem::path& path,
  const PatternRegionState& region,
  const std::string& region_label,
  int region_index,
  std::string& out_error) {
  const size_t point_count = std::min(
    region.output_pattern_points_uv.size(),
    region.output_pattern_points_3d.size());
  if (point_count == 0) {
    out_error = "Region has no output points.";
    return false;
  }

  std::ofstream out(path, std::ios::trunc);
  if (!out.is_open()) {
    out_error = "Failed to open " + path.string() + " for writing.";
    return false;
  }

  out << kOutputPatternFileHeader << "\n";
  out << "region_index " << region_index << "\n";
  out << "region_id " << region.region_id << "\n";
  out << "region_label " << region_label << "\n";
  out << "region_mode " << pattern_region_mode_label(region) << "\n";
  out << "point_count " << point_count << "\n";
  out << "columns index sample_index class_id uv_x uv_y x y z\n";
  out << std::setprecision(17);
  for (size_t i = 0; i < point_count; ++i) {
    const int sample_idx =
      (i < region.output_pattern_sample_indices.size())
        ? region.output_pattern_sample_indices[i]
        : -1;
    const int class_id =
      (i < region.output_pattern_class_ids.size())
        ? sanitize_pattern_class_id(region.output_pattern_class_ids[i])
        : 0;
    const Eigen::Vector2d& uv = region.output_pattern_points_uv[i];
    const Eigen::Vector3d& p3 = region.output_pattern_points_3d[i];
    out << i << " "
        << sample_idx << " "
        << class_id << " "
        << uv.x() << " "
        << uv.y() << " "
        << p3.x() << " "
        << p3.y() << " "
        << p3.z() << "\n";
  }

  if (!out.good()) {
    out_error = "Write error while saving " + path.string() + ".";
    return false;
  }
  return true;
}

bool save_output_patterns_by_region(
  const std::string& directory_path,
  const InteractionState& root_state,
  std::string& out_error,
  int& out_saved_count) {
  out_saved_count = 0;
  if (directory_path.empty()) {
    out_error = "Please provide an output directory.";
    return false;
  }

  const std::filesystem::path output_dir(directory_path);
  std::error_code ec;
  std::filesystem::create_directories(output_dir, ec);
  if (ec) {
    out_error = "Failed to create output directory: " + ec.message();
    return false;
  }
  ec.clear();
  if (!std::filesystem::is_directory(output_dir, ec)) {
    out_error = ec
      ? "Failed to inspect output directory: " + ec.message()
      : "Output path is not a directory.";
    return false;
  }

  std::vector<std::string> skipped_regions;
  for (int region_index = 0; region_index < region_count(root_state); ++region_index) {
    const PatternRegionState& region = region_state(root_state, region_index);
    const size_t point_count = std::min(
      region.output_pattern_points_uv.size(),
      region.output_pattern_points_3d.size());
    if (point_count == 0) {
      skipped_regions.push_back(pattern_region_label(root_state, region_index));
      continue;
    }

    const std::string label = pattern_region_label(root_state, region_index);
    const std::string file_name =
      "region_" + std::to_string(region.region_id) + "_" +
      sanitize_path_component(label) + "_output_pattern.txt";
    std::string region_error;
    if (!save_output_pattern_region_to_file(
          output_dir / file_name,
          region,
          label,
          region_index,
          region_error)) {
      out_error = region_error;
      return false;
    }
    ++out_saved_count;
  }

  if (out_saved_count == 0) {
    out_error = "No regions have output patterns to save.";
    return false;
  }

  out_error =
    "Saved " + std::to_string(out_saved_count) +
    " output pattern file" + (out_saved_count == 1 ? "" : "s") +
    " to " + output_dir.string() + ".";
  if (!skipped_regions.empty()) {
    out_error += " Skipped " + std::to_string(skipped_regions.size()) +
                 " empty region" + (skipped_regions.size() == 1 ? "" : "s") + ".";
  }
  return true;
}

bool save_input_pattern_to_file(
  const std::string& path,
  const InteractionState& root_state,
  std::string& out_error) {
  const PatternRegionState& state = active_region(root_state);
  const size_t point_count = std::min(
    state.pattern_points_uv.size(),
    state.pattern_points_delaunay_triangle.size());
  if (point_count == 0) {
    out_error = "No input pattern points to save.";
    return false;
  }

  std::ofstream out(path, std::ios::trunc);
  if (!out.is_open()) {
    out_error = "Failed to open file for writing.";
    return false;
  }

  out << kPatternFileHeader << "\n";
  out << point_count << "\n";
  out << "# columns triangle_index sample_index class_id uv_x uv_y\n";
  out << std::setprecision(17);
  for (size_t i = 0; i < point_count; ++i) {
    const int tri_idx = state.pattern_points_delaunay_triangle[i];
    const int sample_idx =
      (i < state.selected_sample_indices.size()) ? state.selected_sample_indices[i] : -1;
    const int class_id =
      (i < state.pattern_point_class_ids.size())
        ? sanitize_pattern_class_id(state.pattern_point_class_ids[i])
        : 0;
    const Eigen::Vector2d& uv = state.pattern_points_uv[i];
    out << tri_idx << " "
        << sample_idx << " "
        << class_id << " "
        << uv.x() << " "
        << uv.y() << "\n";
  }

  if (!out.good()) {
    out_error = "Write error while saving pattern.";
    return false;
  }
  return true;
}

bool load_input_pattern_from_file(
  const std::string& path,
  InteractionState& root_state,
  const DelaunayTraversalHelper* delaunay_helper,
  const Eigen::MatrixXd& sample_points_3d,
  const Eigen::MatrixXd& sample_points_uv,
  std::string& out_error) {
  PatternRegionState& state = active_region(root_state);
  if (!delaunay_helper || !delaunay_helper->is_ready()) {
    out_error = "Delaunay helper is not ready.";
    return false;
  }

  std::ifstream in(path);
  if (!in.is_open()) {
    out_error = "Failed to open file for reading.";
    return false;
  }

  std::string header;
  if (!std::getline(in, header)) {
    out_error = "Empty pattern file.";
    return false;
  }
  if (header != kPatternFileHeader) {
    out_error = "Unsupported pattern file format.";
    return false;
  }

  size_t point_count = 0;
  if (!(in >> point_count)) {
    out_error = "Failed to read pattern point count.";
    return false;
  }

  std::vector<Eigen::Vector3d> loaded_points_3d;
  std::vector<Eigen::Vector2d> loaded_points_uv;
  std::vector<Eigen::Vector2d> loaded_processing_uv;
  std::vector<int> loaded_triangles;
  std::vector<Eigen::Vector3i> loaded_triangle_vertices;
  std::vector<int> loaded_sample_indices;
  std::vector<int> loaded_class_ids;
  loaded_points_3d.reserve(point_count);
  loaded_points_uv.reserve(point_count);
  loaded_processing_uv.reserve(point_count);
  loaded_triangles.reserve(point_count);
  loaded_triangle_vertices.reserve(point_count);
  loaded_sample_indices.reserve(point_count);
  loaded_class_ids.reserve(point_count);

  std::unordered_set<int> seen_triangles;
  seen_triangles.reserve(point_count);

  std::string line;
  std::getline(in, line);
  for (size_t i = 0; i < point_count; ++i) {
    int tri_idx = -1;
    int sample_idx = -1;
    int class_id = 0;
    double uv_x = 0.0;
    double uv_y = 0.0;
    bool got_entry = false;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      std::istringstream row(line);
      std::vector<double> values;
      double value = 0.0;
      while (row >> value) {
        values.push_back(value);
      }
      if (values.size() == 4) {
        tri_idx = static_cast<int>(values[0]);
        sample_idx = static_cast<int>(values[1]);
        uv_x = values[2];
        uv_y = values[3];
        class_id = 0;
        got_entry = true;
        break;
      }
      if (values.size() >= 5) {
        tri_idx = static_cast<int>(values[0]);
        sample_idx = static_cast<int>(values[1]);
        class_id = sanitize_pattern_class_id(static_cast<int>(values[2]));
        uv_x = values[3];
        uv_y = values[4];
        got_entry = true;
        break;
      }
      out_error = "Failed to read pattern point entry.";
      return false;
    }
    if (!got_entry) {
      out_error = "Failed to read pattern point entry.";
      return false;
    }
    if (!seen_triangles.insert(tri_idx).second) {
      out_error = "Pattern file contains duplicate triangle ids.";
      return false;
    }

    Eigen::Vector2d center_uv = Eigen::Vector2d::Zero();
    if (!delaunay_helper->triangle_center(tri_idx, center_uv)) {
      out_error = "Pattern file references an invalid triangle id.";
      return false;
    }

    int resolved_tri_idx = -1;
    Eigen::Vector3i tri_vertices(-1, -1, -1);
    if (!delaunay_helper->find_containing_triangle(center_uv, resolved_tri_idx, tri_vertices)) {
      out_error = "Failed to resolve loaded triangle center.";
      return false;
    }

    Eigen::Vector3d query_3d = Eigen::Vector3d::Zero();
    int valid_center_vertices = 0;
    for (int c = 0; c < 3; ++c) {
      const int sidx = tri_vertices[c];
      if (sidx < 0 || sidx >= sample_points_3d.rows() || sample_points_3d.cols() < 3) {
        continue;
      }
      query_3d += sample_points_3d.row(sidx).head<3>().transpose();
      ++valid_center_vertices;
    }
    if (valid_center_vertices > 0) {
      query_3d /= static_cast<double>(valid_center_vertices);
    }

    if (sample_idx < 0 ||
        sample_idx >= sample_points_uv.rows() ||
        sample_idx >= sample_points_3d.rows()) {
      sample_idx = -1;
      double best_d2 = std::numeric_limits<double>::infinity();
      for (int c = 0; c < 3; ++c) {
        const int sidx = tri_vertices[c];
        if (sidx < 0 ||
            sidx >= sample_points_uv.rows() ||
            sidx >= sample_points_3d.rows()) {
          continue;
        }
        const Eigen::Vector2d suv = sample_points_uv.row(sidx).head<2>().transpose();
        const double d2 = (suv - center_uv).squaredNorm();
        if (d2 < best_d2) {
          best_d2 = d2;
          sample_idx = sidx;
        }
      }
    }

    loaded_points_3d.push_back(query_3d);
    loaded_points_uv.push_back(center_uv);
    loaded_processing_uv.push_back(center_uv);
    loaded_triangles.push_back(resolved_tri_idx);
    loaded_triangle_vertices.push_back(tri_vertices);
    loaded_sample_indices.push_back(sample_idx);
    loaded_class_ids.push_back(class_id);
  }

  state.pattern_points_3d = std::move(loaded_points_3d);
  state.pattern_points_uv = std::move(loaded_points_uv);
  state.pattern_processing_uv = std::move(loaded_processing_uv);
  state.pattern_points_delaunay_triangle = std::move(loaded_triangles);
  state.pattern_points_delaunay_vertices = std::move(loaded_triangle_vertices);
  state.selected_sample_indices = std::move(loaded_sample_indices);
  state.pattern_point_class_ids = std::move(loaded_class_ids);
  state.last_crossed_triangle_count = -1;
  reset_voronoi_pcf(root_state);
  invalidate_transition_regions(root_state, state.region_id);
  state.selected_dirty = true;
  return true;
}

} // namespace

std::vector<int> generated_patch_family_region_indices(
  const InteractionState& state,
  int family_id) {
  std::vector<int> region_indices;
  if (family_id < 0) {
    return region_indices;
  }

  for (int region_index = 0; region_index < region_count(state); ++region_index) {
    const PatternRegionState& region = region_state(state, region_index);
    if (!region_is_generated_patch_exemplar(region) ||
        region.generated_patch_family_id != family_id) {
      continue;
    }
    region_indices.push_back(region_index);
  }

  std::sort(
    region_indices.begin(),
    region_indices.end(),
    [&](int lhs_index, int rhs_index) {
      const PatternRegionState& lhs = region_state(state, lhs_index);
      const PatternRegionState& rhs = region_state(state, rhs_index);
      if (lhs.generated_patch_index != rhs.generated_patch_index) {
        return lhs.generated_patch_index < rhs.generated_patch_index;
      }
      return lhs_index < rhs_index;
    });
  return region_indices;
}

void set_generated_patch_family_optimize_requested(
  InteractionState& state,
  int family_id,
  bool requested) {
  for (int region_index = 0; region_index < region_count(state); ++region_index) {
    PatternRegionState& region = region_state(state, region_index);
    if (!region_is_generated_patch_exemplar(region) ||
        region.generated_patch_family_id != family_id) {
      continue;
    }
    region.generated_patch_batch_optimize_requested = requested;
  }
}

void clear_generated_patch_batch_run(
  InteractionState& state,
  bool clear_optimize_requested) {
  if (clear_optimize_requested) {
    for (int region_index = 0; region_index < region_count(state); ++region_index) {
      region_state(state, region_index).generated_patch_batch_optimize_requested = false;
    }
  }
  state.generated_patch_batch_run = GeneratedPatchBatchRunState{};
  state.generated_patch_batch_cancel_requested = false;
}

bool begin_generated_patch_batch_run(
  InteractionState& state,
  int initiating_region_id,
  GeneratedPatchBatchAction action,
  int requested_point_count,
  bool current_region_started,
  bool current_region_completed) {
  if (action == GeneratedPatchBatchAction::None) {
    clear_generated_patch_batch_run(state);
    return false;
  }

  const PatternRegionState* initiating_region =
    find_region_by_id(state, initiating_region_id);
  if (!initiating_region || !region_is_generated_patch_exemplar(*initiating_region)) {
    clear_generated_patch_batch_run(state);
    return false;
  }

  const std::vector<int> family_region_indices =
    generated_patch_family_region_indices(
      state,
      initiating_region->generated_patch_family_id);
  if (family_region_indices.empty()) {
    clear_generated_patch_batch_run(state);
    return false;
  }

  GeneratedPatchBatchRunState run;
  run.active = true;
  run.family_id = initiating_region->generated_patch_family_id;
  run.initiating_region_id = initiating_region_id;
  run.action = static_cast<int>(action);
  run.current_region_started = current_region_started;
  run.current_region_completed = current_region_completed;
  run.requested_point_count = requested_point_count;
  run.region_ids.reserve(family_region_indices.size());
  for (int region_index : family_region_indices) {
    run.region_ids.push_back(region_state(state, region_index).region_id);
  }

  size_t initiating_offset = 0;
  for (size_t offset = 0; offset < run.region_ids.size(); ++offset) {
    if (run.region_ids[offset] == initiating_region_id) {
      initiating_offset = offset;
      break;
    }
  }
  run.current_region_offset = initiating_offset;

  state.generated_patch_batch_run = std::move(run);
  state.generated_patch_batch_cancel_requested = false;
  if (action == GeneratedPatchBatchAction::Optimize) {
    set_generated_patch_family_optimize_requested(
      state,
      state.generated_patch_batch_run.family_id,
      true);
  }
  return true;
}

void init_interaction(
  InteractionState& state,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F) {
  ensure_region_metadata(state);

  Eigen::MatrixXd centers = compute_face_centers(V, F);
  if (polyscope::hasPointCloud(kFaceCentersName)) {
    state.face_centers = polyscope::getPointCloud(kFaceCentersName);
    if (state.face_centers) {
      state.face_centers->updatePointPositions(centers);
    }
  } else {
    state.face_centers = polyscope::registerPointCloud(kFaceCentersName, centers);
    if (state.face_centers) {
      state.face_centers->setPointRadius(viewer_overlay_radius(state.face_centers, 0.0005), false);
    }
  }

  if (state.face_centers) {
    state.face_centers->setPointColor(glm::vec3(0.2f, 0.7f, 0.9f));
    state.face_centers->setEnabled(false);
  }
}

void reset_interaction_for_new_model(
  InteractionState& state,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh) {
  ensure_region_metadata(state);
  clear_whole_model_patch_preview(state, surfaceMesh, uvMesh);
  for (int region_index = 0; region_index < region_count(state); ++region_index) {
    remove_region_visuals(region_state(state, region_index), surfaceMesh, uvMesh);
  }
  if (polyscope::hasPointCloud(kFaceCentersName)) {
    polyscope::removePointCloud(kFaceCentersName);
  }
  state = InteractionState{};
}

void handle_interaction_input(
  InteractionState& root_state,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& sample_points_3d,
  const Eigen::MatrixXd& sample_points_uv,
  const DelaunayTraversalHelper* delaunay_helper) {
  ensure_region_metadata(root_state);
  ImGuiIO& io = ImGui::GetIO();
  if (!io.WantCaptureKeyboard &&
      !io.WantTextInput &&
      !root_state.generated_patch_batch_run.active &&
      region_count(root_state) > 1) {
    const bool previous_region_pressed = ImGui::IsKeyPressed(ImGuiKey_LeftBracket, false);
    const bool next_region_pressed = ImGui::IsKeyPressed(ImGuiKey_RightBracket, false);
    if (previous_region_pressed || next_region_pressed) {
      const int direction = next_region_pressed ? 1 : -1;
      const int count = region_count(root_state);
      root_state.active_region_index =
        (root_state.active_region_index + direction + count) % count;
    }
  }
  PatternRegionState& state = active_region(root_state);
  if (io.WantCaptureMouse) {
    clear_paint_brush_preview(state);
    return;
  }
  const bool leftDown = ImGui::IsMouseDown(ImGuiMouseButton_Left);
  const bool mouseClicked = ImGui::IsMouseClicked(ImGuiMouseButton_Left);
  if (!leftDown) {
    state.last_painted_face = -1;
    state.last_painted_input_face = -1;
  }

  glm::vec2 mousePos(io.MousePos.x, io.MousePos.y);
  auto pick = polyscope::pick::pickAtScreenCoords(mousePos);

  const bool input_paint_active = region_uses_exemplar_input(state) && state.enable_input_paint;
  const bool output_paint_active = state.enable_output_paint;
  auto resolve_paint_pick = [&](int& out_face_index, Eigen::Vector2d& out_uv) {
    out_face_index = -1;
    out_uv = Eigen::Vector2d::Zero();
    if (!pick.first || (pick.first != surfaceMesh && pick.first != uvMesh)) {
      return false;
    }
    polyscope::SurfaceMesh* pickedMesh =
      (pick.first == surfaceMesh) ? surfaceMesh : uvMesh;
    if (!pickedMesh || F.cols() < 3 || V.rows() <= 0 || UV.rows() <= 0 || V.cols() < 3 || UV.cols() < 2) {
      return false;
    }
    const size_t localPick = pick.second;
    const size_t faceStart = pickedMesh->nVertices();
    const size_t faceEnd = faceStart + pickedMesh->nFaces();
    if (localPick < faceStart || localPick >= faceEnd) {
      return false;
    }
    const int face_index = static_cast<int>(localPick - faceStart);
    if (face_index < 0 || face_index >= F.rows()) {
      return false;
    }

    const int v0 = F(face_index, 0);
    const int v1 = F(face_index, 1);
    const int v2 = F(face_index, 2);
    if (v0 < 0 || v1 < 0 || v2 < 0 ||
        v0 >= V.rows() || v1 >= V.rows() || v2 >= V.rows() ||
        v0 >= UV.rows() || v1 >= UV.rows() || v2 >= UV.rows()) {
      return false;
    }

    const glm::vec3 click_world_glm = polyscope::view::screenCoordsToWorldPosition(mousePos);
    const Eigen::Vector3d click_world(
      static_cast<double>(click_world_glm.x),
      static_cast<double>(click_world_glm.y),
      static_cast<double>(click_world_glm.z));
    const Eigen::Vector3d p3d0 = V.row(v0).head<3>().transpose();
    const Eigen::Vector3d p3d1 = V.row(v1).head<3>().transpose();
    const Eigen::Vector3d p3d2 = V.row(v2).head<3>().transpose();
    const Eigen::Vector2d puv0 = UV.row(v0).head<2>().transpose();
    const Eigen::Vector2d puv1 = UV.row(v1).head<2>().transpose();
    const Eigen::Vector2d puv2 = UV.row(v2).head<2>().transpose();

    Eigen::Vector3d bary = Eigen::Vector3d::Constant(1.0 / 3.0);
    if (pick.first == surfaceMesh) {
      bary = barycentric_from_3d(click_world, p3d0, p3d1, p3d2);
    } else {
      const Eigen::Vector2d click_uv_display(click_world.x(), click_world.y());
      bary = barycentric_from_2d(
        click_uv_display,
        uv_to_display_2d(puv0, state.uv_display_offset),
        uv_to_display_2d(puv1, state.uv_display_offset),
        uv_to_display_2d(puv2, state.uv_display_offset));
    }

    out_face_index = face_index;
    out_uv = bary[0] * puv0 + bary[1] * puv1 + bary[2] * puv2;
    return true;
  };

  int hovered_face_index = -1;
  Eigen::Vector2d hovered_brush_center_uv = Eigen::Vector2d::Zero();
  std::vector<int> hovered_brush_faces;
  if ((input_paint_active || output_paint_active) &&
      resolve_paint_pick(hovered_face_index, hovered_brush_center_uv)) {
    hovered_brush_faces = collect_face_brush_faces(
      UV,
      F,
      hovered_brush_center_uv,
      static_cast<double>(state.face_paint_brush_radius));
    update_paint_brush_preview(
      state,
      surfaceMesh,
      uvMesh,
      V,
      UV,
      F,
      hovered_brush_center_uv,
      static_cast<double>(state.face_paint_brush_radius),
      hovered_brush_faces,
      input_paint_active ? kInputPaintPreviewColor : kOutputPaintPreviewColor);
  } else {
    clear_paint_brush_preview(state);
  }

  if (!pick.first || (!leftDown && !mouseClicked)) {
    return;
  }

  if (region_uses_exemplar_input(state) && state.enable_input_selection && mouseClicked) {
    if (pick.first == surfaceMesh || pick.first == uvMesh) {
      const size_t localPick = pick.second;
      polyscope::SurfaceMesh* pickedMesh =
        (pick.first == surfaceMesh) ? surfaceMesh : uvMesh;
      if (pickedMesh) {
        const size_t faceStart = pickedMesh->nVertices();
        const size_t faceEnd = faceStart + pickedMesh->nFaces();
        if (localPick >= faceStart && localPick < faceEnd) {
          const int fidx = static_cast<int>(localPick - faceStart);
          if (fidx >= 0 && fidx < F.rows() && V.rows() > 0 && UV.rows() > 0) {
            if (F.cols() < 3) {
              return;
            }
            if (!delaunay_helper || !delaunay_helper->is_ready() ||
                sample_points_uv.rows() <= 0 || sample_points_uv.cols() < 2 ||
                sample_points_3d.rows() <= 0 || sample_points_3d.cols() < 3) {
              return;
            }

            const int v0 = F(fidx, 0);
            const int v1 = F(fidx, 1);
            const int v2 = F(fidx, 2);
            if (v0 < 0 || v1 < 0 || v2 < 0 ||
                v0 >= V.rows() || v1 >= V.rows() || v2 >= V.rows() ||
                v0 >= UV.rows() || v1 >= UV.rows() || v2 >= UV.rows()) {
              return;
            }

            const glm::vec3 click_world_glm = polyscope::view::screenCoordsToWorldPosition(mousePos);
            const Eigen::Vector3d click_world(
              static_cast<double>(click_world_glm.x),
              static_cast<double>(click_world_glm.y),
              static_cast<double>(click_world_glm.z));

            const Eigen::Vector3d p3d0 = V.row(v0).head<3>().transpose();
            const Eigen::Vector3d p3d1 = V.row(v1).head<3>().transpose();
            const Eigen::Vector3d p3d2 = V.row(v2).head<3>().transpose();
            const Eigen::Vector2d puv0 = UV.row(v0).head<2>().transpose();
            const Eigen::Vector2d puv1 = UV.row(v1).head<2>().transpose();
            const Eigen::Vector2d puv2 = UV.row(v2).head<2>().transpose();

            Eigen::Vector3d bary = Eigen::Vector3d::Constant(1.0 / 3.0);
            if (pick.first == surfaceMesh) {
              bary = barycentric_from_3d(click_world, p3d0, p3d1, p3d2);
            } else {
              const Eigen::Vector2d click_uv_display(click_world.x(), click_world.y());
              bary = barycentric_from_2d(
                click_uv_display,
                uv_to_display_2d(puv0, state.uv_display_offset),
                uv_to_display_2d(puv1, state.uv_display_offset),
                uv_to_display_2d(puv2, state.uv_display_offset));
            }

            const Eigen::Vector3d placed_3d =
              bary[0] * p3d0 + bary[1] * p3d1 + bary[2] * p3d2;
            const Eigen::Vector2d placed_uv =
              bary[0] * puv0 + bary[1] * puv1 + bary[2] * puv2;

            const bool remove_mode = io.KeyShift;
            if (remove_mode) {
              if (!state.pattern_points_uv.empty()) {
                size_t remove_idx = 0;
                double best_d2 = std::numeric_limits<double>::infinity();
                for (size_t i = 0; i < state.pattern_points_uv.size(); ++i) {
                  const double d2 = (state.pattern_points_uv[i] - placed_uv).squaredNorm();
                  if (d2 < best_d2) {
                    best_d2 = d2;
                    remove_idx = i;
                  }
                }
                erase_pattern_point_at(root_state, remove_idx);
                recompute_last_crossed_triangle_count(root_state, delaunay_helper);
                reset_voronoi_pcf(root_state);
                invalidate_transition_regions(root_state, state.region_id);
                state.selected_dirty = true;
              }
              return;
            }

            int best_delaunay_tri = -1;
            Eigen::Vector2d query_uv = Eigen::Vector2d::Zero();
            double best_center_d2 = std::numeric_limits<double>::infinity();
            const int delaunay_tri_count = delaunay_helper->triangle_count();
            for (int tri = 0; tri < delaunay_tri_count; ++tri) {
              Eigen::Vector2d center_uv = Eigen::Vector2d::Zero();
              if (!delaunay_helper->triangle_center(tri, center_uv)) {
                continue;
              }
              const double d2 = (center_uv - placed_uv).squaredNorm();
              if (d2 < best_center_d2) {
                best_center_d2 = d2;
                best_delaunay_tri = tri;
                query_uv = center_uv;
              }
            }
            if (best_delaunay_tri < 0) {
              return;
            }

            int tri_idx = -1;
            Eigen::Vector3i tri_vertices(-1, -1, -1);
            if (!delaunay_helper->find_containing_triangle(query_uv, tri_idx, tri_vertices)) {
              return;
            }

            int nearest_sample_idx = -1;
            double nearest_d2 = std::numeric_limits<double>::infinity();
            for (int c = 0; c < 3; ++c) {
              const int sidx = tri_vertices[c];
              if (sidx < 0 ||
                  sidx >= sample_points_uv.rows() ||
                  sidx >= sample_points_3d.rows()) {
                continue;
              }
              const Eigen::Vector2d suv = sample_points_uv.row(sidx).head<2>().transpose();
              const double d2 = (suv - query_uv).squaredNorm();
              if (d2 < nearest_d2) {
                nearest_d2 = d2;
                nearest_sample_idx = sidx;
              }
            }

            const auto tri_it = std::find(
              state.pattern_points_delaunay_triangle.begin(),
              state.pattern_points_delaunay_triangle.end(),
              tri_idx);
            if (tri_it != state.pattern_points_delaunay_triangle.end()) {
              // Triangle already occupied by an input point; ignore new placement.
              return;
            }

            // Snap stored placed point to the selected Delaunay triangle center.
            Eigen::Vector3d query_3d = Eigen::Vector3d::Zero();
            int valid_center_vertices = 0;
            for (int c = 0; c < 3; ++c) {
              const int sidx = tri_vertices[c];
              if (sidx < 0 || sidx >= sample_points_3d.rows() || sample_points_3d.cols() < 3) {
                continue;
              }
              query_3d += sample_points_3d.row(sidx).head<3>().transpose();
              ++valid_center_vertices;
            }
            if (valid_center_vertices > 0) {
              query_3d /= static_cast<double>(valid_center_vertices);
            } else {
              query_3d = placed_3d;
            }

            state.pattern_points_3d.push_back(query_3d);
            state.pattern_points_uv.push_back(query_uv);
            state.pattern_processing_uv.push_back(query_uv);
            state.pattern_points_delaunay_triangle.push_back(tri_idx);
            state.pattern_points_delaunay_vertices.push_back(tri_vertices);
            state.pattern_point_class_ids.push_back(0);
            state.selected_sample_indices.push_back(nearest_sample_idx);

            recompute_last_crossed_triangle_count(root_state, delaunay_helper);
            reset_voronoi_pcf(root_state);
            invalidate_transition_regions(root_state, state.region_id);
            state.selected_dirty = true;
          }
        }
      }
    }
  }

  if (region_uses_exemplar_input(state) && state.enable_input_paint && leftDown) {
    bool selectMode = io.KeyCtrl && !io.KeyAlt;
    bool deselectMode = io.KeyCtrl && io.KeyAlt;
    if (selectMode || deselectMode) {
      if (hovered_face_index >= 0) {
        const size_t painted_face_count_before = state.input_painted_face_indices.size();
        apply_face_brush(
          state.input_painted_face_indices,
          hovered_brush_faces,
          selectMode,
          deselectMode);
        if (state.input_painted_face_indices.size() != painted_face_count_before) {
          state.last_painted_input_face = hovered_face_index;
          state.input_boundary_pending_edits = true;

          // Keep the boundary preview in sync while deferring expensive
          // downstream processing until the user explicitly stops or finishes painting.
          update_input_boundary_path_from_faces(root_state, V, UV, F);
          update_input_boundary_curve(root_state, surfaceMesh, uvMesh);
        }
      }
    }
  } else if (state.enable_output_paint && leftDown) {
    bool selectMode = io.KeyCtrl && !io.KeyAlt;
    bool deselectMode = io.KeyCtrl && io.KeyAlt;
    if (selectMode || deselectMode) {
      if (hovered_face_index >= 0) {
        const size_t painted_face_count_before = state.painted_face_indices.size();
        apply_face_brush(
          state.painted_face_indices,
          hovered_brush_faces,
          selectMode,
          deselectMode);
        if (state.painted_face_indices.size() != painted_face_count_before) {
          state.last_painted_face = hovered_face_index;
          state.output_boundary_pending_edits = true;

          // Keep the boundary preview in sync while deferring expensive
          // downstream processing until the user explicitly finishes painting.
          update_output_boundary_preview_from_faces(root_state, V, UV, F);
          update_output_boundary_curve(root_state, surfaceMesh, uvMesh);
        }
      }
    }
  }

  if (state.selected_dirty) {
    update_selected_samples_clouds(root_state, surfaceMesh, uvMesh);
    state.selected_dirty = false;
  }
}

void build_interaction_ui(
  InteractionState& root_state,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh,
  const std::vector<DartSample>& samples_uv,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXd& points_3d,
  const Eigen::MatrixXd& points_uv,
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F,
  const DelaunayTraversalHelper* delaunay_helper,
  const Eigen::VectorXd* face_distortion) {
  ensure_region_metadata(root_state);
  ImGuiIO& io = ImGui::GetIO();
  const auto update_generated_patch_batch_status =
    [&](const std::string& status, bool is_error) {
      root_state.generated_patch_batch_status = status;
      root_state.generated_patch_batch_status_is_error =
        !status.empty() && is_error;
    };
  const auto sync_generated_patch_batch_active_region = [&]() {
    GeneratedPatchBatchRunState& batch_run = root_state.generated_patch_batch_run;
    if (!batch_run.active ||
        batch_run.current_region_offset >= batch_run.region_ids.size()) {
      return false;
    }
    const int region_id = batch_run.region_ids[batch_run.current_region_offset];
    const int region_index = region_index_from_id(root_state, region_id);
    if (region_index < 0) {
      return false;
    }
    const PatternRegionState& region = region_state(root_state, region_index);
    if (!region_is_generated_patch_exemplar(region) ||
        region.generated_patch_family_id != batch_run.family_id) {
      return false;
    }
    root_state.active_region_index = region_index;
    return true;
  };
  const auto advance_generated_patch_batch_region = [&]() {
    GeneratedPatchBatchRunState& batch_run = root_state.generated_patch_batch_run;
    if (!batch_run.active) {
      return false;
    }
    while (batch_run.current_region_offset + 1 < batch_run.region_ids.size()) {
      ++batch_run.current_region_offset;
      batch_run.current_region_started = false;
      batch_run.current_region_completed = false;
      if (sync_generated_patch_batch_active_region()) {
        return true;
      }
    }
    const GeneratedPatchBatchAction action =
      static_cast<GeneratedPatchBatchAction>(batch_run.action);
    const size_t completed_count = batch_run.region_ids.size();
    clear_generated_patch_batch_run(root_state, false);
    std::ostringstream status;
    if (action == GeneratedPatchBatchAction::GeneratePoints) {
      status << "Generate Points";
    } else if (action == GeneratedPatchBatchAction::Optimize) {
      status << "Optimize";
    } else {
      status << "Batch";
    }
    status << " finished across " << completed_count
           << " generated exemplar patches.";
    update_generated_patch_batch_status(status.str(), false);
    return false;
  };
  if (root_state.generated_patch_batch_run.active) {
    if (root_state.generated_patch_batch_run.current_region_completed) {
      if (root_state.generated_patch_batch_cancel_requested) {
        clear_generated_patch_batch_run(root_state);
        update_generated_patch_batch_status(
          "Generated patch batch cancelled.",
          false);
      } else {
        (void)advance_generated_patch_batch_region();
      }
    } else if (!sync_generated_patch_batch_active_region()) {
      clear_generated_patch_batch_run(root_state);
      update_generated_patch_batch_status(
        "Generated patch batch stopped because the active patch family changed.",
        true);
    }
  }
  for (int region_index = 0; region_index < region_count(root_state); ++region_index) {
    PatternRegionState& region = region_state(root_state, region_index);
    if (UV.rows() > 0 && UV.cols() >= 2) {
      region.uv_display_offset = uv_display_offset_from_vertices(UV);
    } else {
      region.uv_display_offset.setZero();
    }
  }

  if (UV.rows() > 0 && UV.cols() >= 2) {
    // UV display offsets are updated per region above.
  }

  if (root_state.face_centers) {
    root_state.face_centers->setEnabled(false);
  }

  for (int region_index = 0; region_index < region_count(root_state); ++region_index) {
    PatternRegionState& region = region_state(root_state, region_index);
    const bool is_active_region = (region_index == root_state.active_region_index);
    const bool show_exemplar_input_visuals = region_uses_exemplar_input(region);
    if (region.selected_samples_3d) {
      region.selected_samples_3d->setEnabled(
        is_active_region &&
        show_exemplar_input_visuals &&
        surfaceMesh &&
        surfaceMesh->isEnabled());
    }
    if (region.selected_samples_uv) {
      region.selected_samples_uv->setEnabled(
        is_active_region &&
        show_exemplar_input_visuals &&
        uvMesh &&
        uvMesh->isEnabled());
    }
    if (region.selected_samples_3d_class_1) {
      region.selected_samples_3d_class_1->setEnabled(
        is_active_region &&
        show_exemplar_input_visuals &&
        region_is_two_class(region) &&
        surfaceMesh &&
        surfaceMesh->isEnabled());
    }
    if (region.selected_samples_uv_class_1) {
      region.selected_samples_uv_class_1->setEnabled(
        is_active_region &&
        show_exemplar_input_visuals &&
        region_is_two_class(region) &&
        uvMesh &&
        uvMesh->isEnabled());
    }
    if (region.output_pattern_3d) {
      region.output_pattern_3d->setEnabled(surfaceMesh && surfaceMesh->isEnabled());
    }
    if (region.output_pattern_uv) {
      region.output_pattern_uv->setEnabled(uvMesh && uvMesh->isEnabled());
    }
    if (region.output_pattern_3d_class_1) {
      region.output_pattern_3d_class_1->setEnabled(
        region_is_two_class(region) && surfaceMesh && surfaceMesh->isEnabled());
    }
    if (region.output_pattern_uv_class_1) {
      region.output_pattern_uv_class_1->setEnabled(
        region_is_two_class(region) && uvMesh && uvMesh->isEnabled());
    }
    if (region.input_boundary_curve) {
      region.input_boundary_curve->setEnabled(
        is_active_region &&
        show_exemplar_input_visuals &&
        uvMesh &&
        uvMesh->isEnabled());
    }
    if (region.input_boundary_curve_3d) {
      region.input_boundary_curve_3d->setEnabled(
        is_active_region &&
        show_exemplar_input_visuals &&
        surfaceMesh &&
        surfaceMesh->isEnabled());
    }
    if (region.input_reference_3d) {
      region.input_reference_3d->setEnabled(
        is_active_region &&
        show_exemplar_input_visuals &&
        surfaceMesh &&
        surfaceMesh->isEnabled());
    }
    if (region.input_reference_uv) {
      region.input_reference_uv->setEnabled(
        is_active_region &&
        show_exemplar_input_visuals &&
        uvMesh &&
        uvMesh->isEnabled());
    }
    if (region.output_boundary_curve_3d) {
      region.output_boundary_curve_3d->setEnabled(surfaceMesh && surfaceMesh->isEnabled());
    }
    if (region.output_boundary_curve_uv) {
      region.output_boundary_curve_uv->setEnabled(uvMesh && uvMesh->isEnabled());
    }
  }

  if (!ImGui::CollapsingHeader("Interaction", ImGuiTreeNodeFlags_DefaultOpen)) {
    return;
  }

  ImGui::Indent();
  ImGui::TextUnformatted("Active region:");
  for (int region_index = 0; region_index < region_count(root_state); ++region_index) {
    if (region_index > 0) {
      ImGui::SameLine();
    }
    const bool is_active = (root_state.active_region_index == region_index);
    const std::string region_label = pattern_region_label(root_state, region_index);
    if (is_active) {
      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.25f, 0.55f, 0.9f, 1.0f));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.35f, 0.65f, 1.0f, 1.0f));
    }
    const std::string button_label =
      std::string("Region ") + region_label + "###region_select_" + std::to_string(region_index);
    if (ImGui::Button(button_label.c_str(), ImVec2(90, 0))) {
      root_state.active_region_index = region_index;
    }
    if (is_active) {
      ImGui::PopStyleColor(2);
    }
  }
  ImGui::SameLine();
  if (ImGui::Button("Add Region", ImVec2(110, 0))) {
    root_state.regions.emplace_back();
    ensure_region_identity(
      root_state,
      root_state.regions.back(),
      static_cast<int>(root_state.regions.size()) - 1);
    root_state.active_region_index = static_cast<int>(root_state.regions.size()) - 1;
  }
  ImGui::TextDisabled("Shortcut: [ previous region, ] next region");
  for (int region_index = 0; region_index < region_count(root_state); ++region_index) {
    const PatternRegionState& region = region_state(root_state, region_index);
    const std::string region_label = pattern_region_label(root_state, region_index);
    const char* region_role = pattern_region_mode_label(region);
    ImGui::BulletText(
      "Region %s (%s): input=%zu, refs=%zu, output=%zu",
      region_label.c_str(),
      region_role,
      region.pattern_points_uv.size(),
      region.input_reference_indices.size(),
      region.output_pattern_points_uv.size());
  }

  ImGui::Spacing();
  ImGui::TextUnformatted("Output pattern export:");
  ImGui::SetNextItemWidth(220.0f);
  ImGui::InputText(
    "Directory##output_pattern_export_dir",
    root_state.output_pattern_export_dir,
    sizeof(root_state.output_pattern_export_dir));
  if (ImGui::Button("Save Output Patterns", ImVec2(-1, 0))) {
    std::string status;
    int saved_count = 0;
    if (save_output_patterns_by_region(
          std::string(root_state.output_pattern_export_dir),
          root_state,
          status,
          saved_count)) {
      root_state.output_pattern_export_status = status;
      root_state.output_pattern_export_status_is_error = false;
    } else {
      root_state.output_pattern_export_status = status;
      root_state.output_pattern_export_status_is_error = true;
    }
  }
  if (!root_state.output_pattern_export_status.empty()) {
    const ImVec4 color = root_state.output_pattern_export_status_is_error
      ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f)
      : ImVec4(0.5f, 0.9f, 0.5f, 1.0f);
    ImGui::TextColored(color, "%s", root_state.output_pattern_export_status.c_str());
  }
  ImGui::Spacing();

  PatternRegionState& state = active_region(root_state);
  static int whole_model_patch_count = 4;
  static int whole_model_patch_source_region_id = -1;
  static int whole_model_patch_balance_mode = 0;
  static int whole_model_patch_support_gap_steps = 1;
  const auto set_generated_patch_batch_status =
    [&](const std::string& status, bool is_error) {
      root_state.generated_patch_batch_status = status;
      root_state.generated_patch_batch_status_is_error =
        !status.empty() && is_error;
    };
  const auto cancel_generated_patch_batch =
    [&](const std::string& status = std::string()) {
      clear_generated_patch_batch_run(root_state);
      root_state.generated_patch_batch_cancel_requested = false;
      if (!status.empty()) {
        set_generated_patch_batch_status(status, false);
      }
    };
  const auto start_generated_patch_batch =
    [&](GeneratedPatchBatchAction action,
        bool current_region_started,
        bool current_region_completed,
        int requested_point_count,
        const char* action_label) {
      if (!region_is_generated_patch_exemplar(state)) {
        set_generated_patch_batch_status(
          "Generated patch batching is only available on whole-model exemplar patches.",
          true);
        return false;
      }
      const std::vector<int> family_region_indices =
        generated_patch_family_region_indices(
          root_state,
          state.generated_patch_family_id);
      if (family_region_indices.size() < 2) {
        set_generated_patch_batch_status(
          "This generated patch family needs at least two exemplar regions.",
          true);
        return false;
      }
      if (!begin_generated_patch_batch_run(
            root_state,
            state.region_id,
            action,
            requested_point_count,
            current_region_started,
            current_region_completed)) {
        set_generated_patch_batch_status(
          "Unable to start generated patch batch.",
          true);
        return false;
      }
      std::ostringstream status;
      status << action_label << " across " << family_region_indices.size()
             << " generated exemplar patches.";
      set_generated_patch_batch_status(status.str(), false);
      return true;
    };
  const auto compute_input_pcf_for_generated_patch_family = [&]() {
    if (!region_is_generated_patch_exemplar(state)) {
      set_generated_patch_batch_status(
        "Generated patch batching is only available on whole-model exemplar patches.",
        true);
      return false;
    }
    const std::vector<int> family_region_indices =
      generated_patch_family_region_indices(
        root_state,
        state.generated_patch_family_id);
    if (family_region_indices.size() < 2) {
      set_generated_patch_batch_status(
        "This generated patch family needs at least two exemplar regions.",
        true);
      return false;
    }
    const int saved_active_region_index = root_state.active_region_index;
    int effective_bin_updates = 0;
    for (int region_index : family_region_indices) {
      root_state.active_region_index = region_index;
      compute_voronoi_pcf_histogram(root_state, delaunay_helper);
      PatternRegionState& family_region = active_region(root_state);
      if (!family_region.voronoi_pcf_ready) {
        continue;
      }
      const int effective_bins = std::max(1, family_region.voronoi_pcf_max_k + 1);
      if (effective_bins > 0 &&
          effective_bins < family_region.voronoi_pcf_bin_count) {
        family_region.voronoi_pcf_bin_count = effective_bins;
        reset_voronoi_pcf(root_state);
        compute_voronoi_pcf_histogram(root_state, delaunay_helper);
        ++effective_bin_updates;
      }
    }
    root_state.active_region_index = saved_active_region_index;
    std::ostringstream status;
    status << "Computed input PCF for " << family_region_indices.size()
           << " generated exemplar patches";
    if (effective_bin_updates > 0) {
      status << " and set effective bins on " << effective_bin_updates
             << " of them";
    }
    status << ".";
    set_generated_patch_batch_status(status.str(), false);
    return true;
  };
  const std::string active_region_title =
    std::string("Editing Region ") + pattern_region_label(root_state, root_state.active_region_index);
  ImGui::TextUnformatted(active_region_title.c_str());
  ImGui::SetNextItemWidth(180.0f);
  {
    const std::string region_name_id =
      std::string("Region name##region_name_") + std::to_string(state.region_id);
    ImGui::InputText(region_name_id.c_str(), state.display_name, sizeof(state.display_name));
  }
  ImGui::SameLine();
  if (region_count(root_state) > 1 &&
      ImGui::Button("Remove Region", ImVec2(120, 0))) {
    const int remove_index = root_state.active_region_index;
    cancel_generated_patch_batch("Generated patch batch cancelled after region removal.");
    remove_region_visuals(state, surfaceMesh, uvMesh);
    root_state.regions.erase(root_state.regions.begin() + static_cast<std::ptrdiff_t>(remove_index));
    root_state.active_region_index =
      std::min(remove_index, region_count(root_state) - 1);
    ensure_region_metadata(root_state);
    invalidate_transition_regions(root_state);
    return;
  }
  ImGui::Spacing();

  state.region_mode = static_cast<int>(PatternRegionMode::Exemplar);
  state.transition_source_a_region_id = -1;
  state.transition_source_b_region_id = -1;
  state.active_pattern_class_id = 0;

  ImGui::TextUnformatted("Pattern creation:");
  ImGui::BulletText("Enable pattern creation, then click mesh/UV faces to place points.");
  ImGui::BulletText("Shift + left-click on mesh/UV faces to remove the nearest placed point.");
  ImGui::BulletText("Click is used to choose a containing sample-Delaunay triangle in UV.");
  ImGui::BulletText("Placed points are exact click positions; processing uses nearest Delaunay triangle centers.");
  ImGui::Spacing();

  ImGui::Checkbox("Enable pattern creation", &state.enable_input_selection);
  ImGui::SameLine();
  if (ImGui::Button("Load", ImVec2(56, 0))) {
    std::string error;
    if (load_input_pattern_from_file(
          std::string(state.pattern_file_path),
          root_state,
          delaunay_helper,
          points_3d,
          points_uv,
          error)) {
      state.pattern_file_status =
        "Loaded pattern from " + std::string(state.pattern_file_path);
      state.pattern_file_status_is_error = false;
    } else {
      state.pattern_file_status = error;
      state.pattern_file_status_is_error = true;
    }
  }
  ImGui::SameLine();
  if (ImGui::Button("Save", ImVec2(56, 0))) {
    std::string error;
    if (save_input_pattern_to_file(std::string(state.pattern_file_path), root_state, error)) {
      state.pattern_file_status =
        "Saved pattern to " + std::string(state.pattern_file_path);
      state.pattern_file_status_is_error = false;
    } else {
      state.pattern_file_status = error;
      state.pattern_file_status_is_error = true;
    }
  }
  ImGui::TextUnformatted("Pattern file:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(220.0f);
  {
    const std::string pattern_file_input_id =
      std::string("##pattern_file_region_") + std::to_string(root_state.active_region_index);
    ImGui::InputText(
      pattern_file_input_id.c_str(),
      state.pattern_file_path,
      sizeof(state.pattern_file_path));
  }
  if (!state.pattern_file_status.empty()) {
    const ImVec4 color = state.pattern_file_status_is_error
      ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f)
      : ImVec4(0.5f, 0.9f, 0.5f, 1.0f);
    ImGui::TextColored(color, "%s", state.pattern_file_status.c_str());
  }
  ImGui::Text("Placed pattern points: %zu", state.pattern_points_uv.size());
  ImGui::Text("Input reference: %zu", state.input_reference_indices.size());
  if (!state.pattern_points_delaunay_triangle.empty()) {
    const int tri = state.pattern_points_delaunay_triangle.back();
    ImGui::Text("Last point triangle id: %d", tri);
  }
  if (state.last_crossed_triangle_count >= 0) {
    ImGui::Text("Triangles crossed (last segment): %d", state.last_crossed_triangle_count);
  }

  if (ImGui::Button("Clear pattern points (toggle off)", ImVec2(-1, 0))) {
    state.pattern_points_3d.clear();
    state.pattern_points_uv.clear();
    state.pattern_processing_uv.clear();
    state.pattern_points_delaunay_triangle.clear();
    state.pattern_points_delaunay_vertices.clear();
    state.pattern_point_class_ids.clear();
    state.last_crossed_triangle_count = -1;
    reset_voronoi_pcf(root_state);
    state.selected_sample_indices.clear();
    state.input_reference_indices.clear();
    state.enable_input_selection = false;
    state.selected_dirty = true;
    state.input_boundary_dirty = true;
    state.input_boundary_pending_edits = false;
    invalidate_transition_regions(root_state, state.region_id);
  }

  if (region_is_generated_patch_exemplar(state)) {
    const std::vector<int> family_region_indices =
      generated_patch_family_region_indices(
        root_state,
        state.generated_patch_family_id);
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextUnformatted("Generated Patch Batch:");
    ImGui::TextDisabled(
      "Sequential only. These family actions run one patch at a time.");
    ImGui::Text(
      "Patch family: %zu generated exemplar regions",
      family_region_indices.size());

    if (family_region_indices.size() > 1) {
      if (ImGui::Button("Compute Input PCF For Patch Family", ImVec2(-1, 0))) {
        (void)compute_input_pcf_for_generated_patch_family();
      }
      if (ImGui::Button("Generate Patch Family Sequentially", ImVec2(-1, 0))) {
        (void)start_generated_patch_batch(
          GeneratedPatchBatchAction::GeneratePoints,
          false,
          false,
          -1,
          "Sequentially generating points");
      }
      if (ImGui::Button("Optimize Patch Family Sequentially", ImVec2(-1, 0))) {
        (void)start_generated_patch_batch(
          GeneratedPatchBatchAction::Optimize,
          false,
          false,
          -1,
          "Sequentially optimizing");
      }
    }

    const GeneratedPatchBatchRunState& batch_run = root_state.generated_patch_batch_run;
    if (batch_run.active && batch_run.family_id == state.generated_patch_family_id) {
      const char* action_label = "Batch";
      if (batch_run.action == static_cast<int>(GeneratedPatchBatchAction::GeneratePoints)) {
        action_label = "Generate";
      } else if (batch_run.action == static_cast<int>(GeneratedPatchBatchAction::Optimize)) {
        action_label = "Optimize";
      }
      const size_t current_step =
        std::min(batch_run.current_region_offset + 1, batch_run.region_ids.size());
      ImGui::TextDisabled(
        "%s step %zu / %zu",
        action_label,
        current_step,
        batch_run.region_ids.size());
      if (root_state.generated_patch_batch_cancel_requested) {
        ImGui::TextDisabled("Cancellation requested. The current patch will stop before the batch advances.");
      } else if (ImGui::Button("Cancel Generated Patch Batch", ImVec2(-1, 0))) {
        root_state.generated_patch_batch_cancel_requested = true;
        set_generated_patch_batch_status(
          "Stopping generated patch batch after the current patch.",
          false);
      }
    }

    if (!root_state.generated_patch_batch_status.empty()) {
      const ImVec4 color = root_state.generated_patch_batch_status_is_error
        ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f)
        : ImVec4(0.5f, 0.9f, 0.5f, 1.0f);
      ImGui::TextColored(color, "%s", root_state.generated_patch_batch_status.c_str());
    }
  }

  draw_voronoi_pcf_ui(root_state, delaunay_helper, points_3d, points_uv);

  ImGui::Separator();
  double uv_brush_radius_max = 0.25;
  if (UV.rows() > 0 && UV.cols() >= 2) {
    const Eigen::Vector2d uv_min(
      UV.col(0).minCoeff(),
      UV.col(1).minCoeff());
    const Eigen::Vector2d uv_max(
      UV.col(0).maxCoeff(),
      UV.col(1).maxCoeff());
    uv_brush_radius_max = std::max(0.05, 0.15 * (uv_max - uv_min).norm());
  }
  ImGui::TextUnformatted("Paint brush radius (UV units, literal hover circle):");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(120.0f);
  ImGui::SliderFloat(
    "##face_paint_brush_radius",
    &state.face_paint_brush_radius,
    0.0f,
    static_cast<float>(uv_brush_radius_max),
    "%.2f");

  if (region_uses_exemplar_input(state)) {
    ImGui::TextUnformatted("Input boundary (paint region in 3D/UV):");
    ImGui::BulletText("Ctrl + left-drag to select faces");
    ImGui::BulletText("Ctrl + Option + left-drag to deselect faces");
    ImGui::TextUnformatted("Input paint:");
    ImGui::SameLine();
    ImGui::TextColored(
      state.enable_input_paint ? ImVec4(0.5f, 0.9f, 0.5f, 1.0f) : ImVec4(0.8f, 0.8f, 0.8f, 1.0f),
      "%s",
      state.enable_input_paint ? "ON" : "off");
    ImGui::SameLine();
    if (ImGui::Button(state.enable_input_paint ? "Stop##input_paint" : "Start##input_paint", ImVec2(70, 0))) {
      const bool was_input_paint_enabled = state.enable_input_paint;
      state.enable_input_paint = !state.enable_input_paint;
      if (state.enable_input_paint) {
        if (state.enable_output_paint && state.output_boundary_pending_edits) {
          commit_output_boundary_preview(state);
          state.output_boundary_dirty = true;
          state.output_boundary_pending_edits = false;
        }
        state.enable_output_paint = false;
      } else if (was_input_paint_enabled && state.input_boundary_pending_edits) {
        state.input_boundary_dirty = true;
        state.input_boundary_pending_edits = false;
      }
    }

    ImGui::SameLine();
    if (ImGui::Button("Finish##input_boundary", ImVec2(70, 0))) {
      state.input_boundary_dirty = true;
      state.input_boundary_pending_edits = false;
    }

    ImGui::SameLine();
    if (ImGui::Button("Clear##input_boundary", ImVec2(70, 0))) {
      state.input_painted_face_indices.clear();
      state.input_boundary_uv.resize(0, 2);
      state.input_boundary_3d.resize(0, 3);
      state.input_reference_indices.clear();
      state.input_boundary_dirty = true;
      state.input_boundary_pending_edits = false;
    }

    ImGui::Separator();
  }

  ImGui::TextUnformatted("Output boundary (paint region in 3D/UV):");
  ImGui::BulletText("Ctrl + left-drag to select faces");
  ImGui::BulletText("Ctrl + Option + left-drag to deselect faces");
  ImGui::TextUnformatted("Output paint:");
  ImGui::SameLine();
  ImGui::TextColored(
    state.enable_output_paint ? ImVec4(0.5f, 0.9f, 0.5f, 1.0f) : ImVec4(0.8f, 0.8f, 0.8f, 1.0f),
    "%s",
    state.enable_output_paint ? "ON" : "off");
  ImGui::SameLine();
  if (ImGui::Button(state.enable_output_paint ? "Stop##output_paint" : "Start##output_paint", ImVec2(70, 0))) {
    const bool was_output_paint_enabled = state.enable_output_paint;
    state.enable_output_paint = !state.enable_output_paint;
    if (state.enable_output_paint) {
      if (state.enable_input_paint && state.input_boundary_pending_edits) {
        state.input_boundary_dirty = true;
        state.input_boundary_pending_edits = false;
      }
      state.enable_input_paint = false;
    } else if (was_output_paint_enabled && state.output_boundary_pending_edits) {
      commit_output_boundary_preview(state);
      state.output_boundary_dirty = true;
      state.output_boundary_pending_edits = false;
    }
  }

  ImGui::SameLine();
  if (ImGui::Button("Finish##output_boundary", ImVec2(70, 0))) {
    if (state.output_boundary_pending_edits) {
      commit_output_boundary_preview(state);
      state.output_boundary_dirty = true;
      state.output_boundary_pending_edits = false;
    }
    state.enable_output_paint = false;
  }
  ImGui::SameLine();
  if (ImGui::Button("Clear##output_boundary", ImVec2(70, 0))) {
    state.painted_face_indices.clear();
    state.output_boundary_preview_uv_poly.resize(0, 2);
    state.output_boundary_preview_3d_poly.resize(0, 3);
    state.output_boundary_dirty = true;
    state.output_boundary_pending_edits = false;
  }
  ImGui::SameLine();
  if (ImGui::Button("Whole model##output_boundary", ImVec2(110, 0))) {
    whole_model_patch_source_region_id = state.region_id;
    root_state.whole_model_patch_preview_source_region_id = state.region_id;
    whole_model_patch_count = std::max(1, whole_model_patch_count);
    ImGui::OpenPopup("Whole Model Patches");
  }

  if (ImGui::BeginPopupModal("Whole Model Patches", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    static const char* kWholeModelPatchBalanceLabels[] = {
      "Triangle count",
      "UV area"
    };
    ImGui::TextWrapped(
      "Create almost-equal non-overlapping whole-model patches as exemplar regions that reuse this region's input PCF.");
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputInt("Patch count", &whole_model_patch_count);
    ImGui::SetNextItemWidth(180.0f);
    ImGui::Combo(
      "Balance by",
      &whole_model_patch_balance_mode,
      kWholeModelPatchBalanceLabels,
      IM_ARRAYSIZE(kWholeModelPatchBalanceLabels));
    ImGui::SetNextItemWidth(180.0f);
    ImGui::SliderInt(
      "Support center gap",
      &whole_model_patch_support_gap_steps,
      0,
      4);
    ImGui::TextDisabled(
      "1 rejects support centers closer than about one local center spacing to the patch boundary.");
    whole_model_patch_count = std::clamp(
      whole_model_patch_count,
      1,
      std::max(1, static_cast<int>(F.rows())));
    whole_model_patch_balance_mode = std::clamp(whole_model_patch_balance_mode, 0, 1);
    whole_model_patch_support_gap_steps = std::clamp(whole_model_patch_support_gap_steps, 0, 4);

    std::vector<std::vector<int>> preview_face_partitions;
    const int preview_source_region_index =
      region_index_from_id(root_state, whole_model_patch_source_region_id);
    const bool preview_source_valid =
      preview_source_region_index >= 0 &&
      region_is_exemplar(region_state(root_state, preview_source_region_index));
    if (preview_source_valid) {
      const Eigen::MatrixXd face_centers_uv = compute_face_centers_uv(UV, F);
      Eigen::VectorXd face_weights = Eigen::VectorXd::Ones(F.rows());
      if (whole_model_patch_balance_mode == 1) {
        face_weights = compute_face_areas_uv(UV, F);
      }
      preview_face_partitions = partition_face_indices_into_patches(
        F,
        face_centers_uv,
        face_weights,
        whole_model_patch_count);
      update_whole_model_patch_preview(
        root_state,
        surfaceMesh,
        uvMesh,
        V,
        UV,
        F,
        preview_face_partitions);
    } else {
      clear_whole_model_patch_preview(root_state, surfaceMesh, uvMesh);
    }

    ImGui::Spacing();
    if (preview_source_valid) {
      const int preview_patch_count = static_cast<int>(preview_face_partitions.size());
      int covered_face_count = 0;
      for (const auto& patch_faces : preview_face_partitions) {
        covered_face_count += static_cast<int>(patch_faces.size());
      }
      ImGui::Text("Preview patches: %d", preview_patch_count);
      ImGui::Text("Assigned faces: %d / %d", covered_face_count, static_cast<int>(F.rows()));
      ImGui::TextDisabled(
        "Preview updates live while this dialog is open. Support gap is applied later during point generation.");
    } else {
      ImGui::TextColored(
        ImVec4(1.0f, 0.45f, 0.45f, 1.0f),
        "Whole-model patch preview requires an exemplar source region.");
    }

    const auto refresh_region_input_visuals = [&](int region_index) {
      const int saved_region_index = root_state.active_region_index;
      root_state.active_region_index = region_index;
      update_selected_samples_clouds(root_state, surfaceMesh, uvMesh);
      update_input_boundary_curve(root_state, surfaceMesh, uvMesh);
      update_input_reference_clouds(root_state, points_3d, points_uv, surfaceMesh, uvMesh);
      update_input_boundary_quantities(root_state, surfaceMesh, uvMesh, F);
      root_state.active_region_index = saved_region_index;
    };

    const auto create_whole_model_patch_regions = [&]() {
      const int source_region_index =
        region_index_from_id(root_state, whole_model_patch_source_region_id);
      if (source_region_index < 0) {
        return false;
      }

      const PatternRegionState source_snapshot = region_state(root_state, source_region_index);
      if (!region_is_exemplar(source_snapshot)) {
        return false;
      }

      const std::vector<std::vector<int>>& face_partitions =
        root_state.whole_model_patch_preview_face_partitions;
      if (face_partitions.empty()) {
        return false;
      }

      clear_generated_patch_batch_run(root_state);
      root_state.generated_patch_batch_cancel_requested = false;
      const int generated_patch_family_id =
        root_state.next_generated_patch_family_id++;
      const std::vector<PatchInterfaceSegments> patch_interface_segments =
        build_patch_interface_segments_uv(face_partitions, UV, F);

      const std::string source_label = pattern_region_label(root_state, source_region_index);
      remove_region_visuals(source_snapshot, surfaceMesh, uvMesh);

      PatternRegionState rebuilt_source = clone_patch_region_template(source_snapshot);
      rebuilt_source.region_id = source_snapshot.region_id;
      root_state.regions[static_cast<size_t>(source_region_index)] = std::move(rebuilt_source);

      for (size_t patch_index = 0; patch_index < face_partitions.size(); ++patch_index) {
        PatternRegionState* target_region = nullptr;
        int target_region_index = -1;

        if (patch_index == 0) {
          target_region_index = source_region_index;
          target_region = &region_state(root_state, source_region_index);
        } else {
          root_state.regions.push_back(clone_patch_region_template(source_snapshot));
          target_region_index = static_cast<int>(root_state.regions.size()) - 1;
          target_region = &root_state.regions.back();
          ensure_region_identity(root_state, *target_region, target_region_index);
        }

        target_region->region_mode = static_cast<int>(PatternRegionMode::Exemplar);
        target_region->generated_patch_family_id = generated_patch_family_id;
        target_region->generated_patch_source_region_id = source_snapshot.region_id;
        target_region->generated_patch_index = static_cast<int>(patch_index);
        target_region->generated_patch_support_gap_steps = whole_model_patch_support_gap_steps;
        target_region->painted_face_indices = face_partitions[patch_index];
        target_region->generated_patch_interface_segment_uv_starts.clear();
        target_region->generated_patch_interface_segment_uv_ends.clear();
        if (patch_index < patch_interface_segments.size()) {
          target_region->generated_patch_interface_segment_uv_starts =
            patch_interface_segments[patch_index].segment_uv_starts;
          target_region->generated_patch_interface_segment_uv_ends =
            patch_interface_segments[patch_index].segment_uv_ends;
        }
        target_region->selected_dirty = false;
        target_region->input_boundary_dirty = false;
        target_region->output_boundary_dirty = true;
        target_region->output_pattern_dirty = false;
        target_region->transition_source_a_region_id = -1;
        target_region->transition_source_b_region_id = -1;
        target_region->enable_input_selection = false;
        target_region->enable_input_paint = false;
        target_region->enable_output_paint = false;
        target_region->last_painted_face = -1;
        target_region->last_painted_input_face = -1;

        const std::string patch_label = source_label + " P" + std::to_string(patch_index + 1);
        std::snprintf(
          target_region->display_name,
          sizeof(target_region->display_name),
          "%s",
          patch_label.c_str());

        refresh_region_input_visuals(target_region_index);
      }

      root_state.active_region_index = source_region_index;
      ensure_region_metadata(root_state);
      invalidate_transition_regions(root_state, source_snapshot.region_id);
      clear_whole_model_patch_preview(root_state, surfaceMesh, uvMesh);
      set_generated_patch_batch_status(
        "Created generated exemplar patches. Enable batch mode to mirror synthesis actions across siblings.",
        false);
      std::cout << "Created " << face_partitions.size()
                << " whole-model exemplar patch regions from region "
                << source_label << "\n";
      return true;
    };

    if (ImGui::Button("Create Regions From Preview", ImVec2(220, 0))) {
      const bool created = create_whole_model_patch_regions();
      whole_model_patch_source_region_id = -1;
      root_state.whole_model_patch_preview_source_region_id = -1;
      ImGui::CloseCurrentPopup();
      if (created) {
        return;
      }
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(100, 0))) {
      whole_model_patch_source_region_id = -1;
      root_state.whole_model_patch_preview_source_region_id = -1;
      clear_whole_model_patch_preview(root_state, surfaceMesh, uvMesh);
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  } else if (root_state.whole_model_patch_preview_active) {
    clear_whole_model_patch_preview(root_state, surfaceMesh, uvMesh);
  }

  ImGui::Unindent();

  const int saved_active_region_index = root_state.active_region_index;
  for (int region_index = 0; region_index < region_count(root_state); ++region_index) {
    root_state.active_region_index = region_index;
    PatternRegionState& region = active_region(root_state);
    const bool active_paint_drag =
      region_index == root_state.active_region_index &&
      !io.WantCaptureMouse &&
      ImGui::IsMouseDown(ImGuiMouseButton_Left) &&
      ((region_uses_exemplar_input(region) && region.enable_input_paint) ||
       region.enable_output_paint);

    if (region.selected_dirty) {
      update_selected_samples_clouds(root_state, surfaceMesh, uvMesh);
      region.selected_dirty = false;
    }

    if (region.output_pattern_dirty) {
      update_output_pattern_clouds(root_state, surfaceMesh, uvMesh);
      region.output_pattern_dirty = false;
    }

    if (region.input_boundary_dirty && !active_paint_drag) {
      reset_voronoi_pcf(root_state);
      update_input_boundary_from_faces(root_state, V, UV, F, samples_uv);
      update_input_boundary_curve(root_state, surfaceMesh, uvMesh);
      update_input_reference_clouds(root_state, points_3d, points_uv, surfaceMesh, uvMesh);
      update_input_boundary_quantities(root_state, surfaceMesh, uvMesh, F);
      region.input_boundary_dirty = false;
      region.input_boundary_pending_edits = false;
      invalidate_transition_regions(root_state, region.region_id);
    }

    if (region.output_boundary_dirty && !active_paint_drag) {
      clear_region_output_state(region);
      region.output_pattern_dirty = true;

      update_output_boundary_from_faces(root_state, V, UV, F);
      update_output_boundary_curve(root_state, surfaceMesh, uvMesh);
      update_output_boundary_quantities(root_state, surfaceMesh, uvMesh, F);
      region.output_boundary_dirty = false;
      region.output_boundary_pending_edits = false;
      region.output_boundary_preview_uv_poly.resize(0, 2);
      region.output_boundary_preview_3d_poly.resize(0, 3);
      invalidate_transition_regions(root_state, region.region_id);
    }
  }
  root_state.active_region_index = saved_active_region_index;
}
