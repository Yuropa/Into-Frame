#include "voronoi-pcf.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <imgui.h>
#include <polyscope/polyscope.h>

#include "interaction.h"
#include "lloyd_relaxation.h"

namespace {

bool point_in_or_on_polygon_for_pcf(
  const Eigen::Vector2d& uv,
  const Eigen::MatrixXd& boundary_poly);

bool collect_triangle_center_candidates_in_polygon(
  const Eigen::MatrixXd& boundary_poly,
  const DelaunayTraversalHelper* delaunay_helper,
  std::vector<Eigen::Vector2d>& out_uv_centers,
  std::vector<int>* out_triangle_indices = nullptr);

bool collect_output_triangle_center_candidates(
  const InteractionState& state,
  const DelaunayTraversalHelper* delaunay_helper,
  std::vector<Eigen::Vector2d>& out_uv_centers,
  std::vector<int>* out_triangle_indices = nullptr);

double point_to_segment_distance_uv(
  const Eigen::Vector2d& point_uv,
  const Eigen::Vector2d& segment_start_uv,
  const Eigen::Vector2d& segment_end_uv) {
  const Eigen::Vector2d segment = segment_end_uv - segment_start_uv;
  const double segment_length_sq = segment.squaredNorm();
  if (segment_length_sq <= 1e-20) {
    return (point_uv - segment_start_uv).norm();
  }
  const double t = std::clamp(
    (point_uv - segment_start_uv).dot(segment) / segment_length_sq,
    0.0,
    1.0);
  const Eigen::Vector2d closest_uv = segment_start_uv + t * segment;
  return (point_uv - closest_uv).norm();
}

double min_distance_to_generated_patch_interface_uv(
  const PatternRegionState& state,
  const Eigen::Vector2d& point_uv) {
  const size_t segment_count = std::min(
    state.generated_patch_interface_segment_uv_starts.size(),
    state.generated_patch_interface_segment_uv_ends.size());
  double min_distance = std::numeric_limits<double>::infinity();
  for (size_t segment_index = 0; segment_index < segment_count; ++segment_index) {
    min_distance = std::min(
      min_distance,
      point_to_segment_distance_uv(
        point_uv,
        state.generated_patch_interface_segment_uv_starts[segment_index],
        state.generated_patch_interface_segment_uv_ends[segment_index]));
  }
  return min_distance;
}

double triangle_center_local_spacing_uv(
  const DelaunayTraversalHelper* delaunay_helper,
  int triangle_index,
  const Eigen::Vector2d& center_uv) {
  if (!delaunay_helper || !delaunay_helper->is_ready() || triangle_index < 0) {
    return std::numeric_limits<double>::infinity();
  }

  std::array<int, 3> neighbors = { -1, -1, -1 };
  delaunay_helper->get_triangle_neighbors(triangle_index, neighbors);
  double min_spacing = std::numeric_limits<double>::infinity();
  for (int neighbor_triangle_index : neighbors) {
    if (neighbor_triangle_index < 0) {
      continue;
    }
    Eigen::Vector2d neighbor_center_uv = Eigen::Vector2d::Zero();
    if (!delaunay_helper->triangle_center(neighbor_triangle_index, neighbor_center_uv)) {
      continue;
    }
    const double spacing = (center_uv - neighbor_center_uv).norm();
    if (spacing > 1e-8) {
      min_spacing = std::min(min_spacing, spacing);
    }
  }
  return min_spacing;
}

void filter_generated_patch_output_support_candidates(
  const PatternRegionState& state,
  const DelaunayTraversalHelper* delaunay_helper,
  std::vector<Eigen::Vector2d>& inout_uv_centers,
  std::vector<int>* inout_triangle_indices) {
  if (!region_is_generated_patch_exemplar(state) ||
      state.generated_patch_support_gap_steps <= 0 ||
      !delaunay_helper ||
      !delaunay_helper->is_ready() ||
      !inout_triangle_indices ||
      inout_triangle_indices->size() != inout_uv_centers.size() ||
      state.generated_patch_interface_segment_uv_starts.empty() ||
      state.generated_patch_interface_segment_uv_ends.empty()) {
    return;
  }

  const size_t candidate_count = inout_uv_centers.size();
  std::vector<double> local_spacings(candidate_count, std::numeric_limits<double>::infinity());
  std::vector<double> valid_local_spacings;
  valid_local_spacings.reserve(candidate_count);
  for (size_t candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
    const int triangle_index = (*inout_triangle_indices)[candidate_index];
    const double local_spacing = triangle_center_local_spacing_uv(
      delaunay_helper,
      triangle_index,
      inout_uv_centers[candidate_index]);
    local_spacings[candidate_index] = local_spacing;
    if (std::isfinite(local_spacing) && local_spacing > 1e-8) {
      valid_local_spacings.push_back(local_spacing);
    }
  }

  double fallback_local_spacing = 0.0;
  if (!valid_local_spacings.empty()) {
    const size_t median_index = valid_local_spacings.size() / 2;
    std::nth_element(
      valid_local_spacings.begin(),
      valid_local_spacings.begin() + static_cast<std::ptrdiff_t>(median_index),
      valid_local_spacings.end());
    fallback_local_spacing = valid_local_spacings[median_index];
  }
  if (fallback_local_spacing <= 1e-8) {
    return;
  }

  std::vector<Eigen::Vector2d> kept_uv_centers;
  std::vector<int> kept_triangle_indices;
  kept_uv_centers.reserve(candidate_count);
  kept_triangle_indices.reserve(candidate_count);
  size_t fallback_best_index = candidate_count;
  double fallback_best_clearance_ratio = -std::numeric_limits<double>::infinity();

  for (size_t candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
    double local_spacing = local_spacings[candidate_index];
    if (!std::isfinite(local_spacing) || local_spacing <= 1e-8) {
      local_spacing = fallback_local_spacing;
    }
    const double boundary_distance = min_distance_to_generated_patch_interface_uv(
      state,
      inout_uv_centers[candidate_index]);
    const double clearance_ratio = boundary_distance / std::max(local_spacing, 1e-8);
    if (clearance_ratio > fallback_best_clearance_ratio) {
      fallback_best_clearance_ratio = clearance_ratio;
      fallback_best_index = candidate_index;
    }
    if (boundary_distance + 1e-12 <
        static_cast<double>(state.generated_patch_support_gap_steps) * local_spacing) {
      continue;
    }
    kept_uv_centers.push_back(inout_uv_centers[candidate_index]);
    kept_triangle_indices.push_back((*inout_triangle_indices)[candidate_index]);
  }

  if (kept_uv_centers.empty() && fallback_best_index < candidate_count) {
    kept_uv_centers.push_back(inout_uv_centers[fallback_best_index]);
    kept_triangle_indices.push_back((*inout_triangle_indices)[fallback_best_index]);
  }

  inout_uv_centers = std::move(kept_uv_centers);
  *inout_triangle_indices = std::move(kept_triangle_indices);
}

std::string sanitize_plot_export_component(const std::string& value) {
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

std::string escape_csv_field(const std::string& value) {
  bool needs_quotes = false;
  std::string escaped;
  escaped.reserve(value.size() + 2);
  for (char ch : value) {
    if (ch == '"') {
      escaped.push_back('"');
      escaped.push_back('"');
      needs_quotes = true;
      continue;
    }
    if (ch == ',' || ch == '\n' || ch == '\r') {
      needs_quotes = true;
    }
    escaped.push_back(ch);
  }
  if (!needs_quotes) {
    return escaped;
  }
  return '"' + escaped + '"';
}

bool save_hist_plot_file(
  const std::filesystem::path& path,
  const PatternRegionState& region,
  const std::string& region_label,
  int region_index,
  const char* plot_role,
  const std::vector<float>& hist_plot,
  int point_count,
  int pair_count,
  int max_k,
  std::string& out_error) {
  if (hist_plot.empty()) {
    out_error = std::string("No ") + plot_role + " plot is available to export.";
    return false;
  }

  std::ofstream out(path, std::ios::trunc);
  if (!out.is_open()) {
    out_error = "Failed to open " + path.string() + " for writing.";
    return false;
  }

  out << "region_index,region_id,region_label,region_mode,plot_role,point_count,pair_count,max_k,bin_count,bin_index,normalized_value\n";
  out << std::setprecision(17);
  const std::string escaped_region_label = escape_csv_field(region_label);
  const std::string escaped_region_mode = escape_csv_field(pattern_region_mode_label(region));
  const std::string escaped_plot_role = escape_csv_field(plot_role);
  for (size_t i = 0; i < hist_plot.size(); ++i) {
    out << region_index << ","
        << region.region_id << ","
        << escaped_region_label << ","
        << escaped_region_mode << ","
        << escaped_plot_role << ","
        << point_count << ","
        << pair_count << ","
        << max_k << ","
        << hist_plot.size() << ","
        << i << ","
        << hist_plot[i] << "\n";
  }

  if (!out.good()) {
    out_error = "Write error while saving " + path.string() + ".";
    return false;
  }

  return true;
}

bool save_exemplar_region_hist_plots(
  const std::string& directory_path,
  const PatternRegionState& region,
  const std::string& region_label,
  int region_index,
  std::string& out_status) {
  out_status.clear();

  if (!region_is_exemplar(region)) {
    out_status = "Plot export is only available for exemplar regions.";
    return false;
  }
  if (directory_path.empty()) {
    out_status = "Please provide an output directory.";
    return false;
  }

  const bool has_input = !region.voronoi_pcf_hist_plot.empty();
  const bool has_output = !region.output_voronoi_pcf_hist_plot.empty();
  if (!has_input && !has_output) {
    out_status = "No current input or output plots are available to export.";
    return false;
  }

  const std::filesystem::path output_dir(directory_path);
  std::error_code ec;
  std::filesystem::create_directories(output_dir, ec);
  if (ec) {
    out_status = "Failed to create output directory: " + ec.message();
    return false;
  }

  const std::string safe_label = sanitize_plot_export_component(region_label);
  int saved_count = 0;
  std::vector<std::string> skipped;

  if (has_input) {
    std::string error;
    const std::filesystem::path input_path =
      output_dir /
      ("region_" + std::to_string(region.region_id) + "_" + safe_label + "_input_plot.csv");
    if (!save_hist_plot_file(
          input_path,
          region,
          region_label,
          region_index,
          "input",
          region.voronoi_pcf_hist_plot,
          region.voronoi_pcf_points_inside,
          region.voronoi_pcf_pair_count,
          region.voronoi_pcf_max_k,
          error)) {
      out_status = error;
      return false;
    }
    ++saved_count;
  } else {
    skipped.push_back("input");
  }

  if (has_output) {
    std::string error;
    const std::filesystem::path output_path =
      output_dir /
      ("region_" + std::to_string(region.region_id) + "_" + safe_label + "_output_plot.csv");
    if (!save_hist_plot_file(
          output_path,
          region,
          region_label,
          region_index,
          "output",
          region.output_voronoi_pcf_hist_plot,
          static_cast<int>(region.output_pattern_points_uv.size()),
          region.output_voronoi_pcf_pair_count,
          region.output_voronoi_pcf_max_k,
          error)) {
      out_status = error;
      return false;
    }
    ++saved_count;
  } else {
    skipped.push_back("output");
  }

  out_status =
    "Saved " + std::to_string(saved_count) +
    " plot file" + (saved_count == 1 ? "" : "s") +
    " to " + output_dir.string() + ".";
  if (!skipped.empty()) {
    out_status += " Missing " + skipped.front();
    if (skipped.size() > 1) {
      out_status += " and " + skipped.back();
    }
    out_status += " plot.";
  }

  return true;
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

double point_segment_distance_2d(
  const Eigen::Vector2d& p,
  const Eigen::Vector2d& a,
  const Eigen::Vector2d& b) {
  const Eigen::Vector2d ab = b - a;
  const double ab2 = ab.squaredNorm();
  if (ab2 <= 1e-20) {
    return (p - a).norm();
  }
  const double t = std::clamp((p - a).dot(ab) / ab2, 0.0, 1.0);
  const Eigen::Vector2d closest = a + t * ab;
  return (p - closest).norm();
}

// Unified point-in-polygon test with explicit boundary handling
enum class BoundaryMode {
  EXCLUDE,  // Boundary points are considered outside
  INCLUDE   // Boundary points are considered inside
};

static bool point_in_polygon_with_boundary(
  const Eigen::Vector2d& p,
  const Eigen::MatrixXd& poly,
  BoundaryMode mode) {
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

  // First check if point is on boundary
  bool is_on_boundary = false;
  for (int i = 0; i < poly.rows(); ++i) {
    const int j = (i + 1) % poly.rows();
    const Eigen::Vector2d a = poly.row(i).head<2>().transpose();
    const Eigen::Vector2d b = poly.row(j).head<2>().transpose();
    if (point_on_segment_2d(p, a, b, eps)) {
      is_on_boundary = true;
      break;
    }
  }
  // Handle boundary based on mode
  if (is_on_boundary) {
    return mode == BoundaryMode::INCLUDE;
  }

  // Winding number test for interior points
  int winding_number = 0;
  for (int i = 0; i < poly.rows(); ++i) {
    const int j = (i + 1) % poly.rows();
    const Eigen::Vector2d a = poly.row(i).head<2>().transpose();
    const Eigen::Vector2d b = poly.row(j).head<2>().transpose();

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

// Wrapper for backward compatibility: boundary points excluded
bool point_in_polygon_for_pcf(const Eigen::Vector2d& p, const Eigen::MatrixXd& poly) {
  return point_in_polygon_with_boundary(p, poly, BoundaryMode::EXCLUDE);
}

bool polygon_bbox(
  const Eigen::MatrixXd& poly,
  Eigen::Vector2d& out_min,
  Eigen::Vector2d& out_max) {
  if (poly.rows() < 3 || poly.cols() < 2) {
    out_min.setZero();
    out_max.setZero();
    return false;
  }
  out_min = poly.leftCols<2>().colwise().minCoeff().transpose();
  out_max = poly.leftCols<2>().colwise().maxCoeff().transpose();
  return true;
}

const Eigen::MatrixXd* preferred_density_boundary_uv(const PatternRegionState& region) {
  if (region.input_boundary_uv.rows() >= 3 && region.input_boundary_uv.cols() >= 2) {
    return &region.input_boundary_uv;
  }
  if (region.output_boundary_uv_poly.rows() >= 3 && region.output_boundary_uv_poly.cols() >= 2) {
    return &region.output_boundary_uv_poly;
  }
  return nullptr;
}

const Eigen::MatrixXd* preferred_transition_reference_boundary_uv(
  const PatternRegionState& region) {
  if (region.output_boundary_uv_poly.rows() >= 3 && region.output_boundary_uv_poly.cols() >= 2) {
    return &region.output_boundary_uv_poly;
  }
  if (region.input_boundary_uv.rows() >= 3 && region.input_boundary_uv.cols() >= 2) {
    return &region.input_boundary_uv;
  }
  return nullptr;
}

bool collect_region_reference_support_candidates(
  const PatternRegionState& region,
  const DelaunayTraversalHelper* delaunay_helper,
  std::vector<Eigen::Vector2d>& out_uv_centers,
  std::vector<int>* out_triangle_indices = nullptr) {
  out_uv_centers.clear();
  const Eigen::MatrixXd* boundary_poly = preferred_density_boundary_uv(region);
  if (!boundary_poly) {
    return false;
  }
  return collect_triangle_center_candidates_in_polygon(
    *boundary_poly,
    delaunay_helper,
    out_uv_centers,
    out_triangle_indices);
}

double point_to_polygon_distance_2d(
  const Eigen::Vector2d& uv,
  const Eigen::MatrixXd& boundary_poly) {
  if (boundary_poly.rows() < 3 || boundary_poly.cols() < 2) {
    return std::numeric_limits<double>::infinity();
  }
  if (point_in_polygon_with_boundary(uv, boundary_poly, BoundaryMode::INCLUDE)) {
    return 0.0;
  }
  double best_distance = std::numeric_limits<double>::infinity();
  for (int i = 0; i < boundary_poly.rows(); ++i) {
    const int j = (i + 1) % boundary_poly.rows();
    const Eigen::Vector2d a = boundary_poly.row(i).head<2>().transpose();
    const Eigen::Vector2d b = boundary_poly.row(j).head<2>().transpose();
    best_distance = std::min(best_distance, point_segment_distance_2d(uv, a, b));
  }
  return best_distance;
}

Eigen::Vector2d region_reference_center_uv(const PatternRegionState& region) {
  const Eigen::MatrixXd* boundary_poly = preferred_transition_reference_boundary_uv(region);
  if (boundary_poly && boundary_poly->rows() > 0 && boundary_poly->cols() >= 2) {
    return boundary_poly->leftCols<2>().colwise().mean().transpose();
  }
  if (!region.output_pattern_points_uv.empty()) {
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    for (const Eigen::Vector2d& uv : region.output_pattern_points_uv) {
      center += uv;
    }
    return center / static_cast<double>(region.output_pattern_points_uv.size());
  }
  if (!region.pattern_points_uv.empty()) {
    Eigen::Vector2d center = Eigen::Vector2d::Zero();
    for (const Eigen::Vector2d& uv : region.pattern_points_uv) {
      center += uv;
    }
    return center / static_cast<double>(region.pattern_points_uv.size());
  }
  return Eigen::Vector2d::Zero();
}

bool compute_region_reference_density(
  const PatternRegionState& region,
  const DelaunayTraversalHelper* delaunay_helper,
  double& out_density,
  int* out_support_count = nullptr) {
  out_density = 0.0;
  if (out_support_count) {
    *out_support_count = 0;
  }
  if (!delaunay_helper || !delaunay_helper->is_ready()) {
    return false;
  }
  if (!region.voronoi_pcf_ready || region.voronoi_pcf_points_inside < 2) {
    return false;
  }
  std::vector<Eigen::Vector2d> support_uv;
  if (!collect_region_reference_support_candidates(region, delaunay_helper, support_uv) ||
      support_uv.empty()) {
    return false;
  }
  if (out_support_count) {
    *out_support_count = static_cast<int>(support_uv.size());
  }
  out_density =
    static_cast<double>(region.voronoi_pcf_points_inside) /
    static_cast<double>(support_uv.size());
  return std::isfinite(out_density) && out_density > 0.0;
}

std::vector<float> blend_distribution_plots(
  const std::vector<float>& source_a,
  const std::vector<float>& source_b,
  double blend_to_b) {
  const size_t n = std::max(source_a.size(), source_b.size());
  if (n == 0) {
    return {};
  }
  const double w = std::clamp(blend_to_b, 0.0, 1.0);
  std::vector<float> blended(n, 0.0f);
  for (size_t i = 0; i < n; ++i) {
    const double a = (i < source_a.size()) ? static_cast<double>(source_a[i]) : 0.0;
    const double b = (i < source_b.size()) ? static_cast<double>(source_b[i]) : 0.0;
    blended[i] = static_cast<float>((1.0 - w) * a + w * b);
  }
  return blended;
}

int max_nonzero_hist_bin(
  const std::vector<int>& hist_counts,
  const std::vector<float>& hist_plot) {
  int max_k = 0;
  for (int k = 0; k < static_cast<int>(hist_counts.size()); ++k) {
    if (hist_counts[static_cast<size_t>(k)] > 0) {
      max_k = k;
    }
  }
  if (!hist_counts.empty()) {
    return max_k;
  }
  for (int k = 0; k < static_cast<int>(hist_plot.size()); ++k) {
    if (std::abs(hist_plot[static_cast<size_t>(k)]) > 1e-8f) {
      max_k = k;
    }
  }
  return max_k;
}

struct TransitionTargetProfile {
  std::vector<float> hist_plot;
  std::vector<std::vector<float>> individual_plots;
  std::vector<std::vector<int>> raw_point_hist_counts;
  int effective_point_count = 0;
};

struct TransitionSupportRowTargets {
  std::vector<std::vector<float>> individual_plots;
  std::vector<std::vector<int>> raw_point_hist_counts;
  std::vector<double> blend_weights_to_b;
};

struct TransitionBlendField {
  Eigen::Vector2d axis_dir = Eigen::Vector2d::Zero();
  double axis_min_proj = 0.0;
  double axis_max_proj = 0.0;
  const Eigen::MatrixXd* boundary_a = nullptr;
  const Eigen::MatrixXd* boundary_b = nullptr;
  bool use_normalized_axis = false;
};

bool build_transition_blend_field(
  const std::vector<Eigen::Vector2d>& transition_support_uv,
  const PatternRegionState& source_a,
  const PatternRegionState& source_b,
  TransitionBlendField& out_field) {
  out_field = TransitionBlendField{};
  out_field.boundary_a = preferred_transition_reference_boundary_uv(source_a);
  out_field.boundary_b = preferred_transition_reference_boundary_uv(source_b);

  if (transition_support_uv.empty()) {
    return (out_field.boundary_a != nullptr) || (out_field.boundary_b != nullptr);
  }

  const Eigen::Vector2d center_a = region_reference_center_uv(source_a);
  const Eigen::Vector2d center_b = region_reference_center_uv(source_b);
  const Eigen::Vector2d axis = center_b - center_a;
  const double axis_len = axis.norm();
  if (axis_len > 1e-12) {
    out_field.axis_dir = axis / axis_len;
    out_field.axis_min_proj = std::numeric_limits<double>::infinity();
    out_field.axis_max_proj = -std::numeric_limits<double>::infinity();
    for (const Eigen::Vector2d& support_uv : transition_support_uv) {
      const double proj = support_uv.dot(out_field.axis_dir);
      out_field.axis_min_proj = std::min(out_field.axis_min_proj, proj);
      out_field.axis_max_proj = std::max(out_field.axis_max_proj, proj);
    }
    if (std::isfinite(out_field.axis_min_proj) &&
        std::isfinite(out_field.axis_max_proj) &&
        (out_field.axis_max_proj - out_field.axis_min_proj) > 1e-12) {
      out_field.use_normalized_axis = true;
      return true;
    }
  }

  return (out_field.boundary_a != nullptr) || (out_field.boundary_b != nullptr);
}

double transition_distribution_sort_key(const std::vector<float>& distribution) {
  double key = 0.0;
  for (int k = 0; k < static_cast<int>(distribution.size()); ++k) {
    const double bin_weight = 1.0 / static_cast<double>(k + 1);
    key += bin_weight * static_cast<double>(distribution[static_cast<size_t>(k)]);
  }
  return key;
}

double transition_count_sort_key(const std::vector<int>& counts) {
  double key = 0.0;
  for (int k = 0; k < static_cast<int>(counts.size()); ++k) {
    const double bin_weight = 1.0 / static_cast<double>(k + 1);
    key += bin_weight * static_cast<double>(counts[static_cast<size_t>(k)]);
  }
  return key;
}

std::vector<size_t> sorted_distribution_indices_by_transition_key(
  const std::vector<std::vector<float>>& distributions) {
  std::vector<size_t> sorted_indices(distributions.size(), 0);
  for (size_t i = 0; i < sorted_indices.size(); ++i) {
    sorted_indices[i] = i;
  }
  std::sort(
    sorted_indices.begin(),
    sorted_indices.end(),
    [&](size_t lhs, size_t rhs) {
      const double lhs_key =
        transition_distribution_sort_key(distributions[lhs]);
      const double rhs_key =
        transition_distribution_sort_key(distributions[rhs]);
      if (lhs_key != rhs_key) {
        return lhs_key < rhs_key;
      }
      return lhs < rhs;
    });
  return sorted_indices;
}

std::vector<size_t> sorted_count_indices_by_transition_key(
  const std::vector<std::vector<int>>& count_rows) {
  std::vector<size_t> sorted_indices(count_rows.size(), 0);
  for (size_t i = 0; i < sorted_indices.size(); ++i) {
    sorted_indices[i] = i;
  }
  std::sort(
    sorted_indices.begin(),
    sorted_indices.end(),
    [&](size_t lhs, size_t rhs) {
      const double lhs_key = transition_count_sort_key(count_rows[lhs]);
      const double rhs_key = transition_count_sort_key(count_rows[rhs]);
      if (lhs_key != rhs_key) {
        return lhs_key < rhs_key;
      }
      return lhs < rhs;
    });
  return sorted_indices;
}

size_t quantile_index_for_transition_rows(size_t count, double t) {
  if (count == 0) {
    return 0;
  }
  if (count == 1) {
    return 0;
  }
  const double clamped_t = std::clamp(t, 0.0, 1.0);
  return static_cast<size_t>(std::llround(
    clamped_t * static_cast<double>(count - 1)));
}

double scheduled_transition_blend_weight(double raw_weight_to_b) {
  const double t = std::clamp(raw_weight_to_b, 0.0, 1.0);
  constexpr int kTransitionTargetBandCount = 5;
  if (kTransitionTargetBandCount <= 1) {
    return t;
  }

  const double band_count_minus_one =
    static_cast<double>(kTransitionTargetBandCount - 1);
  const double banded =
    std::round(t * band_count_minus_one) / band_count_minus_one;
  // Keep the target visibly row-ordered without making hard discontinuities at
  // band boundaries; the small continuous term avoids all-or-nothing jumps.
  return std::clamp(0.8 * banded + 0.2 * t, 0.0, 1.0);
}

double transition_semantic_quantile(
  double scheduled_weight_to_b,
  const PatternRegionState& source_a,
  const PatternRegionState& source_b) {
  const double source_a_key =
    transition_distribution_sort_key(source_a.voronoi_pcf_hist_plot);
  const double source_b_key =
    transition_distribution_sort_key(source_b.voronoi_pcf_hist_plot);
  if (!std::isfinite(source_a_key) || !std::isfinite(source_b_key)) {
    return std::clamp(scheduled_weight_to_b, 0.0, 1.0);
  }
  const double semantic_t =
    (source_a_key <= source_b_key)
      ? scheduled_weight_to_b
      : (1.0 - scheduled_weight_to_b);
  return std::clamp(semantic_t, 0.0, 1.0);
}

std::vector<float> average_transition_distributions(
  const std::vector<std::vector<float>>& distributions) {
  size_t max_bins = 0;
  for (const auto& distribution : distributions) {
    max_bins = std::max(max_bins, distribution.size());
  }
  if (max_bins == 0 || distributions.empty()) {
    return {};
  }
  std::vector<float> average(max_bins, 0.0f);
  for (const auto& distribution : distributions) {
    for (size_t k = 0; k < distribution.size(); ++k) {
      average[k] += distribution[k];
    }
  }
  const float inv_count = 1.0f / static_cast<float>(distributions.size());
  for (float& value : average) {
    value *= inv_count;
  }
  return average;
}

std::vector<int> blend_count_signatures(
  const std::vector<int>& source_a,
  const std::vector<int>& source_b,
  double blend_to_b) {
  const size_t n = std::max(source_a.size(), source_b.size());
  if (n == 0) {
    return {};
  }
  const double w = std::clamp(blend_to_b, 0.0, 1.0);
  std::vector<int> blended(n, 0);
  for (size_t i = 0; i < n; ++i) {
    const double a = (i < source_a.size()) ? static_cast<double>(source_a[i]) : 0.0;
    const double b = (i < source_b.size()) ? static_cast<double>(source_b[i]) : 0.0;
    blended[i] = std::max(
      0,
      static_cast<int>(std::llround((1.0 - w) * a + w * b)));
  }
  return blended;
}

double transition_blend_weight_to_source_b(
  const Eigen::Vector2d& uv,
  const TransitionBlendField& blend_field) {
  if (blend_field.use_normalized_axis) {
    const double proj = uv.dot(blend_field.axis_dir);
    const double denom = blend_field.axis_max_proj - blend_field.axis_min_proj;
    if (std::isfinite(denom) && denom > 1e-12) {
      return std::clamp((proj - blend_field.axis_min_proj) / denom, 0.0, 1.0);
    }
  }

  double distance_a = std::numeric_limits<double>::infinity();
  double distance_b = std::numeric_limits<double>::infinity();
  if (blend_field.boundary_a) {
    distance_a = point_to_polygon_distance_2d(uv, *blend_field.boundary_a);
  }
  if (blend_field.boundary_b) {
    distance_b = point_to_polygon_distance_2d(uv, *blend_field.boundary_b);
  }

  const double denom = distance_a + distance_b;
  if (std::isfinite(denom) && denom > 1e-12) {
    return std::clamp(distance_a / denom, 0.0, 1.0);
  }
  return 0.5;
}

bool build_transition_support_row_targets(
  const PatternRegionState& source_a,
  const PatternRegionState& source_b,
  const std::vector<Eigen::Vector2d>& output_support_uv,
  TransitionSupportRowTargets& out_targets) {
  out_targets = TransitionSupportRowTargets{};
  if (output_support_uv.empty() ||
      source_a.voronoi_pcf_hist_plot.empty() ||
      source_b.voronoi_pcf_hist_plot.empty()) {
    return false;
  }

  TransitionBlendField blend_field;
  build_transition_blend_field(output_support_uv, source_a, source_b, blend_field);

  const bool use_source_local_distributions =
    !source_a.voronoi_pcf_individual_plots.empty() &&
    !source_b.voronoi_pcf_individual_plots.empty();
  const std::vector<size_t> source_a_distribution_order =
    use_source_local_distributions
      ? sorted_distribution_indices_by_transition_key(source_a.voronoi_pcf_individual_plots)
      : std::vector<size_t>{};
  const std::vector<size_t> source_b_distribution_order =
    use_source_local_distributions
      ? sorted_distribution_indices_by_transition_key(source_b.voronoi_pcf_individual_plots)
      : std::vector<size_t>{};

  const bool use_source_raw_signatures =
    !source_a.voronoi_pcf_raw_point_hist_counts.empty() &&
    !source_b.voronoi_pcf_raw_point_hist_counts.empty();
  const std::vector<size_t> source_a_raw_order =
    use_source_raw_signatures
      ? sorted_count_indices_by_transition_key(source_a.voronoi_pcf_raw_point_hist_counts)
      : std::vector<size_t>{};
  const std::vector<size_t> source_b_raw_order =
    use_source_raw_signatures
      ? sorted_count_indices_by_transition_key(source_b.voronoi_pcf_raw_point_hist_counts)
      : std::vector<size_t>{};

  out_targets.blend_weights_to_b.resize(output_support_uv.size(), 0.5);
  out_targets.individual_plots.reserve(output_support_uv.size());
  if (use_source_raw_signatures) {
    out_targets.raw_point_hist_counts.reserve(output_support_uv.size());
  }

  for (size_t support_idx = 0; support_idx < output_support_uv.size(); ++support_idx) {
    const double weight_to_b = transition_blend_weight_to_source_b(
      output_support_uv[support_idx],
      blend_field);
    out_targets.blend_weights_to_b[support_idx] = weight_to_b;
    const double target_weight_to_b =
      scheduled_transition_blend_weight(weight_to_b);
    const double semantic_quantile =
      transition_semantic_quantile(target_weight_to_b, source_a, source_b);

    if (use_source_local_distributions) {
      const size_t source_a_idx =
        source_a_distribution_order[
          quantile_index_for_transition_rows(
            source_a_distribution_order.size(),
            semantic_quantile)];
      const size_t source_b_idx =
        source_b_distribution_order[
          quantile_index_for_transition_rows(
            source_b_distribution_order.size(),
            semantic_quantile)];
      out_targets.individual_plots.push_back(
        blend_distribution_plots(
          source_a.voronoi_pcf_individual_plots[source_a_idx],
          source_b.voronoi_pcf_individual_plots[source_b_idx],
          target_weight_to_b));
    } else {
      out_targets.individual_plots.push_back(
        blend_distribution_plots(
          source_a.voronoi_pcf_hist_plot,
          source_b.voronoi_pcf_hist_plot,
          target_weight_to_b));
    }

    if (use_source_raw_signatures) {
      const size_t source_a_raw_idx =
        source_a_raw_order[
          quantile_index_for_transition_rows(
            source_a_raw_order.size(),
            semantic_quantile)];
      const size_t source_b_raw_idx =
        source_b_raw_order[
          quantile_index_for_transition_rows(
            source_b_raw_order.size(),
            semantic_quantile)];
      out_targets.raw_point_hist_counts.push_back(
        blend_count_signatures(
          source_a.voronoi_pcf_raw_point_hist_counts[source_a_raw_idx],
          source_b.voronoi_pcf_raw_point_hist_counts[source_b_raw_idx],
          target_weight_to_b));
    }
  }

  return !out_targets.individual_plots.empty();
}

bool build_transition_target_profile(
  const InteractionState& root_state,
  const PatternRegionState& transition_region,
  const DelaunayTraversalHelper* delaunay_helper,
  TransitionTargetProfile& out_profile) {
  out_profile = TransitionTargetProfile{};
  if (!region_is_transition(transition_region) ||
      !delaunay_helper ||
      !delaunay_helper->is_ready()) {
    return false;
  }

  const PatternRegionState* source_a =
    find_region_by_id(root_state, transition_region.transition_source_a_region_id);
  const PatternRegionState* source_b =
    find_region_by_id(root_state, transition_region.transition_source_b_region_id);
  if (!source_a || !source_b ||
      region_is_transition(*source_a) ||
      region_is_transition(*source_b) ||
      source_a->region_id == source_b->region_id) {
    return false;
  }
  if (source_a->voronoi_pcf_hist_plot.empty() || source_b->voronoi_pcf_hist_plot.empty()) {
    return false;
  }

  double source_a_density = 0.0;
  double source_b_density = 0.0;
  if (!compute_region_reference_density(*source_a, delaunay_helper, source_a_density) ||
      !compute_region_reference_density(*source_b, delaunay_helper, source_b_density)) {
    return false;
  }

  std::vector<Eigen::Vector2d> output_support_uv;
  std::vector<int> output_support_tri_indices;
  if (transition_region.output_boundary_uv_poly.rows() < 3 ||
      transition_region.output_boundary_uv_poly.cols() < 2 ||
      !collect_triangle_center_candidates_in_polygon(
        transition_region.output_boundary_uv_poly,
        delaunay_helper,
        output_support_uv,
        &output_support_tri_indices) ||
      output_support_uv.empty()) {
    return false;
  }

  TransitionSupportRowTargets support_row_targets;
  if (!build_transition_support_row_targets(
        *source_a,
        *source_b,
        output_support_uv,
        support_row_targets)) {
    return false;
  }

  const double blend_weight_sum = std::accumulate(
    support_row_targets.blend_weights_to_b.begin(),
    support_row_targets.blend_weights_to_b.end(),
    0.0);
  const double average_weight_to_b =
    support_row_targets.blend_weights_to_b.empty()
      ? 0.5
      : (blend_weight_sum /
         static_cast<double>(support_row_targets.blend_weights_to_b.size()));

  const double effective_density =
    (1.0 - average_weight_to_b) * source_a_density +
    average_weight_to_b * source_b_density;
  const int output_support_count = static_cast<int>(output_support_uv.size());
  out_profile.effective_point_count = std::clamp(
    static_cast<int>(std::llround(effective_density * static_cast<double>(output_support_count))),
    0,
    output_support_count);

  // The saved target profile stays count-sized for assignment scoring; the live
  // optimizer separately keeps full support-row targets for directional guidance.
  std::vector<size_t> profile_support_indices;
  if (out_profile.effective_point_count > 0) {
    std::vector<size_t> support_order(output_support_uv.size(), 0);
    std::iota(support_order.begin(), support_order.end(), size_t{0});
    std::sort(
      support_order.begin(),
      support_order.end(),
      [&](size_t lhs, size_t rhs) {
        const double lw = support_row_targets.blend_weights_to_b[lhs];
        const double rw = support_row_targets.blend_weights_to_b[rhs];
        if (std::abs(lw - rw) > 1e-9) {
          return lw < rw;
        }
        const Eigen::Vector2d& lu = output_support_uv[lhs];
        const Eigen::Vector2d& ru = output_support_uv[rhs];
        if (std::abs(lu.x() - ru.x()) > 1e-9) {
          return lu.x() < ru.x();
        }
        return lu.y() < ru.y();
      });

    const int target_profile_count = std::min(
      out_profile.effective_point_count,
      static_cast<int>(support_order.size()));
    profile_support_indices.reserve(static_cast<size_t>(target_profile_count));
    if (target_profile_count >= static_cast<int>(support_order.size())) {
      profile_support_indices = std::move(support_order);
    } else {
      const double inv_target_count =
        1.0 / static_cast<double>(std::max(1, target_profile_count));
      for (int i = 0; i < target_profile_count; ++i) {
        const double t = (static_cast<double>(i) + 0.5) * inv_target_count;
        const int ordered_idx = std::clamp(
          static_cast<int>(std::floor(t * static_cast<double>(support_order.size()))),
          0,
          static_cast<int>(support_order.size()) - 1);
        profile_support_indices.push_back(support_order[static_cast<size_t>(ordered_idx)]);
      }
    }
  }
  if (profile_support_indices.empty() && !output_support_uv.empty()) {
    profile_support_indices.push_back(output_support_uv.size() / 2);
  }

  out_profile.individual_plots.reserve(profile_support_indices.size());
  if (!support_row_targets.raw_point_hist_counts.empty()) {
    out_profile.raw_point_hist_counts.reserve(profile_support_indices.size());
  }
  for (size_t support_idx : profile_support_indices) {
    if (support_idx < support_row_targets.individual_plots.size()) {
      out_profile.individual_plots.push_back(
        support_row_targets.individual_plots[support_idx]);
    }
    if (support_idx < support_row_targets.raw_point_hist_counts.size()) {
      out_profile.raw_point_hist_counts.push_back(
        support_row_targets.raw_point_hist_counts[support_idx]);
    }
  }

  out_profile.hist_plot = average_transition_distributions(out_profile.individual_plots);
  if (out_profile.hist_plot.empty()) {
    out_profile.hist_plot = blend_distribution_plots(
      source_a->voronoi_pcf_hist_plot,
      source_b->voronoi_pcf_hist_plot,
      average_weight_to_b);
  }

  return !out_profile.hist_plot.empty();
}

void add_hist_count_if_in_range(std::vector<int>& hist_counts, int k, int delta) {
  if (hist_counts.empty()) {
    return;
  }
  if (k < 0) {
    return;
  }
  const int bin = std::min(k, static_cast<int>(hist_counts.size()) - 1);
  hist_counts[static_cast<size_t>(bin)] += delta;
}


bool k_in_hist_range(const std::vector<int>& hist_counts, int k) {
  return (k >= 0) && (k < static_cast<int>(hist_counts.size()));
}

int histogram_total_count(const std::vector<int>& hist_counts) {
  int total = 0;
  for (int v : hist_counts) {
    if (v > 0) {
      total += v;
    }
  }
  return total;
}

std::vector<float> normalized_histogram(
  const std::vector<int>& hist_counts,
  int pair_count) {
  if (pair_count <= 0 || hist_counts.empty()) {
    return {};
  }
  std::vector<float> normalized(hist_counts.size(), 0.0f);
  const float denom = static_cast<float>(pair_count);
  for (size_t i = 0; i < hist_counts.size(); ++i) {
    normalized[i] = static_cast<float>(hist_counts[i]) / denom;
  }
  return normalized;
}

std::vector<float> average_individual_histogram(
  const std::vector<std::vector<int>>& point_hist_counts,
  const std::vector<std::vector<int>>& point_support_counts,
  int bin_count) {
  if (bin_count <= 0 || point_hist_counts.empty() || point_support_counts.empty()) {
    return {};
  }
  const int n = std::min(
    static_cast<int>(point_hist_counts.size()),
    static_cast<int>(point_support_counts.size()));
  if (n <= 0) {
    return {};
  }
  std::vector<float> out(static_cast<size_t>(bin_count), 0.0f);
  int valid_points = 0;
  for (int i = 0; i < n; ++i) {
    const std::vector<int>& row = point_hist_counts[static_cast<size_t>(i)];
    const std::vector<int>& support = point_support_counts[static_cast<size_t>(i)];
    const int row_bin_count = std::min(bin_count, static_cast<int>(row.size()));
    const int support_bin_count = std::min(bin_count, static_cast<int>(support.size()));
    const int eval_bin_count = std::min(row_bin_count, support_bin_count);
    bool has_valid_support = false;
    for (int k = 0; k < eval_bin_count; ++k) {
      if (support[static_cast<size_t>(k)] > 0) {
        has_valid_support = true;
        break;
      }
    }
    if (!has_valid_support) {
      continue;
    }
    ++valid_points;
    for (int k = 0; k < row_bin_count; ++k) {
      const int denom = (k < support_bin_count) ? support[static_cast<size_t>(k)] : 0;
      if (denom <= 0) {
        continue;
      }
      out[static_cast<size_t>(k)] +=
        static_cast<float>(row[static_cast<size_t>(k)]) / static_cast<float>(denom);
    }
  }
  if (valid_points > 0) {
    const float inv_valid = 1.0f / static_cast<float>(valid_points);
    for (float& v : out) {
      v *= inv_valid;
    }
  }
  return out;
}

double weighted_distribution_l2(
  const std::vector<float>& current_distribution,
  const std::vector<float>& target_distribution) {
  const size_t n = std::max(current_distribution.size(), target_distribution.size());
  double energy = 0.0;
  double cdf_current = 0.0;
  double cdf_target = 0.0;
  double zero_bin_leakage_mass = 0.0;
  constexpr double kZeroBinTol = 1e-12;
  constexpr double kZeroBinQuadraticPenalty = 80.0;
  constexpr double kZeroBinLeakagePenalty = 400.0;
  const size_t strong_prefix_span = std::min(n, std::max<size_t>(3, n / 4));
  const size_t early_span = std::max<size_t>(strong_prefix_span, n / 2);
  for (size_t i = 0; i < n; ++i) {
    const double a = (i < current_distribution.size()) ? current_distribution[i] : 0.0;
    const double b = (i < target_distribution.size()) ? target_distribution[i] : 0.0;
    const double d = a - b;
    // Put much stronger pressure on the first few bins, then taper.
    double early_focus = 0.75;
    if (i < strong_prefix_span) {
      const double t =
        static_cast<double>(i) /
        static_cast<double>(std::max<size_t>(1, strong_prefix_span));
      early_focus = 12.0 - 7.0 * t;
    } else if (i < early_span) {
      const double t =
        static_cast<double>(i - strong_prefix_span) /
        static_cast<double>(std::max<size_t>(1, early_span - strong_prefix_span));
      early_focus = 5.0 - 3.0 * t;
    }
    const double bin_weight = early_focus * (1.0 + 4.0 * b);
    energy += bin_weight * d * d;
    if (b <= kZeroBinTol) {
      zero_bin_leakage_mass += a;
      energy += kZeroBinQuadraticPenalty * a * a;
    }
    cdf_current += a;
    cdf_target += b;
    const double cdf_d = cdf_current - cdf_target;
    const double cdf_weight = (i < strong_prefix_span) ? 1.5 : ((i < early_span) ? 0.6 : 0.2);
    energy += cdf_weight * cdf_d * cdf_d;
  }
  // Strongly discourage placing any mass in bins absent in the input histogram.
  energy += kZeroBinLeakagePenalty * zero_bin_leakage_mass * zero_bin_leakage_mass;
  return energy;
}

int two_class_pair_channel(int class_a, int class_b) {
  const int a = sanitize_pattern_class_id(class_a);
  const int b = sanitize_pattern_class_id(class_b);
  if (a == 0 && b == 0) {
    return 0;
  }
  if (a == 1 && b == 1) {
    return 1;
  }
  return 2;
}

const char* two_class_pair_channel_label(int channel) {
  switch (channel) {
    case 0: return "Class 0 - Class 0";
    case 1: return "Class 1 - Class 1";
    case 2: return "Cross class";
    default: return "Unknown";
  }
}

constexpr int kFixedAnchorTemplateMaxHopRadius = 4;

struct TwoClassPCFStats {
  std::array<std::vector<int>, kTwoClassPairChannelCount> hist_counts;
  std::array<std::vector<float>, kTwoClassPairChannelCount> hist_plot;
  std::array<std::vector<std::vector<float>>, kTwoClassPairChannelCount> individual_plots;
  std::array<std::vector<float>, kTwoClassPairChannelCount> hist_min_plot;
  std::array<std::vector<float>, kTwoClassPairChannelCount> hist_max_plot;
  std::array<int, kPatternClassCount> class_counts = {0, 0};
  std::array<int, kTwoClassPairChannelCount> pair_counts = {0, 0, 0};
  std::vector<int> combined_hist_counts;
  std::vector<float> combined_hist_plot;
  int combined_pair_count = 0;
};

struct FixedAnchorTwoClassEvaluationCache {
  bool valid = false;
  int anchor_class_id = 0;
  int dependent_class_id = 1;
  std::vector<Eigen::Vector2d> anchor_uv_points;
  std::vector<int> anchor_support_rows;
  std::vector<std::vector<int>> anchor_support_counts;
  std::vector<std::vector<float>> anchor_cross_target_distributions;
  std::vector<float> anchor_cross_target_avg_plot;
  std::vector<float> anchor_cross_target_min_plot;
  std::vector<float> anchor_cross_target_max_plot;
  int anchor_cross_near_hop_radius = 2;
  double anchor_cross_target_near_count = 0.0;
  int anchor_cross_target_near_lower_count = 0;
  std::vector<std::vector<float>> dependent_cross_target_distributions;
  std::vector<float> dependent_cross_target_avg_plot;
  std::vector<float> dependent_cross_target_min_plot;
  std::vector<float> dependent_cross_target_max_plot;
  int dependent_cross_near_hop_radius = 2;
  double dependent_cross_target_near_count = 0.0;
  int dependent_cross_target_near_lower_count = 0;
  std::vector<std::vector<Eigen::Vector2d>> anchor_template_target_offsets;
  std::vector<int> anchor_template_target_hop_radii;
  std::vector<double> anchor_template_target_scale_sq;
};

struct FixedAnchorTwoClassProposalCache {
  bool valid = false;
  int anchor_class_id = 0;
  int dependent_class_id = 1;
  int bin_count = 0;
  std::vector<size_t> dependent_point_indices;
  std::vector<int> dependent_row_for_point;
  std::vector<int> dependent_support_rows;
  std::vector<std::vector<int>> dependent_support_counts;
  std::vector<std::vector<int>> dependent_self_hist;
  std::vector<std::vector<int>> dependent_cross_hist;
};

struct FixedAnchorTwoClassObjectiveBreakdown {
  double total_error = std::numeric_limits<double>::infinity();
  double global_symmetric_error = std::numeric_limits<double>::infinity();
  double total_weight = 0.0;
  double dependent_self_distribution_error = 0.0;
  double dependent_self_envelope_error = 0.0;
  double dependent_self_weight = 0.0;
  double dependent_self_weighted_error = 0.0;
  double directional_cross_distribution_error = 0.0;
  double directional_cross_envelope_error = 0.0;
  double directional_cross_near_deficit_error = 0.0;
  double directional_cross_near_excess_error = 0.0;
  double directional_cross_weight = 0.0;
  double directional_cross_weighted_error = 0.0;
  double dependent_cross_distribution_error = 0.0;
  double dependent_cross_near_deficit_error = 0.0;
  double dependent_cross_weight = 0.0;
  double dependent_cross_weighted_error = 0.0;
  double template_offset_error = 0.0;
  double template_offset_weight = 0.0;
  double template_offset_weighted_error = 0.0;
  int directional_cross_near_hop_radius = 2;
  double directional_cross_target_near_count = 0.0;
  int directional_cross_target_near_lower_count = 0;
  int directional_cross_lonely_count = 0;
  int dependent_cross_near_hop_radius = 2;
  double dependent_cross_target_near_count = 0.0;
  int dependent_cross_target_near_lower_count = 0;
  int dependent_cross_lonely_count = 0;
  int anchor_class_id = 0;
  int dependent_class_id = 1;
  int dependent_channel = 1;
  int cross_channel = 2;
  int anchor_count = 0;
  int dependent_count = 0;
  int dependent_valid_anchors = 0;
  int cross_valid_anchors = 0;
  int template_valid_anchors = 0;
};

bool build_two_class_local_distribution(
  const std::vector<int>& row_counts,
  const std::vector<int>& support_counts,
  std::vector<float>& out_distribution);

double compute_two_class_near_count_band_error(
  int near_count,
  int target_lower_count,
  double target_near_count,
  double* out_deficit_error,
  double* out_excess_error);

void finalize_fixed_anchor_near_count_target(
  std::array<std::vector<int>, kFixedAnchorTemplateMaxHopRadius + 1>& near_counts_by_radius,
  int default_radius,
  int& out_radius,
  int& out_lower_count,
  double& out_target_count);

bool build_two_class_pair_histograms(
  const std::vector<Eigen::Vector2d>& uv_points,
  const std::vector<int>& class_ids,
  const DelaunayTraversalHelper* delaunay_helper,
  int bin_count,
  const std::vector<Eigen::Vector2d>& support_uv_points,
  TwoClassPCFStats& out_stats) {
  out_stats = TwoClassPCFStats{};
  if (bin_count <= 0) {
    return false;
  }
  for (auto& hist : out_stats.hist_counts) {
    hist.assign(static_cast<size_t>(bin_count), 0);
  }
  for (auto& plot : out_stats.hist_plot) {
    plot.assign(static_cast<size_t>(bin_count), 0.0f);
  }
  for (auto& channel_plots : out_stats.individual_plots) {
    channel_plots.clear();
  }
  for (auto& plot : out_stats.hist_min_plot) {
    plot.assign(
      static_cast<size_t>(bin_count),
      std::numeric_limits<float>::infinity());
  }
  for (auto& plot : out_stats.hist_max_plot) {
    plot.assign(static_cast<size_t>(bin_count), 0.0f);
  }
  out_stats.combined_hist_counts.assign(static_cast<size_t>(bin_count), 0);
  out_stats.combined_hist_plot.assign(static_cast<size_t>(bin_count), 0.0f);

  if (!delaunay_helper || !delaunay_helper->is_ready() || uv_points.empty()) {
    return false;
  }

  const int n = static_cast<int>(uv_points.size());
  std::vector<int> sanitized_classes(static_cast<size_t>(n), 0);
  for (int i = 0; i < n; ++i) {
    sanitized_classes[static_cast<size_t>(i)] =
      (i < static_cast<int>(class_ids.size()))
        ? sanitize_pattern_class_id(class_ids[static_cast<size_t>(i)])
        : 0;
    ++out_stats.class_counts[static_cast<size_t>(sanitized_classes[static_cast<size_t>(i)])];
  }

  std::array<std::vector<std::vector<int>>, kTwoClassPairChannelCount> point_hist_by_channel;
  for (auto& channel_hist : point_hist_by_channel) {
    channel_hist.assign(
      static_cast<size_t>(n),
      std::vector<int>(static_cast<size_t>(bin_count), 0));
  }

  for (int i = 0; i + 1 < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const int k = delaunay_helper->count_triangles_crossed(
        uv_points[static_cast<size_t>(i)],
        uv_points[static_cast<size_t>(j)]);
      if (k < 0) {
        continue;
      }
      const int bin = std::min(k, bin_count - 1);
      const int channel = two_class_pair_channel(
        sanitized_classes[static_cast<size_t>(i)],
        sanitized_classes[static_cast<size_t>(j)]);
      ++out_stats.hist_counts[static_cast<size_t>(channel)][static_cast<size_t>(bin)];
      ++out_stats.pair_counts[static_cast<size_t>(channel)];
      ++out_stats.combined_hist_counts[static_cast<size_t>(bin)];
      ++out_stats.combined_pair_count;
      ++point_hist_by_channel[static_cast<size_t>(channel)][static_cast<size_t>(i)][static_cast<size_t>(bin)];
      ++point_hist_by_channel[static_cast<size_t>(channel)][static_cast<size_t>(j)][static_cast<size_t>(bin)];
    }
  }

  const std::vector<Eigen::Vector2d>& support_points =
    support_uv_points.empty() ? uv_points : support_uv_points;
  std::vector<std::vector<int>> point_support(
    static_cast<size_t>(n),
    std::vector<int>(static_cast<size_t>(bin_count), 0));
  for (int i = 0; i < n; ++i) {
    for (const Eigen::Vector2d& support_uv : support_points) {
      const int k = delaunay_helper->count_triangles_crossed(
        uv_points[static_cast<size_t>(i)],
        support_uv);
      if (k >= 0) {
        const int bin = std::min(k, bin_count - 1);
        ++point_support[static_cast<size_t>(i)][static_cast<size_t>(bin)];
      }
    }
  }

  for (int channel = 0; channel < kTwoClassPairChannelCount; ++channel) {
    int valid_anchors = 0;
    for (int i = 0; i < n; ++i) {
      const int class_id = sanitized_classes[static_cast<size_t>(i)];
      const bool anchor_is_relevant =
        (channel == 0 && class_id == 0) ||
        (channel == 1 && class_id == 1) ||
        (channel == 2);
      if (!anchor_is_relevant) {
        continue;
      }
      bool has_valid_support = false;
      std::vector<float> local_plot(static_cast<size_t>(bin_count), 0.0f);
      std::vector<char> local_has_bin(static_cast<size_t>(bin_count), 0);
      for (int k = 0; k < bin_count; ++k) {
        if (point_support[static_cast<size_t>(i)][static_cast<size_t>(k)] > 0) {
          has_valid_support = true;
          const int denom = point_support[static_cast<size_t>(i)][static_cast<size_t>(k)];
          local_plot[static_cast<size_t>(k)] =
            static_cast<float>(
              point_hist_by_channel[static_cast<size_t>(channel)][static_cast<size_t>(i)][static_cast<size_t>(k)]) /
            static_cast<float>(denom);
          local_has_bin[static_cast<size_t>(k)] = 1;
        }
      }
      if (!has_valid_support) {
        continue;
      }
      out_stats.individual_plots[static_cast<size_t>(channel)].push_back(local_plot);
      ++valid_anchors;
      for (int k = 0; k < bin_count; ++k) {
        if (local_has_bin[static_cast<size_t>(k)] == 0) {
          continue;
        }
        out_stats.hist_plot[static_cast<size_t>(channel)][static_cast<size_t>(k)] +=
          local_plot[static_cast<size_t>(k)];
        out_stats.hist_min_plot[static_cast<size_t>(channel)][static_cast<size_t>(k)] =
          std::min(
            out_stats.hist_min_plot[static_cast<size_t>(channel)][static_cast<size_t>(k)],
            local_plot[static_cast<size_t>(k)]);
        out_stats.hist_max_plot[static_cast<size_t>(channel)][static_cast<size_t>(k)] =
          std::max(
            out_stats.hist_max_plot[static_cast<size_t>(channel)][static_cast<size_t>(k)],
            local_plot[static_cast<size_t>(k)]);
      }
    }
    if (valid_anchors > 0) {
      const float inv_valid = 1.0f / static_cast<float>(valid_anchors);
      for (float& v : out_stats.hist_plot[static_cast<size_t>(channel)]) {
        v *= inv_valid;
      }
    }
    for (int k = 0; k < bin_count; ++k) {
      float& min_v = out_stats.hist_min_plot[static_cast<size_t>(channel)][static_cast<size_t>(k)];
      if (!std::isfinite(min_v)) {
        min_v = 0.0f;
      }
    }
  }

  int combined_channels = 0;
  for (int channel = 0; channel < kTwoClassPairChannelCount; ++channel) {
    const bool channel_possible =
      (channel == 0 && out_stats.class_counts[0] >= 2) ||
      (channel == 1 && out_stats.class_counts[1] >= 2) ||
      (channel == 2 && out_stats.class_counts[0] > 0 && out_stats.class_counts[1] > 0);
    if (!channel_possible) {
      continue;
    }
    ++combined_channels;
    for (int k = 0; k < bin_count; ++k) {
      out_stats.combined_hist_plot[static_cast<size_t>(k)] +=
        out_stats.hist_plot[static_cast<size_t>(channel)][static_cast<size_t>(k)];
    }
  }
  if (combined_channels > 0) {
    const float inv_channels = 1.0f / static_cast<float>(combined_channels);
    for (float& v : out_stats.combined_hist_plot) {
      v *= inv_channels;
    }
  }

  return true;
}

std::vector<int> build_two_class_support_counts(
  const PatternRegionState& state,
  const Eigen::Vector2d& uv,
  int support_row,
  const std::vector<Eigen::Vector2d>& output_support_uv,
  const DelaunayTraversalHelper* delaunay_helper) {
  const int bin_count = std::max(1, state.voronoi_pcf_bin_count);
  std::vector<int> support_counts(static_cast<size_t>(bin_count), 0);
  if (support_row >= 0 &&
      state.output_support_denominator_cache_valid &&
      support_row < static_cast<int>(state.output_support_k_denominator_cache.size())) {
    const std::vector<int>& cached =
      state.output_support_k_denominator_cache[static_cast<size_t>(support_row)];
    const int copy_bins = std::min(bin_count, static_cast<int>(cached.size()));
    for (int k = 0; k < copy_bins; ++k) {
      support_counts[static_cast<size_t>(k)] = cached[static_cast<size_t>(k)];
    }
    return support_counts;
  }
  if (!delaunay_helper || !delaunay_helper->is_ready()) {
    return support_counts;
  }
  for (const Eigen::Vector2d& support_uv : output_support_uv) {
    const int k = delaunay_helper->count_triangles_crossed(uv, support_uv);
    if (k >= 0) {
      const int bin = std::min(k, bin_count - 1);
      ++support_counts[static_cast<size_t>(bin)];
    }
  }
  return support_counts;
}

bool build_fixed_anchor_two_class_evaluation_cache(
  const PatternRegionState& state,
  const std::vector<Eigen::Vector2d>& uv_points,
  const std::vector<int>& class_ids,
  const std::vector<int>& triangle_indices,
  const std::vector<int>& support_row_for_triangle,
  const std::vector<Eigen::Vector2d>& output_support_uv,
  const DelaunayTraversalHelper* delaunay_helper,
  int anchor_class_id,
  FixedAnchorTwoClassEvaluationCache& out_cache) {
  out_cache = FixedAnchorTwoClassEvaluationCache{};
  out_cache.anchor_class_id = sanitize_pattern_class_id(anchor_class_id);
  out_cache.dependent_class_id = 1 - out_cache.anchor_class_id;

  for (size_t i = 0; i < uv_points.size(); ++i) {
    const int class_id =
      (i < class_ids.size())
        ? sanitize_pattern_class_id(class_ids[i])
        : 0;
    if (class_id != out_cache.anchor_class_id) {
      continue;
    }
    int support_row = -1;
    const int tri_idx =
      (i < triangle_indices.size())
        ? triangle_indices[i]
        : -1;
    if (tri_idx >= 0 &&
        tri_idx < static_cast<int>(support_row_for_triangle.size())) {
      support_row = support_row_for_triangle[static_cast<size_t>(tri_idx)];
    }
    out_cache.anchor_uv_points.push_back(uv_points[i]);
    out_cache.anchor_support_rows.push_back(support_row);
    out_cache.anchor_support_counts.push_back(
      build_two_class_support_counts(
        state,
        uv_points[i],
        support_row,
        output_support_uv,
        delaunay_helper));
  }

  const int bin_count = std::max(1, state.voronoi_pcf_bin_count);
  out_cache.anchor_cross_target_avg_plot.assign(static_cast<size_t>(bin_count), 0.0f);
  out_cache.anchor_cross_target_min_plot.assign(
    static_cast<size_t>(bin_count),
    std::numeric_limits<float>::infinity());
  out_cache.anchor_cross_target_max_plot.assign(static_cast<size_t>(bin_count), 0.0f);
  out_cache.anchor_cross_near_hop_radius =
    std::min(std::max(0, bin_count - 1), 2);
  out_cache.dependent_cross_target_avg_plot.assign(static_cast<size_t>(bin_count), 0.0f);
  out_cache.dependent_cross_target_min_plot.assign(
    static_cast<size_t>(bin_count),
    std::numeric_limits<float>::infinity());
  out_cache.dependent_cross_target_max_plot.assign(static_cast<size_t>(bin_count), 0.0f);
  out_cache.dependent_cross_near_hop_radius =
    std::min(std::max(0, bin_count - 1), 2);

  std::vector<Eigen::Vector2d> input_inside_uv;
  std::vector<int> input_inside_class_ids;
  const size_t input_point_count = std::min(
    state.pattern_points_uv.size(),
    state.pattern_processing_uv.size());
  input_inside_uv.reserve(input_point_count);
  input_inside_class_ids.reserve(input_point_count);
  for (size_t i = 0; i < input_point_count; ++i) {
    if (!point_in_polygon_for_pcf(state.pattern_points_uv[i], state.input_boundary_uv)) {
      continue;
    }
    input_inside_uv.push_back(state.pattern_processing_uv[i]);
    input_inside_class_ids.push_back(
      (i < state.pattern_point_class_ids.size())
        ? sanitize_pattern_class_id(state.pattern_point_class_ids[i])
        : 0);
  }

  if (!input_inside_uv.empty()) {
    std::vector<Eigen::Vector2d> input_support_uv;
    if (!collect_triangle_center_candidates_in_polygon(
          state.input_boundary_uv,
          delaunay_helper,
          input_support_uv)) {
      input_support_uv = input_inside_uv;
    }

    std::vector<std::vector<int>> point_support(
      input_inside_uv.size(),
      std::vector<int>(static_cast<size_t>(bin_count), 0));
    std::vector<std::vector<int>> anchor_cross_hist(
      input_inside_uv.size(),
      std::vector<int>(static_cast<size_t>(bin_count), 0));

    for (size_t i = 0; i < input_inside_uv.size(); ++i) {
      for (const Eigen::Vector2d& support_uv : input_support_uv) {
        const int k = delaunay_helper->count_triangles_crossed(
          input_inside_uv[i],
          support_uv);
        if (k < 0) {
          continue;
        }
        const int bin = std::min(k, bin_count - 1);
        ++point_support[i][static_cast<size_t>(bin)];
      }
    }

    for (size_t i = 0; i + 1 < input_inside_uv.size(); ++i) {
      for (size_t j = i + 1; j < input_inside_uv.size(); ++j) {
        if (input_inside_class_ids[i] == input_inside_class_ids[j]) {
          continue;
        }
        const int k = delaunay_helper->count_triangles_crossed(
          input_inside_uv[i],
          input_inside_uv[j]);
        if (k < 0) {
          continue;
        }
        const int bin = std::min(k, bin_count - 1);
        ++anchor_cross_hist[i][static_cast<size_t>(bin)];
        ++anchor_cross_hist[j][static_cast<size_t>(bin)];
      }
    }

    int valid_anchor_targets = 0;
    int valid_dependent_targets = 0;
    std::array<std::vector<int>, kFixedAnchorTemplateMaxHopRadius + 1>
      anchor_near_counts_by_radius;
    std::array<std::vector<int>, kFixedAnchorTemplateMaxHopRadius + 1>
      dependent_near_counts_by_radius;
    for (size_t i = 0; i < input_inside_uv.size(); ++i) {
      const int class_id = input_inside_class_ids[i];
      if (class_id != out_cache.anchor_class_id &&
          class_id != out_cache.dependent_class_id) {
        continue;
      }
      std::vector<float> local_plot;
      if (!build_two_class_local_distribution(
            anchor_cross_hist[i],
            point_support[i],
            local_plot)) {
        continue;
      }
      if (class_id == out_cache.anchor_class_id) {
        out_cache.anchor_cross_target_distributions.push_back(local_plot);
        ++valid_anchor_targets;
        int cumulative_near_count = 0;
        const int max_near_radius = std::min(
          kFixedAnchorTemplateMaxHopRadius,
          static_cast<int>(anchor_cross_hist[i].size()) - 1);
        for (int k = 0; k <= max_near_radius; ++k) {
          cumulative_near_count += anchor_cross_hist[i][static_cast<size_t>(k)];
          anchor_near_counts_by_radius[static_cast<size_t>(k)].push_back(
            cumulative_near_count);
        }
      } else {
        out_cache.dependent_cross_target_distributions.push_back(local_plot);
        ++valid_dependent_targets;
        int cumulative_near_count = 0;
        const int max_near_radius = std::min(
          kFixedAnchorTemplateMaxHopRadius,
          static_cast<int>(anchor_cross_hist[i].size()) - 1);
        for (int k = 0; k <= max_near_radius; ++k) {
          cumulative_near_count += anchor_cross_hist[i][static_cast<size_t>(k)];
          dependent_near_counts_by_radius[static_cast<size_t>(k)].push_back(
            cumulative_near_count);
        }
      }
      for (int k = 0; k < bin_count; ++k) {
        const float value =
          (k < static_cast<int>(local_plot.size()))
            ? local_plot[static_cast<size_t>(k)]
            : 0.0f;
        if (class_id == out_cache.anchor_class_id) {
          out_cache.anchor_cross_target_avg_plot[static_cast<size_t>(k)] += value;
          out_cache.anchor_cross_target_min_plot[static_cast<size_t>(k)] =
            std::min(
              out_cache.anchor_cross_target_min_plot[static_cast<size_t>(k)],
              value);
          out_cache.anchor_cross_target_max_plot[static_cast<size_t>(k)] =
            std::max(
              out_cache.anchor_cross_target_max_plot[static_cast<size_t>(k)],
              value);
        } else {
          out_cache.dependent_cross_target_avg_plot[static_cast<size_t>(k)] += value;
          out_cache.dependent_cross_target_min_plot[static_cast<size_t>(k)] =
            std::min(
              out_cache.dependent_cross_target_min_plot[static_cast<size_t>(k)],
              value);
          out_cache.dependent_cross_target_max_plot[static_cast<size_t>(k)] =
            std::max(
              out_cache.dependent_cross_target_max_plot[static_cast<size_t>(k)],
              value);
        }
      }
    }

    if (valid_anchor_targets > 0) {
      const float inv_valid = 1.0f / static_cast<float>(valid_anchor_targets);
      for (float& value : out_cache.anchor_cross_target_avg_plot) {
        value *= inv_valid;
      }
    }
    if (valid_dependent_targets > 0) {
      const float inv_valid = 1.0f / static_cast<float>(valid_dependent_targets);
      for (float& value : out_cache.dependent_cross_target_avg_plot) {
        value *= inv_valid;
      }
    }
    for (float& value : out_cache.anchor_cross_target_min_plot) {
      if (!std::isfinite(value)) {
        value = 0.0f;
      }
    }
    for (float& value : out_cache.dependent_cross_target_min_plot) {
      if (!std::isfinite(value)) {
        value = 0.0f;
      }
    }
    finalize_fixed_anchor_near_count_target(
      anchor_near_counts_by_radius,
      out_cache.anchor_cross_near_hop_radius,
      out_cache.anchor_cross_near_hop_radius,
      out_cache.anchor_cross_target_near_lower_count,
      out_cache.anchor_cross_target_near_count);
    finalize_fixed_anchor_near_count_target(
      dependent_near_counts_by_radius,
      out_cache.dependent_cross_near_hop_radius,
      out_cache.dependent_cross_near_hop_radius,
      out_cache.dependent_cross_target_near_lower_count,
      out_cache.dependent_cross_target_near_count);

    std::vector<size_t> input_anchor_indices;
    std::vector<size_t> input_dependent_indices;
    input_anchor_indices.reserve(input_inside_uv.size());
    input_dependent_indices.reserve(input_inside_uv.size());
    for (size_t i = 0; i < input_inside_class_ids.size(); ++i) {
      const int class_id = input_inside_class_ids[i];
      if (class_id == out_cache.anchor_class_id) {
        input_anchor_indices.push_back(i);
      } else if (class_id == out_cache.dependent_class_id) {
        input_dependent_indices.push_back(i);
      }
    }

    for (size_t anchor_point_idx : input_anchor_indices) {
      std::vector<float> anchor_distribution;
      int template_hop_radius = 2;
      if (anchor_point_idx < anchor_cross_hist.size() &&
          anchor_point_idx < point_support.size() &&
          build_two_class_local_distribution(
            anchor_cross_hist[anchor_point_idx],
            point_support[anchor_point_idx],
            anchor_distribution)) {
        double target_peak = 0.0;
        const int eval_bins = std::min(
          kFixedAnchorTemplateMaxHopRadius,
          static_cast<int>(anchor_distribution.size()));
        for (int k = 0; k < eval_bins; ++k) {
          target_peak = std::max(
            target_peak,
            static_cast<double>(anchor_distribution[static_cast<size_t>(k)]));
        }
        const double meaningful_floor = std::max(0.02, 0.15 * target_peak);
        int farthest_meaningful_bin = -1;
        for (int k = 0; k < eval_bins; ++k) {
          if (static_cast<double>(anchor_distribution[static_cast<size_t>(k)]) >=
              meaningful_floor) {
            farthest_meaningful_bin = k;
          }
        }
        if (farthest_meaningful_bin >= 0) {
          template_hop_radius = std::clamp(
            farthest_meaningful_bin + 1,
            2,
            kFixedAnchorTemplateMaxHopRadius);
        }
      }

      std::vector<Eigen::Vector2d> local_target_offsets;
      local_target_offsets.reserve(input_dependent_indices.size());
      int nearest_dependent_k = std::numeric_limits<int>::max();
      Eigen::Vector2d nearest_dependent_offset = Eigen::Vector2d::Zero();
      bool have_nearest_dependent = false;
      for (size_t dependent_point_idx : input_dependent_indices) {
        const int k = delaunay_helper->count_triangles_crossed(
          input_inside_uv[anchor_point_idx],
          input_inside_uv[dependent_point_idx]);
        if (k < 0) {
          continue;
        }
        const Eigen::Vector2d offset =
          input_inside_uv[dependent_point_idx] - input_inside_uv[anchor_point_idx];
        if (k <= template_hop_radius) {
          local_target_offsets.push_back(offset);
        }
        if (k < nearest_dependent_k) {
          nearest_dependent_k = k;
          nearest_dependent_offset = offset;
          have_nearest_dependent = true;
        }
      }

      if (local_target_offsets.empty() && have_nearest_dependent) {
        local_target_offsets.push_back(nearest_dependent_offset);
        template_hop_radius = std::clamp(
          std::max(2, nearest_dependent_k),
          2,
          kFixedAnchorTemplateMaxHopRadius);
      }

      if (local_target_offsets.empty()) {
        continue;
      }

      double scale_sq = 0.0;
      for (const Eigen::Vector2d& offset : local_target_offsets) {
        scale_sq += offset.squaredNorm();
      }
      scale_sq /= static_cast<double>(std::max<size_t>(1, local_target_offsets.size()));
      scale_sq = std::max(1e-6, scale_sq);

      out_cache.anchor_template_target_offsets.push_back(std::move(local_target_offsets));
      out_cache.anchor_template_target_hop_radii.push_back(template_hop_radius);
      out_cache.anchor_template_target_scale_sq.push_back(scale_sq);
    }
  }

  out_cache.valid = !out_cache.anchor_uv_points.empty();
  return out_cache.valid;
}

static int get_support_pairwise_dist(
  const PatternRegionState& state,
  int si,
  int sj) noexcept;

bool build_fixed_anchor_two_class_proposal_cache(
  const PatternRegionState& state,
  const std::vector<Eigen::Vector2d>& uv_points,
  const std::vector<int>& class_ids,
  const std::vector<int>& triangle_indices,
  const std::vector<int>& support_row_for_triangle,
  const std::vector<Eigen::Vector2d>& output_support_uv,
  const DelaunayTraversalHelper* delaunay_helper,
  const FixedAnchorTwoClassEvaluationCache& fixed_anchor_cache,
  FixedAnchorTwoClassProposalCache& out_cache) {
  out_cache = FixedAnchorTwoClassProposalCache{};
  if (!fixed_anchor_cache.valid ||
      !delaunay_helper ||
      !delaunay_helper->is_ready()) {
    return false;
  }

  out_cache.anchor_class_id = fixed_anchor_cache.anchor_class_id;
  out_cache.dependent_class_id = fixed_anchor_cache.dependent_class_id;
  out_cache.bin_count = std::max(1, state.voronoi_pcf_bin_count);
  out_cache.dependent_row_for_point.assign(uv_points.size(), -1);

  for (size_t i = 0; i < uv_points.size(); ++i) {
    const int class_id =
      (i < class_ids.size())
        ? sanitize_pattern_class_id(class_ids[i])
        : 0;
    if (class_id != out_cache.dependent_class_id) {
      continue;
    }
    int support_row = -1;
    const int tri_idx =
      (i < triangle_indices.size())
        ? triangle_indices[i]
        : -1;
    if (tri_idx >= 0 &&
        tri_idx < static_cast<int>(support_row_for_triangle.size())) {
      support_row = support_row_for_triangle[static_cast<size_t>(tri_idx)];
    }
    out_cache.dependent_row_for_point[i] =
      static_cast<int>(out_cache.dependent_point_indices.size());
    out_cache.dependent_point_indices.push_back(i);
    out_cache.dependent_support_rows.push_back(support_row);
    out_cache.dependent_support_counts.push_back(
      build_two_class_support_counts(
        state,
        uv_points[i],
        support_row,
        output_support_uv,
        delaunay_helper));
  }

  const size_t dependent_count = out_cache.dependent_point_indices.size();
  if (dependent_count == 0) {
    return false;
  }

  out_cache.dependent_self_hist.assign(
    dependent_count,
    std::vector<int>(static_cast<size_t>(out_cache.bin_count), 0));
  out_cache.dependent_cross_hist.assign(
    dependent_count,
    std::vector<int>(static_cast<size_t>(out_cache.bin_count), 0));

  const auto pair_distance = [&](const Eigen::Vector2d& lhs_uv,
                                 int lhs_support_row,
                                 const Eigen::Vector2d& rhs_uv,
                                 int rhs_support_row) {
    if (lhs_support_row >= 0 && rhs_support_row >= 0) {
      return get_support_pairwise_dist(state, lhs_support_row, rhs_support_row);
    }
    return delaunay_helper->count_triangles_crossed(lhs_uv, rhs_uv);
  };

  for (size_t dep_idx = 0; dep_idx < dependent_count; ++dep_idx) {
    const size_t point_idx = out_cache.dependent_point_indices[dep_idx];
    for (size_t anchor_idx = 0;
         anchor_idx < fixed_anchor_cache.anchor_uv_points.size();
         ++anchor_idx) {
      const int k = pair_distance(
        uv_points[point_idx],
        out_cache.dependent_support_rows[dep_idx],
        fixed_anchor_cache.anchor_uv_points[anchor_idx],
        fixed_anchor_cache.anchor_support_rows[anchor_idx]);
      if (k < 0) {
        continue;
      }
      const int bin = std::min(k, out_cache.bin_count - 1);
      ++out_cache.dependent_cross_hist[dep_idx][static_cast<size_t>(bin)];
    }
  }

  for (size_t lhs_idx = 0; lhs_idx + 1 < dependent_count; ++lhs_idx) {
    const size_t lhs_point_idx = out_cache.dependent_point_indices[lhs_idx];
    for (size_t rhs_idx = lhs_idx + 1; rhs_idx < dependent_count; ++rhs_idx) {
      const size_t rhs_point_idx = out_cache.dependent_point_indices[rhs_idx];
      const int k = pair_distance(
        uv_points[lhs_point_idx],
        out_cache.dependent_support_rows[lhs_idx],
        uv_points[rhs_point_idx],
        out_cache.dependent_support_rows[rhs_idx]);
      if (k < 0) {
        continue;
      }
      const int bin = std::min(k, out_cache.bin_count - 1);
      ++out_cache.dependent_self_hist[lhs_idx][static_cast<size_t>(bin)];
      ++out_cache.dependent_self_hist[rhs_idx][static_cast<size_t>(bin)];
    }
  }

  out_cache.valid = true;
  return true;
}

double two_class_count_penalty(
  const std::array<int, kPatternClassCount>& output_counts,
  const std::array<int, kPatternClassCount>& target_counts);

double two_class_distribution_error_for_active_channels(
  const PatternRegionState& state,
  const TwoClassPCFStats& output_stats,
  const std::array<bool, kTwoClassPairChannelCount>& active_channels,
  const std::array<int, kPatternClassCount>& target_counts,
  bool include_count_penalty) {
  double error = 0.0;
  double active_weight_sum = 0.0;
  for (int channel = 0; channel < kTwoClassPairChannelCount; ++channel) {
    if (!active_channels[static_cast<size_t>(channel)] ||
        !state.two_class_pair_channel_enabled[static_cast<size_t>(channel)]) {
      continue;
    }
    const bool target_channel_possible =
      (channel == 0 && state.two_class_voronoi_pcf_points_inside[0] >= 2) ||
      (channel == 1 && state.two_class_voronoi_pcf_points_inside[1] >= 2) ||
      (channel == 2 &&
       state.two_class_voronoi_pcf_points_inside[0] > 0 &&
       state.two_class_voronoi_pcf_points_inside[1] > 0);
    if (!target_channel_possible) {
      continue;
    }
    const double channel_weight = std::max(
      0.0f,
      state.two_class_pair_channel_weights[static_cast<size_t>(channel)]);
    if (channel_weight <= 0.0) {
      continue;
    }
    error += channel_weight * weighted_distribution_l2(
      output_stats.hist_plot[static_cast<size_t>(channel)],
      state.two_class_voronoi_pcf_hist_plot[static_cast<size_t>(channel)]);
    active_weight_sum += channel_weight;

    if (state.two_class_envelope_penalty_enabled) {
      const auto& target_min =
        state.two_class_voronoi_pcf_hist_min_plot[static_cast<size_t>(channel)];
      const auto& target_max =
        state.two_class_voronoi_pcf_hist_max_plot[static_cast<size_t>(channel)];
      const auto& output_min =
        output_stats.hist_min_plot[static_cast<size_t>(channel)];
      const auto& output_max =
        output_stats.hist_max_plot[static_cast<size_t>(channel)];
      const int eval_bins = std::max(
        std::max(static_cast<int>(target_min.size()), static_cast<int>(target_max.size())),
        std::max(static_cast<int>(output_min.size()), static_cast<int>(output_max.size())));
      double envelope_error = 0.0;
      constexpr double kEnvelopeSlack = 0.015;
      for (int k = 0; k < eval_bins; ++k) {
        const double t_min =
          (k < static_cast<int>(target_min.size()))
            ? static_cast<double>(target_min[static_cast<size_t>(k)])
            : 0.0;
        const double t_max =
          (k < static_cast<int>(target_max.size()))
            ? static_cast<double>(target_max[static_cast<size_t>(k)])
            : 0.0;
        const double o_min =
          (k < static_cast<int>(output_min.size()))
            ? static_cast<double>(output_min[static_cast<size_t>(k)])
            : 0.0;
        const double o_max =
          (k < static_cast<int>(output_max.size()))
            ? static_cast<double>(output_max[static_cast<size_t>(k)])
            : 0.0;
        const double under = std::max(0.0, (t_min - kEnvelopeSlack) - o_min);
        const double over = std::max(0.0, o_max - (t_max + kEnvelopeSlack));
        envelope_error += under * under + over * over;
      }
      error +=
        channel_weight *
        static_cast<double>(std::max(0.0f, state.two_class_envelope_penalty_weight)) *
        envelope_error;
    }
  }
  if (active_weight_sum > 0.0) {
    error /= active_weight_sum;
  } else if (!include_count_penalty) {
    return std::numeric_limits<double>::infinity();
  }

  if (!include_count_penalty) {
    return error;
  }
  return error + two_class_count_penalty(output_stats.class_counts, target_counts);
}

double two_class_distribution_error(
  const PatternRegionState& state,
  const TwoClassPCFStats& output_stats,
  const std::array<int, kPatternClassCount>& target_counts) {
  const std::array<bool, kTwoClassPairChannelCount> active_channels = {
    true,
    true,
    true,
  };
  return two_class_distribution_error_for_active_channels(
    state,
    output_stats,
    active_channels,
    target_counts,
    true);
}

void clear_two_class_output_stats(PatternRegionState& state) {
  state.output_voronoi_pcf_hist_counts.clear();
  state.output_voronoi_pcf_hist_plot.clear();
  for (auto& hist : state.two_class_output_voronoi_pcf_hist_counts) {
    hist.clear();
  }
  for (auto& plot : state.two_class_output_voronoi_pcf_hist_plot) {
    plot.clear();
  }
  state.two_class_output_counts = {0, 0};
  state.two_class_output_voronoi_pcf_pair_count = {0, 0, 0};
  state.output_voronoi_pcf_max_k = 0;
  state.output_voronoi_pcf_pair_count = 0;
  state.output_voronoi_pcf_ready = false;
  state.output_voronoi_pcf_energy = 0.0;
  state.output_voronoi_objective_energy = 0.0;
}

std::array<int, kPatternClassCount> estimate_two_class_target_counts(
  const PatternRegionState& state,
  int input_support_count,
  int output_support_count) {
  std::array<int, kPatternClassCount> target_counts = {0, 0};
  if (output_support_count <= 0) {
    return target_counts;
  }
  const int input_point_count =
    std::max(0, state.two_class_voronoi_pcf_points_inside[0]) +
    std::max(0, state.two_class_voronoi_pcf_points_inside[1]);
  const int safe_input_support =
    std::max(1, input_support_count > 0 ? input_support_count : input_point_count);
  int total = 0;
  for (int class_id = 0; class_id < kPatternClassCount; ++class_id) {
    const int input_count =
      std::max(0, state.two_class_voronoi_pcf_points_inside[static_cast<size_t>(class_id)]);
    if (input_count <= 0) {
      target_counts[static_cast<size_t>(class_id)] = 0;
      continue;
    }
    const double density =
      static_cast<double>(input_count) / static_cast<double>(safe_input_support);
    int target =
      static_cast<int>(std::llround(density * static_cast<double>(output_support_count)));
    target = std::max(input_count >= 2 ? 2 : 1, target);
    target_counts[static_cast<size_t>(class_id)] = target;
    total += target;
  }
  while (total > output_support_count) {
    int reduce_class = -1;
    for (int class_id = 0; class_id < kPatternClassCount; ++class_id) {
      const int min_count =
        state.two_class_voronoi_pcf_points_inside[static_cast<size_t>(class_id)] >= 2
          ? 2
          : (state.two_class_voronoi_pcf_points_inside[static_cast<size_t>(class_id)] > 0 ? 1 : 0);
      if (target_counts[static_cast<size_t>(class_id)] > min_count &&
          (reduce_class < 0 ||
           target_counts[static_cast<size_t>(class_id)] >
             target_counts[static_cast<size_t>(reduce_class)])) {
        reduce_class = class_id;
      }
    }
    if (reduce_class < 0) {
      for (int class_id = 0; class_id < kPatternClassCount; ++class_id) {
        if (target_counts[static_cast<size_t>(class_id)] > 0 &&
            (reduce_class < 0 ||
             target_counts[static_cast<size_t>(class_id)] >
               target_counts[static_cast<size_t>(reduce_class)])) {
          reduce_class = class_id;
        }
      }
      if (reduce_class < 0) {
        break;
      }
    }
    --target_counts[static_cast<size_t>(reduce_class)];
    --total;
  }
  return target_counts;
}

int two_class_locked_anchor_count(
  const PatternRegionState& state,
  int anchor_class_id) {
  const int safe_anchor_class_id = sanitize_pattern_class_id(anchor_class_id);
  if (state.two_class_locked_anchor_class_id == safe_anchor_class_id &&
      !state.two_class_locked_anchor_points_uv.empty()) {
    return static_cast<int>(state.two_class_locked_anchor_points_uv.size());
  }

  int anchor_count = 0;
  for (size_t i = 0; i < state.output_pattern_points_uv.size(); ++i) {
    const int class_id =
      (i < state.output_pattern_class_ids.size())
        ? sanitize_pattern_class_id(state.output_pattern_class_ids[i])
        : 0;
    if (class_id == safe_anchor_class_id) {
      ++anchor_count;
    }
  }
  return anchor_count;
}

std::array<int, kPatternClassCount> adapt_two_class_target_counts_for_locked_anchor(
  const PatternRegionState& state,
  const std::array<int, kPatternClassCount>& base_target_counts,
  int output_support_count) {
  std::array<int, kPatternClassCount> adapted_target_counts = base_target_counts;
  if (!state.two_class_sequential_dependency_enabled) {
    return adapted_target_counts;
  }

  const int anchor_class_id =
    sanitize_pattern_class_id(state.two_class_anchor_class_id);
  const int dependent_class_id = 1 - anchor_class_id;
  const int locked_anchor_count =
    two_class_locked_anchor_count(state, anchor_class_id);
  if (locked_anchor_count <= 0) {
    return adapted_target_counts;
  }

  adapted_target_counts[static_cast<size_t>(anchor_class_id)] = locked_anchor_count;

  const int input_anchor_count =
    std::max(0, state.two_class_voronoi_pcf_points_inside[static_cast<size_t>(anchor_class_id)]);
  const int input_dependent_count =
    std::max(0, state.two_class_voronoi_pcf_points_inside[static_cast<size_t>(dependent_class_id)]);
  const int dependent_min_count =
    input_dependent_count >= 2 ? 2 : (input_dependent_count > 0 ? 1 : 0);

  int adapted_dependent_count =
    std::max(0, adapted_target_counts[static_cast<size_t>(dependent_class_id)]);
  if (input_dependent_count <= 0) {
    adapted_dependent_count = 0;
  } else if (input_anchor_count > 0) {
    adapted_dependent_count = static_cast<int>(std::llround(
      static_cast<double>(locked_anchor_count) *
      static_cast<double>(input_dependent_count) /
      static_cast<double>(input_anchor_count)));
  }
  adapted_dependent_count = std::max(dependent_min_count, adapted_dependent_count);
  if (output_support_count > 0) {
    adapted_dependent_count = std::min(
      adapted_dependent_count,
      std::max(0, output_support_count - locked_anchor_count));
  }
  adapted_target_counts[static_cast<size_t>(dependent_class_id)] = adapted_dependent_count;
  return adapted_target_counts;
}

int two_class_near_field_split_for_bins(int bin_count) {
  return std::max(1, std::max(1, bin_count) / 2);
}

int infer_two_class_local_proposal_radius_from_distributions(
  const std::vector<float>& current_distribution,
  const std::vector<float>& target_distribution,
  int eval_bins,
  int hist_bin_count) {
  if (target_distribution.empty() || eval_bins <= 0) {
    return 1;
  }

  const int safe_eval_bins = std::max(1, eval_bins);
  const int strong_prefix_bins = std::min(
    safe_eval_bins,
    std::max(2, std::min(6, two_class_near_field_split_for_bins(hist_bin_count))));
  const int radius_eval_bins = std::min(
    safe_eval_bins,
    std::max(3, std::min(5, strong_prefix_bins + 1)));

  double target_peak = 0.0;
  for (int k = 0; k < radius_eval_bins; ++k) {
    const double tgt_v =
      (k < static_cast<int>(target_distribution.size()))
        ? static_cast<double>(target_distribution[static_cast<size_t>(k)])
        : 0.0;
    target_peak = std::max(target_peak, tgt_v);
  }

  double positive_deficit_mass = 0.0;
  double positive_deficit_center = 0.0;
  int farthest_meaningful_deficit_bin = -1;
  for (int k = 0; k < radius_eval_bins; ++k) {
    const double out_v =
      (k < static_cast<int>(current_distribution.size()))
        ? static_cast<double>(current_distribution[static_cast<size_t>(k)])
        : 0.0;
    const double tgt_v =
      (k < static_cast<int>(target_distribution.size()))
        ? static_cast<double>(target_distribution[static_cast<size_t>(k)])
        : 0.0;
    const double deficit = std::max(0.0, tgt_v - out_v);
    positive_deficit_mass += deficit;
    positive_deficit_center += static_cast<double>(k + 1) * deficit;
    const double meaningful_deficit_floor = std::max(0.02, 0.15 * target_peak);
    if (deficit >= meaningful_deficit_floor) {
      farthest_meaningful_deficit_bin = k;
    }
  }

  if (positive_deficit_mass <= 0.05) {
    return 1;
  }

  const double deficit_centroid =
    positive_deficit_center / std::max(1e-9, positive_deficit_mass);
  int proposal_radius = 1;
  if (farthest_meaningful_deficit_bin >= 1 ||
      deficit_centroid >= 1.25 ||
      positive_deficit_mass >= 0.10) {
    proposal_radius = 2;
  }
  if (farthest_meaningful_deficit_bin >= 3 ||
      deficit_centroid >= 2.6) {
    proposal_radius = 3;
  }
  return std::clamp(proposal_radius, 1, 3);
}

double normalized_histogram_l2(
  const std::vector<int>& hist_counts,
  int pair_count,
  const std::vector<float>& target_normalized) {
  const std::vector<float> current_normalized = normalized_histogram(hist_counts, pair_count);
  return weighted_distribution_l2(current_normalized, target_normalized);
}

double average_individual_histogram_l2(
  const std::vector<std::vector<int>>& point_hist_counts,
  const std::vector<std::vector<int>>& point_support_counts,
  int bin_count,
  const std::vector<float>& target_avg_individual) {
  const std::vector<float> current_avg_individual =
    average_individual_histogram(point_hist_counts, point_support_counts, bin_count);
  return weighted_distribution_l2(current_avg_individual, target_avg_individual);
}

double min_cost_perfect_assignment(const std::vector<std::vector<double>>& cost_matrix) {
  const int n = static_cast<int>(cost_matrix.size());
  if (n <= 0) {
    return 0.0;
  }
  for (const auto& row : cost_matrix) {
    if (static_cast<int>(row.size()) != n) {
      return std::numeric_limits<double>::infinity();
    }
  }

  const double kInf = std::numeric_limits<double>::infinity();
  std::vector<double> u(static_cast<size_t>(n + 1), 0.0);
  std::vector<double> v(static_cast<size_t>(n + 1), 0.0);
  std::vector<int> p(static_cast<size_t>(n + 1), 0);
  std::vector<int> way(static_cast<size_t>(n + 1), 0);

  for (int i = 1; i <= n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(static_cast<size_t>(n + 1), kInf);
    std::vector<char> used(static_cast<size_t>(n + 1), 0);
    do {
      used[static_cast<size_t>(j0)] = 1;
      const int i0 = p[static_cast<size_t>(j0)];
      double delta = kInf;
      int j1 = 0;
      for (int j = 1; j <= n; ++j) {
        if (used[static_cast<size_t>(j)] != 0) {
          continue;
        }
        const double cur =
          cost_matrix[static_cast<size_t>(i0 - 1)][static_cast<size_t>(j - 1)] -
          u[static_cast<size_t>(i0)] -
          v[static_cast<size_t>(j)];
        if (cur < minv[static_cast<size_t>(j)]) {
          minv[static_cast<size_t>(j)] = cur;
          way[static_cast<size_t>(j)] = j0;
        }
        if (minv[static_cast<size_t>(j)] < delta) {
          delta = minv[static_cast<size_t>(j)];
          j1 = j;
        }
      }
      if (!std::isfinite(delta)) {
        return std::numeric_limits<double>::infinity();
      }
      for (int j = 0; j <= n; ++j) {
        if (used[static_cast<size_t>(j)] != 0) {
          u[static_cast<size_t>(p[static_cast<size_t>(j)])] += delta;
          v[static_cast<size_t>(j)] -= delta;
        } else {
          minv[static_cast<size_t>(j)] -= delta;
        }
      }
      j0 = j1;
    } while (p[static_cast<size_t>(j0)] != 0);

    do {
      const int j1 = way[static_cast<size_t>(j0)];
      p[static_cast<size_t>(j0)] = p[static_cast<size_t>(j1)];
      j0 = j1;
    } while (j0 != 0);
  }

  double total_cost = 0.0;
  for (int j = 1; j <= n; ++j) {
    const int i = p[static_cast<size_t>(j)];
    if (i <= 0) {
      continue;
    }
    total_cost += cost_matrix[static_cast<size_t>(i - 1)][static_cast<size_t>(j - 1)];
  }
  return total_cost;
}

double min_cost_assignment_with_unmatched(
  const std::vector<std::vector<double>>& match_costs,
  const std::vector<double>& unmatched_output_costs,
  const std::vector<double>& unmatched_target_costs) {
  const int out_count = static_cast<int>(match_costs.size());
  const int tgt_count = static_cast<int>(unmatched_target_costs.size());
  if (out_count <= 0 || tgt_count <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  for (const auto& row : match_costs) {
    if (static_cast<int>(row.size()) != tgt_count) {
      return std::numeric_limits<double>::infinity();
    }
  }
  if (static_cast<int>(unmatched_output_costs.size()) != out_count) {
    return std::numeric_limits<double>::infinity();
  }

  const int assignment_size = out_count + tgt_count;
  constexpr double kForbiddenAssignmentCost = 1e12;
  std::vector<std::vector<double>> assignment_cost(
    static_cast<size_t>(assignment_size),
    std::vector<double>(static_cast<size_t>(assignment_size), kForbiddenAssignmentCost));

  for (int oi = 0; oi < out_count; ++oi) {
    for (int ti = 0; ti < tgt_count; ++ti) {
      const double cost = match_costs[static_cast<size_t>(oi)][static_cast<size_t>(ti)];
      assignment_cost[static_cast<size_t>(oi)][static_cast<size_t>(ti)] =
        std::isfinite(cost) ? cost : kForbiddenAssignmentCost;
    }
    assignment_cost[static_cast<size_t>(oi)][static_cast<size_t>(tgt_count + oi)] =
      std::isfinite(unmatched_output_costs[static_cast<size_t>(oi)])
        ? unmatched_output_costs[static_cast<size_t>(oi)]
        : kForbiddenAssignmentCost;
  }

  for (int ti = 0; ti < tgt_count; ++ti) {
    assignment_cost[static_cast<size_t>(out_count + ti)][static_cast<size_t>(ti)] =
      std::isfinite(unmatched_target_costs[static_cast<size_t>(ti)])
        ? unmatched_target_costs[static_cast<size_t>(ti)]
        : kForbiddenAssignmentCost;
    for (int oi = 0; oi < out_count; ++oi) {
      assignment_cost[static_cast<size_t>(out_count + ti)][static_cast<size_t>(tgt_count + oi)] = 0.0;
    }
  }

  return min_cost_perfect_assignment(assignment_cost);
}

bool build_two_class_local_distribution(
  const std::vector<int>& row_counts,
  const std::vector<int>& support_counts,
  std::vector<float>& out_distribution) {
  const int eval_bins = std::min(
    static_cast<int>(row_counts.size()),
    static_cast<int>(support_counts.size()));
  out_distribution.assign(static_cast<size_t>(std::max(0, eval_bins)), 0.0f);
  bool has_valid_support = false;
  for (int k = 0; k < eval_bins; ++k) {
    const int denom = support_counts[static_cast<size_t>(k)];
    if (denom <= 0) {
      continue;
    }
    has_valid_support = true;
    out_distribution[static_cast<size_t>(k)] =
      static_cast<float>(row_counts[static_cast<size_t>(k)]) /
      static_cast<float>(denom);
  }
  return has_valid_support;
}

double compute_two_class_near_count_band_error(
  int near_count,
  int target_lower_count,
  double target_near_count,
  double* out_deficit_error = nullptr,
  double* out_excess_error = nullptr) {
  if (out_deficit_error != nullptr) {
    *out_deficit_error = 0.0;
  }
  if (out_excess_error != nullptr) {
    *out_excess_error = 0.0;
  }

  double deficit_error = 0.0;
  if (target_lower_count > 0 && near_count < target_lower_count) {
    const double deficit =
      static_cast<double>(target_lower_count - near_count) /
      static_cast<double>(std::max(1, target_lower_count));
    deficit_error = deficit * deficit;
  }

  const double upper_target = std::max(
    static_cast<double>(std::max(0, target_lower_count)),
    target_near_count);
  double excess_error = 0.0;
  if (upper_target > 0.0 && static_cast<double>(near_count) > upper_target) {
    const double excess =
      (static_cast<double>(near_count) - upper_target) /
      std::max(1.0, upper_target);
    excess_error = excess * excess;
  }

  if (out_deficit_error != nullptr) {
    *out_deficit_error = deficit_error;
  }
  if (out_excess_error != nullptr) {
    *out_excess_error = excess_error;
  }
  return deficit_error + excess_error;
}

void finalize_fixed_anchor_near_count_target(
  std::array<std::vector<int>, kFixedAnchorTemplateMaxHopRadius + 1>& near_counts_by_radius,
  int default_radius,
  int& out_radius,
  int& out_lower_count,
  double& out_target_count) {
  out_radius = std::clamp(default_radius, 0, kFixedAnchorTemplateMaxHopRadius);
  out_lower_count = 0;
  out_target_count = 0.0;

  int fallback_radius = out_radius;
  int fallback_median = 0;
  for (int radius = 0; radius <= kFixedAnchorTemplateMaxHopRadius; ++radius) {
    std::vector<int>& counts = near_counts_by_radius[static_cast<size_t>(radius)];
    if (counts.empty()) {
      continue;
    }
    std::sort(counts.begin(), counts.end());
    const size_t lower_idx = std::min(counts.size() - 1, counts.size() / 4);
    const size_t median_idx = counts.size() / 2;
    const int lower_count = counts[lower_idx];
    const int median_count = counts[median_idx];
    if (median_count > fallback_median) {
      fallback_median = median_count;
      fallback_radius = radius;
    }
    if (lower_count > 0) {
      out_radius = radius;
      out_lower_count = lower_count;
      out_target_count = static_cast<double>(median_count);
      return;
    }
  }

  if (fallback_median > 0) {
    std::vector<int>& counts =
      near_counts_by_radius[static_cast<size_t>(fallback_radius)];
    out_radius = fallback_radius;
    out_lower_count = 1;
    out_target_count =
      counts.empty()
        ? static_cast<double>(fallback_median)
        : static_cast<double>(counts[counts.size() / 2]);
  }
}

double two_class_individual_target_match_cost(
  const std::vector<float>& out_dist,
  const std::vector<float>& tgt_dist,
  int hist_bin_count) {
  const double weighted_error = weighted_distribution_l2(out_dist, tgt_dist);
  const int m = std::min(
    static_cast<int>(tgt_dist.size()),
    std::max(1, hist_bin_count));
  const int strong_prefix_bins = std::min(
    m,
    std::max(2, std::min(6, two_class_near_field_split_for_bins(hist_bin_count))));
  double prefix_error = 0.0;
  double prefix_mass_error = 0.0;
  for (int k = 0; k < strong_prefix_bins; ++k) {
    const double out_v =
      (k < static_cast<int>(out_dist.size()))
        ? static_cast<double>(out_dist[static_cast<size_t>(k)])
        : 0.0;
    const double tgt_v =
      (k < static_cast<int>(tgt_dist.size()))
        ? static_cast<double>(tgt_dist[static_cast<size_t>(k)])
        : 0.0;
    const double d = out_v - tgt_v;
    prefix_error += d * d;
    prefix_mass_error += std::abs(d);
  }
  return weighted_error +
         12.0 * prefix_error +
         6.0 * prefix_mass_error;
}

double two_class_best_target_distribution_cost_from_set(
  const std::vector<float>& output_distribution,
  const std::vector<std::vector<float>>& target_distributions,
  const std::vector<float>& average_target_distribution,
  int hist_bin_count,
  const std::vector<float>** out_best_target_distribution = nullptr) {
  const std::vector<float>* best_target_distribution = nullptr;
  double best_cost = std::numeric_limits<double>::infinity();

  for (const auto& target_distribution : target_distributions) {
    const double cost = two_class_individual_target_match_cost(
      output_distribution,
      target_distribution,
      hist_bin_count);
    if (cost < best_cost) {
      best_cost = cost;
      best_target_distribution = &target_distribution;
    }
  }

  if ((best_target_distribution == nullptr || !std::isfinite(best_cost)) &&
      !average_target_distribution.empty()) {
    best_target_distribution = &average_target_distribution;
    best_cost = two_class_individual_target_match_cost(
      output_distribution,
      average_target_distribution,
      hist_bin_count);
  }

  if (out_best_target_distribution != nullptr) {
    *out_best_target_distribution = best_target_distribution;
  }
  return best_cost;
}

double two_class_best_target_distribution_cost(
  const PatternRegionState& state,
  int channel,
  const std::vector<float>& output_distribution,
  const std::vector<float>** out_best_target_distribution = nullptr) {
  const int safe_channel = std::clamp(channel, 0, kTwoClassPairChannelCount - 1);
  return two_class_best_target_distribution_cost_from_set(
    output_distribution,
    state.two_class_voronoi_pcf_individual_plots[static_cast<size_t>(safe_channel)],
    state.two_class_voronoi_pcf_hist_plot[static_cast<size_t>(safe_channel)],
    state.voronoi_pcf_bin_count,
    out_best_target_distribution);
}

double two_class_perfect_assignment_distribution_cost(
  const std::vector<std::vector<float>>& output_distributions,
  const std::vector<std::vector<float>>& target_distributions,
  int hist_bin_count) {
  const int out_count = static_cast<int>(output_distributions.size());
  const int tgt_count = static_cast<int>(target_distributions.size());
  if (out_count <= 0 || tgt_count <= 0 || out_count != tgt_count) {
    return std::numeric_limits<double>::infinity();
  }

  std::vector<std::vector<double>> match_costs(
    static_cast<size_t>(out_count),
    std::vector<double>(static_cast<size_t>(tgt_count), std::numeric_limits<double>::infinity()));
  for (int oi = 0; oi < out_count; ++oi) {
    for (int ti = 0; ti < tgt_count; ++ti) {
      match_costs[static_cast<size_t>(oi)][static_cast<size_t>(ti)] =
        two_class_individual_target_match_cost(
          output_distributions[static_cast<size_t>(oi)],
          target_distributions[static_cast<size_t>(ti)],
          hist_bin_count);
    }
  }

  const double assignment_cost = min_cost_perfect_assignment(match_costs);
  return assignment_cost / static_cast<double>(std::max(1, out_count));
}

double two_class_assignment_distribution_cost_with_unmatched_penalty(
  const std::vector<std::vector<float>>& output_distributions,
  const std::vector<std::vector<float>>& target_distributions,
  int hist_bin_count,
  double unmatched_penalty) {
  const int out_count = static_cast<int>(output_distributions.size());
  const int tgt_count = static_cast<int>(target_distributions.size());
  if (out_count <= 0 || tgt_count <= 0) {
    return std::numeric_limits<double>::infinity();
  }

  std::vector<std::vector<double>> match_costs(
    static_cast<size_t>(out_count),
    std::vector<double>(static_cast<size_t>(tgt_count), std::numeric_limits<double>::infinity()));
  std::vector<double> best_out_cost(
    static_cast<size_t>(out_count),
    std::numeric_limits<double>::infinity());
  std::vector<double> best_tgt_cost(
    static_cast<size_t>(tgt_count),
    std::numeric_limits<double>::infinity());

  for (int oi = 0; oi < out_count; ++oi) {
    for (int ti = 0; ti < tgt_count; ++ti) {
      const double cost = two_class_individual_target_match_cost(
        output_distributions[static_cast<size_t>(oi)],
        target_distributions[static_cast<size_t>(ti)],
        hist_bin_count);
      match_costs[static_cast<size_t>(oi)][static_cast<size_t>(ti)] = cost;
      best_out_cost[static_cast<size_t>(oi)] =
        std::min(best_out_cost[static_cast<size_t>(oi)], cost);
      best_tgt_cost[static_cast<size_t>(ti)] =
        std::min(best_tgt_cost[static_cast<size_t>(ti)], cost);
    }
  }

  std::vector<double> unmatched_output_costs(
    static_cast<size_t>(out_count),
    unmatched_penalty);
  std::vector<double> unmatched_target_costs(
    static_cast<size_t>(tgt_count),
    unmatched_penalty);
  for (int oi = 0; oi < out_count; ++oi) {
    unmatched_output_costs[static_cast<size_t>(oi)] =
      best_out_cost[static_cast<size_t>(oi)] + unmatched_penalty;
  }
  for (int ti = 0; ti < tgt_count; ++ti) {
    unmatched_target_costs[static_cast<size_t>(ti)] =
      best_tgt_cost[static_cast<size_t>(ti)] + unmatched_penalty;
  }

  const double assignment_cost = min_cost_assignment_with_unmatched(
    match_costs,
    unmatched_output_costs,
    unmatched_target_costs);
  const int normalizer = std::max(out_count, tgt_count);
  return assignment_cost / static_cast<double>(std::max(1, normalizer));
}

double two_class_assignment_distribution_cost(
  const std::vector<std::vector<float>>& output_distributions,
  const std::vector<std::vector<float>>& target_distributions,
  int hist_bin_count) {
  return two_class_assignment_distribution_cost_with_unmatched_penalty(
    output_distributions,
    target_distributions,
    hist_bin_count,
    1.0);
}

double fixed_anchor_template_offset_match_cost(
  const Eigen::Vector2d& output_offset,
  const Eigen::Vector2d& target_offset,
  double scale_sq) {
  const double safe_scale_sq = std::max(1e-6, scale_sq);
  const Eigen::Vector2d delta = output_offset - target_offset;
  const double radial_delta = output_offset.norm() - target_offset.norm();
  return
    delta.squaredNorm() / safe_scale_sq +
    0.35 * (radial_delta * radial_delta) / safe_scale_sq;
}

double fixed_anchor_template_offset_set_cost(
  const std::vector<Eigen::Vector2d>& output_offsets,
  const std::vector<Eigen::Vector2d>& target_offsets,
  double scale_sq,
  double unmatched_penalty) {
  if (output_offsets.empty() && target_offsets.empty()) {
    return 0.0;
  }
  if (output_offsets.empty()) {
    return unmatched_penalty * static_cast<double>(target_offsets.size());
  }
  if (target_offsets.empty()) {
    return unmatched_penalty * static_cast<double>(output_offsets.size());
  }

  const int out_count = static_cast<int>(output_offsets.size());
  const int tgt_count = static_cast<int>(target_offsets.size());
  std::vector<std::vector<double>> match_costs(
    static_cast<size_t>(out_count),
    std::vector<double>(static_cast<size_t>(tgt_count), std::numeric_limits<double>::infinity()));
  for (int oi = 0; oi < out_count; ++oi) {
    for (int ti = 0; ti < tgt_count; ++ti) {
      match_costs[static_cast<size_t>(oi)][static_cast<size_t>(ti)] =
        fixed_anchor_template_offset_match_cost(
          output_offsets[static_cast<size_t>(oi)],
          target_offsets[static_cast<size_t>(ti)],
          scale_sq);
    }
  }

  if (out_count == tgt_count) {
    return min_cost_perfect_assignment(match_costs) /
           static_cast<double>(std::max(1, out_count));
  }

  std::vector<double> unmatched_output_costs(
    static_cast<size_t>(out_count),
    unmatched_penalty);
  std::vector<double> unmatched_target_costs(
    static_cast<size_t>(tgt_count),
    unmatched_penalty);
  for (int oi = 0; oi < out_count; ++oi) {
    const auto row_min_it = std::min_element(
      match_costs[static_cast<size_t>(oi)].begin(),
      match_costs[static_cast<size_t>(oi)].end());
    const double best_cost =
      (row_min_it != match_costs[static_cast<size_t>(oi)].end())
        ? *row_min_it
        : 0.0;
    unmatched_output_costs[static_cast<size_t>(oi)] =
      best_cost + unmatched_penalty;
  }
  for (int ti = 0; ti < tgt_count; ++ti) {
    double best_cost = std::numeric_limits<double>::infinity();
    for (int oi = 0; oi < out_count; ++oi) {
      best_cost = std::min(
        best_cost,
        match_costs[static_cast<size_t>(oi)][static_cast<size_t>(ti)]);
    }
    unmatched_target_costs[static_cast<size_t>(ti)] =
      (std::isfinite(best_cost) ? best_cost : 0.0) + unmatched_penalty;
  }

  const double assignment_cost = min_cost_assignment_with_unmatched(
    match_costs,
    unmatched_output_costs,
    unmatched_target_costs);
  return assignment_cost /
         static_cast<double>(std::max(1, std::max(out_count, tgt_count)));
}

double fixed_anchor_best_template_cost(
  const FixedAnchorTwoClassEvaluationCache& fixed_anchor_cache,
  const std::array<std::vector<Eigen::Vector2d>, kFixedAnchorTemplateMaxHopRadius + 1>& output_offsets_by_radius,
  int* out_best_template_idx = nullptr) {
  constexpr double kTemplateUnmatchedPenalty = 4.0;
  if (out_best_template_idx != nullptr) {
    *out_best_template_idx = -1;
  }
  if (fixed_anchor_cache.anchor_template_target_offsets.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  double best_cost = std::numeric_limits<double>::infinity();
  int best_template_idx = -1;
  for (size_t template_idx = 0;
       template_idx < fixed_anchor_cache.anchor_template_target_offsets.size();
       ++template_idx) {
    const int hop_radius =
      (template_idx < fixed_anchor_cache.anchor_template_target_hop_radii.size())
        ? std::clamp(
            fixed_anchor_cache.anchor_template_target_hop_radii[template_idx],
            1,
            kFixedAnchorTemplateMaxHopRadius)
        : 2;
    const double scale_sq =
      (template_idx < fixed_anchor_cache.anchor_template_target_scale_sq.size())
        ? fixed_anchor_cache.anchor_template_target_scale_sq[template_idx]
        : 1e-4;
    const double template_cost = fixed_anchor_template_offset_set_cost(
      output_offsets_by_radius[static_cast<size_t>(hop_radius)],
      fixed_anchor_cache.anchor_template_target_offsets[template_idx],
      scale_sq,
      kTemplateUnmatchedPenalty);
    if (template_cost < best_cost) {
      best_cost = template_cost;
      best_template_idx = static_cast<int>(template_idx);
    }
  }

  if (out_best_template_idx != nullptr) {
    *out_best_template_idx = best_template_idx;
  }
  return best_cost;
}

double two_class_count_penalty(
  const std::array<int, kPatternClassCount>& output_counts,
  const std::array<int, kPatternClassCount>& target_counts) {
  constexpr double kClassCountPenalty = 0.15;
  double error = 0.0;
  for (int class_id = 0; class_id < kPatternClassCount; ++class_id) {
    const int target = std::max(0, target_counts[static_cast<size_t>(class_id)]);
    const int output_count = output_counts[static_cast<size_t>(class_id)];
    if (target <= 0) {
      if (output_count > 0) {
        error += kClassCountPenalty * static_cast<double>(output_count * output_count);
      }
      continue;
    }
    const double gap = static_cast<double>(output_count - target);
    const double scale = static_cast<double>(std::max(1, target));
    error += kClassCountPenalty * (gap * gap) / scale;
  }
  return error;
}

double two_class_distribution_error_for_active_channels_with_individual_targets(
  const PatternRegionState& state,
  const TwoClassPCFStats& output_stats,
  const std::array<std::vector<std::vector<float>>, kTwoClassPairChannelCount>& output_individual_plots,
  const std::array<bool, kTwoClassPairChannelCount>& active_channels,
  const std::array<int, kPatternClassCount>& target_counts,
  bool include_count_penalty) {
  double error = 0.0;
  double active_weight_sum = 0.0;
  const int bin_count = std::max(1, state.voronoi_pcf_bin_count);
  for (int channel = 0; channel < kTwoClassPairChannelCount; ++channel) {
    if (!active_channels[static_cast<size_t>(channel)] ||
        !state.two_class_pair_channel_enabled[static_cast<size_t>(channel)]) {
      continue;
    }
    const bool target_channel_possible =
      (channel == 0 && state.two_class_voronoi_pcf_points_inside[0] >= 2) ||
      (channel == 1 && state.two_class_voronoi_pcf_points_inside[1] >= 2) ||
      (channel == 2 &&
       state.two_class_voronoi_pcf_points_inside[0] > 0 &&
       state.two_class_voronoi_pcf_points_inside[1] > 0);
    if (!target_channel_possible) {
      continue;
    }
    const double channel_weight = std::max(
      0.0f,
      state.two_class_pair_channel_weights[static_cast<size_t>(channel)]);
    if (channel_weight <= 0.0) {
      continue;
    }

    double channel_error = std::numeric_limits<double>::infinity();
    const auto& target_distributions =
      state.two_class_voronoi_pcf_individual_plots[static_cast<size_t>(channel)];
    const auto& output_distributions =
      output_individual_plots[static_cast<size_t>(channel)];
    if (!target_distributions.empty() && !output_distributions.empty()) {
      channel_error = two_class_assignment_distribution_cost(
        output_distributions,
        target_distributions,
        bin_count);
    }
    if (!std::isfinite(channel_error)) {
      channel_error = weighted_distribution_l2(
        output_stats.hist_plot[static_cast<size_t>(channel)],
        state.two_class_voronoi_pcf_hist_plot[static_cast<size_t>(channel)]);
    }

    error += channel_weight * channel_error;
    active_weight_sum += channel_weight;

    if (state.two_class_envelope_penalty_enabled) {
      const auto& target_min =
        state.two_class_voronoi_pcf_hist_min_plot[static_cast<size_t>(channel)];
      const auto& target_max =
        state.two_class_voronoi_pcf_hist_max_plot[static_cast<size_t>(channel)];
      const auto& output_min =
        output_stats.hist_min_plot[static_cast<size_t>(channel)];
      const auto& output_max =
        output_stats.hist_max_plot[static_cast<size_t>(channel)];
      const int eval_bins = std::max(
        std::max(static_cast<int>(target_min.size()), static_cast<int>(target_max.size())),
        std::max(static_cast<int>(output_min.size()), static_cast<int>(output_max.size())));
      double envelope_error = 0.0;
      constexpr double kEnvelopeSlack = 0.015;
      for (int k = 0; k < eval_bins; ++k) {
        const double t_min =
          (k < static_cast<int>(target_min.size()))
            ? static_cast<double>(target_min[static_cast<size_t>(k)])
            : 0.0;
        const double t_max =
          (k < static_cast<int>(target_max.size()))
            ? static_cast<double>(target_max[static_cast<size_t>(k)])
            : 0.0;
        const double o_min =
          (k < static_cast<int>(output_min.size()))
            ? static_cast<double>(output_min[static_cast<size_t>(k)])
            : 0.0;
        const double o_max =
          (k < static_cast<int>(output_max.size()))
            ? static_cast<double>(output_max[static_cast<size_t>(k)])
            : 0.0;
        const double under = std::max(0.0, (t_min - kEnvelopeSlack) - o_min);
        const double over = std::max(0.0, o_max - (t_max + kEnvelopeSlack));
        envelope_error += under * under + over * over;
      }
      error +=
        channel_weight *
        static_cast<double>(std::max(0.0f, state.two_class_envelope_penalty_weight)) *
        envelope_error;
    }
  }

  if (active_weight_sum > 0.0) {
    error /= active_weight_sum;
  } else if (!include_count_penalty) {
    return std::numeric_limits<double>::infinity();
  }

  if (include_count_penalty) {
    error += two_class_count_penalty(output_stats.class_counts, target_counts);
  }
  return error;
}

double zero_bin_leakage_mass(
  const std::vector<int>& hist_counts,
  int pair_count,
  const std::vector<int>& target_hist_counts) {
  if (pair_count <= 0 || hist_counts.empty() || target_hist_counts.empty()) {
    return 0.0;
  }
  const size_t n = std::max(hist_counts.size(), target_hist_counts.size());
  double leakage = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const int target_count = (i < target_hist_counts.size()) ? target_hist_counts[i] : 0;
    if (target_count != 0) {
      continue;
    }
    const int count = (i < hist_counts.size()) ? hist_counts[i] : 0;
    if (count > 0) {
      leakage += static_cast<double>(count) / static_cast<double>(pair_count);
    }
  }
  return leakage;
}

Eigen::Vector3d normalize_barycentric(const Eigen::Vector3d& bary) {
  Eigen::Vector3d clamped = bary.cwiseMax(0.0);
  const double sum = clamped.sum();
  if (sum <= 1e-14) {
    return Eigen::Vector3d::Constant(1.0 / 3.0);
  }
  return clamped / sum;
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

bool lift_uv_to_output_3d(
  const Eigen::Vector2d& uv,
  const DelaunayTraversalHelper* delaunay_helper,
  const Eigen::MatrixXd& points_3d,
  const Eigen::MatrixXd& points_uv,
  Eigen::Vector3d& out_3d) {
  if (!delaunay_helper || !delaunay_helper->is_ready()) {
    return false;
  }
  if (points_uv.cols() < 2 || points_3d.cols() < 3) {
    return false;
  }

  int tri_idx = -1;
  Eigen::Vector3i tri_vertices(-1, -1, -1);
  if (!delaunay_helper->find_containing_triangle(uv, tri_idx, tri_vertices)) {
    return false;
  }

  const int max_rows = std::min(points_uv.rows(), points_3d.rows());
  for (int c = 0; c < 3; ++c) {
    const int v = tri_vertices[c];
    if (v < 0 || v >= max_rows) {
      return false;
    }
  }

  const Eigen::Vector2d a = points_uv.row(tri_vertices[0]).head<2>().transpose();
  const Eigen::Vector2d b = points_uv.row(tri_vertices[1]).head<2>().transpose();
  const Eigen::Vector2d c = points_uv.row(tri_vertices[2]).head<2>().transpose();
  const Eigen::Vector3d bary = barycentric_from_2d(uv, a, b, c);

  const Eigen::Vector3d p0 = points_3d.row(tri_vertices[0]).head<3>().transpose();
  const Eigen::Vector3d p1 = points_3d.row(tri_vertices[1]).head<3>().transpose();
  const Eigen::Vector3d p2 = points_3d.row(tri_vertices[2]).head<3>().transpose();
  out_3d = bary[0] * p0 + bary[1] * p1 + bary[2] * p2;
  return true;
}

Eigen::Vector3d nearest_sample_3d(
  const Eigen::Vector2d& uv,
  const Eigen::MatrixXd& points_3d,
  const Eigen::MatrixXd& points_uv) {
  const int max_rows = std::min(points_uv.rows(), points_3d.rows());
  if (max_rows <= 0 || points_uv.cols() < 2 || points_3d.cols() < 3) {
    return Eigen::Vector3d::Zero();
  }

  int best_idx = 0;
  double best_d2 = std::numeric_limits<double>::infinity();
  for (int i = 0; i < max_rows; ++i) {
    const Eigen::Vector2d suv = points_uv.row(i).head<2>().transpose();
    const double d2 = (suv - uv).squaredNorm();
    if (d2 < best_d2) {
      best_d2 = d2;
      best_idx = i;
    }
  }
  return points_3d.row(best_idx).head<3>().transpose();
}

void clear_output_pattern_and_hist(PatternRegionState& state) {
  state.output_pattern_sample_indices.clear();
  state.output_pattern_points_3d.clear();
  state.output_pattern_points_uv.clear();
  state.output_pattern_class_ids.clear();
  clear_two_class_output_stats(state);
  state.optimizer_improvements = 0;
  state.optimizer_iterations_ran = 0;
  state.output_pattern_dirty = true;
}

void clear_output_pattern_and_hist(InteractionState& root_state) {
  clear_output_pattern_and_hist(active_region(root_state));
}

bool build_histogram_for_uv_points(
  const std::vector<Eigen::Vector2d>& uv_points,
  const DelaunayTraversalHelper* delaunay_helper,
  int bin_count,
  std::vector<int>& out_hist) {
  if (bin_count <= 0) {
    return false;
  }
  out_hist.assign(static_cast<size_t>(bin_count), 0);
  if (!delaunay_helper || !delaunay_helper->is_ready()) {
    return false;
  }

  for (size_t i = 0; i + 1 < uv_points.size(); ++i) {
    for (size_t j = i + 1; j < uv_points.size(); ++j) {
      const int k = delaunay_helper->count_triangles_crossed(
        uv_points[i], uv_points[j]);
      if (k >= 0) {
        add_hist_count_if_in_range(out_hist, k, 1);
      }
    }
  }
  return true;
}

bool build_pair_hist_and_average_individual_plot(
  const std::vector<Eigen::Vector2d>& uv_points,
  const DelaunayTraversalHelper* delaunay_helper,
  int bin_count,
  std::vector<int>& out_hist_counts,
  int& out_in_range_pair_count,
  std::vector<float>& out_avg_individual_plot,
  const std::vector<Eigen::Vector2d>* support_uv_points = nullptr,
  std::vector<std::vector<float>>* out_individual_distributions = nullptr,
  const std::vector<int>* cached_support_triangle_indices = nullptr,
  const std::vector<std::vector<int>>* cached_support_counts = nullptr,
  std::vector<float>* out_avg_shell_counts = nullptr,
  std::vector<std::vector<int>>* out_point_hist_counts = nullptr) {
  out_hist_counts.clear();
  out_avg_individual_plot.clear();
  out_in_range_pair_count = 0;
  if (out_individual_distributions) {
    out_individual_distributions->clear();
  }
  if (out_avg_shell_counts) {
    out_avg_shell_counts->clear();
  }
  if (out_point_hist_counts) {
    out_point_hist_counts->clear();
  }

  if (bin_count <= 0 || uv_points.size() < 2) {
    return false;
  }
  if (!delaunay_helper || !delaunay_helper->is_ready()) {
    return false;
  }

  out_hist_counts.assign(static_cast<size_t>(bin_count), 0);
  const int n = static_cast<int>(uv_points.size());
  const std::vector<Eigen::Vector2d>& support_points =
    (support_uv_points && !support_uv_points->empty())
      ? *support_uv_points
      : uv_points;
  std::vector<std::vector<int>> point_counts(
    static_cast<size_t>(n),
    std::vector<int>(static_cast<size_t>(bin_count), 0));
  std::vector<std::vector<int>> point_support_counts(
    static_cast<size_t>(n),
    std::vector<int>(static_cast<size_t>(bin_count), 0));

  std::unordered_map<int, int> tri_to_support_row;
  const bool can_use_cached_support =
    cached_support_triangle_indices &&
    cached_support_counts &&
    cached_support_triangle_indices->size() == support_points.size() &&
    cached_support_counts->size() == support_points.size();
  if (can_use_cached_support) {
    tri_to_support_row.reserve(cached_support_triangle_indices->size());
    for (size_t si = 0; si < cached_support_triangle_indices->size(); ++si) {
      tri_to_support_row.emplace((*cached_support_triangle_indices)[si], static_cast<int>(si));
    }
  }

  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const int k = delaunay_helper->count_triangles_crossed(
        uv_points[static_cast<size_t>(i)],
        uv_points[static_cast<size_t>(j)]);
      if (k < 0) {
        continue;  // Only skip invalid k values
      }
      // Use distortion-weighted effective distance if distortion is provided
      double d_eff = static_cast<double>(k);
      
      // Track in histogram if within range
      const int bin_idx = std::min(
        static_cast<int>(std::floor(d_eff)),
        bin_count - 1);
      if (bin_idx >= 0) {
        ++out_hist_counts[static_cast<size_t>(bin_idx)];
        ++point_counts[static_cast<size_t>(i)][static_cast<size_t>(bin_idx)];
        ++point_counts[static_cast<size_t>(j)][static_cast<size_t>(bin_idx)];
      }
    }
  }

  const int support_point_count = static_cast<int>(support_points.size());
#if defined(_OPENMP)
  #pragma omp parallel for schedule(static) if (n >= 64 && support_point_count >= 512)
#endif
  for (int i = 0; i < n; ++i) {
    bool loaded_from_cache = false;
    if (can_use_cached_support) {
      int tri_idx = -1;
      Eigen::Vector3i tri_vertices(-1, -1, -1);
      if (delaunay_helper->find_containing_triangle(
            uv_points[static_cast<size_t>(i)], tri_idx, tri_vertices)) {
        const auto it = tri_to_support_row.find(tri_idx);
        if (it != tri_to_support_row.end()) {
          const std::vector<int>& cached_row =
            (*cached_support_counts)[static_cast<size_t>(it->second)];
          const int copy_bins = std::min(bin_count, static_cast<int>(cached_row.size()));
          for (int k = 0; k < copy_bins; ++k) {
            point_support_counts[static_cast<size_t>(i)][static_cast<size_t>(k)] =
              cached_row[static_cast<size_t>(k)];
          }
          loaded_from_cache = true;
        }
      }
    }

    if (!loaded_from_cache) {
      for (int si = 0; si < support_point_count; ++si) {
        const Eigen::Vector2d& support_uv =
          support_points[static_cast<size_t>(si)];
        const int k = delaunay_helper->count_triangles_crossed(
          uv_points[static_cast<size_t>(i)],
          support_uv);
        if (k >= 0) {
          const int bin = std::min(k, bin_count - 1);
          ++point_support_counts[static_cast<size_t>(i)][static_cast<size_t>(bin)];
        }
      }
    }
  }

  out_in_range_pair_count = histogram_total_count(out_hist_counts);
  out_avg_individual_plot.assign(static_cast<size_t>(bin_count), 0.0f);
  if (out_individual_distributions) {
    out_individual_distributions->assign(
      static_cast<size_t>(n),
      std::vector<float>(static_cast<size_t>(bin_count), 0.0f));
  }
  if (out_avg_shell_counts) {
    out_avg_shell_counts->assign(static_cast<size_t>(bin_count), 0.0f);
  }
  if (out_point_hist_counts) {
    *out_point_hist_counts = point_counts;
  }
  int valid_points = 0;
  for (int i = 0; i < n; ++i) {
    bool has_valid_support = false;
    for (int k = 0; k < bin_count; ++k) {
      if (point_support_counts[static_cast<size_t>(i)][static_cast<size_t>(k)] > 0) {
        has_valid_support = true;
        break;
      }
    }
    if (!has_valid_support) {
      continue;
    }
    ++valid_points;
    for (int k = 0; k < bin_count; ++k) {
      if (out_avg_shell_counts) {
        (*out_avg_shell_counts)[static_cast<size_t>(k)] +=
          static_cast<float>(point_counts[static_cast<size_t>(i)][static_cast<size_t>(k)]);
      }
      const int denom = point_support_counts[static_cast<size_t>(i)][static_cast<size_t>(k)];
      if (denom <= 0) {
        continue;
      }
      const float p =
        static_cast<float>(point_counts[static_cast<size_t>(i)][static_cast<size_t>(k)]) /
        static_cast<float>(denom);
      out_avg_individual_plot[static_cast<size_t>(k)] +=
        p;
      if (out_individual_distributions) {
        (*out_individual_distributions)[static_cast<size_t>(i)][static_cast<size_t>(k)] = p;
      }
    }
  }
  if (valid_points > 0) {
    const float inv_valid = 1.0f / static_cast<float>(valid_points);
    for (float& v : out_avg_individual_plot) {
      v *= inv_valid;
    }
    if (out_avg_shell_counts) {
      for (float& v : *out_avg_shell_counts) {
        v *= inv_valid;
      }
    }
  }
  return true;
}

// Wrapper for point-in-or-on-polygon: boundary points included
bool point_in_or_on_polygon_for_pcf(
  const Eigen::Vector2d& uv,
  const Eigen::MatrixXd& boundary_poly) {
  return point_in_polygon_with_boundary(uv, boundary_poly, BoundaryMode::INCLUDE);
}

bool collect_triangle_center_candidates_in_polygon(
  const Eigen::MatrixXd& boundary_poly,
  const DelaunayTraversalHelper* delaunay_helper,
  std::vector<Eigen::Vector2d>& out_uv_centers,
  std::vector<int>* out_triangle_indices) {
  out_uv_centers.clear();
  if (out_triangle_indices) {
    out_triangle_indices->clear();
  }
  if (!delaunay_helper || !delaunay_helper->is_ready()) {
    return false;
  }
  if (boundary_poly.rows() < 3 || boundary_poly.cols() < 2) {
    return false;
  }

  const int tri_count = delaunay_helper->triangle_count();
  out_uv_centers.reserve(static_cast<size_t>(tri_count));
  if (out_triangle_indices) {
    out_triangle_indices->reserve(static_cast<size_t>(tri_count));
  }
  for (int tri = 0; tri < tri_count; ++tri) {
    Eigen::Vector2d center_uv = Eigen::Vector2d::Zero();
    if (!delaunay_helper->triangle_center(tri, center_uv)) {
      continue;
    }
    if (!point_in_or_on_polygon_for_pcf(center_uv, boundary_poly)) {
      continue;
    }
    out_uv_centers.push_back(center_uv);
    if (out_triangle_indices) {
      out_triangle_indices->push_back(tri);
    }
  }
  return !out_uv_centers.empty();
}

bool collect_output_triangle_center_candidates(
  const InteractionState& root_state,
  const DelaunayTraversalHelper* delaunay_helper,
  std::vector<Eigen::Vector2d>& out_uv_centers,
  std::vector<int>* out_triangle_indices) {
  const PatternRegionState& state = active_region(root_state);
  std::vector<int> local_triangle_indices;
  std::vector<int>* triangle_indices =
    out_triangle_indices ? out_triangle_indices : &local_triangle_indices;
  if (!collect_triangle_center_candidates_in_polygon(
    state.output_boundary_uv_poly,
    delaunay_helper,
    out_uv_centers,
    triangle_indices)) {
    return false;
  }
  filter_generated_patch_output_support_candidates(
    state,
    delaunay_helper,
    out_uv_centers,
    triangle_indices);
  return !out_uv_centers.empty();
}

bool same_boundary_polygon(
  const Eigen::MatrixXd& a,
  const Eigen::MatrixXd& b,
  double eps = 1e-12) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    return false;
  }
  if (a.size() == 0) {
    return true;
  }
  return ((a - b).cwiseAbs().maxCoeff() <= eps);
}

// Retrieve the cached hop-distance between two support candidates (O(1)).
// Returns -1 if the cache is invalid or indices are out of range.
static int get_support_pairwise_dist(
  const PatternRegionState& state,
  int si,
  int sj) noexcept {
  if (si == sj) {
    return 0;
  }
  if (si > sj) {
    std::swap(si, sj); // ensure sj > si
  }
  const size_t idx =
    static_cast<size_t>(sj) * (static_cast<size_t>(sj) - 1) / 2 +
    static_cast<size_t>(si);
  if (!state.output_support_pairwise_cache_valid ||
      idx >= state.output_support_pairwise_distances.size()) {
    return -1;
  }
  return static_cast<int>(state.output_support_pairwise_distances[idx]);
}

bool ensure_output_support_denominator_cache(
  InteractionState& root_state,
  const DelaunayTraversalHelper* delaunay_helper,
  int bin_count) {
  PatternRegionState& state = active_region(root_state);
  const int safe_bin_count = std::max(1, bin_count);
  if (!delaunay_helper || !delaunay_helper->is_ready()) {
    state.output_support_denominator_cache_valid = false;
    state.output_support_pairwise_cache_valid = false;
    return false;
  }
  if (state.output_boundary_uv_poly.rows() < 3 || state.output_boundary_uv_poly.cols() < 2) {
    state.output_support_denominator_cache_valid = false;
    state.output_support_pairwise_cache_valid = false;
    return false;
  }

  const int tri_count = delaunay_helper->triangle_count();
  const bool cache_shape_ok =
    state.output_support_uv_cache.size() == state.output_support_tri_indices_cache.size() &&
    state.output_support_uv_cache.size() == state.output_support_k_denominator_cache.size();
  const bool cache_is_current =
    state.output_support_denominator_cache_valid &&
    state.output_support_pairwise_cache_valid &&
    cache_shape_ok &&
    state.output_support_denominator_cache_bin_count == safe_bin_count &&
    state.output_support_denominator_cache_triangle_count == tri_count &&
    same_boundary_polygon(
      state.output_support_denominator_cache_boundary_uv,
      state.output_boundary_uv_poly);

  if (cache_is_current) {
    return true;
  }

  std::vector<Eigen::Vector2d> support_uv;
  std::vector<int> support_tri_indices;
  if (!collect_output_triangle_center_candidates(
        root_state,
        delaunay_helper,
        support_uv,
        &support_tri_indices)) {
    state.output_support_denominator_cache_valid = false;
    state.output_support_pairwise_cache_valid = false;
    state.output_support_uv_cache.clear();
    state.output_support_tri_indices_cache.clear();
    state.output_support_k_denominator_cache.clear();
    state.output_support_pairwise_distances.clear();
    return false;
  }

  const int support_count = static_cast<int>(support_uv.size());
  std::vector<std::vector<int>> support_counts(
    static_cast<size_t>(support_count),
    std::vector<int>(static_cast<size_t>(safe_bin_count), 0));

  // Allocate compact triangular pairwise distance matrix.
  // Entry for pair (sj > si): distances[sj*(sj-1)/2 + si].
  const size_t pairwise_size =
    static_cast<size_t>(support_count) *
    (support_count > 0 ? static_cast<size_t>(support_count - 1) : 0) / 2;
  state.output_support_pairwise_distances.assign(pairwise_size, int16_t(-1));
  state.output_support_pairwise_cache_valid = false;

#if defined(_OPENMP)
  #pragma omp parallel for schedule(static) if (support_count >= 512)
#endif
  for (int i = 0; i < support_count; ++i) {
    for (int j = 0; j < support_count; ++j) {
      const Eigen::Vector2d& support_pt =
        support_uv[static_cast<size_t>(j)];
      const int k = delaunay_helper->count_triangles_crossed(
        support_uv[static_cast<size_t>(i)], support_pt);
      if (k >= 0) {
        const int bin = std::min(k, safe_bin_count - 1);
        ++support_counts[static_cast<size_t>(i)][static_cast<size_t>(bin)];
      }
      // Store in triangular matrix: only when j < i (i.e. sj=i > si=j).
      // Each (i,j) pair is written by exactly one thread, so this is safe.
      // int16_t is sufficient: hop counts above ~32K only occur on meshes with
      // tens of thousands of triangles, which are far beyond practical use.
      if (j < i) {
        const size_t idx =
          static_cast<size_t>(i) * (static_cast<size_t>(i) - 1) / 2 +
          static_cast<size_t>(j);
        state.output_support_pairwise_distances[idx] =
          static_cast<int16_t>(std::clamp(k, -1, 32767));
      }
    }
  }

  state.output_support_uv_cache = std::move(support_uv);
  state.output_support_tri_indices_cache = std::move(support_tri_indices);
  state.output_support_k_denominator_cache = std::move(support_counts);
  state.output_support_denominator_cache_boundary_uv = state.output_boundary_uv_poly;
  state.output_support_denominator_cache_bin_count = safe_bin_count;
  state.output_support_denominator_cache_triangle_count = tri_count;
  state.output_support_denominator_cache_valid = true;
  state.output_support_pairwise_cache_valid = true;
  return true;
}

bool generate_baseline_pattern_on_graph(
  InteractionState& root_state,
  const DelaunayTraversalHelper* delaunay_helper,
  const Eigen::MatrixXd& points_3d,
  const Eigen::MatrixXd& points_uv) {
  PatternRegionState& state = active_region(root_state);
  clear_output_pattern_and_hist(root_state);

  if (!delaunay_helper || !delaunay_helper->is_ready()) {
    return false;
  }
  if (points_uv.rows() <= 0 || points_uv.cols() < 2 ||
      points_3d.rows() <= 0 || points_3d.cols() < 3) {
    return false;
  }
  if (state.output_boundary_uv_poly.rows() < 3 || state.output_boundary_uv_poly.cols() < 2) {
    return false;
  }

  if (!state.voronoi_pcf_ready) {
    compute_voronoi_pcf_histogram(root_state, delaunay_helper);
  }

  std::vector<Eigen::Vector2d> candidate_uv;
  std::vector<int> candidate_triangle_indices;
  if (!collect_output_triangle_center_candidates(
        root_state, delaunay_helper, candidate_uv, &candidate_triangle_indices)) {
    return false;
  }

  const int output_support_count = static_cast<int>(candidate_uv.size());
  if (output_support_count < 2) {
    return false;
  }

  int target_count = state.baseline_graph_point_count;
  target_count = std::max(2, target_count);
  target_count = std::min(target_count, output_support_count);

  // Precompute all pairwise graph distances (triangle crossing counts)
  std::vector<std::vector<double>> distance_matrix(output_support_count, 
      std::vector<double>(output_support_count, std::numeric_limits<double>::infinity()));

  int valid_pairs = 0;
  int invalid_pairs = 0;
  std::vector<double> all_finite_distances;
  all_finite_distances.reserve(static_cast<size_t>(output_support_count * (output_support_count - 1) / 2));

  // Compute all pairwise distances
  for (int i = 0; i < output_support_count; ++i) {
    distance_matrix[i][i] = 0.0;
    for (int j = i + 1; j < output_support_count; ++j) {
      const int k = delaunay_helper->count_triangles_crossed(
          candidate_uv[static_cast<size_t>(i)], 
          candidate_uv[static_cast<size_t>(j)]);
      if (k >= 0) {
        const double dist = static_cast<double>(k);
        distance_matrix[i][j] = dist;
        distance_matrix[j][i] = dist;
        all_finite_distances.push_back(dist);
        ++valid_pairs;
      } else {
        ++invalid_pairs;
      }
    }
  }

  // Validate graph connectivity
  const int total_pairs = valid_pairs + invalid_pairs;
  const double connectivity_ratio = (total_pairs > 0) 
      ? static_cast<double>(valid_pairs) / static_cast<double>(total_pairs)
      : 0.0;
  
  if (connectivity_ratio < 0.8) {
    std::cerr << "Warning: Only " << (connectivity_ratio * 100.0) << "% of point pairs "
              << "have valid graph distance. Graph may be poorly connected." << std::endl;
  }
  
  if (valid_pairs == 0) {
    std::cerr << "Error: No valid graph distances found. Cannot generate baseline pattern." << std::endl;
    return false;
  }

  // Fast lookup using precomputed matrix
  const auto graph_distance = [&](int a_pos, int b_pos) -> double {
    return distance_matrix[a_pos][b_pos];
  };

  std::random_device rd;
  std::mt19937 rng(rd());
  const int baseline_mode = std::clamp(state.baseline_graph_mode, 0, 3);
  std::vector<int> selected_positions;
  selected_positions.reserve(static_cast<size_t>(target_count));

  if (baseline_mode == 0) {
    // CSR on graph nodes: uniformly random subset.
    std::vector<int> order(output_support_count);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    selected_positions.assign(
      order.begin(),
      order.begin() + target_count);
  } else if (baseline_mode == 1) {
    // Clustered process: generate synthetic cluster centers in UV space,
    // then sample nearby triangle centers to each synthetic center
    const int center_count = std::clamp(
      static_cast<int>(std::llround(std::sqrt(static_cast<double>(target_count)) * state.clustered_num_centers_multiplier)),
      1,
      std::min(target_count, state.clustered_num_centers_max));

    // Compute adaptive radius based on graph distance distribution
    double cluster_radius = 1.0;
    
    if (!all_finite_distances.empty()) {
      std::vector<double> sorted_distances = all_finite_distances;
      std::sort(sorted_distances.begin(), sorted_distances.end());
      
      const double median_dist = sorted_distances[sorted_distances.size() / 2];
      const double percentile_25 = sorted_distances[static_cast<size_t>(sorted_distances.size() * 0.25)];
      const double percentile_75 = sorted_distances[static_cast<size_t>(sorted_distances.size() * 0.75)];
      
      // Use interactive parameters from UI
      const double sigma_multiplier = static_cast<double>(state.clustered_sigma_multiplier);
      const double maxk_multiplier = static_cast<double>(state.clustered_max_k_multiplier);
      
      cluster_radius = std::max(0.5, percentile_25 * sigma_multiplier);
      
      std::cout << "Clustered mode: " << center_count << " synthetic centers, median=" << median_dist 
                << ", p25=" << percentile_25 << ", p75=" << percentile_75
                << " | cluster_radius=" << cluster_radius << " (" << sigma_multiplier << "×p25)" << std::endl;
    } else {
      std::cerr << "Warning: No finite distances for adaptive clustering. Using defaults." << std::endl;
    }

    // STEP 1: Generate K random cluster centers in UV space (within output boundary)
    std::vector<Eigen::Vector2d> synthetic_centers;
    std::uniform_real_distribution<double> uv_dist(0.0, 1.0);
    
    // Get bounding box of output boundary
    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();
    for (int i = 0; i < state.output_boundary_uv_poly.rows(); ++i) {
      min_x = std::min(min_x, state.output_boundary_uv_poly(i, 0));
      max_x = std::max(max_x, state.output_boundary_uv_poly(i, 0));
      min_y = std::min(min_y, state.output_boundary_uv_poly(i, 1));
      max_y = std::max(max_y, state.output_boundary_uv_poly(i, 1));
    }
    
    // Rejection sampling: try to generate K cluster centers
    int attempts = 0;
    const int max_attempts = center_count * 100;
    while (static_cast<int>(synthetic_centers.size()) < center_count && attempts < max_attempts) {
      const double x = min_x + uv_dist(rng) * (max_x - min_x);
      const double y = min_y + uv_dist(rng) * (max_y - min_y);
      const Eigen::Vector2d pt(x, y);
      
      if (point_in_polygon_for_pcf(pt, state.output_boundary_uv_poly)) {
        synthetic_centers.push_back(pt);
      }
      ++attempts;
    }
    
    std::cout << "Generated " << synthetic_centers.size() << " cluster centers" << std::endl;
    
    if (synthetic_centers.size() < 1) {
      std::cerr << "Error: Could not generate any cluster centers." << std::endl;
      return false;
    }
    
    // STEP 2: For each support point, assign to nearest cluster anchor (Voronoi partition)
    // This ensures non-overlapping, balanced clusters
    std::vector<std::vector<int>> cluster_members(synthetic_centers.size());
    std::vector<int> nearest_anchor_idx(output_support_count, -1);
    
    for (int pos = 0; pos < output_support_count; ++pos) {
      int best_anchor_idx = -1;
      double best_dist = std::numeric_limits<double>::infinity();
      
      // Find nearest cluster anchor in graph distance
      for (size_t ci = 0; ci < synthetic_centers.size(); ++ci) {
        // Use the anchor point (nearest triangle center to synth center)
        // We need to find/store this - for now, use a simple approach:
        // Find closest support point to each synthetic center once, then reuse
        // This is done implicitly when we find nearest_idx below...
      }
    }
    
    // Better approach: precompute anchor points
    std::vector<int> cluster_anchors(synthetic_centers.size(), -1);
    for (size_t ci = 0; ci < synthetic_centers.size(); ++ci) {
      // Find nearest triangle center to this synthetic center (Euclidean in UV)
      int nearest_idx = 0;
      double nearest_dist = std::numeric_limits<double>::infinity();
      
      for (int pos = 0; pos < output_support_count; ++pos) {
        const Eigen::Vector2d& support_uv = candidate_uv[static_cast<size_t>(pos)];
        const double eu_dist = (support_uv - synthetic_centers[ci]).norm();
        
        if (eu_dist < nearest_dist) {
          nearest_dist = eu_dist;
          nearest_idx = pos;
        }
      }
      cluster_anchors[ci] = nearest_idx;
    }
    
    // Now assign each point to its nearest anchor (Voronoi partition)
    for (int pos = 0; pos < output_support_count; ++pos) {
      int best_cluster_idx = 0;
      double best_dist_to_anchor = std::numeric_limits<double>::infinity();
      
      for (size_t ci = 0; ci < cluster_anchors.size(); ++ci) {
        const double d = graph_distance(pos, cluster_anchors[ci]);
        if (std::isfinite(d) && d < best_dist_to_anchor) {
          best_dist_to_anchor = d;
          best_cluster_idx = static_cast<int>(ci);
        }
      }
      
      // Assign to nearest cluster
      cluster_members[static_cast<size_t>(best_cluster_idx)].push_back(pos);
    }
    
    // DIAGNOSTIC: Show cluster sizes
    std::cout << "Cluster membership:" << std::endl;
    for (size_t ci = 0; ci < synthetic_centers.size(); ++ci) {
      std::cout << "  Cluster " << ci << ": " << cluster_members[ci].size() << " members" << std::endl;
    }
    
    // STEP 3: Allocate target points evenly across clusters
    selected_positions.clear();
    
    // Calculate target per cluster (distribute evenly)
    const int target_per_cluster = target_count / static_cast<int>(synthetic_centers.size());
    const int remainder = target_count % static_cast<int>(synthetic_centers.size());
    
    std::vector<int> cluster_quota(synthetic_centers.size());
    for (size_t ci = 0; ci < synthetic_centers.size(); ++ci) {
      cluster_quota[ci] = target_per_cluster + (static_cast<int>(ci) < remainder ? 1 : 0);
    }
    
    std::cout << "Cluster quota:" << std::endl;
    for (size_t ci = 0; ci < synthetic_centers.size(); ++ci) {
      std::cout << "  Cluster " << ci << ": " << cluster_quota[ci] << " points (available: " 
                << cluster_members[ci].size() << ")" << std::endl;
    }
    
    // Sample from each cluster
    for (size_t ci = 0; ci < synthetic_centers.size(); ++ci) {
      std::vector<int> available = cluster_members[ci];
      int selected_from_this = 0;
      
      for (int quota = 0; quota < cluster_quota[ci]; ++quota) {
        if (available.empty()) {
          std::cerr << "Warning: Cluster " << ci << " ran out of points after " << quota 
                    << " / " << cluster_quota[ci] << std::endl;
          break;
        }
        
        // Random selection from available in this cluster
        std::uniform_int_distribution<size_t> pick(0, available.size() - 1);
        const size_t idx = pick(rng);
        const int chosen_pos = available[idx];
        selected_positions.push_back(chosen_pos);
        ++selected_from_this;
        
        // Mark as used by removing from available
        available.erase(available.begin() + static_cast<int>(idx));
      }
      
      if (selected_from_this < cluster_quota[ci]) {
        std::cerr << "  Actually selected: " << selected_from_this << std::endl;
      }
    }
    
    std::cout << "Total selected_positions size: " << selected_positions.size() << std::endl;
    
    // DIAGNOSTIC: Show final distribution
    std::vector<int> selected_per_cluster(synthetic_centers.size(), 0);
    for (int pos : selected_positions) {
      for (size_t ci = 0; ci < cluster_members.size(); ++ci) {
        for (int member : cluster_members[ci]) {
          if (pos == member) {
            selected_per_cluster[ci]++;
            goto found_in_cluster;
          }
        }
      }
      found_in_cluster:;
    }
    
    std::cout << "Final selected points per cluster:" << std::endl;
    for (size_t ci = 0; ci < synthetic_centers.size(); ++ci) {
      std::cout << "  Cluster " << ci << ": " << selected_per_cluster[ci] << " selected" << std::endl;
    }
  } else if (baseline_mode == 2) {
    // Regularly dispersed process: greedy farthest-point sampling in graph distance.
    std::uniform_int_distribution<int> pick_start(0, output_support_count - 1);
    const int first = pick_start(rng);
    std::vector<char> used(static_cast<size_t>(output_support_count), 0);
    used[static_cast<size_t>(first)] = 1;
    selected_positions.push_back(first);

    std::vector<double> min_dist(
      static_cast<size_t>(output_support_count),
      std::numeric_limits<double>::infinity());
    
    // Initialize distances from first point
    for (int i = 0; i < output_support_count; ++i) {
      if (i == first) {
        min_dist[static_cast<size_t>(i)] = 0.0;
      } else {
        min_dist[static_cast<size_t>(i)] = graph_distance(i, first);
      }
    }

    while (static_cast<int>(selected_positions.size()) < target_count) {
      int best_pos = -1;
      double best_dist = -1.0;
      
      // Find point with maximum distance to nearest selected point
      for (int i = 0; i < output_support_count; ++i) {
        if (used[static_cast<size_t>(i)]) {
          continue;
        }
        const double d = min_dist[static_cast<size_t>(i)];
        
        // Only consider finite distances, or if all are infinite, accept anything
        if (std::isfinite(d)) {
          if (d > best_dist) {
            best_dist = d;
            best_pos = i;
          }
        } else if (best_pos < 0) {
          // No finite option yet, take this as fallback
          best_pos = i;
          best_dist = d;
        }
      }
      
      if (best_pos < 0) {
        std::cerr << "Warning: Farthest-point sampling terminated early - no reachable points." << std::endl;
        break;
      }
      
      used[static_cast<size_t>(best_pos)] = 1;
      selected_positions.push_back(best_pos);
      
      // Update minimum distances to account for new point
      for (int i = 0; i < output_support_count; ++i) {
        if (used[static_cast<size_t>(i)]) {
          continue;
        }
        const double d = graph_distance(i, best_pos);
        if (d < min_dist[static_cast<size_t>(i)]) {
          min_dist[static_cast<size_t>(i)] = d;
        }
      }
    }

    // Fill remaining with random selection if needed
    if (static_cast<int>(selected_positions.size()) < target_count) {
      std::vector<int> leftovers;
      leftovers.reserve(static_cast<size_t>(output_support_count));
      for (int i = 0; i < output_support_count; ++i) {
        if (!used[static_cast<size_t>(i)]) {
          leftovers.push_back(i);
        }
      }
      std::shuffle(leftovers.begin(), leftovers.end(), rng);
      for (int pos : leftovers) {
        if (static_cast<int>(selected_positions.size()) >= target_count) {
          break;
        }
        selected_positions.push_back(pos);
      }
    }
    
    // Validation: check connectivity of selected points
    int disconnected_pairs = 0;
    for (size_t i = 0; i < selected_positions.size(); ++i) {
      for (size_t j = i + 1; j < selected_positions.size(); ++j) {
        const double d = graph_distance(selected_positions[i], selected_positions[j]);
        if (!std::isfinite(d)) {
          ++disconnected_pairs;
        }
      }
    }
    if (disconnected_pairs > 0) {
      const int total_selected_pairs = static_cast<int>(selected_positions.size() * (selected_positions.size() - 1) / 2);
      std::cerr << "Warning: " << disconnected_pairs << " / " << total_selected_pairs 
                << " pairs in selected pattern are disconnected in graph." << std::endl;
    }
  } else if (baseline_mode == 3) {
    // Theoretical g(r)=1: Compute expected pair distribution for CSR in graph distance
    // This accounts for graph geometry - g(r)=1 means actual = expected for random
    
    std::cout << "g(r)=1 mode: Computing expected baseline for CSR in graph distance" << std::endl;
    
    // Generate a large random sample to estimate expected distribution
    const int sample_size = std::min(target_count * 3, output_support_count);
    std::vector<int> order(output_support_count);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    
    std::vector<int> random_sample;
    random_sample.assign(order.begin(), order.begin() + sample_size);
    
    // Compute pairwise graph distances for this random sample
    std::map<int, int> distance_histogram;
    int total_pairs = 0;
    
    for (size_t i = 0; i < random_sample.size(); ++i) {
      for (size_t j = i + 1; j < random_sample.size(); ++j) {
        const double d = graph_distance(random_sample[i], random_sample[j]);
        if (std::isfinite(d)) {
          const int k = static_cast<int>(d);
          distance_histogram[k]++;
          ++total_pairs;
        }
      }
    }
    
    std::cout << "Sampled " << sample_size << " random points, " << total_pairs << " pairs" << std::endl;
    
    // Now generate the actual pattern using the same approach
    selected_positions.clear();
    std::shuffle(order.begin(), order.end(), rng);
    selected_positions.assign(order.begin(), order.begin() + target_count);
    
    std::cout << "Generated g(r)=1 baseline with " << selected_positions.size() << " points (uniform random)" << std::endl;
    std::cout << "Note: g(r)=1 in graph distance produces a naturally decaying histogram" << std::endl;
    std::cout << "      due to graph geometry (more neighbors at small k, fewer at large k)" << std::endl;
  }

  // Final validation of selected pattern (for modes 0-3)
  if (selected_positions.size() != static_cast<size_t>(target_count)) {
      std::cerr << "Warning: Generated " << selected_positions.size() << " points, "
                << "requested " << target_count << std::endl;
    }

    // Check graph distance statistics for generated pattern
    std::vector<double> pattern_distances;
    int pattern_disconnected = 0;
    for (size_t i = 0; i < selected_positions.size(); ++i) {
      for (size_t j = i + 1; j < selected_positions.size(); ++j) {
        const double d = graph_distance(selected_positions[i], selected_positions[j]);
        if (std::isfinite(d)) {
          pattern_distances.push_back(d);
        } else {
          ++pattern_disconnected;
        }
      }
    }

    if (!pattern_distances.empty()) {
      std::sort(pattern_distances.begin(), pattern_distances.end());
      const double min_dist = pattern_distances.front();
      const double max_dist = pattern_distances.back();
      const double median_dist = pattern_distances[pattern_distances.size() / 2];
      std::cout << "Generated pattern graph distances: min=" << min_dist 
                << ", median=" << median_dist << ", max=" << max_dist << std::endl;
    }
    
    if (pattern_disconnected > 0) {
      const int total = static_cast<int>(selected_positions.size() * (selected_positions.size() - 1) / 2);
      std::cout << "Pattern has " << pattern_disconnected << " / " << total 
                << " disconnected pairs (" << (100.0 * pattern_disconnected / total) << "%)" << std::endl;
    }

  // Continue with histogram computation for all modes
  std::vector<int> selected_indices;
  selected_indices.reserve(selected_positions.size());
  for (int pos : selected_positions) {
    if (pos >= 0 && pos < output_support_count) {
      selected_indices.push_back(candidate_triangle_indices[static_cast<size_t>(pos)]);
    }
  }

  state.output_pattern_sample_indices = selected_indices;
  state.output_pattern_points_uv.clear();
  state.output_pattern_points_3d.clear();
  state.output_pattern_points_uv.reserve(selected_indices.size());
  state.output_pattern_points_3d.reserve(selected_indices.size());
  for (int pos : selected_positions) {
    if (pos < 0 || pos >= output_support_count) {
      continue;
    }
    const Eigen::Vector2d uv = candidate_uv[static_cast<size_t>(pos)];
    state.output_pattern_points_uv.push_back(uv);
    Eigen::Vector3d lifted_3d = Eigen::Vector3d::Zero();
    if (!lift_uv_to_output_3d(uv, delaunay_helper, points_3d, points_uv, lifted_3d)) {
      lifted_3d = nearest_sample_3d(uv, points_3d, points_uv);
    }
    state.output_pattern_points_3d.push_back(lifted_3d);
  }

  const int hist_bin_count = std::max(1, state.voronoi_pcf_bin_count);
  std::vector<int> hist_counts;
  int in_range_pair_count = 0;
  std::vector<float> avg_individual_plot;
  if (state.output_pattern_points_uv.size() >= 2 &&
      build_pair_hist_and_average_individual_plot(
        state.output_pattern_points_uv,
        delaunay_helper,
        hist_bin_count,
        hist_counts,
        in_range_pair_count,
        avg_individual_plot,
        &candidate_uv)) {
    state.output_voronoi_pcf_hist_counts = hist_counts;
    state.output_voronoi_pcf_hist_plot = avg_individual_plot;
    state.output_voronoi_pcf_pair_count = in_range_pair_count;
    state.output_voronoi_pcf_ready = true;
    state.output_voronoi_pcf_max_k = 0;
    for (int k = 0; k < static_cast<int>(hist_counts.size()); ++k) {
      if (hist_counts[static_cast<size_t>(k)] > 0) {
        state.output_voronoi_pcf_max_k = k;
      }
    }

    if (!state.voronoi_pcf_hist_plot.empty()) {
      state.output_voronoi_pcf_energy =
        weighted_distribution_l2(state.output_voronoi_pcf_hist_plot, state.voronoi_pcf_hist_plot);
    }
  }

  state.output_voronoi_objective_energy = state.output_voronoi_pcf_energy;
  state.optimizer_improvements = 0;
  state.optimizer_iterations_ran = 0;
  state.output_pattern_dirty = true;
  
  return true;
}

void draw_distribution_plot(
  const char* plot_id,
  const char* label,
  const std::vector<float>& values,
  float y_max_override = -1.0f) {
  if (values.empty()) {
    ImGui::Text("Histogram unavailable.");
    return;
  }

  float y_max = y_max_override;
  if (y_max_override < 0.0f) {
    y_max = 1e-6f;
    for (float v : values) {
      y_max = std::max(y_max, v);
    }
    y_max *= 1.05f;
  }
  ImGui::PlotHistogram(
    plot_id,
    values.data(),
    static_cast<int>(values.size()),
    0,
    label,
    0.0f,
    y_max,
    ImVec2(-1, 180));
}

} // namespace

void reset_voronoi_pcf(InteractionState& root_state) {
  PatternRegionState& state = active_region(root_state);
  state.voronoi_pcf_hist_counts.clear();
  state.voronoi_pcf_hist_plot.clear();
  state.voronoi_pcf_individual_plots.clear();
  state.voronoi_pcf_avg_shell_counts.clear();
  state.voronoi_pcf_raw_point_hist_counts.clear();
  for (auto& hist : state.two_class_voronoi_pcf_hist_counts) {
    hist.clear();
  }
  for (auto& plot : state.two_class_voronoi_pcf_hist_plot) {
    plot.clear();
  }
  for (auto& plots : state.two_class_voronoi_pcf_individual_plots) {
    plots.clear();
  }
  for (auto& plot : state.two_class_voronoi_pcf_hist_min_plot) {
    plot.clear();
  }
  for (auto& plot : state.two_class_voronoi_pcf_hist_max_plot) {
    plot.clear();
  }
  state.two_class_voronoi_pcf_points_inside = {0, 0};
  state.two_class_voronoi_pcf_pair_count = {0, 0, 0};
  state.two_class_target_output_counts = {0, 0};
  state.voronoi_pcf_position_targets_enabled = false;
  state.voronoi_pcf_max_k = 0;
  state.voronoi_pcf_points_inside = 0;
  state.voronoi_pcf_pair_count = 0;
  state.voronoi_pcf_ready = false;
  clear_output_pattern_and_hist(root_state);
}

void compute_voronoi_pcf_histogram(
  InteractionState& root_state,
  const DelaunayTraversalHelper* delaunay_helper) {
  PatternRegionState& state = active_region(root_state);

  state.voronoi_pcf_hist_counts.clear();
  state.voronoi_pcf_hist_plot.clear();
  state.voronoi_pcf_individual_plots.clear();
  state.voronoi_pcf_avg_shell_counts.clear();
  state.voronoi_pcf_raw_point_hist_counts.clear();
  for (auto& hist : state.two_class_voronoi_pcf_hist_counts) {
    hist.clear();
  }
  for (auto& plot : state.two_class_voronoi_pcf_hist_plot) {
    plot.clear();
  }
  for (auto& plots : state.two_class_voronoi_pcf_individual_plots) {
    plots.clear();
  }
  for (auto& plot : state.two_class_voronoi_pcf_hist_min_plot) {
    plot.clear();
  }
  for (auto& plot : state.two_class_voronoi_pcf_hist_max_plot) {
    plot.clear();
  }
  state.two_class_voronoi_pcf_points_inside = {0, 0};
  state.two_class_voronoi_pcf_pair_count = {0, 0, 0};
  state.two_class_target_output_counts = {0, 0};
  state.voronoi_pcf_position_targets_enabled = false;
  state.voronoi_pcf_max_k = 0;
  state.voronoi_pcf_points_inside = 0;
  state.voronoi_pcf_pair_count = 0;
  if (!delaunay_helper || !delaunay_helper->is_ready()) {
    state.voronoi_pcf_ready = false;
    return;
  }
  state.voronoi_pcf_ready = true;

  if (region_is_two_class(state)) {
    const auto mark_two_class_target_unready = [&]() {
      state.voronoi_pcf_ready = false;
    };
    if (state.input_boundary_uv.rows() < 3 || state.input_boundary_uv.cols() < 2) {
      mark_two_class_target_unready();
      return;
    }

    const size_t n_points =
      std::min(state.pattern_points_uv.size(), state.pattern_processing_uv.size());
    if (n_points < 2) {
      mark_two_class_target_unready();
      return;
    }

    std::vector<Eigen::Vector2d> input_inside_uv;
    std::vector<int> input_inside_class_ids;
    input_inside_uv.reserve(n_points);
    input_inside_class_ids.reserve(n_points);
    for (size_t i = 0; i < n_points; ++i) {
      if (!point_in_polygon_for_pcf(state.pattern_points_uv[i], state.input_boundary_uv)) {
        continue;
      }
      input_inside_uv.push_back(state.pattern_processing_uv[i]);
      input_inside_class_ids.push_back(
        (i < state.pattern_point_class_ids.size())
          ? sanitize_pattern_class_id(state.pattern_point_class_ids[i])
          : 0);
    }

    if (input_inside_uv.size() < 2) {
      mark_two_class_target_unready();
      return;
    }

    std::vector<Eigen::Vector2d> input_support_uv;
    if (!collect_triangle_center_candidates_in_polygon(
          state.input_boundary_uv,
          delaunay_helper,
          input_support_uv)) {
      input_support_uv = input_inside_uv;
    }

    const int bin_count = std::max(1, state.voronoi_pcf_bin_count);
    TwoClassPCFStats stats;
    if (!build_two_class_pair_histograms(
          input_inside_uv,
          input_inside_class_ids,
          delaunay_helper,
          bin_count,
          input_support_uv,
          stats)) {
      mark_two_class_target_unready();
      return;
    }

    state.two_class_voronoi_pcf_hist_counts = stats.hist_counts;
    state.two_class_voronoi_pcf_hist_plot = stats.hist_plot;
    state.two_class_voronoi_pcf_individual_plots = stats.individual_plots;
    state.two_class_voronoi_pcf_hist_min_plot = stats.hist_min_plot;
    state.two_class_voronoi_pcf_hist_max_plot = stats.hist_max_plot;
    state.two_class_voronoi_pcf_points_inside = stats.class_counts;
    state.two_class_voronoi_pcf_pair_count = stats.pair_counts;
    state.voronoi_pcf_hist_counts = std::move(stats.combined_hist_counts);
    state.voronoi_pcf_hist_plot = std::move(stats.combined_hist_plot);
    state.voronoi_pcf_points_inside =
      state.two_class_voronoi_pcf_points_inside[0] +
      state.two_class_voronoi_pcf_points_inside[1];
    state.voronoi_pcf_pair_count =
      state.two_class_voronoi_pcf_pair_count[0] +
      state.two_class_voronoi_pcf_pair_count[1] +
      state.two_class_voronoi_pcf_pair_count[2];
    state.voronoi_pcf_max_k =
      max_nonzero_hist_bin(state.voronoi_pcf_hist_counts, state.voronoi_pcf_hist_plot);
    return;
  }

  if (region_is_transition(state)) {
    TransitionTargetProfile transition_profile;
    if (!build_transition_target_profile(
          root_state,
          state,
          delaunay_helper,
          transition_profile)) {
      return;
    }
    state.voronoi_pcf_hist_plot = std::move(transition_profile.hist_plot);
    state.voronoi_pcf_individual_plots = std::move(transition_profile.individual_plots);
    state.voronoi_pcf_raw_point_hist_counts =
      std::move(transition_profile.raw_point_hist_counts);
    // Transition regions now build a blended non-positional target profile and
    // then use the same optimizer path as normal regions.
    state.voronoi_pcf_position_targets_enabled = false;
    state.voronoi_pcf_points_inside = transition_profile.effective_point_count;
    state.voronoi_pcf_max_k =
      max_nonzero_hist_bin(state.voronoi_pcf_hist_counts, state.voronoi_pcf_hist_plot);
    return;
  }

  if (state.input_boundary_uv.rows() < 3 || state.input_boundary_uv.cols() < 2) {
    return;
  }

  const size_t n_points =
    std::min(state.pattern_points_uv.size(), state.pattern_processing_uv.size());
  if (n_points < 2) {
    return;
  }

  std::vector<int> inside_indices;
  inside_indices.reserve(n_points);
  for (size_t i = 0; i < n_points; ++i) {
    if (point_in_polygon_for_pcf(state.pattern_points_uv[i], state.input_boundary_uv)) {
      inside_indices.push_back(static_cast<int>(i));
    }
  }

  state.voronoi_pcf_points_inside =
    static_cast<int>(std::min<size_t>(inside_indices.size(), static_cast<size_t>(std::numeric_limits<int>::max())));
  if (inside_indices.size() < 2) {
    return;
  }

  const int bin_count = std::max(1, state.voronoi_pcf_bin_count);
  std::vector<Eigen::Vector2d> input_inside_uv;
  input_inside_uv.reserve(inside_indices.size());
  for (int idx : inside_indices) {
    input_inside_uv.push_back(state.pattern_processing_uv[static_cast<size_t>(idx)]);
  }

  std::vector<Eigen::Vector2d> input_support_uv;
  if (!collect_triangle_center_candidates_in_polygon(
        state.input_boundary_uv,
        delaunay_helper,
        input_support_uv)) {
    input_support_uv = input_inside_uv;
  }

  std::vector<int> hist_counts;
  int in_range_pair_count = 0;
  std::vector<float> avg_individual_plot;
  std::vector<std::vector<float>> individual_distributions;
  std::vector<float> avg_shell_count_plot;
  std::vector<std::vector<int>> raw_point_hist_counts;
  
  if (!build_pair_hist_and_average_individual_plot(
        input_inside_uv,
        delaunay_helper,
        bin_count,
        hist_counts,
        in_range_pair_count,
        avg_individual_plot,
        &input_support_uv,
        &individual_distributions,
        nullptr,
        nullptr,
        &avg_shell_count_plot,
        &raw_point_hist_counts)) {
    return;
  }

  state.voronoi_pcf_hist_counts = hist_counts;
  state.voronoi_pcf_hist_plot = avg_individual_plot;
  state.voronoi_pcf_individual_plots = std::move(individual_distributions);
  state.voronoi_pcf_avg_shell_counts = std::move(avg_shell_count_plot);
  state.voronoi_pcf_raw_point_hist_counts = std::move(raw_point_hist_counts);
  state.voronoi_pcf_max_k =
    max_nonzero_hist_bin(state.voronoi_pcf_hist_counts, state.voronoi_pcf_hist_plot);
  state.voronoi_pcf_pair_count = in_range_pair_count;

}

namespace {

void hash_combine_u64(std::uint64_t& seed, std::uint64_t value) {
  seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

std::uint64_t quantized_uv_hash_value(double value) {
  return static_cast<std::uint64_t>(
    static_cast<std::int64_t>(std::llround(value * 1000000000.0)));
}

std::uint64_t two_class_output_signature(const PatternRegionState& state) {
  std::uint64_t seed = 1469598103934665603ULL;
  hash_combine_u64(seed, static_cast<std::uint64_t>(state.region_id));
  hash_combine_u64(seed, static_cast<std::uint64_t>(state.voronoi_pcf_bin_count));
  hash_combine_u64(seed, static_cast<std::uint64_t>(sanitize_pattern_class_id(state.two_class_anchor_class_id)));
  hash_combine_u64(seed, state.two_class_sequential_dependency_enabled ? 1ULL : 0ULL);
  hash_combine_u64(seed, state.two_class_optimize_anchor_points ? 1ULL : 0ULL);
  hash_combine_u64(seed, state.two_class_envelope_penalty_enabled ? 1ULL : 0ULL);
  hash_combine_u64(
    seed,
    quantized_uv_hash_value(static_cast<double>(state.two_class_envelope_penalty_weight)));
  for (int class_id = 0; class_id < kPatternClassCount; ++class_id) {
    hash_combine_u64(
      seed,
      static_cast<std::uint64_t>(
        static_cast<std::int64_t>(state.two_class_target_output_counts[static_cast<size_t>(class_id)])));
    hash_combine_u64(
      seed,
      static_cast<std::uint64_t>(
        static_cast<std::int64_t>(state.two_class_voronoi_pcf_points_inside[static_cast<size_t>(class_id)])));
  }
  for (int channel = 0; channel < kTwoClassPairChannelCount; ++channel) {
    hash_combine_u64(
      seed,
      state.two_class_pair_channel_enabled[static_cast<size_t>(channel)] ? 1ULL : 0ULL);
    hash_combine_u64(
      seed,
      quantized_uv_hash_value(
        static_cast<double>(state.two_class_pair_channel_weights[static_cast<size_t>(channel)])));
    const auto& target_plot =
      state.two_class_voronoi_pcf_hist_plot[static_cast<size_t>(channel)];
    hash_combine_u64(seed, static_cast<std::uint64_t>(target_plot.size()));
    for (float value : target_plot) {
      hash_combine_u64(seed, quantized_uv_hash_value(static_cast<double>(value)));
    }
    const auto& target_min =
      state.two_class_voronoi_pcf_hist_min_plot[static_cast<size_t>(channel)];
    const auto& target_max =
      state.two_class_voronoi_pcf_hist_max_plot[static_cast<size_t>(channel)];
    hash_combine_u64(seed, static_cast<std::uint64_t>(target_min.size()));
    for (float value : target_min) {
      hash_combine_u64(seed, quantized_uv_hash_value(static_cast<double>(value)));
    }
    hash_combine_u64(seed, static_cast<std::uint64_t>(target_max.size()));
    for (float value : target_max) {
      hash_combine_u64(seed, quantized_uv_hash_value(static_cast<double>(value)));
    }
  }
  hash_combine_u64(seed, static_cast<std::uint64_t>(state.output_pattern_points_uv.size()));
  for (size_t i = 0; i < state.output_pattern_points_uv.size(); ++i) {
    const Eigen::Vector2d& uv = state.output_pattern_points_uv[i];
    const int class_id =
      (i < state.output_pattern_class_ids.size())
        ? sanitize_pattern_class_id(state.output_pattern_class_ids[i])
        : 0;
    const int tri_idx =
      (i < state.output_pattern_sample_indices.size())
        ? state.output_pattern_sample_indices[i]
        : -1;
    hash_combine_u64(seed, quantized_uv_hash_value(uv.x()));
    hash_combine_u64(seed, quantized_uv_hash_value(uv.y()));
    hash_combine_u64(seed, static_cast<std::uint64_t>(class_id));
    hash_combine_u64(seed, static_cast<std::uint64_t>(static_cast<std::int64_t>(tri_idx)));
  }
  return seed;
}

void sync_two_class_points_to_state(
  PatternRegionState& state,
  const std::vector<Eigen::Vector2d>& uv_points,
  const std::vector<int>& class_ids,
  const std::vector<int>& triangle_indices,
  const DelaunayTraversalHelper* delaunay_helper,
  const Eigen::MatrixXd& points_3d,
  const Eigen::MatrixXd& points_uv) {
  state.output_pattern_sample_indices.clear();
  state.output_pattern_points_uv.clear();
  state.output_pattern_points_3d.clear();
  state.output_pattern_class_ids.clear();
  state.output_pattern_sample_indices.reserve(uv_points.size());
  state.output_pattern_points_uv.reserve(uv_points.size());
  state.output_pattern_points_3d.reserve(uv_points.size());
  state.output_pattern_class_ids.reserve(uv_points.size());

  for (size_t i = 0; i < uv_points.size(); ++i) {
    const Eigen::Vector2d& uv = uv_points[i];
    state.output_pattern_points_uv.push_back(uv);
    state.output_pattern_class_ids.push_back(
      (i < class_ids.size()) ? sanitize_pattern_class_id(class_ids[i]) : 0);
    state.output_pattern_sample_indices.push_back(
      (i < triangle_indices.size()) ? triangle_indices[i] : -1);

    Eigen::Vector3d lifted_3d = Eigen::Vector3d::Zero();
    if (!lift_uv_to_output_3d(uv, delaunay_helper, points_3d, points_uv, lifted_3d)) {
      lifted_3d = nearest_sample_3d(uv, points_3d, points_uv);
    }
    state.output_pattern_points_3d.push_back(lifted_3d);
  }
  state.output_pattern_dirty = true;
}

void clear_two_class_locked_anchor(PatternRegionState& state) {
  state.two_class_locked_anchor_points_uv.clear();
  state.two_class_locked_anchor_sample_indices.clear();
}

bool collect_two_class_locked_anchor_points(
  const PatternRegionState& state,
  int anchor_class_id,
  std::vector<Eigen::Vector2d>& anchor_uv,
  std::vector<int>& anchor_triangles) {
  anchor_uv.clear();
  anchor_triangles.clear();

  if (state.two_class_locked_anchor_class_id == anchor_class_id &&
      !state.two_class_locked_anchor_points_uv.empty()) {
    anchor_uv = state.two_class_locked_anchor_points_uv;
    anchor_triangles = state.two_class_locked_anchor_sample_indices;
    if (anchor_triangles.size() != anchor_uv.size()) {
      anchor_triangles.resize(anchor_uv.size(), -1);
    }
    return true;
  }

  for (size_t i = 0; i < state.output_pattern_points_uv.size(); ++i) {
    const int class_id =
      (i < state.output_pattern_class_ids.size())
        ? sanitize_pattern_class_id(state.output_pattern_class_ids[i])
        : -1;
    if (class_id != anchor_class_id) {
      continue;
    }
    anchor_uv.push_back(state.output_pattern_points_uv[i]);
    anchor_triangles.push_back(
      (i < state.output_pattern_sample_indices.size())
        ? state.output_pattern_sample_indices[i]
        : -1);
  }

  return !anchor_uv.empty();
}

bool update_two_class_output_stats_in_state(
  PatternRegionState& state,
  const std::vector<Eigen::Vector2d>& uv_points,
  const std::vector<int>& class_ids,
  const std::vector<Eigen::Vector2d>& output_support_uv,
  const DelaunayTraversalHelper* delaunay_helper,
  const std::vector<int>* output_support_tri_indices,
  double* out_error);

double compute_two_class_output_error_with_fixed_anchor(
  const PatternRegionState& state,
  const std::vector<Eigen::Vector2d>& uv_points,
  const std::vector<int>& class_ids,
  const std::vector<int>* triangle_indices,
  const std::vector<Eigen::Vector2d>& output_support_uv,
  const std::vector<int>& support_row_for_triangle,
  const DelaunayTraversalHelper* delaunay_helper,
  const FixedAnchorTwoClassEvaluationCache& fixed_anchor_cache,
  FixedAnchorTwoClassObjectiveBreakdown* out_breakdown = nullptr);

bool set_single_class_anchor_target(
  InteractionState& root_state,
  const DelaunayTraversalHelper* delaunay_helper) {
  PatternRegionState& state = active_region(root_state);
  if (!delaunay_helper || !delaunay_helper->is_ready()) {
    return false;
  }
  if (state.input_boundary_uv.rows() < 3 || state.input_boundary_uv.cols() < 2) {
    return false;
  }

  const int anchor_class_id =
    sanitize_pattern_class_id(state.two_class_anchor_class_id);
  const size_t n_points =
    std::min(state.pattern_points_uv.size(), state.pattern_processing_uv.size());
  if (n_points < 2) {
    return false;
  }

  std::vector<Eigen::Vector2d> input_inside_uv;
  input_inside_uv.reserve(n_points);
  for (size_t i = 0; i < n_points; ++i) {
    if (!point_in_polygon_for_pcf(state.pattern_points_uv[i], state.input_boundary_uv)) {
      continue;
    }
    const int class_id =
      (i < state.pattern_point_class_ids.size())
        ? sanitize_pattern_class_id(state.pattern_point_class_ids[i])
        : 0;
    if (class_id != anchor_class_id) {
      continue;
    }
    input_inside_uv.push_back(state.pattern_processing_uv[i]);
  }

  if (input_inside_uv.size() < 2) {
    return false;
  }

  std::vector<Eigen::Vector2d> input_support_uv;
  if (!collect_triangle_center_candidates_in_polygon(
        state.input_boundary_uv,
        delaunay_helper,
        input_support_uv)) {
    input_support_uv = input_inside_uv;
  }

  const int bin_count = std::max(1, state.voronoi_pcf_bin_count);
  std::vector<int> hist_counts;
  int in_range_pair_count = 0;
  std::vector<float> avg_individual_plot;
  std::vector<std::vector<float>> individual_distributions;
  std::vector<float> avg_shell_count_plot;
  std::vector<std::vector<int>> raw_point_hist_counts;
  if (!build_pair_hist_and_average_individual_plot(
        input_inside_uv,
        delaunay_helper,
        bin_count,
        hist_counts,
        in_range_pair_count,
        avg_individual_plot,
        &input_support_uv,
        &individual_distributions,
        nullptr,
        nullptr,
        &avg_shell_count_plot,
        &raw_point_hist_counts)) {
    return false;
  }

  state.voronoi_pcf_hist_counts = std::move(hist_counts);
  state.voronoi_pcf_hist_plot = std::move(avg_individual_plot);
  state.voronoi_pcf_individual_plots = std::move(individual_distributions);
  state.voronoi_pcf_avg_shell_counts = std::move(avg_shell_count_plot);
  state.voronoi_pcf_raw_point_hist_counts = std::move(raw_point_hist_counts);
  state.voronoi_pcf_position_targets_enabled = false;
  state.voronoi_pcf_points_inside = static_cast<int>(input_inside_uv.size());
  state.voronoi_pcf_pair_count = in_range_pair_count;
  state.voronoi_pcf_max_k =
    max_nonzero_hist_bin(state.voronoi_pcf_hist_counts, state.voronoi_pcf_hist_plot);
  state.voronoi_pcf_ready = true;
  return true;
}

bool finish_two_class_anchor_normal_stage(
  InteractionState& root_state,
  const DelaunayTraversalHelper* delaunay_helper) {
  PatternRegionState& state = active_region(root_state);
  if (!delaunay_helper || !delaunay_helper->is_ready()) {
    return false;
  }
  const int anchor_class_id =
    sanitize_pattern_class_id(state.two_class_anchor_class_id);
  if (state.output_pattern_points_uv.empty()) {
    return false;
  }

  std::vector<int> anchor_classes(state.output_pattern_points_uv.size(), anchor_class_id);
  std::vector<int> triangle_indices(state.output_pattern_points_uv.size(), -1);
  for (size_t i = 0; i < state.output_pattern_points_uv.size(); ++i) {
    int tri_idx = -1;
    Eigen::Vector3i tri_vertices(-1, -1, -1);
    if (delaunay_helper->find_containing_triangle(
          state.output_pattern_points_uv[i],
          tri_idx,
          tri_vertices)) {
      triangle_indices[i] = tri_idx;
    }
  }
  state.output_pattern_class_ids = anchor_classes;
  state.output_pattern_sample_indices = triangle_indices;
  state.two_class_locked_anchor_class_id = anchor_class_id;
  state.two_class_locked_anchor_points_uv = state.output_pattern_points_uv;
  state.two_class_locked_anchor_sample_indices = triangle_indices;

  std::vector<Eigen::Vector2d> output_support_uv;
  std::vector<int> output_support_tri_indices;
  if (!collect_output_triangle_center_candidates(
        root_state,
        delaunay_helper,
        output_support_uv,
        &output_support_tri_indices)) {
    return false;
  }

  (void)update_two_class_output_stats_in_state(
    state,
    state.output_pattern_points_uv,
    state.output_pattern_class_ids,
    output_support_uv,
    delaunay_helper,
    &output_support_tri_indices,
    nullptr);
  state.two_class_anchor_normal_stage_active = false;
  state.region_mode = static_cast<int>(PatternRegionMode::TwoClass);
  state.output_pattern_dirty = true;
  return true;
}

bool update_two_class_output_stats_in_state(
  PatternRegionState& state,
  const std::vector<Eigen::Vector2d>& uv_points,
  const std::vector<int>& class_ids,
  const std::vector<Eigen::Vector2d>& output_support_uv,
  const DelaunayTraversalHelper* delaunay_helper,
  const std::vector<int>* output_support_tri_indices,
  double* out_error = nullptr) {
  if (out_error) {
    *out_error = std::numeric_limits<double>::infinity();
  }
  const int bin_count = std::max(1, state.voronoi_pcf_bin_count);
  TwoClassPCFStats output_stats;
  if (!build_two_class_pair_histograms(
        uv_points,
        class_ids,
        delaunay_helper,
        bin_count,
        output_support_uv,
        output_stats)) {
    clear_two_class_output_stats(state);
    return false;
  }

  state.two_class_output_voronoi_pcf_hist_counts = output_stats.hist_counts;
  state.two_class_output_voronoi_pcf_hist_plot = output_stats.hist_plot;
  state.two_class_output_counts = output_stats.class_counts;
  state.two_class_output_voronoi_pcf_pair_count = output_stats.pair_counts;
  state.output_voronoi_pcf_hist_counts = std::move(output_stats.combined_hist_counts);
  state.output_voronoi_pcf_hist_plot = std::move(output_stats.combined_hist_plot);
  state.output_voronoi_pcf_pair_count = output_stats.combined_pair_count;
  state.output_voronoi_pcf_max_k =
    max_nonzero_hist_bin(state.output_voronoi_pcf_hist_counts, state.output_voronoi_pcf_hist_plot);
  state.output_voronoi_pcf_ready = true;

  double error = two_class_distribution_error(
    state,
    output_stats,
    state.two_class_target_output_counts);

  if (state.two_class_sequential_dependency_enabled &&
      delaunay_helper &&
      delaunay_helper->is_ready()) {
    const int anchor_class_id =
      sanitize_pattern_class_id(state.two_class_anchor_class_id);
    const int dependent_class_id = 1 - anchor_class_id;
    if (output_stats.class_counts[static_cast<size_t>(anchor_class_id)] > 0 &&
        output_stats.class_counts[static_cast<size_t>(dependent_class_id)] > 0) {
      const std::vector<int>* support_tri_indices = output_support_tri_indices;
      if ((support_tri_indices == nullptr ||
           support_tri_indices->size() != output_support_uv.size()) &&
          state.output_support_tri_indices_cache.size() == output_support_uv.size()) {
        support_tri_indices = &state.output_support_tri_indices_cache;
      }

      if (support_tri_indices != nullptr &&
          support_tri_indices->size() == output_support_uv.size()) {
        const int triangle_count = delaunay_helper->triangle_count();
        std::vector<int> support_row_for_triangle(
          static_cast<size_t>(std::max(0, triangle_count)),
          -1);
        for (size_t si = 0; si < support_tri_indices->size(); ++si) {
          const int tri_idx = (*support_tri_indices)[si];
          if (tri_idx >= 0 && tri_idx < triangle_count) {
            support_row_for_triangle[static_cast<size_t>(tri_idx)] = static_cast<int>(si);
          }
        }

        std::vector<int> triangle_indices(uv_points.size(), -1);
        if (state.output_pattern_sample_indices.size() == uv_points.size()) {
          triangle_indices = state.output_pattern_sample_indices;
        } else {
          for (size_t i = 0; i < uv_points.size(); ++i) {
            int tri_idx = -1;
            Eigen::Vector3i tri_vertices(-1, -1, -1);
            if (delaunay_helper->find_containing_triangle(
                  uv_points[i],
                  tri_idx,
                  tri_vertices)) {
              triangle_indices[i] = tri_idx;
            }
          }
        }

        FixedAnchorTwoClassEvaluationCache fixed_anchor_cache;
        if (build_fixed_anchor_two_class_evaluation_cache(
              state,
              uv_points,
              class_ids,
              triangle_indices,
              support_row_for_triangle,
              output_support_uv,
              delaunay_helper,
              anchor_class_id,
              fixed_anchor_cache)) {
          const double fixed_anchor_error =
            compute_two_class_output_error_with_fixed_anchor(
              state,
              uv_points,
              class_ids,
              &triangle_indices,
              output_support_uv,
              support_row_for_triangle,
              delaunay_helper,
              fixed_anchor_cache);
          if (std::isfinite(fixed_anchor_error)) {
            error = fixed_anchor_error;
          }
        }
      }
    }
  }

  state.output_voronoi_pcf_energy = error;
  state.output_voronoi_objective_energy = error;
  if (out_error) {
    *out_error = error;
  }
  return true;
}

double compute_two_class_output_error(
  const PatternRegionState& state,
  const std::vector<Eigen::Vector2d>& uv_points,
  const std::vector<int>& class_ids,
  const std::vector<Eigen::Vector2d>& output_support_uv,
  const DelaunayTraversalHelper* delaunay_helper) {
  const int bin_count = std::max(1, state.voronoi_pcf_bin_count);
  TwoClassPCFStats output_stats;
  if (!build_two_class_pair_histograms(
        uv_points,
        class_ids,
        delaunay_helper,
        bin_count,
        output_support_uv,
        output_stats)) {
    return std::numeric_limits<double>::infinity();
  }
  return two_class_distribution_error(
    state,
    output_stats,
    state.two_class_target_output_counts);
}

double compute_two_class_output_error_with_fixed_anchor(
  const PatternRegionState& state,
  const std::vector<Eigen::Vector2d>& uv_points,
  const std::vector<int>& class_ids,
  const std::vector<int>* triangle_indices,
  const std::vector<Eigen::Vector2d>& output_support_uv,
  const std::vector<int>& support_row_for_triangle,
  const DelaunayTraversalHelper* delaunay_helper,
  const FixedAnchorTwoClassEvaluationCache& fixed_anchor_cache,
  FixedAnchorTwoClassObjectiveBreakdown* out_breakdown) {
  if (out_breakdown != nullptr) {
    *out_breakdown = FixedAnchorTwoClassObjectiveBreakdown{};
  }
  if (!fixed_anchor_cache.valid ||
      !delaunay_helper ||
      !delaunay_helper->is_ready()) {
    const double fallback_error = compute_two_class_output_error(
      state,
      uv_points,
      class_ids,
      output_support_uv,
      delaunay_helper);
    if (out_breakdown != nullptr) {
      out_breakdown->total_error = fallback_error;
      out_breakdown->global_symmetric_error = fallback_error;
    }
    return fallback_error;
  }

  const int bin_count = std::max(1, state.voronoi_pcf_bin_count);
  const int dependent_class_id = fixed_anchor_cache.dependent_class_id;
  const int dependent_channel =
    two_class_pair_channel(dependent_class_id, dependent_class_id);
  const int cross_channel =
    two_class_pair_channel(fixed_anchor_cache.anchor_class_id, dependent_class_id);
  if (out_breakdown != nullptr) {
    out_breakdown->anchor_class_id = fixed_anchor_cache.anchor_class_id;
    out_breakdown->dependent_class_id = dependent_class_id;
    out_breakdown->dependent_channel = dependent_channel;
    out_breakdown->cross_channel = cross_channel;
    out_breakdown->anchor_count =
      static_cast<int>(fixed_anchor_cache.anchor_uv_points.size());
  }

  std::vector<Eigen::Vector2d> dependent_uv_points;
  std::vector<int> dependent_support_rows;
  std::vector<std::vector<int>> dependent_support_counts;
  dependent_uv_points.reserve(uv_points.size());
  dependent_support_rows.reserve(uv_points.size());
  dependent_support_counts.reserve(uv_points.size());

  for (size_t i = 0; i < uv_points.size(); ++i) {
    const int class_id =
      (i < class_ids.size())
        ? sanitize_pattern_class_id(class_ids[i])
        : 0;
    if (class_id != dependent_class_id) {
      continue;
    }
    int support_row = -1;
    if (triangle_indices && i < triangle_indices->size()) {
      const int tri_idx = (*triangle_indices)[i];
      if (tri_idx >= 0 &&
          tri_idx < static_cast<int>(support_row_for_triangle.size())) {
        support_row = support_row_for_triangle[static_cast<size_t>(tri_idx)];
      }
    }
    dependent_uv_points.push_back(uv_points[i]);
    dependent_support_rows.push_back(support_row);
    dependent_support_counts.push_back(
      build_two_class_support_counts(
        state,
        uv_points[i],
        support_row,
        output_support_uv,
        delaunay_helper));
  }

  TwoClassPCFStats output_stats;
  for (auto& hist : output_stats.hist_counts) {
    hist.assign(static_cast<size_t>(bin_count), 0);
  }
  for (auto& plot : output_stats.hist_plot) {
    plot.assign(static_cast<size_t>(bin_count), 0.0f);
  }
  std::array<std::vector<std::vector<float>>, kTwoClassPairChannelCount>
    output_individual_plots;
  for (auto& plots : output_individual_plots) {
    plots.clear();
  }
  for (auto& plot : output_stats.hist_min_plot) {
    plot.assign(
      static_cast<size_t>(bin_count),
      std::numeric_limits<float>::infinity());
  }
  for (auto& plot : output_stats.hist_max_plot) {
    plot.assign(static_cast<size_t>(bin_count), 0.0f);
  }
  output_stats.class_counts[static_cast<size_t>(fixed_anchor_cache.anchor_class_id)] =
    static_cast<int>(fixed_anchor_cache.anchor_uv_points.size());
  output_stats.class_counts[static_cast<size_t>(dependent_class_id)] =
    static_cast<int>(dependent_uv_points.size());
  if (out_breakdown != nullptr) {
    out_breakdown->dependent_count = static_cast<int>(dependent_uv_points.size());
  }

  std::vector<std::vector<int>> anchor_cross_hist(
    fixed_anchor_cache.anchor_uv_points.size(),
    std::vector<int>(static_cast<size_t>(bin_count), 0));
  std::vector<std::array<std::vector<Eigen::Vector2d>, kFixedAnchorTemplateMaxHopRadius + 1>>
    anchor_template_output_offsets(fixed_anchor_cache.anchor_uv_points.size());
  std::vector<std::vector<int>> dependent_self_hist(
    dependent_uv_points.size(),
    std::vector<int>(static_cast<size_t>(bin_count), 0));
  std::vector<std::vector<int>> dependent_cross_hist(
    dependent_uv_points.size(),
    std::vector<int>(static_cast<size_t>(bin_count), 0));

  const auto pair_distance = [&](const Eigen::Vector2d& lhs_uv,
                                 int lhs_support_row,
                                 const Eigen::Vector2d& rhs_uv,
                                 int rhs_support_row) {
    if (lhs_support_row >= 0 && rhs_support_row >= 0) {
      return get_support_pairwise_dist(state, lhs_support_row, rhs_support_row);
    }
    return delaunay_helper->count_triangles_crossed(lhs_uv, rhs_uv);
  };

  for (size_t anchor_idx = 0; anchor_idx < fixed_anchor_cache.anchor_uv_points.size(); ++anchor_idx) {
    for (size_t dep_idx = 0; dep_idx < dependent_uv_points.size(); ++dep_idx) {
      const int k = pair_distance(
        fixed_anchor_cache.anchor_uv_points[anchor_idx],
        fixed_anchor_cache.anchor_support_rows[anchor_idx],
        dependent_uv_points[dep_idx],
        dependent_support_rows[dep_idx]);
      if (k < 0) {
        continue;
      }
      const int bin = std::min(k, bin_count - 1);
      ++output_stats.hist_counts[static_cast<size_t>(cross_channel)][static_cast<size_t>(bin)];
      ++output_stats.pair_counts[static_cast<size_t>(cross_channel)];
      ++anchor_cross_hist[anchor_idx][static_cast<size_t>(bin)];
      ++dependent_cross_hist[dep_idx][static_cast<size_t>(bin)];
      if (k <= kFixedAnchorTemplateMaxHopRadius) {
        const Eigen::Vector2d offset =
          dependent_uv_points[dep_idx] - fixed_anchor_cache.anchor_uv_points[anchor_idx];
        for (int radius = std::max(1, k);
             radius <= kFixedAnchorTemplateMaxHopRadius;
             ++radius) {
          anchor_template_output_offsets[anchor_idx][static_cast<size_t>(radius)].push_back(offset);
        }
      }
    }
  }

  for (size_t lhs_idx = 0; lhs_idx + 1 < dependent_uv_points.size(); ++lhs_idx) {
    for (size_t rhs_idx = lhs_idx + 1; rhs_idx < dependent_uv_points.size(); ++rhs_idx) {
      const int k = pair_distance(
        dependent_uv_points[lhs_idx],
        dependent_support_rows[lhs_idx],
        dependent_uv_points[rhs_idx],
        dependent_support_rows[rhs_idx]);
      if (k < 0) {
        continue;
      }
      const int bin = std::min(k, bin_count - 1);
      ++output_stats.hist_counts[static_cast<size_t>(dependent_channel)][static_cast<size_t>(bin)];
      ++output_stats.pair_counts[static_cast<size_t>(dependent_channel)];
      ++dependent_self_hist[lhs_idx][static_cast<size_t>(bin)];
      ++dependent_self_hist[rhs_idx][static_cast<size_t>(bin)];
    }
  }

  const auto accumulate_channel_row = [&](int channel,
                                          const std::vector<int>& row_counts,
                                          const std::vector<int>& support_counts,
                                          int& valid_anchor_count) {
    bool has_valid_support = false;
    std::vector<float> local_plot(static_cast<size_t>(bin_count), 0.0f);
    std::vector<char> local_has_bin(static_cast<size_t>(bin_count), 0);
    for (int k = 0; k < bin_count; ++k) {
      const int denom =
        (k < static_cast<int>(support_counts.size()))
          ? support_counts[static_cast<size_t>(k)]
          : 0;
      if (denom <= 0) {
        continue;
      }
      has_valid_support = true;
      const int count =
        (k < static_cast<int>(row_counts.size()))
          ? row_counts[static_cast<size_t>(k)]
          : 0;
      local_plot[static_cast<size_t>(k)] =
        static_cast<float>(count) / static_cast<float>(denom);
      local_has_bin[static_cast<size_t>(k)] = 1;
    }
    if (!has_valid_support) {
      return;
    }
    output_individual_plots[static_cast<size_t>(channel)].push_back(local_plot);
    ++valid_anchor_count;
    for (int k = 0; k < bin_count; ++k) {
      if (local_has_bin[static_cast<size_t>(k)] == 0) {
        continue;
      }
      output_stats.hist_plot[static_cast<size_t>(channel)][static_cast<size_t>(k)] +=
        local_plot[static_cast<size_t>(k)];
      output_stats.hist_min_plot[static_cast<size_t>(channel)][static_cast<size_t>(k)] =
        std::min(
          output_stats.hist_min_plot[static_cast<size_t>(channel)][static_cast<size_t>(k)],
          local_plot[static_cast<size_t>(k)]);
      output_stats.hist_max_plot[static_cast<size_t>(channel)][static_cast<size_t>(k)] =
        std::max(
          output_stats.hist_max_plot[static_cast<size_t>(channel)][static_cast<size_t>(k)],
          local_plot[static_cast<size_t>(k)]);
    }
  };

  const auto finalize_channel = [&](int channel, int valid_anchor_count) {
    if (valid_anchor_count > 0) {
      const float inv_valid = 1.0f / static_cast<float>(valid_anchor_count);
      for (float& value : output_stats.hist_plot[static_cast<size_t>(channel)]) {
        value *= inv_valid;
      }
    }
    for (int k = 0; k < bin_count; ++k) {
      float& min_value =
        output_stats.hist_min_plot[static_cast<size_t>(channel)][static_cast<size_t>(k)];
      if (!std::isfinite(min_value)) {
        min_value = 0.0f;
      }
    }
  };

  int dependent_valid_anchors = 0;
  for (size_t dep_idx = 0; dep_idx < dependent_uv_points.size(); ++dep_idx) {
    accumulate_channel_row(
      dependent_channel,
      dependent_self_hist[dep_idx],
      dependent_support_counts[dep_idx],
      dependent_valid_anchors);
  }
  finalize_channel(dependent_channel, dependent_valid_anchors);
  if (out_breakdown != nullptr) {
    out_breakdown->dependent_valid_anchors = dependent_valid_anchors;
  }

  int cross_valid_anchors = 0;
  int directional_cross_lonely_count = 0;
  double directional_cross_near_deficit_error = 0.0;
  double directional_cross_near_excess_error = 0.0;
  const int directional_cross_near_radius = std::clamp(
    fixed_anchor_cache.anchor_cross_near_hop_radius,
    0,
    std::max(0, bin_count - 1));
  const int directional_cross_target_lower_count =
    std::max(0, fixed_anchor_cache.anchor_cross_target_near_lower_count);
  const double directional_cross_target_near_count =
    std::max(0.0, fixed_anchor_cache.anchor_cross_target_near_count);
  for (size_t anchor_idx = 0; anchor_idx < fixed_anchor_cache.anchor_uv_points.size(); ++anchor_idx) {
    accumulate_channel_row(
      cross_channel,
      anchor_cross_hist[anchor_idx],
      fixed_anchor_cache.anchor_support_counts[anchor_idx],
      cross_valid_anchors);
    int near_count = 0;
    const int near_eval_bins = std::min(
      static_cast<int>(anchor_cross_hist[anchor_idx].size()),
      directional_cross_near_radius + 1);
    for (int k = 0; k < near_eval_bins; ++k) {
      near_count += anchor_cross_hist[anchor_idx][static_cast<size_t>(k)];
    }
    if (near_count <= 0) {
      ++directional_cross_lonely_count;
    }
    double near_deficit_error = 0.0;
    double near_excess_error = 0.0;
    compute_two_class_near_count_band_error(
      near_count,
      directional_cross_target_lower_count,
      directional_cross_target_near_count,
      &near_deficit_error,
      &near_excess_error);
    directional_cross_near_deficit_error += near_deficit_error;
    directional_cross_near_excess_error += near_excess_error;
  }
  finalize_channel(cross_channel, cross_valid_anchors);
  if (cross_valid_anchors > 0) {
    directional_cross_near_deficit_error /= static_cast<double>(cross_valid_anchors);
    directional_cross_near_excess_error /= static_cast<double>(cross_valid_anchors);
  }
  if (out_breakdown != nullptr) {
    out_breakdown->cross_valid_anchors = cross_valid_anchors;
  }

  std::vector<float> dependent_cross_avg_plot(static_cast<size_t>(bin_count), 0.0f);
  std::vector<std::vector<float>> dependent_cross_output_distributions;
  dependent_cross_output_distributions.reserve(dependent_cross_hist.size());
  int dependent_cross_valid_anchors = 0;
  int dependent_cross_lonely_count = 0;
  double dependent_cross_near_deficit_error = 0.0;
  const int dependent_cross_near_radius = std::clamp(
    fixed_anchor_cache.dependent_cross_near_hop_radius,
    0,
    std::max(0, bin_count - 1));
  const int dependent_cross_target_lower_count =
    std::max(0, fixed_anchor_cache.dependent_cross_target_near_lower_count);
  for (size_t dep_idx = 0; dep_idx < dependent_cross_hist.size(); ++dep_idx) {
    std::vector<float> local_plot;
    if (!build_two_class_local_distribution(
          dependent_cross_hist[dep_idx],
          dependent_support_counts[dep_idx],
          local_plot)) {
      continue;
    }
    dependent_cross_output_distributions.push_back(local_plot);
    ++dependent_cross_valid_anchors;
    for (int k = 0; k < bin_count; ++k) {
      dependent_cross_avg_plot[static_cast<size_t>(k)] +=
        (k < static_cast<int>(local_plot.size()))
          ? local_plot[static_cast<size_t>(k)]
          : 0.0f;
    }
    int near_count = 0;
    const int near_eval_bins = std::min(
      static_cast<int>(dependent_cross_hist[dep_idx].size()),
      dependent_cross_near_radius + 1);
    for (int k = 0; k < near_eval_bins; ++k) {
      near_count += dependent_cross_hist[dep_idx][static_cast<size_t>(k)];
    }
    if (near_count <= 0) {
      ++dependent_cross_lonely_count;
    }
    if (dependent_cross_target_lower_count > 0 &&
        near_count < dependent_cross_target_lower_count) {
      const double deficit =
        static_cast<double>(dependent_cross_target_lower_count - near_count) /
        static_cast<double>(std::max(1, dependent_cross_target_lower_count));
      dependent_cross_near_deficit_error += deficit * deficit;
    }
  }
  if (dependent_cross_valid_anchors > 0) {
    const float inv_valid = 1.0f / static_cast<float>(dependent_cross_valid_anchors);
    for (float& value : dependent_cross_avg_plot) {
      value *= inv_valid;
    }
    dependent_cross_near_deficit_error /=
      static_cast<double>(dependent_cross_valid_anchors);
  }

  const auto compute_fixed_anchor_channel_error =
    [&](int channel,
        const std::vector<std::vector<float>>& output_distributions,
        const std::vector<std::vector<float>>& target_distributions,
        const std::vector<float>& target_avg_plot,
        const std::vector<float>& target_min_plot,
        const std::vector<float>& target_max_plot,
        double* out_distribution_error = nullptr,
        double* out_envelope_error = nullptr) {
      if (out_distribution_error != nullptr) {
        *out_distribution_error = 0.0;
      }
      if (out_envelope_error != nullptr) {
        *out_envelope_error = 0.0;
      }
      if (!state.two_class_pair_channel_enabled[static_cast<size_t>(channel)]) {
        return 0.0;
      }
      const double channel_weight = std::max(
        0.0f,
        state.two_class_pair_channel_weights[static_cast<size_t>(channel)]);
      if (channel_weight <= 0.0) {
        return 0.0;
      }

      double channel_error = std::numeric_limits<double>::infinity();
      if (!target_distributions.empty() && !output_distributions.empty()) {
        channel_error = two_class_assignment_distribution_cost(
          output_distributions,
          target_distributions,
          state.voronoi_pcf_bin_count);
      }
      if (!std::isfinite(channel_error)) {
        if (target_avg_plot.empty()) {
          return 0.0;
        }
        channel_error = weighted_distribution_l2(
          output_stats.hist_plot[static_cast<size_t>(channel)],
          target_avg_plot);
      }
      if (out_distribution_error != nullptr) {
        *out_distribution_error = channel_error;
      }

      double weighted_channel_error = channel_weight * channel_error;
      if (state.two_class_envelope_penalty_enabled &&
          !target_min_plot.empty() &&
          !target_max_plot.empty()) {
        const auto& output_min =
          output_stats.hist_min_plot[static_cast<size_t>(channel)];
        const auto& output_max =
          output_stats.hist_max_plot[static_cast<size_t>(channel)];
        const int eval_bins = std::max(
          std::max(
            static_cast<int>(target_min_plot.size()),
            static_cast<int>(target_max_plot.size())),
          std::max(
            static_cast<int>(output_min.size()),
            static_cast<int>(output_max.size())));
        double envelope_error = 0.0;
        constexpr double kEnvelopeSlack = 0.015;
        for (int k = 0; k < eval_bins; ++k) {
          const double t_min =
            (k < static_cast<int>(target_min_plot.size()))
              ? static_cast<double>(target_min_plot[static_cast<size_t>(k)])
              : 0.0;
          const double t_max =
            (k < static_cast<int>(target_max_plot.size()))
              ? static_cast<double>(target_max_plot[static_cast<size_t>(k)])
              : 0.0;
          const double o_min =
            (k < static_cast<int>(output_min.size()))
              ? static_cast<double>(output_min[static_cast<size_t>(k)])
              : 0.0;
          const double o_max =
            (k < static_cast<int>(output_max.size()))
              ? static_cast<double>(output_max[static_cast<size_t>(k)])
              : 0.0;
          const double low_gap = std::max(0.0, t_min - kEnvelopeSlack - o_min);
          const double high_gap = std::max(0.0, o_max - (t_max + kEnvelopeSlack));
          envelope_error += low_gap * low_gap + high_gap * high_gap;
        }
        if (out_envelope_error != nullptr) {
          *out_envelope_error = envelope_error;
        }
        weighted_channel_error +=
          channel_weight *
          static_cast<double>(state.two_class_envelope_penalty_weight) *
          envelope_error;
      }
      return weighted_channel_error;
    };

  double total_error = 0.0;
  double total_weight = 0.0;
  double dependent_self_raw_error = 0.0;
  double dependent_self_envelope_error = 0.0;
  double dependent_self_weighted_error = 0.0;
  if (state.two_class_pair_channel_enabled[static_cast<size_t>(dependent_channel)] &&
      state.two_class_pair_channel_weights[static_cast<size_t>(dependent_channel)] > 0.0f) {
    dependent_self_weighted_error = compute_fixed_anchor_channel_error(
      dependent_channel,
      output_individual_plots[static_cast<size_t>(dependent_channel)],
      state.two_class_voronoi_pcf_individual_plots[static_cast<size_t>(dependent_channel)],
      state.two_class_voronoi_pcf_hist_plot[static_cast<size_t>(dependent_channel)],
      state.two_class_voronoi_pcf_hist_min_plot[static_cast<size_t>(dependent_channel)],
      state.two_class_voronoi_pcf_hist_max_plot[static_cast<size_t>(dependent_channel)],
      &dependent_self_raw_error,
      &dependent_self_envelope_error);
    total_error += dependent_self_weighted_error;
    total_weight += std::max(
      0.0f,
      state.two_class_pair_channel_weights[static_cast<size_t>(dependent_channel)]);
  }
  double directional_cross_raw_error = 0.0;
  double directional_cross_envelope_error = 0.0;
  double directional_cross_weighted_error = 0.0;
  if (state.two_class_pair_channel_enabled[static_cast<size_t>(cross_channel)] &&
      state.two_class_pair_channel_weights[static_cast<size_t>(cross_channel)] > 0.0f) {
    const double directional_cross_channel_weight = std::max(
      0.0f,
      state.two_class_pair_channel_weights[static_cast<size_t>(cross_channel)]);
    const double directional_cross_base_weighted_error =
      compute_fixed_anchor_channel_error(
      cross_channel,
      output_individual_plots[static_cast<size_t>(cross_channel)],
      fixed_anchor_cache.anchor_cross_target_distributions,
      fixed_anchor_cache.anchor_cross_target_avg_plot,
      fixed_anchor_cache.anchor_cross_target_min_plot,
      fixed_anchor_cache.anchor_cross_target_max_plot,
      &directional_cross_raw_error,
      &directional_cross_envelope_error);
    constexpr double kDirectionalCrossNearDeficitWeight = 2.0;
    constexpr double kDirectionalCrossNearExcessWeight = 1.0;
    const double directional_cross_base_error =
      directional_cross_channel_weight > 0.0
        ? (directional_cross_base_weighted_error / directional_cross_channel_weight)
        : directional_cross_raw_error;
    const double directional_cross_combined_error =
      directional_cross_base_error +
      kDirectionalCrossNearDeficitWeight * directional_cross_near_deficit_error +
      kDirectionalCrossNearExcessWeight * directional_cross_near_excess_error;
    directional_cross_weighted_error =
      directional_cross_channel_weight * directional_cross_combined_error;
    total_error += directional_cross_weighted_error;
    total_weight += directional_cross_channel_weight;
  }
  double dependent_cross_raw_error = 0.0;
  double dependent_cross_weighted_error = 0.0;
  if (state.two_class_pair_channel_enabled[static_cast<size_t>(cross_channel)] &&
      state.two_class_pair_channel_weights[static_cast<size_t>(cross_channel)] > 0.0f &&
      !fixed_anchor_cache.dependent_cross_target_avg_plot.empty() &&
      dependent_cross_valid_anchors > 0) {
    if (!dependent_cross_output_distributions.empty() &&
        !fixed_anchor_cache.dependent_cross_target_distributions.empty()) {
      dependent_cross_raw_error = two_class_assignment_distribution_cost(
        dependent_cross_output_distributions,
        fixed_anchor_cache.dependent_cross_target_distributions,
        state.voronoi_pcf_bin_count);
    }
    if (!std::isfinite(dependent_cross_raw_error)) {
      dependent_cross_raw_error = weighted_distribution_l2(
        dependent_cross_avg_plot,
        fixed_anchor_cache.dependent_cross_target_avg_plot);
    }
    constexpr double kDependentCrossNearDeficitWeight = 2.0;
    const double dependent_cross_combined_error =
      dependent_cross_raw_error +
      kDependentCrossNearDeficitWeight * dependent_cross_near_deficit_error;
    dependent_cross_weighted_error =
      std::max(
        0.0f,
        state.two_class_pair_channel_weights[static_cast<size_t>(cross_channel)]) *
      dependent_cross_combined_error;
    total_error += dependent_cross_weighted_error;
    total_weight += std::max(
      0.0f,
      state.two_class_pair_channel_weights[static_cast<size_t>(cross_channel)]);
  }

  double template_average_error = 0.0;
  double template_weight = 0.0;
  double template_weighted_error = 0.0;
  int valid_template_anchors = 0;
  if (state.two_class_pair_channel_enabled[static_cast<size_t>(cross_channel)] &&
      state.two_class_pair_channel_weights[static_cast<size_t>(cross_channel)] > 0.0f &&
      !fixed_anchor_cache.anchor_template_target_offsets.empty()) {
    double template_error_sum = 0.0;
    for (size_t anchor_idx = 0;
         anchor_idx < anchor_template_output_offsets.size();
         ++anchor_idx) {
      const double template_cost = fixed_anchor_best_template_cost(
        fixed_anchor_cache,
        anchor_template_output_offsets[anchor_idx],
        nullptr);
      if (!std::isfinite(template_cost)) {
        continue;
      }
      template_error_sum += template_cost;
      ++valid_template_anchors;
    }
    if (valid_template_anchors > 0) {
      template_average_error =
        template_error_sum / static_cast<double>(valid_template_anchors);
      // Template offsets are now a proposal prior only; the exact objective
      // remains dependent-self plus directional cross distribution fidelity.
      constexpr double kAnchorTemplateObjectiveWeightScale = 0.0;
      template_weight =
        std::max(
          0.0f,
          state.two_class_pair_channel_weights[static_cast<size_t>(cross_channel)]) *
        kAnchorTemplateObjectiveWeightScale;
      template_weighted_error = template_weight * template_average_error;
      if (template_weight > 0.0) {
        total_error += template_weighted_error;
        total_weight += template_weight;
      }
    }
  }

  const double final_error = total_weight > 0.0
    ? total_error / total_weight
    : std::numeric_limits<double>::infinity();
  if (out_breakdown != nullptr) {
    out_breakdown->total_error = final_error;
    out_breakdown->total_weight = total_weight;
    out_breakdown->dependent_self_distribution_error = dependent_self_raw_error;
    out_breakdown->dependent_self_envelope_error = dependent_self_envelope_error;
    out_breakdown->dependent_self_weight = std::max(
      0.0f,
      state.two_class_pair_channel_weights[static_cast<size_t>(dependent_channel)]);
    out_breakdown->dependent_self_weighted_error = dependent_self_weighted_error;
    out_breakdown->directional_cross_distribution_error = directional_cross_raw_error;
    out_breakdown->directional_cross_envelope_error = directional_cross_envelope_error;
    out_breakdown->directional_cross_near_deficit_error =
      directional_cross_near_deficit_error;
    out_breakdown->directional_cross_near_excess_error =
      directional_cross_near_excess_error;
    out_breakdown->directional_cross_weight = std::max(
      0.0f,
      state.two_class_pair_channel_weights[static_cast<size_t>(cross_channel)]);
    out_breakdown->directional_cross_weighted_error = directional_cross_weighted_error;
    out_breakdown->directional_cross_near_hop_radius = directional_cross_near_radius;
    out_breakdown->directional_cross_target_near_count =
      fixed_anchor_cache.anchor_cross_target_near_count;
    out_breakdown->directional_cross_target_near_lower_count =
      fixed_anchor_cache.anchor_cross_target_near_lower_count;
    out_breakdown->directional_cross_lonely_count = directional_cross_lonely_count;
    out_breakdown->dependent_cross_distribution_error = dependent_cross_raw_error;
    out_breakdown->dependent_cross_near_deficit_error =
      dependent_cross_near_deficit_error;
    out_breakdown->dependent_cross_weight = std::max(
      0.0f,
      state.two_class_pair_channel_weights[static_cast<size_t>(cross_channel)]);
    out_breakdown->dependent_cross_weighted_error = dependent_cross_weighted_error;
    out_breakdown->dependent_cross_near_hop_radius = dependent_cross_near_radius;
    out_breakdown->dependent_cross_target_near_count =
      fixed_anchor_cache.dependent_cross_target_near_count;
    out_breakdown->dependent_cross_target_near_lower_count =
      fixed_anchor_cache.dependent_cross_target_near_lower_count;
    out_breakdown->dependent_cross_lonely_count = dependent_cross_lonely_count;
    out_breakdown->template_offset_error = template_average_error;
    out_breakdown->template_offset_weight = template_weight;
    out_breakdown->template_offset_weighted_error = template_weighted_error;
    out_breakdown->template_valid_anchors = valid_template_anchors;
    out_breakdown->global_symmetric_error = compute_two_class_output_error(
      state,
      uv_points,
      class_ids,
      output_support_uv,
      delaunay_helper);
  }
  return final_error;
}

double two_class_channel_only_error(
  const PatternRegionState& state,
  const TwoClassPCFStats& stats,
  const std::array<bool, kTwoClassPairChannelCount>& active_channels) {
  double error = 0.0;
  double weight_sum = 0.0;
  for (int channel = 0; channel < kTwoClassPairChannelCount; ++channel) {
    if (!active_channels[static_cast<size_t>(channel)] ||
        !state.two_class_pair_channel_enabled[static_cast<size_t>(channel)]) {
      continue;
    }
    const bool target_channel_possible =
      (channel == 0 && state.two_class_voronoi_pcf_points_inside[0] >= 2) ||
      (channel == 1 && state.two_class_voronoi_pcf_points_inside[1] >= 2) ||
      (channel == 2 &&
       state.two_class_voronoi_pcf_points_inside[0] > 0 &&
       state.two_class_voronoi_pcf_points_inside[1] > 0);
    if (!target_channel_possible) {
      continue;
    }
    const double weight = std::max(
      0.0f,
      state.two_class_pair_channel_weights[static_cast<size_t>(channel)]);
    if (weight <= 0.0) {
      continue;
    }
    error += weight * weighted_distribution_l2(
      stats.hist_plot[static_cast<size_t>(channel)],
      state.two_class_voronoi_pcf_hist_plot[static_cast<size_t>(channel)]);
    weight_sum += weight;
  }
  return weight_sum > 0.0
    ? error / weight_sum
    : std::numeric_limits<double>::infinity();
}

double compute_two_class_partial_channel_error(
  const PatternRegionState& state,
  const std::vector<Eigen::Vector2d>& uv_points,
  const std::vector<int>& class_ids,
  const std::vector<Eigen::Vector2d>& output_support_uv,
  const DelaunayTraversalHelper* delaunay_helper,
  const std::array<bool, kTwoClassPairChannelCount>& active_channels) {
  const int bin_count = std::max(1, state.voronoi_pcf_bin_count);
  TwoClassPCFStats stats;
  if (!build_two_class_pair_histograms(
        uv_points,
        class_ids,
        delaunay_helper,
        bin_count,
        output_support_uv,
        stats)) {
    return std::numeric_limits<double>::infinity();
  }
  return two_class_channel_only_error(state, stats, active_channels);
}

double compute_two_class_strategic_distribution_energy(
  const std::vector<float>& distribution,
  const std::vector<float>& target_distribution) {
  if (distribution.empty() || target_distribution.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  const int bin_count = std::max(
    static_cast<int>(distribution.size()),
    static_cast<int>(target_distribution.size()));
  const int prefix_bins = std::max(1, (bin_count + 1) / 2);

  double energy = weighted_distribution_l2(distribution, target_distribution);
  double prefix_l2 = 0.0;
  double prefix_cdf = 0.0;
  double prefix_forbidden_mass = 0.0;
  double prefix_curr_cdf = 0.0;
  double prefix_target_cdf = 0.0;
  constexpr double kStrategicPrefixWeight = 7.0;
  constexpr double kStrategicPrefixCdfWeight = 3.0;
  constexpr double kStrategicPrefixLeakPenalty = 300.0;
  constexpr double kStrategicPrefixTol = 1e-10;
  for (int k = 0; k < prefix_bins; ++k) {
    const double current_value =
      (k < static_cast<int>(distribution.size()))
        ? static_cast<double>(distribution[static_cast<size_t>(k)])
        : 0.0;
    const double target_value =
      (k < static_cast<int>(target_distribution.size()))
        ? static_cast<double>(target_distribution[static_cast<size_t>(k)])
        : 0.0;
    const double delta = current_value - target_value;
    const double bin_weight = 2.5 + 10.0 * target_value;
    prefix_l2 += bin_weight * delta * delta;
    prefix_curr_cdf += current_value;
    prefix_target_cdf += target_value;
    const double cdf_delta = prefix_curr_cdf - prefix_target_cdf;
    prefix_cdf += cdf_delta * cdf_delta;
    if (target_value <= kStrategicPrefixTol && current_value > kStrategicPrefixTol) {
      prefix_forbidden_mass += current_value;
    }
  }

  energy +=
    kStrategicPrefixWeight * prefix_l2 +
    kStrategicPrefixCdfWeight * prefix_cdf +
    kStrategicPrefixLeakPenalty * prefix_forbidden_mass * prefix_forbidden_mass;
  return energy;
}

double compute_two_class_strategic_histogram_energy(
  const PatternRegionState& state,
  const std::array<std::vector<int>, kTwoClassPairChannelCount>& channel_hist_counts,
  const std::array<int, kTwoClassPairChannelCount>& channel_pair_counts,
  const std::array<std::vector<float>, kTwoClassPairChannelCount>& target_channel_distributions,
  const std::array<bool, kTwoClassPairChannelCount>& active_channels) {
  double energy = 0.0;
  double weight_sum = 0.0;
  for (int channel = 0; channel < kTwoClassPairChannelCount; ++channel) {
    if (!active_channels[static_cast<size_t>(channel)] ||
        !state.two_class_pair_channel_enabled[static_cast<size_t>(channel)]) {
      continue;
    }
    const double channel_weight = std::max(
      0.0f,
      state.two_class_pair_channel_weights[static_cast<size_t>(channel)]);
    if (channel_weight <= 0.0) {
      continue;
    }
    if (channel_pair_counts[static_cast<size_t>(channel)] <= 0 ||
        target_channel_distributions[static_cast<size_t>(channel)].empty()) {
      continue;
    }

    const std::vector<float> distribution = normalized_histogram(
      channel_hist_counts[static_cast<size_t>(channel)],
      channel_pair_counts[static_cast<size_t>(channel)]);
    if (distribution.empty()) {
      continue;
    }
    const double channel_energy = compute_two_class_strategic_distribution_energy(
      distribution,
      target_channel_distributions[static_cast<size_t>(channel)]);
    if (!std::isfinite(channel_energy)) {
      continue;
    }
    energy += channel_weight * channel_energy;
    weight_sum += channel_weight;
  }

  return weight_sum > 0.0
    ? energy / weight_sum
    : std::numeric_limits<double>::infinity();
}

bool generate_two_class_points_from_support(
  InteractionState& root_state,
  const DelaunayTraversalHelper* delaunay_helper,
  const Eigen::MatrixXd& points_3d,
  const Eigen::MatrixXd& points_uv,
  const std::vector<Eigen::Vector2d>& output_support_uv,
  const std::vector<int>& output_support_tri_indices,
  const std::array<int, kPatternClassCount>& target_counts) {
  PatternRegionState& state = active_region(root_state);
  if (!delaunay_helper || !delaunay_helper->is_ready() ||
      output_support_uv.empty() ||
      output_support_tri_indices.size() != output_support_uv.size()) {
    return false;
  }

  const std::array<int, kPatternClassCount> effective_target_counts =
    adapt_two_class_target_counts_for_locked_anchor(
      state,
      target_counts,
      static_cast<int>(output_support_uv.size()));
  const int requested_total =
    std::max(0, effective_target_counts[0]) +
    std::max(0, effective_target_counts[1]);
  if (requested_total <= 0) {
    return false;
  }

  static std::mt19937 rng(std::random_device{}());
  if (state.two_class_sequential_dependency_enabled) {
    std::vector<int> free_positions(static_cast<size_t>(output_support_uv.size()));
    std::iota(free_positions.begin(), free_positions.end(), 0);
    std::shuffle(free_positions.begin(), free_positions.end(), rng);

    std::vector<Eigen::Vector2d> generated_uv;
    std::vector<int> generated_classes;
    std::vector<int> generated_triangles;
    generated_uv.reserve(static_cast<size_t>(
      std::min<int>(requested_total, static_cast<int>(output_support_uv.size()))));
    generated_classes.reserve(generated_uv.capacity());
    generated_triangles.reserve(generated_uv.capacity());

    const int anchor_class_id =
      sanitize_pattern_class_id(state.two_class_anchor_class_id);
    const int dependent_class_id = 1 - anchor_class_id;
    const int dependent_channel =
      two_class_pair_channel(dependent_class_id, dependent_class_id);
    const int cross_channel =
      two_class_pair_channel(anchor_class_id, dependent_class_id);
    const int greedy_candidate_count = requested_total > 96 ? 16 : 24;
    const int triangle_count = delaunay_helper->triangle_count();
    std::vector<char> used_triangles(static_cast<size_t>(std::max(0, triangle_count)), 0);
    int existing_anchor_count = 0;
    std::vector<Eigen::Vector2d> locked_anchor_uv;
    std::vector<int> locked_anchor_triangles;
    (void)collect_two_class_locked_anchor_points(
      state,
      anchor_class_id,
      locked_anchor_uv,
      locked_anchor_triangles);

    for (size_t i = 0; i < locked_anchor_uv.size(); ++i) {
      int tri_idx =
        (i < locked_anchor_triangles.size())
          ? locked_anchor_triangles[i]
          : -1;
      if (tri_idx < 0 || tri_idx >= triangle_count) {
        Eigen::Vector3i tri_vertices(-1, -1, -1);
        (void)delaunay_helper->find_containing_triangle(
          locked_anchor_uv[i],
          tri_idx,
          tri_vertices);
      }
      generated_uv.push_back(locked_anchor_uv[i]);
      generated_classes.push_back(anchor_class_id);
      generated_triangles.push_back(tri_idx);
      ++existing_anchor_count;
      if (tri_idx >= 0 && tri_idx < triangle_count) {
        used_triangles[static_cast<size_t>(tri_idx)] = 1;
      }
    }
    if (existing_anchor_count <= 0) {
      std::cout << "Sequential 2-class generation skipped: no locked anchor class. "
                   "Use the normal optimizer anchor stage and finish it before "
                   "generating the dependent class.\n";
      return false;
    }

    free_positions.erase(
      std::remove_if(
        free_positions.begin(),
        free_positions.end(),
        [&](int support_pos) {
          if (support_pos < 0 ||
              support_pos >= static_cast<int>(output_support_tri_indices.size())) {
            return true;
          }
          const int tri_idx = output_support_tri_indices[static_cast<size_t>(support_pos)];
          return tri_idx >= 0 &&
                 tri_idx < triangle_count &&
                 used_triangles[static_cast<size_t>(tri_idx)] != 0;
        }),
      free_positions.end());

    const auto erase_free_position_at = [&](size_t index) {
      if (index >= free_positions.size()) {
        return;
      }
      free_positions.erase(
        free_positions.begin() + static_cast<std::ptrdiff_t>(index));
    };

    const int bin_count = std::max(1, state.voronoi_pcf_bin_count);
    std::vector<int> support_row_for_triangle(
      static_cast<size_t>(std::max(0, triangle_count)),
      -1);
    for (size_t si = 0; si < output_support_tri_indices.size(); ++si) {
      const int tri_idx = output_support_tri_indices[si];
      if (tri_idx >= 0 && tri_idx < triangle_count) {
        support_row_for_triangle[static_cast<size_t>(tri_idx)] = static_cast<int>(si);
      }
    }

    if (existing_anchor_count > 0) {
      const int dependent_target =
        std::max(0, effective_target_counts[static_cast<size_t>(dependent_class_id)]);
      const int strategic_target_count = std::min(
        dependent_target,
        static_cast<int>(free_positions.size()));
      if (strategic_target_count > 0) {
        const std::vector<Eigen::Vector2d> anchor_only_uv = generated_uv;
        const std::vector<int> anchor_only_classes = generated_classes;
        const std::vector<int> anchor_only_triangles = generated_triangles;

        FixedAnchorTwoClassEvaluationCache fixed_anchor_cache;
        (void)build_fixed_anchor_two_class_evaluation_cache(
          state,
          anchor_only_uv,
          anchor_only_classes,
          anchor_only_triangles,
          support_row_for_triangle,
          output_support_uv,
          delaunay_helper,
          anchor_class_id,
          fixed_anchor_cache);

        std::array<bool, kTwoClassPairChannelCount> dependent_active_channels = {
          false,
          false,
          false,
        };
        dependent_active_channels[static_cast<size_t>(dependent_channel)] = true;
        dependent_active_channels[static_cast<size_t>(cross_channel)] = true;
        std::array<bool, kTwoClassPairChannelCount> cross_only_channels = {
          false,
          false,
          false,
        };
        cross_only_channels[static_cast<size_t>(cross_channel)] = true;

        std::array<std::vector<float>, kTwoClassPairChannelCount> target_channel_distributions;
        for (int channel = 0; channel < kTwoClassPairChannelCount; ++channel) {
          if (!dependent_active_channels[static_cast<size_t>(channel)]) {
            continue;
          }
          const int pair_count =
            state.two_class_voronoi_pcf_pair_count[static_cast<size_t>(channel)];
          if (pair_count <= 0) {
            continue;
          }
          target_channel_distributions[static_cast<size_t>(channel)] = normalized_histogram(
            state.two_class_voronoi_pcf_hist_counts[static_cast<size_t>(channel)],
            pair_count);
        }

        std::vector<std::vector<int>> cross_hist_by_support(output_support_uv.size());
        std::vector<int> cross_pair_count_by_support(output_support_uv.size(), 0);
        for (int support_pos : free_positions) {
          std::vector<int>& cross_hist =
            cross_hist_by_support[static_cast<size_t>(support_pos)];
          cross_hist.assign(static_cast<size_t>(bin_count), 0);
          int valid_cross_pairs = 0;
          for (size_t anchor_idx = 0;
               anchor_idx < fixed_anchor_cache.anchor_uv_points.size();
               ++anchor_idx) {
            int k = -1;
            const int anchor_support_row =
              (anchor_idx < fixed_anchor_cache.anchor_support_rows.size())
                ? fixed_anchor_cache.anchor_support_rows[anchor_idx]
                : -1;
            if (anchor_support_row >= 0) {
              k = get_support_pairwise_dist(
                state,
                support_pos,
                anchor_support_row);
            }
            if (k < 0) {
              k = delaunay_helper->count_triangles_crossed(
                output_support_uv[static_cast<size_t>(support_pos)],
                fixed_anchor_cache.anchor_uv_points[anchor_idx]);
            }
            if (k < 0) {
              continue;
            }
            ++valid_cross_pairs;
            const int bin = std::min(k, bin_count - 1);
            ++cross_hist[static_cast<size_t>(bin)];
          }
          cross_pair_count_by_support[static_cast<size_t>(support_pos)] = valid_cross_pairs;
        }

        const int strategic_limit = std::min(strategic_target_count, 512);
        const int proposal_probe_limit = std::min(
          static_cast<int>(free_positions.size()),
          std::max(192, std::min(384, strategic_target_count * 4)));
        const int proposal_eval_limit = std::min(
          proposal_probe_limit,
          std::max(96, std::min(192, std::max(1, proposal_probe_limit / 2))));
        const int proposal_explore_limit = std::max(
          4,
          std::min(16, std::max(1, proposal_eval_limit / 8)));
        const int prefix_target_count = std::min(strategic_target_count, strategic_limit);
        const int max_parallel_restarts = std::max(1, std::min(4, omp_get_max_threads()));
        const int requested_restarts =
          (prefix_target_count >= 128) ? 4 :
          ((prefix_target_count >= 48) ? 3 : 2);
        const int strategic_restart_count =
          std::max(1, std::min(max_parallel_restarts, requested_restarts));

        struct DependentStrategicAttemptResult {
          std::vector<int> support_positions;
          double exact_error = std::numeric_limits<double>::infinity();
          bool valid = false;
        };

        const auto run_dependent_strategic_attempt =
          [&](unsigned int seed) -> DependentStrategicAttemptResult {
            DependentStrategicAttemptResult result;
            std::mt19937 attempt_rng(seed);
            std::vector<char> selected_mask(output_support_uv.size(), 0);
            std::vector<int> selected_positions;
            selected_positions.reserve(static_cast<size_t>(strategic_target_count));

            std::array<std::vector<int>, kTwoClassPairChannelCount> running_channel_hist_counts;
            for (auto& hist : running_channel_hist_counts) {
              hist.assign(static_cast<size_t>(bin_count), 0);
            }
            std::array<int, kTwoClassPairChannelCount> running_channel_pair_counts = {0, 0, 0};

            const auto collect_candidate_pair_state =
              [&](int candidate_pos,
                  const std::vector<int>& current_selected_positions,
                  std::vector<int>& add_hist,
                  int& add_pair_count) {
                add_hist.assign(static_cast<size_t>(bin_count), 0);
                add_pair_count = 0;
                for (int selected_pos : current_selected_positions) {
                  int k = get_support_pairwise_dist(state, candidate_pos, selected_pos);
                  if (k < 0) {
                    k = delaunay_helper->count_triangles_crossed(
                      output_support_uv[static_cast<size_t>(candidate_pos)],
                      output_support_uv[static_cast<size_t>(selected_pos)]);
                  }
                  if (k < 0) {
                    continue;
                  }
                  ++add_pair_count;
                  const int bin = std::min(k, bin_count - 1);
                  ++add_hist[static_cast<size_t>(bin)];
                }
              };

            const auto projected_candidate_energy =
              [&](int candidate_pos,
                  const std::vector<int>& current_selected_positions,
                  const std::array<std::vector<int>, kTwoClassPairChannelCount>& current_hist_counts,
                  const std::array<int, kTwoClassPairChannelCount>& current_pair_counts,
                  const std::array<bool, kTwoClassPairChannelCount>& active_channels,
                  std::vector<int>* out_add_hist,
                  int* out_add_pair_count) {
                std::vector<int> local_add_hist;
                int local_add_pair_count = 0;
                collect_candidate_pair_state(
                  candidate_pos,
                  current_selected_positions,
                  local_add_hist,
                  local_add_pair_count);

                std::array<std::vector<int>, kTwoClassPairChannelCount> projected_hist_counts =
                  current_hist_counts;
                std::array<int, kTwoClassPairChannelCount> projected_pair_counts =
                  current_pair_counts;
                for (int k = 0; k < bin_count; ++k) {
                  projected_hist_counts[static_cast<size_t>(dependent_channel)][static_cast<size_t>(k)] +=
                    local_add_hist[static_cast<size_t>(k)];
                  projected_hist_counts[static_cast<size_t>(cross_channel)][static_cast<size_t>(k)] +=
                    cross_hist_by_support[static_cast<size_t>(candidate_pos)][static_cast<size_t>(k)];
                }
                projected_pair_counts[static_cast<size_t>(dependent_channel)] +=
                  local_add_pair_count;
                projected_pair_counts[static_cast<size_t>(cross_channel)] +=
                  cross_pair_count_by_support[static_cast<size_t>(candidate_pos)];

                if (out_add_hist) {
                  *out_add_hist = std::move(local_add_hist);
                }
                if (out_add_pair_count) {
                  *out_add_pair_count = local_add_pair_count;
                }
                return compute_two_class_strategic_histogram_energy(
                  state,
                  projected_hist_counts,
                  projected_pair_counts,
                  target_channel_distributions,
                  active_channels);
              };

            std::vector<int> seed_pool = free_positions;
            constexpr int strategic_seed_pool_limit = 96;
            if (static_cast<int>(seed_pool.size()) > strategic_seed_pool_limit) {
              std::shuffle(seed_pool.begin(), seed_pool.end(), attempt_rng);
              seed_pool.resize(static_cast<size_t>(strategic_seed_pool_limit));
            }

            if (prefix_target_count >= 2 && seed_pool.size() >= 2) {
              int best_seed_a = -1;
              int best_seed_b = -1;
              double best_seed_energy = std::numeric_limits<double>::infinity();
              for (size_t ia = 0; ia < seed_pool.size(); ++ia) {
                for (size_t ib = ia + 1; ib < seed_pool.size(); ++ib) {
                  const int seed_a = seed_pool[ia];
                  const int seed_b = seed_pool[ib];

                  std::array<std::vector<int>, kTwoClassPairChannelCount> seed_hist_counts;
                  for (auto& hist : seed_hist_counts) {
                    hist.assign(static_cast<size_t>(bin_count), 0);
                  }
                  std::array<int, kTwoClassPairChannelCount> seed_pair_counts = {0, 0, 0};
                  for (int k = 0; k < bin_count; ++k) {
                    seed_hist_counts[static_cast<size_t>(cross_channel)][static_cast<size_t>(k)] =
                      cross_hist_by_support[static_cast<size_t>(seed_a)][static_cast<size_t>(k)] +
                      cross_hist_by_support[static_cast<size_t>(seed_b)][static_cast<size_t>(k)];
                  }
                  seed_pair_counts[static_cast<size_t>(cross_channel)] =
                    cross_pair_count_by_support[static_cast<size_t>(seed_a)] +
                    cross_pair_count_by_support[static_cast<size_t>(seed_b)];

                  int bb_k = get_support_pairwise_dist(state, seed_a, seed_b);
                  if (bb_k < 0) {
                    bb_k = delaunay_helper->count_triangles_crossed(
                      output_support_uv[static_cast<size_t>(seed_a)],
                      output_support_uv[static_cast<size_t>(seed_b)]);
                  }
                  if (bb_k >= 0) {
                    ++seed_pair_counts[static_cast<size_t>(dependent_channel)];
                    const int bb_bin = std::min(bb_k, bin_count - 1);
                    ++seed_hist_counts[static_cast<size_t>(dependent_channel)][static_cast<size_t>(bb_bin)];
                  }

                  const double seed_energy = compute_two_class_strategic_histogram_energy(
                    state,
                    seed_hist_counts,
                    seed_pair_counts,
                    target_channel_distributions,
                    dependent_active_channels);
                  if (seed_energy < best_seed_energy) {
                    best_seed_energy = seed_energy;
                    best_seed_a = seed_a;
                    best_seed_b = seed_b;
                  }
                }
              }

              if (best_seed_a >= 0 && best_seed_b >= 0) {
                selected_positions.push_back(best_seed_a);
                selected_positions.push_back(best_seed_b);
                selected_mask[static_cast<size_t>(best_seed_a)] = 1;
                selected_mask[static_cast<size_t>(best_seed_b)] = 1;
                for (int k = 0; k < bin_count; ++k) {
                  running_channel_hist_counts[static_cast<size_t>(cross_channel)][static_cast<size_t>(k)] =
                    cross_hist_by_support[static_cast<size_t>(best_seed_a)][static_cast<size_t>(k)] +
                    cross_hist_by_support[static_cast<size_t>(best_seed_b)][static_cast<size_t>(k)];
                }
                running_channel_pair_counts[static_cast<size_t>(cross_channel)] =
                  cross_pair_count_by_support[static_cast<size_t>(best_seed_a)] +
                  cross_pair_count_by_support[static_cast<size_t>(best_seed_b)];
                int bb_k = get_support_pairwise_dist(state, best_seed_a, best_seed_b);
                if (bb_k < 0) {
                  bb_k = delaunay_helper->count_triangles_crossed(
                    output_support_uv[static_cast<size_t>(best_seed_a)],
                    output_support_uv[static_cast<size_t>(best_seed_b)]);
                }
                if (bb_k >= 0) {
                  ++running_channel_pair_counts[static_cast<size_t>(dependent_channel)];
                  const int bb_bin = std::min(bb_k, bin_count - 1);
                  ++running_channel_hist_counts[static_cast<size_t>(dependent_channel)][static_cast<size_t>(bb_bin)];
                }
              }
            }

            if (selected_positions.empty()) {
              int best_seed_pos = -1;
              double best_seed_energy = std::numeric_limits<double>::infinity();
              for (int candidate_pos : seed_pool) {
                std::array<std::vector<int>, kTwoClassPairChannelCount> projected_hist_counts;
                for (auto& hist : projected_hist_counts) {
                  hist.assign(static_cast<size_t>(bin_count), 0);
                }
                std::array<int, kTwoClassPairChannelCount> projected_pair_counts = {0, 0, 0};
                projected_hist_counts[static_cast<size_t>(cross_channel)] =
                  cross_hist_by_support[static_cast<size_t>(candidate_pos)];
                projected_pair_counts[static_cast<size_t>(cross_channel)] =
                  cross_pair_count_by_support[static_cast<size_t>(candidate_pos)];
                const double seed_energy = compute_two_class_strategic_histogram_energy(
                  state,
                  projected_hist_counts,
                  projected_pair_counts,
                  target_channel_distributions,
                  cross_only_channels);
                if (seed_energy < best_seed_energy) {
                  best_seed_energy = seed_energy;
                  best_seed_pos = candidate_pos;
                }
              }
              if (best_seed_pos < 0 && !seed_pool.empty()) {
                std::uniform_int_distribution<int> seed_pick(
                  0,
                  static_cast<int>(seed_pool.size()) - 1);
                best_seed_pos = seed_pool[static_cast<size_t>(seed_pick(attempt_rng))];
              }
              if (best_seed_pos < 0) {
                return result;
              }
              selected_positions.push_back(best_seed_pos);
              selected_mask[static_cast<size_t>(best_seed_pos)] = 1;
              running_channel_hist_counts[static_cast<size_t>(cross_channel)] =
                cross_hist_by_support[static_cast<size_t>(best_seed_pos)];
              running_channel_pair_counts[static_cast<size_t>(cross_channel)] =
                cross_pair_count_by_support[static_cast<size_t>(best_seed_pos)];
            }

            while (static_cast<int>(selected_positions.size()) < prefix_target_count) {
              std::vector<int> unselected_positions;
              unselected_positions.reserve(free_positions.size());
              for (int support_pos : free_positions) {
                if (!selected_mask[static_cast<size_t>(support_pos)]) {
                  unselected_positions.push_back(support_pos);
                }
              }
              if (unselected_positions.empty()) {
                break;
              }

              std::vector<int> probe_positions = unselected_positions;
              if (static_cast<int>(probe_positions.size()) > proposal_probe_limit) {
                std::shuffle(probe_positions.begin(), probe_positions.end(), attempt_rng);
                probe_positions.resize(static_cast<size_t>(proposal_probe_limit));
              }

              struct RankedCandidate {
                double quick_energy;
                int support_pos;
              };
              std::vector<RankedCandidate> ranked_candidates;
              ranked_candidates.reserve(probe_positions.size());
              for (int candidate_pos : probe_positions) {
                std::array<std::vector<int>, kTwoClassPairChannelCount> projected_hist_counts =
                  running_channel_hist_counts;
                std::array<int, kTwoClassPairChannelCount> projected_pair_counts =
                  running_channel_pair_counts;
                for (int k = 0; k < bin_count; ++k) {
                  projected_hist_counts[static_cast<size_t>(cross_channel)][static_cast<size_t>(k)] +=
                    cross_hist_by_support[static_cast<size_t>(candidate_pos)][static_cast<size_t>(k)];
                }
                projected_pair_counts[static_cast<size_t>(cross_channel)] +=
                  cross_pair_count_by_support[static_cast<size_t>(candidate_pos)];
                const double quick_energy = compute_two_class_strategic_histogram_energy(
                  state,
                  projected_hist_counts,
                  projected_pair_counts,
                  target_channel_distributions,
                  cross_only_channels);
                ranked_candidates.push_back({quick_energy, candidate_pos});
              }

              std::sort(
                ranked_candidates.begin(),
                ranked_candidates.end(),
                [](const RankedCandidate& lhs, const RankedCandidate& rhs) {
                  if (lhs.quick_energy == rhs.quick_energy) {
                    return lhs.support_pos < rhs.support_pos;
                  }
                  return lhs.quick_energy < rhs.quick_energy;
                });

              std::vector<int> evaluation_positions;
              evaluation_positions.reserve(
                std::min(proposal_eval_limit, static_cast<int>(ranked_candidates.size())));
              const int explore_count = std::min(
                proposal_explore_limit,
                std::max(0, static_cast<int>(ranked_candidates.size()) - 1));
              const int greedy_keep = std::min(
                static_cast<int>(ranked_candidates.size()),
                std::max(1, proposal_eval_limit - explore_count));
              for (int i = 0; i < greedy_keep; ++i) {
                evaluation_positions.push_back(
                  ranked_candidates[static_cast<size_t>(i)].support_pos);
              }
              if (static_cast<int>(evaluation_positions.size()) < proposal_eval_limit &&
                  static_cast<int>(ranked_candidates.size()) > greedy_keep) {
                std::vector<int> explore_positions;
                explore_positions.reserve(
                  static_cast<size_t>(ranked_candidates.size() - greedy_keep));
                for (size_t i = static_cast<size_t>(greedy_keep);
                     i < ranked_candidates.size();
                     ++i) {
                  explore_positions.push_back(
                    ranked_candidates[i].support_pos);
                }
                std::shuffle(
                  explore_positions.begin(),
                  explore_positions.end(),
                  attempt_rng);
                const int remaining_slots =
                  proposal_eval_limit - static_cast<int>(evaluation_positions.size());
                const int add_count = std::min(
                  remaining_slots,
                  static_cast<int>(explore_positions.size()));
                for (int i = 0; i < add_count; ++i) {
                  evaluation_positions.push_back(
                    explore_positions[static_cast<size_t>(i)]);
                }
              }

              int best_candidate_pos = -1;
              double best_candidate_energy = std::numeric_limits<double>::infinity();
              std::vector<int> best_candidate_add_hist(static_cast<size_t>(bin_count), 0);
              int best_candidate_add_pair_count = 0;

              for (int candidate_pos : evaluation_positions) {
                std::vector<int> candidate_add_hist;
                int candidate_add_pair_count = 0;
                const double candidate_energy = projected_candidate_energy(
                  candidate_pos,
                  selected_positions,
                  running_channel_hist_counts,
                  running_channel_pair_counts,
                  dependent_active_channels,
                  &candidate_add_hist,
                  &candidate_add_pair_count);
                if (candidate_energy < best_candidate_energy) {
                  best_candidate_energy = candidate_energy;
                  best_candidate_pos = candidate_pos;
                  best_candidate_add_hist = std::move(candidate_add_hist);
                  best_candidate_add_pair_count = candidate_add_pair_count;
                }
              }

              if (best_candidate_pos < 0) {
                std::uniform_int_distribution<int> fallback_pick(
                  0,
                  static_cast<int>(probe_positions.size()) - 1);
                best_candidate_pos =
                  probe_positions[static_cast<size_t>(fallback_pick(attempt_rng))];
                collect_candidate_pair_state(
                  best_candidate_pos,
                  selected_positions,
                  best_candidate_add_hist,
                  best_candidate_add_pair_count);
              }

              selected_positions.push_back(best_candidate_pos);
              selected_mask[static_cast<size_t>(best_candidate_pos)] = 1;
              for (int k = 0; k < bin_count; ++k) {
                running_channel_hist_counts[static_cast<size_t>(dependent_channel)][static_cast<size_t>(k)] +=
                  best_candidate_add_hist[static_cast<size_t>(k)];
                running_channel_hist_counts[static_cast<size_t>(cross_channel)][static_cast<size_t>(k)] +=
                  cross_hist_by_support[static_cast<size_t>(best_candidate_pos)][static_cast<size_t>(k)];
              }
              running_channel_pair_counts[static_cast<size_t>(dependent_channel)] +=
                best_candidate_add_pair_count;
              running_channel_pair_counts[static_cast<size_t>(cross_channel)] +=
                cross_pair_count_by_support[static_cast<size_t>(best_candidate_pos)];
            }

            if (static_cast<int>(selected_positions.size()) < strategic_target_count) {
              std::vector<int> remaining_positions;
              remaining_positions.reserve(free_positions.size());
              for (int support_pos : free_positions) {
                if (!selected_mask[static_cast<size_t>(support_pos)]) {
                  remaining_positions.push_back(support_pos);
                }
              }
              std::shuffle(
                remaining_positions.begin(),
                remaining_positions.end(),
                attempt_rng);
              const int needed =
                strategic_target_count - static_cast<int>(selected_positions.size());
              const int add_count = std::min(
                needed,
                static_cast<int>(remaining_positions.size()));
              for (int i = 0; i < add_count; ++i) {
                selected_positions.push_back(
                  remaining_positions[static_cast<size_t>(i)]);
              }
            }

            if (selected_positions.empty()) {
              return result;
            }

            std::vector<Eigen::Vector2d> attempt_uv = anchor_only_uv;
            std::vector<int> attempt_classes = anchor_only_classes;
            std::vector<int> attempt_triangles = anchor_only_triangles;
            attempt_uv.reserve(anchor_only_uv.size() + selected_positions.size());
            attempt_classes.reserve(anchor_only_classes.size() + selected_positions.size());
            attempt_triangles.reserve(anchor_only_triangles.size() + selected_positions.size());
            for (int support_pos : selected_positions) {
              attempt_uv.push_back(output_support_uv[static_cast<size_t>(support_pos)]);
              attempt_classes.push_back(dependent_class_id);
              attempt_triangles.push_back(
                output_support_tri_indices[static_cast<size_t>(support_pos)]);
            }

            result.support_positions = std::move(selected_positions);
            result.exact_error = compute_two_class_output_error_with_fixed_anchor(
              state,
              attempt_uv,
              attempt_classes,
              &attempt_triangles,
              output_support_uv,
              support_row_for_triangle,
              delaunay_helper,
              fixed_anchor_cache);
            result.valid = !result.support_positions.empty();
            return result;
          };

        std::random_device rd;
        const unsigned int base_seed = rd();
        std::vector<DependentStrategicAttemptResult> attempt_results(
          static_cast<size_t>(strategic_restart_count));
        #pragma omp parallel for schedule(static) if (strategic_restart_count > 1)
        for (int attempt_idx = 0;
             attempt_idx < strategic_restart_count;
             ++attempt_idx) {
          const unsigned int attempt_seed =
            base_seed +
            static_cast<unsigned int>(0x9e3779b9u * static_cast<unsigned int>(attempt_idx + 1));
          attempt_results[static_cast<size_t>(attempt_idx)] =
            run_dependent_strategic_attempt(attempt_seed);
        }

        int best_attempt_idx = -1;
        for (int attempt_idx = 0;
             attempt_idx < strategic_restart_count;
             ++attempt_idx) {
          const auto& candidate = attempt_results[static_cast<size_t>(attempt_idx)];
          if (!candidate.valid) {
            continue;
          }
          if (best_attempt_idx < 0) {
            best_attempt_idx = attempt_idx;
            continue;
          }
          const auto& best = attempt_results[static_cast<size_t>(best_attempt_idx)];
          const bool candidate_finite = std::isfinite(candidate.exact_error);
          const bool best_finite = std::isfinite(best.exact_error);
          if (candidate_finite != best_finite) {
            if (candidate_finite) {
              best_attempt_idx = attempt_idx;
            }
            continue;
          }
          if (candidate_finite &&
              candidate.exact_error + 1e-9 < best.exact_error) {
            best_attempt_idx = attempt_idx;
            continue;
          }
          if ((!best_finite ||
               std::abs(candidate.exact_error - best.exact_error) <= 1e-9) &&
              candidate.support_positions.size() > best.support_positions.size()) {
            best_attempt_idx = attempt_idx;
          }
        }

        if (best_attempt_idx >= 0) {
          const auto& best_attempt = attempt_results[static_cast<size_t>(best_attempt_idx)];
          for (int support_pos : best_attempt.support_positions) {
            generated_uv.push_back(output_support_uv[static_cast<size_t>(support_pos)]);
            generated_classes.push_back(dependent_class_id);
            generated_triangles.push_back(
              output_support_tri_indices[static_cast<size_t>(support_pos)]);
          }

          clear_output_pattern_and_hist(state);
          state.two_class_target_output_counts = effective_target_counts;
          sync_two_class_points_to_state(
            state,
            generated_uv,
            generated_classes,
            generated_triangles,
            delaunay_helper,
            points_3d,
            points_uv);
          double error = std::numeric_limits<double>::infinity();
          update_two_class_output_stats_in_state(
            state,
            generated_uv,
            generated_classes,
            output_support_uv,
            delaunay_helper,
            &output_support_tri_indices,
            &error);
          state.optimizer_improvements = 0;
          state.optimizer_iterations_ran = 0;
          std::cout << "Generated dependent class with strategic placement: anchor=class "
                    << anchor_class_id
                    << ", dependent=class " << dependent_class_id
                    << ", class0=" << state.two_class_output_counts[0]
                    << ", class1=" << state.two_class_output_counts[1]
                    << ", restarts=" << strategic_restart_count
                    << ", probe=" << proposal_probe_limit
                    << ", eval=" << proposal_eval_limit
                    << ", error=" << error << "\n";
          return true;
        }
      }
    }

    const auto append_best_for_class =
      [&](int class_id,
          int class_target,
          const std::array<bool, kTwoClassPairChannelCount>& active_channels) {
        const int safe_target = std::max(0, class_target);
        for (int placed = 0; placed < safe_target && !free_positions.empty(); ++placed) {
          std::shuffle(free_positions.begin(), free_positions.end(), rng);
          const int eval_count = std::min(
            greedy_candidate_count,
            static_cast<int>(free_positions.size()));
          int best_eval_index = 0;
          double best_score = std::numeric_limits<double>::infinity();
          for (int eval_index = 0; eval_index < eval_count; ++eval_index) {
            const int support_pos = free_positions[static_cast<size_t>(eval_index)];
            std::vector<Eigen::Vector2d> test_uv = generated_uv;
            std::vector<int> test_classes = generated_classes;
            test_uv.push_back(output_support_uv[static_cast<size_t>(support_pos)]);
            test_classes.push_back(class_id);

            double candidate_score = compute_two_class_partial_channel_error(
              state,
              test_uv,
              test_classes,
              output_support_uv,
              delaunay_helper,
              active_channels);
            if (!std::isfinite(candidate_score)) {
              candidate_score = 0.0;
            }
            const double fill_fraction =
              static_cast<double>(placed + 1) /
              static_cast<double>(std::max(1, safe_target));
            candidate_score += 1e-6 * fill_fraction *
              static_cast<double>(support_pos % 997);
            if (candidate_score < best_score) {
              best_score = candidate_score;
              best_eval_index = eval_index;
            }
          }

          const int support_pos =
            free_positions[static_cast<size_t>(best_eval_index)];
          generated_uv.push_back(output_support_uv[static_cast<size_t>(support_pos)]);
          generated_classes.push_back(class_id);
          generated_triangles.push_back(
            output_support_tri_indices[static_cast<size_t>(support_pos)]);
          const int tri_idx = output_support_tri_indices[static_cast<size_t>(support_pos)];
          if (tri_idx >= 0 && tri_idx < triangle_count) {
            used_triangles[static_cast<size_t>(tri_idx)] = 1;
          }
          erase_free_position_at(static_cast<size_t>(best_eval_index));
        }
      };

    std::array<bool, kTwoClassPairChannelCount> dependent_channels = {false, false, false};
    dependent_channels[static_cast<size_t>(dependent_channel)] = true;
    dependent_channels[static_cast<size_t>(cross_channel)] = true;

    append_best_for_class(
      dependent_class_id,
      effective_target_counts[static_cast<size_t>(dependent_class_id)],
      dependent_channels);

    if (generated_uv.empty()) {
      return false;
    }

    clear_output_pattern_and_hist(state);
    state.two_class_target_output_counts = effective_target_counts;
    sync_two_class_points_to_state(
      state,
      generated_uv,
      generated_classes,
      generated_triangles,
      delaunay_helper,
      points_3d,
      points_uv);
    double error = std::numeric_limits<double>::infinity();
    update_two_class_output_stats_in_state(
      state,
      generated_uv,
      generated_classes,
      output_support_uv,
      delaunay_helper,
      &output_support_tri_indices,
      &error);
    state.optimizer_improvements = 0;
    state.optimizer_iterations_ran = 0;
    std::cout << "Generated sequential 2-class seed: anchor=class "
              << anchor_class_id
              << ", class0=" << state.two_class_output_counts[0]
              << ", class1=" << state.two_class_output_counts[1]
              << ", error=" << error << "\n";
    return true;
  }

  std::vector<int> all_positions(static_cast<size_t>(output_support_uv.size()));
  std::iota(all_positions.begin(), all_positions.end(), 0);
  std::shuffle(all_positions.begin(), all_positions.end(), rng);

  std::vector<Eigen::Vector2d> generated_uv;
  std::vector<int> generated_classes;
  std::vector<int> generated_triangles;
  generated_uv.reserve(static_cast<size_t>(requested_total));
  generated_classes.reserve(static_cast<size_t>(requested_total));
  generated_triangles.reserve(static_cast<size_t>(requested_total));

  size_t cursor = 0;
  for (int class_id = 0; class_id < kPatternClassCount; ++class_id) {
    const int target = std::max(0, target_counts[static_cast<size_t>(class_id)]);
    for (int count = 0;
         count < target && cursor < all_positions.size();
         ++count, ++cursor) {
      const int support_pos = all_positions[cursor];
      generated_uv.push_back(output_support_uv[static_cast<size_t>(support_pos)]);
      generated_classes.push_back(class_id);
      generated_triangles.push_back(output_support_tri_indices[static_cast<size_t>(support_pos)]);
    }
  }

  if (generated_uv.empty()) {
    return false;
  }

  clear_output_pattern_and_hist(state);
  state.two_class_target_output_counts = effective_target_counts;
  sync_two_class_points_to_state(
    state,
    generated_uv,
    generated_classes,
    generated_triangles,
    delaunay_helper,
    points_3d,
    points_uv);
  double error = std::numeric_limits<double>::infinity();
  update_two_class_output_stats_in_state(
    state,
    generated_uv,
    generated_classes,
    output_support_uv,
    delaunay_helper,
    &output_support_tri_indices,
    &error);
  state.optimizer_improvements = 0;
  state.optimizer_iterations_ran = 0;
  std::cout << "Generated 2-class seed: class0=" << state.two_class_output_counts[0]
            << ", class1=" << state.two_class_output_counts[1]
            << ", error=" << error << "\n";
  return true;
}

} // namespace

void draw_voronoi_pcf_ui(
  InteractionState& root_state,
  const DelaunayTraversalHelper* delaunay_helper,
  const Eigen::MatrixXd& points_3d,
  const Eigen::MatrixXd& points_uv) {
  const auto set_generated_patch_batch_status =
    [&](const std::string& status, bool is_error) {
      root_state.generated_patch_batch_status = status;
      root_state.generated_patch_batch_status_is_error =
        !status.empty() && is_error;
  };

  PatternRegionState& state = active_region(root_state);
  state.region_mode = static_cast<int>(PatternRegionMode::Exemplar);
  state.active_pattern_class_id = 0;
  state.two_class_anchor_normal_stage_active = false;
  static std::unordered_map<int, int> last_input_pcf_bin_count;
  const int region_runtime_id = state.region_id;
  constexpr bool is_transition_region = false;
  const auto active_generated_patch_family_indices = [&]() {
    if (!region_is_generated_patch_exemplar(state)) {
      return std::vector<int>{};
    }
    return generated_patch_family_region_indices(
      root_state,
      state.generated_patch_family_id);
  };
  const auto should_broadcast_to_generated_patch_family = [&]() {
    if (!root_state.generated_patch_batch_mode_enabled ||
        !region_is_generated_patch_exemplar(state) ||
        is_transition_region) {
      return false;
    }
    return active_generated_patch_family_indices().size() > 1;
  };
  ImGui::Separator();
  ImGui::TextUnformatted("Pattern Synthesis - Voronoi PCF");
  
  if (ImGui::SliderInt("Histogram bins##pcf_bin_count", &state.voronoi_pcf_bin_count, 8, 256)) {
    state.voronoi_pcf_bin_count = std::max(1, state.voronoi_pcf_bin_count);
    if (state.voronoi_pcf_ready) {
      reset_voronoi_pcf(root_state);
      compute_voronoi_pcf_histogram(root_state, delaunay_helper);
      last_input_pcf_bin_count[region_runtime_id] = state.voronoi_pcf_bin_count;
    }
  }

  if (ImGui::Button("Compute input PCF", ImVec2(-1, 0))) {
    if (should_broadcast_to_generated_patch_family()) {
      const std::vector<int> family_region_indices =
        active_generated_patch_family_indices();
      const int saved_active_region_index = root_state.active_region_index;
      for (int region_index : family_region_indices) {
        root_state.active_region_index = region_index;
        compute_voronoi_pcf_histogram(root_state, delaunay_helper);
        const PatternRegionState& family_region =
          region_state(root_state, region_index);
        last_input_pcf_bin_count[family_region.region_id] =
          family_region.voronoi_pcf_bin_count;
      }
      root_state.active_region_index = saved_active_region_index;
      std::ostringstream status;
      status << "Computed input PCF for " << family_region_indices.size()
             << " generated exemplar patches.";
      set_generated_patch_batch_status(status.str(), false);
    } else {
      compute_voronoi_pcf_histogram(root_state, delaunay_helper);
      last_input_pcf_bin_count[region_runtime_id] = state.voronoi_pcf_bin_count;
    }
  }
  if (state.voronoi_pcf_ready) {
    const int effective_bins = std::max(1, state.voronoi_pcf_max_k + 1);
    ImGui::Text("Effective bins (nonzero): %d / %d", effective_bins, state.voronoi_pcf_bin_count);
    if (last_input_pcf_bin_count[region_runtime_id] == state.voronoi_pcf_bin_count &&
        effective_bins > 0 &&
        effective_bins < state.voronoi_pcf_bin_count) {
      if (ImGui::Button("Set bins to effective")) {
        state.voronoi_pcf_bin_count = effective_bins;
        reset_voronoi_pcf(root_state);
        compute_voronoi_pcf_histogram(root_state, delaunay_helper);
        last_input_pcf_bin_count[region_runtime_id] = state.voronoi_pcf_bin_count;
      }
      ImGui::SameLine();
      ImGui::TextDisabled("(recomputes input PCF)");
    }
  }
  
  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();

  int estimated_target_count = -1;
  int input_support_count = 0;
  int output_support_count = 0;
  std::vector<Eigen::Vector2d> output_support_uv;
  std::vector<int> output_support_tri_indices;
  if (delaunay_helper && delaunay_helper->is_ready()) {
    const int support_cache_bin_count = std::max(1, state.voronoi_pcf_bin_count);
    if (ensure_output_support_denominator_cache(
          root_state,
          delaunay_helper,
          support_cache_bin_count)) {
      output_support_uv = state.output_support_uv_cache;
      output_support_tri_indices = state.output_support_tri_indices_cache;
    } else if (state.voronoi_pcf_ready && state.voronoi_pcf_points_inside >= 2) {
      (void)collect_output_triangle_center_candidates(
        root_state,
        delaunay_helper,
        output_support_uv,
        &output_support_tri_indices);
    }
    if (!output_support_uv.empty()) {
      output_support_count = static_cast<int>(output_support_uv.size());
    }
  }

  const auto clamp_target_count_to_support = [&](int raw_target_count) {
    if (output_support_count <= 0) {
      return -1;
    }
    if (output_support_count == 1) {
      return 1;
    }
    return std::max(2, std::min(raw_target_count, output_support_count));
  };

  if (delaunay_helper && delaunay_helper->is_ready() &&
      state.voronoi_pcf_ready && state.voronoi_pcf_points_inside >= 2) {
    std::vector<Eigen::Vector2d> input_support_uv;
    const bool got_input_support = collect_triangle_center_candidates_in_polygon(
      state.input_boundary_uv,
      delaunay_helper,
      input_support_uv);
    if (got_input_support && !input_support_uv.empty()) {
      input_support_count = static_cast<int>(input_support_uv.size());
    }
    if (input_support_count > 0 && output_support_count > 0) {
      const double input_density =
        static_cast<double>(state.voronoi_pcf_points_inside) /
        static_cast<double>(input_support_count);
      const double buffered_target_count =
        1. * input_density * static_cast<double>(output_support_count); //changed from 1.5 to 1.0 to reduce target count and speed up optimization
      estimated_target_count = clamp_target_count_to_support(
        static_cast<int>(std::llround(buffered_target_count)));
    }
  }

struct TransitionOptimizerTargets {
  bool ready = false;
  std::vector<std::vector<float>> distribution_for_support_row;
  std::vector<std::vector<int>> raw_point_hist_counts_for_support_row;
};

  enum class VoronoiOptimizerMode {
    Normal,
    StructuredTargets,
  };

  static std::vector<float> live_target_hist;
  static std::vector<std::vector<float>> live_target_individual_distributions;
  static std::vector<std::vector<int>> live_target_raw_connectivity_signatures;
  static bool live_position_targets_enabled = false;
  static TransitionOptimizerTargets live_transition_optimizer_targets;
  static int live_hist_bin_count = 0;
  static std::mt19937 live_rng(std::random_device{}());
  static int live_worst_bin_focus_count = 4;
  static double live_last_worst_bin_residual = std::numeric_limits<double>::infinity();
  static int live_last_worst_bin_index = -1;
  static bool use_bin_weighting = true;
  static int bin_weight_transition = 8;
  static float early_bin_weight = 8.0f;
  static float late_bin_weight = 0.35f;
  static float transition_direction_gradient_weight = 0.45f;
  static bool hard_near_field_objective = true;
  static int hard_near_field_split_bin = 0;  // 0 = auto (half of bins)
  constexpr double kHardNearFieldTol = 1e-6;
  constexpr double kHardNearFieldPenaltyScale = 1e5;

  const auto refresh_transition_optimizer_targets = [&]() {
    live_transition_optimizer_targets.ready = false;
    live_transition_optimizer_targets.distribution_for_support_row.clear();
    live_transition_optimizer_targets.raw_point_hist_counts_for_support_row.clear();
  };
    refresh_transition_optimizer_targets();

    const auto transition_optimizer_targets_active = [&]() {
      return false;
    };

    const auto structured_support_rows_required = [&]() {
      return false;
    };

    const auto current_optimizer_mode = [&]() -> VoronoiOptimizerMode {
      return VoronoiOptimizerMode::Normal;
    };

    const auto optimizer_mode_label =
      [&]() -> const char* {
        return "normal";
    };

  const auto near_field_split_for_bins = [&](int bin_count) {
    const int safe_bins = std::max(1, bin_count);
    if (hard_near_field_split_bin > 0) {
      return std::clamp(hard_near_field_split_bin, 1, safe_bins);
    }
    return std::max(1, safe_bins / 2);
  };

  const auto infer_local_proposal_radius_from_distributions =
    [&](const std::vector<float>& current_distribution,
        const std::vector<float>& target_distribution,
        int eval_bins,
        int hist_bin_count) {
      if (target_distribution.empty() || eval_bins <= 0) {
        return 1;
      }

      const int safe_eval_bins = std::max(1, eval_bins);
      const int strong_prefix_bins = std::min(
        safe_eval_bins,
        std::max(2, std::min(6, near_field_split_for_bins(hist_bin_count))));
      const int radius_eval_bins = std::min(
        safe_eval_bins,
        std::max(3, std::min(5, strong_prefix_bins + 1)));

      double target_peak = 0.0;
      for (int k = 0; k < radius_eval_bins; ++k) {
        const double tgt_v =
          (k < static_cast<int>(target_distribution.size()))
            ? static_cast<double>(target_distribution[static_cast<size_t>(k)])
            : 0.0;
        target_peak = std::max(target_peak, tgt_v);
      }

      double positive_deficit_mass = 0.0;
      double positive_deficit_center = 0.0;
      int farthest_meaningful_deficit_bin = -1;
      for (int k = 0; k < radius_eval_bins; ++k) {
        const double out_v =
          (k < static_cast<int>(current_distribution.size()))
            ? static_cast<double>(current_distribution[static_cast<size_t>(k)])
            : 0.0;
        const double tgt_v =
          (k < static_cast<int>(target_distribution.size()))
            ? static_cast<double>(target_distribution[static_cast<size_t>(k)])
            : 0.0;
        const double deficit = std::max(0.0, tgt_v - out_v);
        positive_deficit_mass += deficit;
        positive_deficit_center +=
          static_cast<double>(k + 1) * deficit;
        const double meaningful_deficit_floor =
          std::max(0.02, 0.15 * target_peak);
        if (deficit >= meaningful_deficit_floor) {
          farthest_meaningful_deficit_bin = k;
        }
      }

      if (positive_deficit_mass <= 0.05) {
        return 1;
      }
      const double deficit_centroid =
        positive_deficit_center / std::max(1e-9, positive_deficit_mass);
      int proposal_radius = 1;
      if (farthest_meaningful_deficit_bin >= 1 ||
          deficit_centroid >= 1.25 ||
          positive_deficit_mass >= 0.10) {
        proposal_radius = 2;
      }
      if (farthest_meaningful_deficit_bin >= 3 ||
          deficit_centroid >= 2.6) {
        proposal_radius = 3;
      }
      return std::clamp(proposal_radius, 1, 3);
    };

  const auto forbidden_near_field_mass =
    [&](const std::vector<float>& distribution,
        const std::vector<float>& reference_distribution,
        int bin_count) {
      if (!hard_near_field_objective) {
        return 0.0;
      }
      const int split_bin = near_field_split_for_bins(bin_count);
      const int eval_bins = std::min(
        split_bin,
        std::max(
          static_cast<int>(distribution.size()),
          static_cast<int>(reference_distribution.size())));
      double forbidden_mass = 0.0;
      for (int k = 0; k < eval_bins; ++k) {
        const double value =
          (k < static_cast<int>(distribution.size()))
            ? static_cast<double>(distribution[static_cast<size_t>(k)])
            : 0.0;
        const double reference =
          (k < static_cast<int>(reference_distribution.size()))
            ? static_cast<double>(reference_distribution[static_cast<size_t>(k)])
            : 0.0;
        if (reference <= kHardNearFieldTol && value > kHardNearFieldTol) {
          forbidden_mass += value;
        }
      }
      return forbidden_mass;
    };

  const auto forbidden_near_field_bin_count =
    [&](const std::vector<float>& distribution,
        const std::vector<float>& reference_distribution,
        int bin_count) {
      if (!hard_near_field_objective) {
        return 0;
      }
      const int split_bin = near_field_split_for_bins(bin_count);
      const int eval_bins = std::min(
        split_bin,
        std::max(
          static_cast<int>(distribution.size()),
          static_cast<int>(reference_distribution.size())));
      int forbidden_bins = 0;
      for (int k = 0; k < eval_bins; ++k) {
        const double value =
          (k < static_cast<int>(distribution.size()))
            ? static_cast<double>(distribution[static_cast<size_t>(k)])
            : 0.0;
        const double reference =
          (k < static_cast<int>(reference_distribution.size()))
            ? static_cast<double>(reference_distribution[static_cast<size_t>(k)])
            : 0.0;
        if (reference <= kHardNearFieldTol && value > kHardNearFieldTol) {
          ++forbidden_bins;
        }
      }
      return forbidden_bins;
    };

  const auto hard_near_field_penalty =
    [&](const std::vector<float>& distribution,
        const std::vector<float>& reference_distribution,
        int bin_count) {
      if (!hard_near_field_objective) {
        return 0.0;
      }
      const double forbidden_mass = forbidden_near_field_mass(
        distribution,
        reference_distribution,
        bin_count);
      const int forbidden_bins = forbidden_near_field_bin_count(
        distribution,
        reference_distribution,
        bin_count);
      if (forbidden_mass <= 0.0 && forbidden_bins <= 0) {
        return 0.0;
      }
      return kHardNearFieldPenaltyScale *
             (forbidden_mass + 0.25 * static_cast<double>(forbidden_bins));
    };

  const auto select_target_local_shape_prototypes =
    [&](const std::vector<std::vector<float>>& target_distributions) {
      std::vector<std::vector<float>> prototypes;
      if (target_distributions.empty()) {
        return prototypes;
      }
      const int target_count = static_cast<int>(target_distributions.size());
      const int max_prototypes = std::max(
        1,
        std::min(12, target_count));
      std::vector<int> selected_indices;
      selected_indices.reserve(static_cast<size_t>(max_prototypes));
      std::vector<double> best_distance(
        static_cast<size_t>(target_count),
        std::numeric_limits<double>::infinity());

      selected_indices.push_back(0);
      while (static_cast<int>(selected_indices.size()) < max_prototypes) {
        const int last_idx = selected_indices.back();
        int farthest_idx = -1;
        double farthest_distance = -1.0;
        for (int i = 0; i < target_count; ++i) {
          if (std::find(selected_indices.begin(), selected_indices.end(), i) != selected_indices.end()) {
            best_distance[static_cast<size_t>(i)] = 0.0;
            continue;
          }
          const double d = weighted_distribution_l2(
            target_distributions[static_cast<size_t>(i)],
            target_distributions[static_cast<size_t>(last_idx)]);
          best_distance[static_cast<size_t>(i)] =
            std::min(best_distance[static_cast<size_t>(i)], d);
          if (best_distance[static_cast<size_t>(i)] > farthest_distance) {
            farthest_distance = best_distance[static_cast<size_t>(i)];
            farthest_idx = i;
          }
        }
        if (farthest_idx < 0) {
          break;
        }
        selected_indices.push_back(farthest_idx);
      }

      prototypes.reserve(selected_indices.size());
      for (int idx : selected_indices) {
        prototypes.push_back(target_distributions[static_cast<size_t>(idx)]);
      }
      return prototypes;
    };

  const auto raw_connectivity_eval_bins_for_hist =
    [&](int bin_count) {
      return std::min(
        bin_count,
        std::max(3, std::min(10, near_field_split_for_bins(bin_count) + 2)));
    };

  const auto raw_connectivity_bin_weight =
    [&](int k, int eval_bins) {
      if (k <= 0) {
        return 8.0;
      }
      if (k == 1) {
        return 6.0;
      }
      if (k == 2) {
        return 4.0;
      }
      if (k < eval_bins / 2) {
        return 2.5;
      }
      return 1.0;
    };

  const auto build_raw_connectivity_signature =
    [&](const std::vector<int>& counts,
        int eval_bins) {
      std::vector<int> signature(static_cast<size_t>(eval_bins), 0);
      const int copy_bins = std::min(eval_bins, static_cast<int>(counts.size()));
      for (int k = 0; k < copy_bins; ++k) {
        signature[static_cast<size_t>(k)] = counts[static_cast<size_t>(k)];
      }
      return signature;
    };

  const auto select_target_raw_connectivity_prototypes =
    [&](const std::vector<std::vector<int>>& target_point_counts,
        int eval_bins,
        std::vector<float>* out_fractions) {
      std::vector<std::vector<int>> prototypes;
      if (out_fractions) {
        out_fractions->clear();
      }
      if (target_point_counts.empty() || eval_bins <= 0) {
        return prototypes;
      }

      std::map<std::vector<int>, int> signature_counts;
      for (const auto& counts : target_point_counts) {
        const std::vector<int> signature =
          build_raw_connectivity_signature(counts, eval_bins);
        ++signature_counts[signature];
      }
      if (signature_counts.empty()) {
        return prototypes;
      }

      std::vector<std::pair<int, std::vector<int>>> ranked_signatures;
      ranked_signatures.reserve(signature_counts.size());
      for (const auto& entry : signature_counts) {
        ranked_signatures.emplace_back(entry.second, entry.first);
      }
      std::sort(
        ranked_signatures.begin(),
        ranked_signatures.end(),
        [](const auto& a, const auto& b) {
          if (a.first != b.first) {
            return a.first > b.first;
          }
          return a.second < b.second;
        });

      const int max_prototypes = std::min(12, static_cast<int>(ranked_signatures.size()));
      int kept_total = 0;
      for (int i = 0; i < max_prototypes; ++i) {
        kept_total += ranked_signatures[static_cast<size_t>(i)].first;
      }
      if (kept_total <= 0) {
        return prototypes;
      }

      prototypes.reserve(static_cast<size_t>(max_prototypes));
      if (out_fractions) {
        out_fractions->reserve(static_cast<size_t>(max_prototypes));
      }
      for (int i = 0; i < max_prototypes; ++i) {
        prototypes.push_back(ranked_signatures[static_cast<size_t>(i)].second);
        if (out_fractions) {
          out_fractions->push_back(
            static_cast<float>(ranked_signatures[static_cast<size_t>(i)].first) /
            static_cast<float>(kept_total));
        }
      }
      return prototypes;
    };

  const auto compute_raw_connectivity_cost_from_signature =
    [&](const std::vector<int>& signature,
        const std::vector<int>& prototype) {
      const int eval_bins = std::max(
        static_cast<int>(signature.size()),
        static_cast<int>(prototype.size()));
      double cost = 0.0;
      double prefix_signature = 0.0;
      double prefix_prototype = 0.0;
      constexpr int kNearFieldDeficitBins = 3;
      constexpr double kNearFieldDeficitLinearPenalty = 6.0;
      constexpr double kNearFieldDeficitQuadraticPenalty = 5.0;
      for (int k = 0; k < eval_bins; ++k) {
        const int current =
          (k < static_cast<int>(signature.size()))
            ? signature[static_cast<size_t>(k)]
            : 0;
        const int target =
          (k < static_cast<int>(prototype.size()))
            ? prototype[static_cast<size_t>(k)]
            : 0;
        const double w = raw_connectivity_bin_weight(k, eval_bins);
        cost += w * std::abs(current - target);
        if (target == 0 && current > 0) {
          cost += 2.5 * w * static_cast<double>(current);
        }
        if (k < std::min(eval_bins, kNearFieldDeficitBins) && current < target) {
          const double deficit = static_cast<double>(target - current);
          cost +=
            kNearFieldDeficitLinearPenalty * w * deficit +
            kNearFieldDeficitQuadraticPenalty * w * deficit * deficit;
        }
        prefix_signature += static_cast<double>(current);
        prefix_prototype += static_cast<double>(target);
        cost += 0.35 * w * std::abs(prefix_signature - prefix_prototype);
      }
      const double near_field_deficit = std::max(0.0, prefix_prototype - prefix_signature);
      if (near_field_deficit > 0.0) {
        constexpr double kNearFieldPrefixPenalty = 7.5;
        cost += kNearFieldPrefixPenalty * near_field_deficit * near_field_deficit;
      }
      return cost;
    };

  const auto compute_raw_connectivity_cost_from_signature_legacy =
    [&](const std::vector<int>& signature,
        const std::vector<int>& prototype) {
      const int eval_bins = std::max(
        static_cast<int>(signature.size()),
        static_cast<int>(prototype.size()));
      double cost = 0.0;
      double prefix_signature = 0.0;
      double prefix_prototype = 0.0;
      for (int k = 0; k < eval_bins; ++k) {
        const int current =
          (k < static_cast<int>(signature.size()))
            ? signature[static_cast<size_t>(k)]
            : 0;
        const int target =
          (k < static_cast<int>(prototype.size()))
            ? prototype[static_cast<size_t>(k)]
            : 0;
        const double w = raw_connectivity_bin_weight(k, eval_bins);
        cost += w * std::abs(current - target);
        if (target == 0 && current > 0) {
          cost += 2.5 * w * static_cast<double>(current);
        }
        prefix_signature += static_cast<double>(current);
        prefix_prototype += static_cast<double>(target);
        cost += 0.35 * w * std::abs(prefix_signature - prefix_prototype);
      }
      return cost;
    };

  const auto compute_raw_connectivity_cost_from_counts =
    [&](const std::vector<int>& counts,
        const std::vector<std::vector<int>>& prototypes,
        int eval_bins,
        int* out_best_proto) {
      if (out_best_proto) {
        *out_best_proto = -1;
      }
      if (prototypes.empty() || eval_bins <= 0) {
        return 0.0;
      }
      const std::vector<int> signature =
        build_raw_connectivity_signature(counts, eval_bins);
      double best_cost = std::numeric_limits<double>::infinity();
      int best_proto = -1;
      for (int p = 0; p < static_cast<int>(prototypes.size()); ++p) {
        const double cost = compute_raw_connectivity_cost_from_signature(
          signature,
          prototypes[static_cast<size_t>(p)]);
        if (cost < best_cost) {
          best_cost = cost;
          best_proto = p;
        }
      }
      if (out_best_proto) {
        *out_best_proto = best_proto;
      }
      return std::isfinite(best_cost) ? best_cost : 0.0;
    };

  const auto compute_raw_connectivity_cost_from_counts_with_delta =
    [&](const std::vector<int>& counts,
        int remove_bin,
        int add_bin,
        const std::vector<std::vector<int>>& prototypes,
        int eval_bins,
        int* out_best_proto) {
      if (out_best_proto) {
        *out_best_proto = -1;
      }
      if (prototypes.empty() || eval_bins <= 0) {
        return 0.0;
      }
      std::vector<int> signature =
        build_raw_connectivity_signature(counts, eval_bins);
      if (remove_bin >= 0 && remove_bin < eval_bins) {
        --signature[static_cast<size_t>(remove_bin)];
      }
      if (add_bin >= 0 && add_bin < eval_bins) {
        ++signature[static_cast<size_t>(add_bin)];
      }
      double best_cost = std::numeric_limits<double>::infinity();
      int best_proto = -1;
      for (int p = 0; p < static_cast<int>(prototypes.size()); ++p) {
        const double cost = compute_raw_connectivity_cost_from_signature(
          signature,
          prototypes[static_cast<size_t>(p)]);
        if (cost < best_cost) {
          best_cost = cost;
          best_proto = p;
        }
      }
      if (out_best_proto) {
        *out_best_proto = best_proto;
      }
      return std::isfinite(best_cost) ? best_cost : 0.0;
    };

  const auto compute_proto_cost_from_distribution =
    [&](const std::vector<float>& distribution,
        const std::vector<std::vector<float>>& prototypes,
        int* out_best_proto) {
      if (out_best_proto) {
        *out_best_proto = -1;
      }
      if (prototypes.empty()) {
        return 0.0;
      }
      double best_cost = std::numeric_limits<double>::infinity();
      int best_proto = -1;
      for (int p = 0; p < static_cast<int>(prototypes.size()); ++p) {
        const double cost = weighted_distribution_l2(
                              distribution,
                              prototypes[static_cast<size_t>(p)]) +
          hard_near_field_penalty(
            distribution,
            prototypes[static_cast<size_t>(p)],
            std::max(
              static_cast<int>(distribution.size()),
              static_cast<int>(prototypes[static_cast<size_t>(p)].size())));
        if (cost < best_cost) {
          best_cost = cost;
          best_proto = p;
        }
      }
      if (out_best_proto) {
        *out_best_proto = best_proto;
      }
      return std::isfinite(best_cost) ? best_cost : kHardNearFieldPenaltyScale;
    };

  const auto compute_proto_cost_from_counts =
    [&](const std::vector<int>& counts,
        const std::vector<int>& support,
        const std::vector<std::vector<float>>& prototypes,
        std::vector<float>* scratch_distribution,
        int* out_best_proto) {
      if (out_best_proto) {
        *out_best_proto = -1;
      }
      if (prototypes.empty()) {
        return 0.0;
      }
      std::vector<float> local_distribution;
      std::vector<float>& distribution =
        scratch_distribution ? *scratch_distribution : local_distribution;
      const int eval_bins = std::min(
        static_cast<int>(counts.size()),
        static_cast<int>(support.size()));
      distribution.assign(static_cast<size_t>(eval_bins), 0.0f);
      for (int k = 0; k < eval_bins; ++k) {
        const int denom = support[static_cast<size_t>(k)];
        if (denom <= 0) {
          continue;
        }
        distribution[static_cast<size_t>(k)] =
          static_cast<float>(counts[static_cast<size_t>(k)]) /
          static_cast<float>(denom);
      }
      return compute_proto_cost_from_distribution(
        distribution,
        prototypes,
        out_best_proto);
    };

  const auto compute_proto_cost_from_counts_with_delta =
    [&](const std::vector<int>& counts,
        const std::vector<int>& support,
        int remove_bin,
        int add_bin,
        const std::vector<std::vector<float>>& prototypes,
        std::vector<float>* scratch_distribution,
        int* out_best_proto) {
      if (out_best_proto) {
        *out_best_proto = -1;
      }
      if (prototypes.empty()) {
        return 0.0;
      }
      std::vector<float> local_distribution;
      std::vector<float>& distribution =
        scratch_distribution ? *scratch_distribution : local_distribution;
      const int eval_bins = std::min(
        static_cast<int>(counts.size()),
        static_cast<int>(support.size()));
      distribution.assign(static_cast<size_t>(eval_bins), 0.0f);
      for (int k = 0; k < eval_bins; ++k) {
        const int denom = support[static_cast<size_t>(k)];
        if (denom <= 0) {
          continue;
        }
        int adjusted_count = counts[static_cast<size_t>(k)];
        if (k == remove_bin) {
          --adjusted_count;
        }
        if (k == add_bin) {
          ++adjusted_count;
        }
        distribution[static_cast<size_t>(k)] =
          static_cast<float>(adjusted_count) /
          static_cast<float>(denom);
      }
      return compute_proto_cost_from_distribution(
        distribution,
        prototypes,
        out_best_proto);
    };

  const auto positional_target_row_for_support_row =
    [&](int support_row) -> int {
      if (!live_position_targets_enabled ||
          support_row < 0 ||
          support_row >= static_cast<int>(live_target_individual_distributions.size())) {
        return -1;
      }
      return support_row;
    };

  const auto positional_target_distribution_for_support_row =
    [&](int support_row) -> const std::vector<float>* {
      const int target_row = positional_target_row_for_support_row(support_row);
      if (target_row >= 0) {
        return &live_target_individual_distributions[static_cast<size_t>(target_row)];
      }
      return nullptr;
    };

    const auto transition_target_distribution_for_support_row =
      [&](int support_row) -> const std::vector<float>* {
        if (!transition_optimizer_targets_active() ||
            support_row < 0 ||
            support_row >=
              static_cast<int>(live_transition_optimizer_targets.distribution_for_support_row.size())) {
          return nullptr;
      }
      const auto& target_distribution =
        live_transition_optimizer_targets.distribution_for_support_row[static_cast<size_t>(support_row)];
      return target_distribution.empty() ? nullptr : &target_distribution;
    };

  const auto positional_target_raw_signature_for_support_row =
    [&](int support_row) -> const std::vector<int>* {
      const int target_row = positional_target_row_for_support_row(support_row);
      if (target_row >= 0 &&
          target_row < static_cast<int>(live_target_raw_connectivity_signatures.size())) {
        return &live_target_raw_connectivity_signatures[static_cast<size_t>(target_row)];
      }
      return nullptr;
    };

    const auto transition_target_raw_signature_for_support_row =
      [&](int support_row) -> const std::vector<int>* {
        if (!transition_optimizer_targets_active() ||
            support_row < 0 ||
            support_row >= static_cast<int>(
              live_transition_optimizer_targets.raw_point_hist_counts_for_support_row.size())) {
        return nullptr;
      }
      const auto& target_signature =
        live_transition_optimizer_targets.raw_point_hist_counts_for_support_row[
          static_cast<size_t>(support_row)];
      return target_signature.empty() ? nullptr : &target_signature;
    };

  const auto active_target_distribution_for_support_row =
    [&](int support_row) -> const std::vector<float>* {
      if (current_optimizer_mode() != VoronoiOptimizerMode::StructuredTargets) {
        return nullptr;
      }
      if (live_position_targets_enabled) {
        return positional_target_distribution_for_support_row(support_row);
      }
      return transition_target_distribution_for_support_row(support_row);
    };

  const auto active_target_raw_signature_for_support_row =
    [&](int support_row) -> const std::vector<int>* {
      if (current_optimizer_mode() != VoronoiOptimizerMode::StructuredTargets) {
        return nullptr;
      }
      if (live_position_targets_enabled) {
        return positional_target_raw_signature_for_support_row(support_row);
      }
      return transition_target_raw_signature_for_support_row(support_row);
    };

  const auto individual_target_match_cost =
    [&](const std::vector<float>& out_dist,
        const std::vector<float>& tgt_dist) {
      const double weighted_error = weighted_distribution_l2(out_dist, tgt_dist);
      const int m = std::min(
        static_cast<int>(tgt_dist.size()),
        live_hist_bin_count);
      const int strong_prefix_bins = std::min(
        m,
        std::max(2, std::min(6, near_field_split_for_bins(live_hist_bin_count))));
      double prefix_error = 0.0;
      double prefix_mass_error = 0.0;
      for (int k = 0; k < strong_prefix_bins; ++k) {
        const double out_v =
          (k < static_cast<int>(out_dist.size()))
            ? static_cast<double>(out_dist[static_cast<size_t>(k)])
            : 0.0;
        const double tgt_v =
          (k < static_cast<int>(tgt_dist.size()))
            ? static_cast<double>(tgt_dist[static_cast<size_t>(k)])
            : 0.0;
        const double d = out_v - tgt_v;
        prefix_error += d * d;
        prefix_mass_error += std::abs(d);
      }
      return weighted_error +
             12.0 * prefix_error +
             6.0 * prefix_mass_error;
    };

  const auto targeted_distribution_cost_from_counts =
    [&](const std::vector<int>& counts,
        const std::vector<int>& support,
        int support_row,
        std::vector<float>* scratch_distribution) {
      const std::vector<float>* target_distribution =
        active_target_distribution_for_support_row(support_row);
      if (target_distribution == nullptr) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      std::vector<float> local_distribution;
      std::vector<float>& distribution =
        scratch_distribution ? *scratch_distribution : local_distribution;
      const int eval_bins = std::min(
        static_cast<int>(counts.size()),
        static_cast<int>(support.size()));
      distribution.assign(static_cast<size_t>(eval_bins), 0.0f);
      for (int k = 0; k < eval_bins; ++k) {
        const int denom = support[static_cast<size_t>(k)];
        if (denom <= 0) {
          continue;
        }
        distribution[static_cast<size_t>(k)] =
          static_cast<float>(counts[static_cast<size_t>(k)]) /
          static_cast<float>(denom);
      }
      return individual_target_match_cost(
        distribution,
        *target_distribution);
    };

  const auto compute_target_proto_fractions =
    [&](const std::vector<std::vector<float>>& target_distributions,
        const std::vector<std::vector<float>>& prototypes) {
      std::vector<float> fractions(prototypes.size(), 0.0f);
      if (target_distributions.empty() || prototypes.empty()) {
        return fractions;
      }
      for (const auto& dist : target_distributions) {
        int best_proto = -1;
        (void)compute_proto_cost_from_distribution(dist, prototypes, &best_proto);
        if (best_proto >= 0 && best_proto < static_cast<int>(fractions.size())) {
          fractions[static_cast<size_t>(best_proto)] += 1.0f;
        }
      }
      const float inv_count = 1.0f / static_cast<float>(target_distributions.size());
      for (float& v : fractions) {
        v *= inv_count;
      }
      return fractions;
    };

  const auto compute_raw_connectivity_objective_from_point_hist =
    [&](const std::vector<std::vector<int>>& point_hist_counts,
        const std::vector<int>* point_support_rows = nullptr) {
      const int eval_bins = raw_connectivity_eval_bins_for_hist(live_hist_bin_count);
      if (eval_bins <= 0 || point_hist_counts.empty()) {
        return 0.0;
      }
      if (live_position_targets_enabled &&
          point_support_rows != nullptr) {
        double cost_sum = 0.0;
        int matched_points = 0;
        const int point_count = std::min(
          static_cast<int>(point_hist_counts.size()),
          static_cast<int>(point_support_rows->size()));
        for (int i = 0; i < point_count; ++i) {
          const int support_row = (*point_support_rows)[static_cast<size_t>(i)];
          const std::vector<int>* target_signature =
            active_target_raw_signature_for_support_row(support_row);
          if (target_signature == nullptr) {
            continue;
          }
          const std::vector<int> signature =
            build_raw_connectivity_signature(
              point_hist_counts[static_cast<size_t>(i)],
              eval_bins);
          cost_sum += compute_raw_connectivity_cost_from_signature_legacy(
            signature,
            build_raw_connectivity_signature(*target_signature, eval_bins));
          ++matched_points;
        }
        if (matched_points > 0) {
          return cost_sum / static_cast<double>(matched_points);
        }
      }
      double transition_directional_raw_cost = 0.0;
      int transition_directional_raw_matches = 0;
      if (transition_optimizer_targets_active() &&
          point_support_rows != nullptr) {
        const int point_count = std::min(
          static_cast<int>(point_hist_counts.size()),
          static_cast<int>(point_support_rows->size()));
        for (int i = 0; i < point_count; ++i) {
          const int support_row = (*point_support_rows)[static_cast<size_t>(i)];
          const std::vector<int>* target_signature =
            active_target_raw_signature_for_support_row(support_row);
          if (target_signature == nullptr) {
            continue;
          }
          const std::vector<int> signature =
            build_raw_connectivity_signature(
              point_hist_counts[static_cast<size_t>(i)],
              eval_bins);
          transition_directional_raw_cost +=
            compute_raw_connectivity_cost_from_signature_legacy(
              signature,
              build_raw_connectivity_signature(*target_signature, eval_bins));
          ++transition_directional_raw_matches;
        }
        if (transition_directional_raw_matches > 0) {
          transition_directional_raw_cost /=
            static_cast<double>(transition_directional_raw_matches);
        }
      }
      std::vector<std::vector<int>> local_prototypes;
      std::vector<float> local_fractions;
      local_prototypes = select_target_raw_connectivity_prototypes(
        state.voronoi_pcf_raw_point_hist_counts,
        eval_bins,
        &local_fractions);
      if (local_prototypes.empty() || local_fractions.empty()) {
        return
          0.28 * static_cast<double>(transition_direction_gradient_weight) *
          transition_directional_raw_cost;
      }

      double cost_sum = 0.0;
      int valid_points = 0;
      std::vector<int> proto_counts(local_prototypes.size(), 0);
      for (const auto& counts : point_hist_counts) {
        int best_proto = -1;
        const double cost = compute_raw_connectivity_cost_from_counts(
          counts,
          local_prototypes,
          eval_bins,
          &best_proto);
        cost_sum += cost;
        ++valid_points;
        if (best_proto >= 0 &&
            best_proto < static_cast<int>(proto_counts.size())) {
          ++proto_counts[static_cast<size_t>(best_proto)];
        }
      }
      if (valid_points <= 0) {
        return 0.0;
      }

      double occupancy_error = 0.0;
      const int proto_count = std::min(
        static_cast<int>(proto_counts.size()),
        static_cast<int>(local_fractions.size()));
      for (int p = 0; p < proto_count; ++p) {
        const double curr_frac =
          static_cast<double>(proto_counts[static_cast<size_t>(p)]) /
          static_cast<double>(valid_points);
        const double target_frac =
          static_cast<double>(local_fractions[static_cast<size_t>(p)]);
        const double d = curr_frac - target_frac;
        occupancy_error += d * d;
      }
      double raw_cost =
        (cost_sum / static_cast<double>(valid_points)) + 4.0 * occupancy_error;
      if (transition_directional_raw_matches > 0 &&
          transition_direction_gradient_weight > 0.0f) {
        constexpr double kTransitionSoftRawDirectionScale = 0.18;
        raw_cost +=
          kTransitionSoftRawDirectionScale *
          static_cast<double>(transition_direction_gradient_weight) *
          transition_directional_raw_cost;
      }
      return raw_cost;
    };

  struct ExactEvalScratch {
    std::vector<std::vector<int>> candidate_point_hist;
    std::vector<std::vector<int>> candidate_point_support;
    std::vector<int> scratch_support_row;
    std::vector<int> candidate_support_rows;
    std::vector<std::vector<float>> current_individual_distributions;
    std::vector<char> point_has_support;
    std::vector<int> valid_output_indices;
    std::vector<std::vector<double>> match_costs;
    std::vector<double> best_out_cost;
    std::vector<double> best_tgt_cost;
    std::vector<double> unmatched_output_costs;
    std::vector<double> unmatched_target_costs;
    std::vector<double> avg_distribution;
  };

  const auto compute_raw_connectivity_objective_from_uv_points =
    [&](const std::vector<Eigen::Vector2d>& uv_points) {
      const int eval_bins = raw_connectivity_eval_bins_for_hist(live_hist_bin_count);
      const int n_points = static_cast<int>(uv_points.size());
      if (eval_bins <= 0 || n_points <= 0) {
        return 0.0;
      }
      std::vector<std::vector<int>> point_hist_counts(
        static_cast<size_t>(n_points),
        std::vector<int>(static_cast<size_t>(eval_bins), 0));
      std::vector<int> point_support_rows(static_cast<size_t>(n_points), -1);
      const bool needs_structured_support_rows =
        current_optimizer_mode() == VoronoiOptimizerMode::StructuredTargets;
      if (needs_structured_support_rows &&
          output_support_tri_indices.size() == output_support_uv.size()) {
        std::unordered_map<int, int> tri_to_support_row;
        tri_to_support_row.reserve(output_support_tri_indices.size());
        for (size_t si = 0; si < output_support_tri_indices.size(); ++si) {
          tri_to_support_row.emplace(
            output_support_tri_indices[si],
            static_cast<int>(si));
        }
        for (int i = 0; i < n_points; ++i) {
          int tri_idx = -1;
          Eigen::Vector3i tri_vertices(-1, -1, -1);
          if (!delaunay_helper->find_containing_triangle(
                uv_points[static_cast<size_t>(i)],
                tri_idx,
                tri_vertices)) {
            continue;
          }
          const auto it = tri_to_support_row.find(tri_idx);
          if (it != tri_to_support_row.end()) {
            point_support_rows[static_cast<size_t>(i)] = it->second;
          }
        }
      }
      for (int i = 0; i < n_points; ++i) {
        for (int j = i + 1; j < n_points; ++j) {
          const int k = delaunay_helper->count_triangles_crossed(
            uv_points[static_cast<size_t>(i)],
            uv_points[static_cast<size_t>(j)]);
          if (k >= 0) {
            const int bin = std::min(k, eval_bins - 1);
            ++point_hist_counts[static_cast<size_t>(i)][static_cast<size_t>(bin)];
            ++point_hist_counts[static_cast<size_t>(j)][static_cast<size_t>(bin)];
          }
        }
      }
      return compute_raw_connectivity_objective_from_point_hist(
        point_hist_counts,
        needs_structured_support_rows
          ? &point_support_rows
          : nullptr);
    };

  const auto get_exact_eval_scratch = [&]() -> ExactEvalScratch& {
    static thread_local ExactEvalScratch scratch;
    return scratch;
  };

  const auto compute_shape_error_from_targets =
    [&](ExactEvalScratch& scratch,
        int n_points,
        int valid_points,
        const std::vector<int>* point_support_rows) -> double {
      if (live_position_targets_enabled &&
          point_support_rows != nullptr) {
        double structured_cost_sum = 0.0;
        int matched_points = 0;
        for (int oi = 0; oi < static_cast<int>(scratch.valid_output_indices.size()); ++oi) {
          const int point_idx =
            scratch.valid_output_indices[static_cast<size_t>(oi)];
          if (point_idx < 0 ||
              point_idx >= static_cast<int>(point_support_rows->size())) {
            continue;
          }
          const int support_row =
            (*point_support_rows)[static_cast<size_t>(point_idx)];
          const std::vector<float>* target_distribution =
            active_target_distribution_for_support_row(support_row);
          if (target_distribution == nullptr) {
            continue;
          }
          structured_cost_sum += individual_target_match_cost(
            scratch.current_individual_distributions[static_cast<size_t>(point_idx)],
            *target_distribution);
          ++matched_points;
        }
        if (matched_points > 0) {
          return structured_cost_sum / static_cast<double>(matched_points);
        }
        return std::numeric_limits<double>::infinity();
      }

      if (!live_target_individual_distributions.empty()) {
        const int out_count = static_cast<int>(scratch.valid_output_indices.size());
        const int tgt_count = static_cast<int>(live_target_individual_distributions.size());
        if (out_count <= 0 || tgt_count <= 0) {
          return std::numeric_limits<double>::infinity();
        }

        scratch.match_costs.resize(static_cast<size_t>(out_count));
        for (int oi = 0; oi < out_count; ++oi) {
          scratch.match_costs[static_cast<size_t>(oi)].assign(
            static_cast<size_t>(tgt_count),
            std::numeric_limits<double>::infinity());
        }
        scratch.best_out_cost.assign(
          static_cast<size_t>(out_count),
          std::numeric_limits<double>::infinity());
        scratch.best_tgt_cost.assign(
          static_cast<size_t>(tgt_count),
          std::numeric_limits<double>::infinity());

        for (int oi = 0; oi < out_count; ++oi) {
          const auto& out_dist =
            scratch.current_individual_distributions[
              static_cast<size_t>(scratch.valid_output_indices[static_cast<size_t>(oi)])];
          for (int ti = 0; ti < tgt_count; ++ti) {
            const double cost = individual_target_match_cost(
              out_dist,
              live_target_individual_distributions[static_cast<size_t>(ti)]);
            scratch.match_costs[static_cast<size_t>(oi)][static_cast<size_t>(ti)] = cost;
            scratch.best_out_cost[static_cast<size_t>(oi)] =
              std::min(scratch.best_out_cost[static_cast<size_t>(oi)], cost);
            scratch.best_tgt_cost[static_cast<size_t>(ti)] =
              std::min(scratch.best_tgt_cost[static_cast<size_t>(ti)], cost);
          }
        }

        constexpr double kUnmatchedAssignmentPenalty = 1.0;
        scratch.unmatched_output_costs.assign(
          static_cast<size_t>(out_count),
          kUnmatchedAssignmentPenalty);
        scratch.unmatched_target_costs.assign(
          static_cast<size_t>(tgt_count),
          kUnmatchedAssignmentPenalty);
        for (int oi = 0; oi < out_count; ++oi) {
          scratch.unmatched_output_costs[static_cast<size_t>(oi)] =
            scratch.best_out_cost[static_cast<size_t>(oi)] + kUnmatchedAssignmentPenalty;
        }
        for (int ti = 0; ti < tgt_count; ++ti) {
          scratch.unmatched_target_costs[static_cast<size_t>(ti)] =
            scratch.best_tgt_cost[static_cast<size_t>(ti)] + kUnmatchedAssignmentPenalty;
        }

        const double assignment_cost = min_cost_assignment_with_unmatched(
          scratch.match_costs,
          scratch.unmatched_output_costs,
          scratch.unmatched_target_costs);
        const int normalizer = std::max(out_count, tgt_count);
        double shape_cost =
          assignment_cost / static_cast<double>(std::max(1, normalizer));

        if (transition_optimizer_targets_active() &&
            point_support_rows != nullptr &&
            transition_direction_gradient_weight > 0.0f) {
          double directional_cost_sum = 0.0;
          int directional_matches = 0;
          for (int oi = 0; oi < out_count; ++oi) {
            const int point_idx =
              scratch.valid_output_indices[static_cast<size_t>(oi)];
            if (point_idx < 0 ||
                point_idx >= static_cast<int>(point_support_rows->size())) {
              continue;
            }
            const int support_row =
              (*point_support_rows)[static_cast<size_t>(point_idx)];
            const std::vector<float>* row_target =
              active_target_distribution_for_support_row(support_row);
            if (row_target == nullptr) {
              continue;
            }
            directional_cost_sum += individual_target_match_cost(
              scratch.current_individual_distributions[static_cast<size_t>(point_idx)],
              *row_target);
            ++directional_matches;
          }
          if (directional_matches > 0) {
            constexpr double kTransitionSoftDirectionExactScale = 0.28;
            shape_cost +=
              kTransitionSoftDirectionExactScale *
              static_cast<double>(transition_direction_gradient_weight) *
              (directional_cost_sum / static_cast<double>(directional_matches));
          }
        }

        return shape_cost;
      }

      scratch.avg_distribution.assign(static_cast<size_t>(live_hist_bin_count), 0.0);
      for (int i = 0; i < n_points; ++i) {
        if (!scratch.point_has_support[static_cast<size_t>(i)]) {
          continue;
        }
        for (int k = 0; k < live_hist_bin_count; ++k) {
          scratch.avg_distribution[static_cast<size_t>(k)] +=
            static_cast<double>(
              scratch.current_individual_distributions[static_cast<size_t>(i)][static_cast<size_t>(k)]);
        }
      }
      const double inv_valid_local = 1.0 / static_cast<double>(valid_points);
      for (double& v : scratch.avg_distribution) {
        v *= inv_valid_local;
      }
      double shape_error = 0.0;
      for (int k = 0; k < live_hist_bin_count; ++k) {
        const double bin_weight = use_bin_weighting
          ? (k < bin_weight_transition ? static_cast<double>(early_bin_weight)
                                       : static_cast<double>(late_bin_weight))
          : 1.0;
        const double p = scratch.avg_distribution[static_cast<size_t>(k)];
        const double t = (k < static_cast<int>(live_target_hist.size()))
          ? static_cast<double>(live_target_hist[static_cast<size_t>(k)])
          : 0.0;
        const double d = p - t;
        shape_error += d * d * bin_weight;
      }
      return shape_error;
    };

  const auto compute_exact_error_from_point_data =
    [&](const std::vector<std::vector<int>>& point_hist,
        const std::vector<std::vector<int>>& point_support,
        const std::vector<int>* point_support_rows = nullptr,
        double* out_raw_connectivity_error = nullptr) -> double {
    const int n_points = std::min(
      static_cast<int>(point_hist.size()),
      static_cast<int>(point_support.size()));
    if (n_points < 2 || live_hist_bin_count <= 0) {
      if (out_raw_connectivity_error) {
        *out_raw_connectivity_error = 0.0;
      }
      return std::numeric_limits<double>::infinity();
    }

    if (out_raw_connectivity_error) {
      *out_raw_connectivity_error =
        compute_raw_connectivity_objective_from_point_hist(
          point_hist,
          point_support_rows);
    }

    ExactEvalScratch& scratch = get_exact_eval_scratch();
    scratch.current_individual_distributions.resize(static_cast<size_t>(n_points));
    for (int i = 0; i < n_points; ++i) {
      scratch.current_individual_distributions[static_cast<size_t>(i)].assign(
        static_cast<size_t>(live_hist_bin_count),
        0.0f);
    }
    scratch.point_has_support.assign(static_cast<size_t>(n_points), 0);
    scratch.valid_output_indices.clear();
    scratch.valid_output_indices.reserve(static_cast<size_t>(n_points));
    int valid_points = 0;
    for (int i = 0; i < n_points; ++i) {
      bool has_valid_support = false;
      for (int k = 0; k < live_hist_bin_count; ++k) {
        if (point_support[static_cast<size_t>(i)][static_cast<size_t>(k)] > 0) {
          has_valid_support = true;
          break;
        }
      }
      if (!has_valid_support) {
        continue;
      }
      scratch.point_has_support[static_cast<size_t>(i)] = 1;
      scratch.valid_output_indices.push_back(i);
      ++valid_points;
      for (int k = 0; k < live_hist_bin_count; ++k) {
        const int denom = point_support[static_cast<size_t>(i)][static_cast<size_t>(k)];
        if (denom <= 0) {
          continue;
        }
        scratch.current_individual_distributions[static_cast<size_t>(i)][static_cast<size_t>(k)] =
          static_cast<float>(
            static_cast<double>(point_hist[static_cast<size_t>(i)][static_cast<size_t>(k)]) /
            static_cast<double>(denom));
      }
    }
    if (valid_points <= 0) {
      return std::numeric_limits<double>::infinity();
    }
    return compute_shape_error_from_targets(
      scratch,
      n_points,
      valid_points,
      point_support_rows);
  };

  const auto compute_live_pcf_error =
    [&](const std::vector<Eigen::Vector2d>& uv_points,
        double* out_raw_connectivity_error = nullptr) -> double {
    const int n_points = static_cast<int>(uv_points.size());
    if (n_points < 2 || live_hist_bin_count <= 0) {
      if (out_raw_connectivity_error) {
        *out_raw_connectivity_error = 0.0;
      }
      return std::numeric_limits<double>::infinity();
    }
    std::vector<std::vector<int>> point_hist(
      static_cast<size_t>(n_points),
      std::vector<int>(static_cast<size_t>(live_hist_bin_count), 0));
    std::vector<std::vector<int>> point_support(
      static_cast<size_t>(n_points),
      std::vector<int>(static_cast<size_t>(live_hist_bin_count), 0));
    std::vector<int> point_support_rows(static_cast<size_t>(n_points), -1);
    const std::vector<Eigen::Vector2d>& support_points =
      output_support_uv.empty() ? uv_points : output_support_uv;
    std::unordered_map<int, int> tri_to_support_row;
    const bool support_row_map_available =
      output_support_tri_indices.size() == support_points.size();
    if (support_row_map_available) {
      tri_to_support_row.reserve(output_support_tri_indices.size());
      for (size_t si = 0; si < output_support_tri_indices.size(); ++si) {
        tri_to_support_row.emplace(
          output_support_tri_indices[si],
          static_cast<int>(si));
      }
    }
    const bool can_use_cached_support =
      state.output_support_denominator_cache_valid &&
      state.output_support_tri_indices_cache.size() == support_points.size() &&
      state.output_support_k_denominator_cache.size() == support_points.size();

    for (int i = 0; i < n_points; ++i) {
      for (int j = i + 1; j < n_points; ++j) {
        int k = delaunay_helper->count_triangles_crossed(
          uv_points[static_cast<size_t>(i)],
          uv_points[static_cast<size_t>(j)]);
        if (k < 0) {
          continue;
        }
        if (k < live_hist_bin_count) {
          ++point_hist[static_cast<size_t>(i)][static_cast<size_t>(k)];
          ++point_hist[static_cast<size_t>(j)][static_cast<size_t>(k)];
        }
      }
    }

    const int live_support_point_count =
      static_cast<int>(support_points.size());
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (n_points >= 64 && live_support_point_count >= 512)
#endif
    for (int i = 0; i < n_points; ++i) {
      bool loaded_from_cache = false;
      int tri_idx = -1;
      Eigen::Vector3i tri_vertices(-1, -1, -1);
      if (support_row_map_available &&
          delaunay_helper->find_containing_triangle(
            uv_points[static_cast<size_t>(i)], tri_idx, tri_vertices)) {
        const auto it = tri_to_support_row.find(tri_idx);
        if (it != tri_to_support_row.end()) {
          point_support_rows[static_cast<size_t>(i)] = it->second;
          if (can_use_cached_support) {
            const std::vector<int>& cached_row =
              state.output_support_k_denominator_cache[static_cast<size_t>(it->second)];
            const int copy_bins = std::min(
              live_hist_bin_count,
              static_cast<int>(cached_row.size()));
            for (int k = 0; k < copy_bins; ++k) {
              point_support[static_cast<size_t>(i)][static_cast<size_t>(k)] =
                cached_row[static_cast<size_t>(k)];
            }
            loaded_from_cache = true;
          }
        }
      }

      if (!loaded_from_cache) {
        for (int si = 0; si < live_support_point_count; ++si) {
          const Eigen::Vector2d& support_uv =
            support_points[static_cast<size_t>(si)];
          const int k = delaunay_helper->count_triangles_crossed(
            uv_points[static_cast<size_t>(i)],
            support_uv);
          if (k >= 0 && k < live_hist_bin_count) {
            ++point_support[static_cast<size_t>(i)][static_cast<size_t>(k)];
          }
        }
      }
    }

    return compute_exact_error_from_point_data(
      point_hist,
      point_support,
      structured_support_rows_required() ? &point_support_rows : nullptr,
      out_raw_connectivity_error);
  };

  const auto compute_live_bin_residual_stats = [&](
    const std::vector<Eigen::Vector2d>& uv_points,
    int focus_count,
    int* out_worst_bin_idx,
    std::vector<int>* out_focus_bins,
    std::vector<float>* out_avg_plot = nullptr) -> double {
    if (out_worst_bin_idx) {
      *out_worst_bin_idx = -1;
    }
    if (out_focus_bins) {
      out_focus_bins->clear();
    }
    if (out_avg_plot) {
      out_avg_plot->clear();
    }
    if (live_hist_bin_count <= 0 || uv_points.size() < 2) {
      return std::numeric_limits<double>::infinity();
    }

    std::vector<int> hist_counts;
    int pair_count = 0;
    std::vector<float> avg_plot;
    const bool ok = build_pair_hist_and_average_individual_plot(
      uv_points,
      delaunay_helper,
      live_hist_bin_count,
      hist_counts,
      pair_count,
      avg_plot,
      &output_support_uv,
      nullptr,
      state.output_support_denominator_cache_valid
        ? &state.output_support_tri_indices_cache
        : nullptr,
      state.output_support_denominator_cache_valid
        ? &state.output_support_k_denominator_cache
        : nullptr);
    if (!ok) {
      return std::numeric_limits<double>::infinity();
    }

    if (out_avg_plot) {
      *out_avg_plot = avg_plot;
    }

    const int n = std::max(live_hist_bin_count, static_cast<int>(live_target_hist.size()));
    if (n <= 0) {
      return std::numeric_limits<double>::infinity();
    }

    std::vector<std::pair<double, int>> residuals;
    residuals.reserve(static_cast<size_t>(n));
    double worst_residual = 0.0;
    int worst_bin_idx = -1;
    for (int k = 0; k < n; ++k) {
      const double p = (k < static_cast<int>(avg_plot.size()))
        ? static_cast<double>(avg_plot[static_cast<size_t>(k)])
        : 0.0;
      const double t = (k < static_cast<int>(live_target_hist.size()))
        ? static_cast<double>(live_target_hist[static_cast<size_t>(k)])
        : 0.0;
      const double residual = std::abs(p - t);
      residuals.emplace_back(residual, k);
      if (residual > worst_residual) {
        worst_residual = residual;
        worst_bin_idx = k;
      }
    }

    if (out_worst_bin_idx) {
      *out_worst_bin_idx = worst_bin_idx;
    }
    if (out_focus_bins && !residuals.empty() && focus_count > 0) {
      std::sort(
        residuals.begin(),
        residuals.end(),
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
          if (a.first == b.first) {
            return a.second < b.second;
          }
          return a.first > b.first;
        });
      const int top_n = std::min(focus_count, static_cast<int>(residuals.size()));
      for (int i = 0; i < top_n; ++i) {
        if (residuals[static_cast<size_t>(i)].first <= 1e-12) {
          break;
        }
        out_focus_bins->push_back(residuals[static_cast<size_t>(i)].second);
      }
    }
    return worst_residual;
  };

  const auto sync_live_points_to_state = [&](const std::vector<Eigen::Vector2d>& uv_points) {
    state.output_pattern_sample_indices.clear();
    state.output_pattern_points_uv.clear();
    state.output_pattern_points_3d.clear();
    state.output_pattern_points_uv.reserve(uv_points.size());
    state.output_pattern_points_3d.reserve(uv_points.size());
    for (const Eigen::Vector2d& uv : uv_points) {
      state.output_pattern_points_uv.push_back(uv);
      Eigen::Vector3d lifted_3d = Eigen::Vector3d::Zero();
      if (!lift_uv_to_output_3d(uv, delaunay_helper, points_3d, points_uv, lifted_3d)) {
        lifted_3d = nearest_sample_3d(uv, points_3d, points_uv);
      }
      state.output_pattern_points_3d.push_back(lifted_3d);
    }
    state.output_pattern_dirty = true;
  };

  const auto update_live_hist_in_state = [&](const std::vector<Eigen::Vector2d>& uv_points) {
    std::vector<int> hist_counts;
    int pair_count = 0;
    std::vector<float> avg_plot;
    const bool ok = build_pair_hist_and_average_individual_plot(
      uv_points,
      delaunay_helper,
      live_hist_bin_count,
      hist_counts,
      pair_count,
      avg_plot,
      &output_support_uv,
      nullptr,
      state.output_support_denominator_cache_valid
        ? &state.output_support_tri_indices_cache
        : nullptr,
      state.output_support_denominator_cache_valid
        ? &state.output_support_k_denominator_cache
        : nullptr);
    if (!ok) {
      return;
    }

    state.output_voronoi_pcf_hist_counts = std::move(hist_counts);
    state.output_voronoi_pcf_hist_plot = std::move(avg_plot);
    state.output_voronoi_pcf_pair_count = pair_count;
    state.output_voronoi_pcf_max_k = 0;
    for (int k = 0; k < static_cast<int>(state.output_voronoi_pcf_hist_counts.size()); ++k) {
      if (state.output_voronoi_pcf_hist_counts[static_cast<size_t>(k)] > 0) {
        state.output_voronoi_pcf_max_k = k;
      }
    }
    state.output_voronoi_pcf_ready = true;
  };
  
  // Point generation controls
  ImGui::TextUnformatted("Point Generation:");
  if (estimated_target_count > 0) {
    ImGui::Text("Estimated target: %d points", estimated_target_count);
  } else {
    ImGui::TextDisabled(
      is_transition_region
        ? "Build the transition target to enable generation"
        : "Compute input PCF to enable generation");
  }

  static int gen_point_count = 100;
  static bool gen_point_count_initialized = false;
  static int gen_mode = 1; // 0 = random, 1 = strategic
  static double last_generated_exact_error = std::numeric_limits<double>::infinity();
  bool reset_after_generate_points = false;
  int auto_generation_start_count = estimated_target_count;
  double auto_generation_overseed_factor = 1.0;
  if (estimated_target_count > 0 &&
      output_support_count > 0 &&
      state.voronoi_pcf_ready) {
    const int bin_count = std::max(1, state.voronoi_pcf_bin_count);
    std::vector<float> zero_distribution(
      static_cast<size_t>(bin_count),
      0.0f);
    const auto infer_target_radius =
      [&](const std::vector<float>& target_distribution) {
        const int eval_bins = std::min(
          bin_count,
          std::max(
            static_cast<int>(zero_distribution.size()),
            static_cast<int>(target_distribution.size())));
        return infer_local_proposal_radius_from_distributions(
          zero_distribution,
          target_distribution,
          eval_bins,
          bin_count);
      };

    int overall_target_radius = 1;
    if (!state.voronoi_pcf_hist_plot.empty()) {
      overall_target_radius = infer_target_radius(state.voronoi_pcf_hist_plot);
    }

    int evaluated_individual_targets = 0;
    int medium_order_target_rows = 0;
    int high_order_target_rows = 0;
    for (const auto& target_distribution : state.voronoi_pcf_individual_plots) {
      if (target_distribution.empty()) {
        continue;
      }
      ++evaluated_individual_targets;
      const int target_radius = infer_target_radius(target_distribution);
      if (target_radius >= 2) {
        ++medium_order_target_rows;
      }
      if (target_radius >= 3) {
        ++high_order_target_rows;
      }
    }

    const double medium_order_fraction =
      (evaluated_individual_targets > 0)
        ? static_cast<double>(medium_order_target_rows) /
            static_cast<double>(evaluated_individual_targets)
        : 0.0;
    const double high_order_fraction =
      (evaluated_individual_targets > 0)
        ? static_cast<double>(high_order_target_rows) /
            static_cast<double>(evaluated_individual_targets)
        : 0.0;

    if (overall_target_radius >= 3 || high_order_fraction >= 0.12) {
      auto_generation_overseed_factor =
        (overall_target_radius >= 3) ? 1.45 : 1.35;
      const double extra_strength = std::clamp(
        std::max(high_order_fraction, 0.5 * medium_order_fraction) - 0.12,
        0.0,
        0.28) / 0.28;
      auto_generation_overseed_factor += 0.10 * extra_strength;
      auto_generation_start_count = clamp_target_count_to_support(
        static_cast<int>(
          std::llround(
            auto_generation_overseed_factor *
            static_cast<double>(estimated_target_count))));
      if (auto_generation_start_count <= 0) {
        auto_generation_start_count = estimated_target_count;
        auto_generation_overseed_factor = 1.0;
      } else {
        auto_generation_start_count = std::max(
          estimated_target_count,
          auto_generation_start_count);
      }
    }
  }
  if (estimated_target_count > 0 && !gen_point_count_initialized) {
    gen_point_count = auto_generation_start_count;
    gen_point_count_initialized = true;
  }
  ImGui::SliderInt("Point count##gen_count", &gen_point_count, 10, 5000);
  if (auto_generation_start_count > estimated_target_count &&
      auto_generation_overseed_factor > 1.0) {
    ImGui::TextDisabled(
      "High-order auto-start: %d points (x%.2f strategic over-seed)",
      auto_generation_start_count,
      auto_generation_overseed_factor);
  }
  ImGui::RadioButton("Random##gen_mode", &gen_mode, 0);
  ImGui::SameLine();
  ImGui::RadioButton("Strategic##gen_mode", &gen_mode, 1);

  const auto generate_points_from_support = [&]() -> bool {
    if (delaunay_helper && delaunay_helper->is_ready() && !output_support_uv.empty()) {
      clear_output_pattern_and_hist(state);

      // Keep the generator, strategic polishing, and optimizer aligned on the
      // same target objective. Without this, compute_live_pcf_error() can read
      // stale static targets and report a seed score the optimizer cannot
      // reproduce, which makes every candidate look worse than the start.
      live_hist_bin_count = state.voronoi_pcf_bin_count;
      live_target_hist = state.voronoi_pcf_hist_plot;
      live_target_individual_distributions = state.voronoi_pcf_individual_plots;
      live_target_raw_connectivity_signatures = state.voronoi_pcf_raw_point_hist_counts;
      state.voronoi_pcf_position_targets_enabled = false;
      live_position_targets_enabled = false;

      // Select points from Delaunay triangle centers with either random or strategic sampling.
      int target_count = std::min(gen_point_count, static_cast<int>(output_support_uv.size()));
      const int bin_count = std::max(1, state.voronoi_pcf_bin_count);
      std::mt19937 rng(std::random_device{}());

      std::vector<float> target_distribution;
      if (!state.voronoi_pcf_hist_counts.empty() && state.voronoi_pcf_pair_count > 0) {
        target_distribution = normalized_histogram(
          state.voronoi_pcf_hist_counts,
          state.voronoi_pcf_pair_count);
      }
      if (target_distribution.empty() && !state.voronoi_pcf_hist_plot.empty()) {
        target_distribution = state.voronoi_pcf_hist_plot;
      }
      std::vector<std::vector<float>> target_local_shape_prototypes;
      std::vector<float> target_local_shape_proto_fractions;
      const double strategic_local_shape_weight = 1.0;
      const int strategic_prefix_bins = std::max(1, (bin_count + 1) / 2);
      bool use_strategic_local_shape = false;
      if (use_strategic_local_shape) {
        target_local_shape_prototypes =
          select_target_local_shape_prototypes(state.voronoi_pcf_individual_plots);
        if (!target_local_shape_prototypes.empty()) {
          std::vector<int> target_proto_counts(
            target_local_shape_prototypes.size(),
            0);
          for (const auto& target_plot : state.voronoi_pcf_individual_plots) {
            int best_proto = -1;
            (void)compute_proto_cost_from_distribution(
              target_plot,
              target_local_shape_prototypes,
              &best_proto);
            if (best_proto >= 0 &&
                best_proto < static_cast<int>(target_proto_counts.size())) {
              ++target_proto_counts[static_cast<size_t>(best_proto)];
            }
          }
          const int total_target_proto_counts = std::accumulate(
            target_proto_counts.begin(),
            target_proto_counts.end(),
            0);
          if (total_target_proto_counts > 0) {
            target_local_shape_proto_fractions.resize(target_proto_counts.size(), 0.0f);
            for (size_t p = 0; p < target_proto_counts.size(); ++p) {
              target_local_shape_proto_fractions[p] =
                static_cast<float>(target_proto_counts[p]) /
                static_cast<float>(total_target_proto_counts);
            }
          }
        }
        use_strategic_local_shape =
          !target_local_shape_prototypes.empty() &&
          !target_local_shape_proto_fractions.empty();
      }

      const auto compute_strategic_histogram_energy =
        [&](const std::vector<float>& distribution) {
          if (target_distribution.empty() || distribution.empty()) {
            return 0.0;
          }

          double energy =
            weighted_distribution_l2(distribution, target_distribution) +
            hard_near_field_penalty(
              distribution,
              target_distribution,
              bin_count);

          const int eval_bins = std::min(
            strategic_prefix_bins,
            std::max(
              static_cast<int>(distribution.size()),
              static_cast<int>(target_distribution.size())));
          double prefix_l2 = 0.0;
          double prefix_cdf = 0.0;
          double prefix_forbidden_mass = 0.0;
          double prefix_curr_cdf = 0.0;
          double prefix_target_cdf = 0.0;
          constexpr double kStrategicPrefixWeight = 7.0;
          constexpr double kStrategicPrefixCdfWeight = 3.0;
          constexpr double kStrategicPrefixLeakPenalty = 300.0;
          constexpr double kStrategicPrefixTol = 1e-10;
          for (int k = 0; k < eval_bins; ++k) {
            const double a =
              (k < static_cast<int>(distribution.size()))
                ? static_cast<double>(distribution[static_cast<size_t>(k)])
                : 0.0;
            const double b =
              (k < static_cast<int>(target_distribution.size()))
                ? static_cast<double>(target_distribution[static_cast<size_t>(k)])
                : 0.0;
            const double d = a - b;
            const double bin_weight = 2.5 + 10.0 * b;
            prefix_l2 += bin_weight * d * d;
            prefix_curr_cdf += a;
            prefix_target_cdf += b;
            const double cdf_d = prefix_curr_cdf - prefix_target_cdf;
            prefix_cdf += cdf_d * cdf_d;
            if (b <= kStrategicPrefixTol && a > kStrategicPrefixTol) {
              prefix_forbidden_mass += a;
            }
          }

          energy +=
            kStrategicPrefixWeight * prefix_l2 +
            kStrategicPrefixCdfWeight * prefix_cdf +
            kStrategicPrefixLeakPenalty * prefix_forbidden_mass * prefix_forbidden_mass;
          return energy;
        };

      const auto compute_strategic_local_shape_energy =
        [&](double proto_cost_sum,
            const std::vector<int>& proto_counts,
            int point_count) {
          if (!use_strategic_local_shape || point_count <= 0) {
            return 0.0;
          }
          const double avg_proto_cost =
            proto_cost_sum / static_cast<double>(point_count);
          double proto_occupancy_error = 0.0;
          const int proto_count = std::min(
            static_cast<int>(proto_counts.size()),
            static_cast<int>(target_local_shape_proto_fractions.size()));
          for (int p = 0; p < proto_count; ++p) {
            const double current_fraction =
              static_cast<double>(proto_counts[static_cast<size_t>(p)]) /
              static_cast<double>(point_count);
            const double target_fraction =
              static_cast<double>(
                target_local_shape_proto_fractions[static_cast<size_t>(p)]);
            const double d = current_fraction - target_fraction;
            proto_occupancy_error += d * d;
          }
          return avg_proto_cost +
                 0.5 * proto_occupancy_error;
        };

      std::vector<int> all_positions(output_support_uv.size());
      std::iota(all_positions.begin(), all_positions.end(), 0);
      // Runtime guard: generate an informed prefix, then fill remainder randomly.
      const int strategic_limit = std::min(target_count, 512);
      const int proposal_pool_limit = std::min(
        static_cast<int>(output_support_uv.size()),
        std::max(192, std::min(384, target_count * 4)));
      const int proposal_probe_limit = std::min(
        static_cast<int>(output_support_uv.size()),
        std::max(
          proposal_pool_limit,
          std::min(768, std::max(proposal_pool_limit * 2, target_count * 6))));
      const int proposal_eval_limit = std::min(
        proposal_pool_limit,
        std::max(96, std::min(192, std::max(1, proposal_pool_limit / 2))));
      const int proposal_explore_limit = std::max(
        4,
        std::min(16, std::max(1, proposal_eval_limit / 8)));
      const int strategic_target_count = std::min(target_count, strategic_limit);
      const int max_parallel_restarts = std::max(1, std::min(4, omp_get_max_threads()));
      const int requested_restarts =
        (strategic_target_count >= 128) ? 4 :
        ((strategic_target_count >= 48) ? 3 : 2);
      const int strategic_restart_count =
        (gen_mode == 1)
          ? std::max(1, std::min(max_parallel_restarts, requested_restarts))
          : 1;

      struct StrategicAttemptResult {
        std::vector<int> support_positions;
        double exact_error = std::numeric_limits<double>::infinity();
        int polish_moves = 0;
        bool valid = false;
      };

      const auto run_strategic_attempt =
        [&](unsigned int seed) -> StrategicAttemptResult {
          StrategicAttemptResult result;
          std::mt19937 attempt_rng(seed);
          std::vector<char> selected_mask(output_support_uv.size(), 0);
          std::vector<int> selected_positions;
          selected_positions.reserve(static_cast<size_t>(target_count));
          std::vector<std::vector<int>> strategic_point_hist_counts;
          std::vector<int> strategic_point_proto_id;
          std::vector<double> strategic_point_proto_cost;
          std::vector<int> strategic_proto_counts(
            target_local_shape_prototypes.size(),
            0);
          double strategic_proto_cost_sum = 0.0;
          std::vector<int> running_hist(static_cast<size_t>(bin_count), 0);
          int running_pair_count = 0;

          const auto recompute_strategic_local_shape_state = [&]() {
            if (!use_strategic_local_shape) {
              return;
            }
            strategic_point_proto_id.assign(
              strategic_point_hist_counts.size(),
              -1);
            strategic_point_proto_cost.assign(
              strategic_point_hist_counts.size(),
              0.0);
            strategic_proto_counts.assign(
              target_local_shape_prototypes.size(),
              0);
            strategic_proto_cost_sum = 0.0;
            std::vector<float> strategic_distribution_scratch;
            for (size_t i = 0; i < strategic_point_hist_counts.size(); ++i) {
              const int support_pos = selected_positions[i];
              if (support_pos < 0 ||
                  support_pos >=
                    static_cast<int>(state.output_support_k_denominator_cache.size())) {
                continue;
              }
              int best_proto = -1;
              const double proto_cost = compute_proto_cost_from_counts(
                strategic_point_hist_counts[i],
                state.output_support_k_denominator_cache[static_cast<size_t>(support_pos)],
                target_local_shape_prototypes,
                &strategic_distribution_scratch,
                &best_proto);
              strategic_point_proto_id[i] = best_proto;
              strategic_point_proto_cost[i] = proto_cost;
              strategic_proto_cost_sum += proto_cost;
              if (best_proto >= 0 &&
                  best_proto < static_cast<int>(strategic_proto_counts.size())) {
                ++strategic_proto_counts[static_cast<size_t>(best_proto)];
              }
            }
          };

          const auto collect_candidate_pair_state =
            [&](int candidate_pos,
                std::vector<int>& add_hist,
                std::vector<int>& candidate_pair_bins) {
              add_hist.assign(static_cast<size_t>(bin_count), 0);
              candidate_pair_bins.clear();
              candidate_pair_bins.reserve(selected_positions.size());
              for (int selected_pos : selected_positions) {
                const int k = delaunay_helper->count_triangles_crossed(
                  output_support_uv[static_cast<size_t>(candidate_pos)],
                  output_support_uv[static_cast<size_t>(selected_pos)]);
                candidate_pair_bins.push_back(k);
                if (k >= 0) {
                  const int bin = std::min(k, bin_count - 1);
                  ++add_hist[static_cast<size_t>(bin)];
                }
              }
            };

          const auto compute_projected_histogram_energy =
            [&](const std::vector<int>& add_hist) {
              const int projected_pair_count =
                running_pair_count + static_cast<int>(selected_positions.size());
              if (projected_pair_count <= 0) {
                return std::numeric_limits<double>::infinity();
              }

              std::vector<int> projected_hist = running_hist;
              for (int k = 0; k < bin_count; ++k) {
                projected_hist[static_cast<size_t>(k)] +=
                  add_hist[static_cast<size_t>(k)];
              }

              const std::vector<float> projected_distribution =
                normalized_histogram(projected_hist, projected_pair_count);
              if (target_distribution.empty() || projected_distribution.empty()) {
                return 0.0;
              }
              return compute_strategic_histogram_energy(projected_distribution);
            };

          std::vector<float> strategic_direction_distribution_scratch;
          const auto compute_strategic_direction_energy =
            [&](int support_pos,
                const std::vector<int>& point_hist) {
              if (current_optimizer_mode() != VoronoiOptimizerMode::StructuredTargets ||
                  transition_direction_gradient_weight <= 0.0f ||
                  support_pos < 0 ||
                  support_pos >=
                    static_cast<int>(state.output_support_k_denominator_cache.size())) {
                return 0.0;
              }
              const double direction_cost = targeted_distribution_cost_from_counts(
                point_hist,
                state.output_support_k_denominator_cache[static_cast<size_t>(support_pos)],
                support_pos,
                &strategic_direction_distribution_scratch);
              if (!std::isfinite(direction_cost)) {
                return 0.0;
              }
              return static_cast<double>(transition_direction_gradient_weight) *
                     direction_cost;
            };

          // Keep the existing pair seed exactly as-is; multistart varies the
          // sampled pool rather than changing the seed rule itself.
          constexpr int strategic_seed_pool_limit = 96;
          if (strategic_target_count >= 2 && all_positions.size() >= 2) {
            std::vector<int> seed_pool = all_positions;
            if (static_cast<int>(seed_pool.size()) > strategic_seed_pool_limit) {
              // Weighted seed pool: prefer candidates whose normalised denominator
              // profile has a high dot-product similarity to the target distribution.
              // Fall back to random shuffle when the cache is unavailable.
              if (state.output_support_denominator_cache_valid &&
                  !target_distribution.empty() &&
                  !state.output_support_k_denominator_cache.empty()) {
                std::vector<std::pair<double, int>> scored;
                scored.reserve(seed_pool.size());
                const int eval_bins = std::min(bin_count, static_cast<int>(target_distribution.size()));
                for (int pos : seed_pool) {
                  if (pos < 0 ||
                      pos >= static_cast<int>(state.output_support_k_denominator_cache.size())) {
                    scored.emplace_back(0.0, pos);
                    continue;
                  }
                  const auto& denom = state.output_support_k_denominator_cache[static_cast<size_t>(pos)];
                  const std::vector<float>* reference_target =
                    active_target_distribution_for_support_row(pos);
                  if (reference_target == nullptr) {
                    reference_target = &target_distribution;
                  }
                  double total = 0.0;
                  for (int d : denom) {
                    total += d;
                  }
                  if (total <= 0.0 || reference_target->empty()) {
                    scored.emplace_back(0.0, pos);
                    continue;
                  }
                  double dot = 0.0;
                  const int reference_eval_bins = std::min(
                    eval_bins,
                    static_cast<int>(reference_target->size()));
                  for (int k = 0;
                       k < reference_eval_bins && k < static_cast<int>(denom.size());
                       ++k) {
                    dot += (static_cast<double>(denom[static_cast<size_t>(k)]) / total) *
                           static_cast<double>((*reference_target)[static_cast<size_t>(k)]);
                  }
                  scored.emplace_back(-dot, pos); // negate: ascending = best first
                }
                std::partial_sort(
                  scored.begin(),
                  scored.begin() +
                    std::min(strategic_seed_pool_limit, static_cast<int>(scored.size())),
                  scored.end());
                seed_pool.clear();
                seed_pool.reserve(static_cast<size_t>(strategic_seed_pool_limit));
                for (int s = 0;
                     s < strategic_seed_pool_limit &&
                     s < static_cast<int>(scored.size());
                     ++s) {
                  seed_pool.push_back(scored[static_cast<size_t>(s)].second);
                }
              } else {
                std::shuffle(seed_pool.begin(), seed_pool.end(), attempt_rng);
                seed_pool.resize(static_cast<size_t>(strategic_seed_pool_limit));
              }
            }

            int best_seed_a = -1;
            int best_seed_b = -1;
            std::vector<int> best_seed_hist(static_cast<size_t>(bin_count), 0);
            double best_seed_energy = std::numeric_limits<double>::infinity();

            for (size_t ia = 0; ia < seed_pool.size(); ++ia) {
              for (size_t ib = ia + 1; ib < seed_pool.size(); ++ib) {
                const int a = seed_pool[ia];
                const int b = seed_pool[ib];
                // Use pairwise cache when available for O(1) lookup instead of traversal.
                const int k =
                  (a >= 0 && b >= 0 && state.output_support_pairwise_cache_valid)
                    ? get_support_pairwise_dist(state, a, b)
                    : delaunay_helper->count_triangles_crossed(
                        output_support_uv[static_cast<size_t>(a)],
                        output_support_uv[static_cast<size_t>(b)]);
                if (k < 0 || k >= bin_count) {
                  continue;
                }

                std::vector<int> seed_hist(static_cast<size_t>(bin_count), 0);
                ++seed_hist[static_cast<size_t>(k)];
                std::vector<float> seed_distribution =
                  normalized_histogram(seed_hist, 1);

                double seed_energy = 0.0;
                if (!target_distribution.empty() && !seed_distribution.empty()) {
                  seed_energy = compute_strategic_histogram_energy(seed_distribution);
                }

                if (use_strategic_local_shape) {
                  std::vector<int> proto_counts(
                    target_local_shape_prototypes.size(),
                    0);
                  double proto_cost_sum = 0.0;
                  std::vector<float> seed_proto_distribution_scratch;
                  for (int pos : {a, b}) {
                    int best_proto = -1;
                    const double proto_cost = compute_proto_cost_from_counts(
                      seed_hist,
                      state.output_support_k_denominator_cache[static_cast<size_t>(pos)],
                      target_local_shape_prototypes,
                      &seed_proto_distribution_scratch,
                      &best_proto);
                    proto_cost_sum += proto_cost;
                    if (best_proto >= 0 &&
                        best_proto < static_cast<int>(proto_counts.size())) {
                      ++proto_counts[static_cast<size_t>(best_proto)];
                    }
                  }
                  seed_energy +=
                    strategic_local_shape_weight *
                    compute_strategic_local_shape_energy(
                      proto_cost_sum,
                      proto_counts,
                      2);
                }
                seed_energy += 0.5 *
                  (compute_strategic_direction_energy(a, seed_hist) +
                   compute_strategic_direction_energy(b, seed_hist));

                if (seed_energy < best_seed_energy) {
                  best_seed_energy = seed_energy;
                  best_seed_a = a;
                  best_seed_b = b;
                  best_seed_hist = std::move(seed_hist);
                }
              }
            }

            if (best_seed_a >= 0 && best_seed_b >= 0) {
              selected_positions.push_back(best_seed_a);
              selected_positions.push_back(best_seed_b);
              selected_mask[static_cast<size_t>(best_seed_a)] = 1;
              selected_mask[static_cast<size_t>(best_seed_b)] = 1;
              running_hist = best_seed_hist;
              running_pair_count = 1;
              if (use_strategic_local_shape) {
                strategic_point_hist_counts.assign(
                  2,
                  std::vector<int>(static_cast<size_t>(bin_count), 0));
                strategic_point_hist_counts[0] = best_seed_hist;
                strategic_point_hist_counts[1] = best_seed_hist;
                recompute_strategic_local_shape_state();
              }
            }
          }

          if (selected_positions.empty()) {
            std::uniform_int_distribution<int> seed_pick(
              0,
              static_cast<int>(all_positions.size()) - 1);
            const int seed_pos =
              all_positions[static_cast<size_t>(seed_pick(attempt_rng))];
            selected_positions.push_back(seed_pos);
            selected_mask[static_cast<size_t>(seed_pos)] = 1;
            if (use_strategic_local_shape) {
              strategic_point_hist_counts.assign(
                1,
                std::vector<int>(static_cast<size_t>(bin_count), 0));
              recompute_strategic_local_shape_state();
            }
          }

          while (static_cast<int>(selected_positions.size()) < strategic_target_count) {
            std::vector<int> unselected_positions;
            unselected_positions.reserve(all_positions.size());
            for (int pos : all_positions) {
              if (!selected_mask[static_cast<size_t>(pos)]) {
                unselected_positions.push_back(pos);
              }
            }
            if (unselected_positions.empty()) {
              break;
            }

            std::vector<int> probe_positions = unselected_positions;
            if (static_cast<int>(probe_positions.size()) > proposal_probe_limit) {
              std::shuffle(
                probe_positions.begin(),
                probe_positions.end(),
                attempt_rng);
              probe_positions.resize(static_cast<size_t>(proposal_probe_limit));
            }

            int best_candidate_pos = -1;
            std::vector<int> best_candidate_add_hist(static_cast<size_t>(bin_count), 0);
            std::vector<int> best_candidate_pair_bins;
            double best_candidate_energy = std::numeric_limits<double>::infinity();

            struct RankedCandidate {
              double histogram_energy;
              int support_pos;
            };
            std::vector<RankedCandidate> ranked_candidates;
            ranked_candidates.reserve(probe_positions.size());

            std::vector<int> candidate_add_hist;
            std::vector<int> candidate_pair_bins;
            for (int candidate_pos : probe_positions) {
              collect_candidate_pair_state(
                candidate_pos,
                candidate_add_hist,
                candidate_pair_bins);
              const double histogram_energy =
                compute_projected_histogram_energy(candidate_add_hist) +
                compute_strategic_direction_energy(candidate_pos, candidate_add_hist);
              ranked_candidates.push_back({histogram_energy, candidate_pos});
              if (!use_strategic_local_shape &&
                  histogram_energy < best_candidate_energy) {
                best_candidate_energy = histogram_energy;
                best_candidate_pos = candidate_pos;
                best_candidate_add_hist = candidate_add_hist;
                best_candidate_pair_bins = candidate_pair_bins;
              }
            }

            if (use_strategic_local_shape && !ranked_candidates.empty()) {
              std::sort(
                ranked_candidates.begin(),
                ranked_candidates.end(),
                [](const RankedCandidate& a, const RankedCandidate& b) {
                  if (a.histogram_energy == b.histogram_energy) {
                    return a.support_pos < b.support_pos;
                  }
                  return a.histogram_energy < b.histogram_energy;
                });

              std::vector<int> evaluation_positions;
              evaluation_positions.reserve(
                std::min(proposal_eval_limit, static_cast<int>(ranked_candidates.size())));
              const int explore_count = std::min(
                proposal_explore_limit,
                std::max(0, static_cast<int>(ranked_candidates.size()) - 1));
              const int greedy_keep = std::min(
                static_cast<int>(ranked_candidates.size()),
                std::max(1, proposal_eval_limit - explore_count));
              for (int i = 0; i < greedy_keep; ++i) {
                evaluation_positions.push_back(
                  ranked_candidates[static_cast<size_t>(i)].support_pos);
              }
              if (static_cast<int>(evaluation_positions.size()) < proposal_eval_limit &&
                  static_cast<int>(ranked_candidates.size()) > greedy_keep) {
                std::vector<int> explore_positions;
                explore_positions.reserve(
                  static_cast<size_t>(ranked_candidates.size() - greedy_keep));
                for (size_t i = static_cast<size_t>(greedy_keep);
                     i < ranked_candidates.size();
                     ++i) {
                  explore_positions.push_back(ranked_candidates[i].support_pos);
                }
                std::shuffle(
                  explore_positions.begin(),
                  explore_positions.end(),
                  attempt_rng);
                const int remaining_slots =
                  proposal_eval_limit - static_cast<int>(evaluation_positions.size());
                const int add_count = std::min(
                  remaining_slots,
                  static_cast<int>(explore_positions.size()));
                for (int i = 0; i < add_count; ++i) {
                  evaluation_positions.push_back(
                    explore_positions[static_cast<size_t>(i)]);
                }
              }

              for (int candidate_pos : evaluation_positions) {
                collect_candidate_pair_state(
                  candidate_pos,
                  candidate_add_hist,
                  candidate_pair_bins);
                double candidate_energy =
                  compute_projected_histogram_energy(candidate_add_hist) +
                  compute_strategic_direction_energy(candidate_pos, candidate_add_hist);
                if (candidate_pos >= 0 &&
                    candidate_pos <
                      static_cast<int>(state.output_support_k_denominator_cache.size())) {
                  double projected_proto_cost_sum = strategic_proto_cost_sum;
                  std::vector<int> projected_proto_counts = strategic_proto_counts;
                  std::vector<float> candidate_distribution;
                  std::vector<float> existing_distribution_scratch;

                  for (size_t j = 0; j < candidate_pair_bins.size(); ++j) {
                    const int k = candidate_pair_bins[j];
                    if (k < 0 || k >= bin_count ||
                        j >= strategic_point_hist_counts.size() ||
                        j >= strategic_point_proto_cost.size() ||
                        j >= strategic_point_proto_id.size()) {
                      continue;
                    }
                    const int support_pos = selected_positions[j];
                    int new_proto_id = -1;
                    const double new_proto_cost =
                      compute_proto_cost_from_counts_with_delta(
                        strategic_point_hist_counts[j],
                        state.output_support_k_denominator_cache[static_cast<size_t>(support_pos)],
                        -1,
                        k,
                        target_local_shape_prototypes,
                        &existing_distribution_scratch,
                        &new_proto_id);

                    const int old_proto_id = strategic_point_proto_id[j];
                    const double old_proto_cost = strategic_point_proto_cost[j];
                    projected_proto_cost_sum += new_proto_cost - old_proto_cost;
                    if (old_proto_id >= 0 &&
                        old_proto_id < static_cast<int>(projected_proto_counts.size())) {
                      --projected_proto_counts[static_cast<size_t>(old_proto_id)];
                    }
                    if (new_proto_id >= 0 &&
                        new_proto_id < static_cast<int>(projected_proto_counts.size())) {
                      ++projected_proto_counts[static_cast<size_t>(new_proto_id)];
                    }
                  }

                  int candidate_proto_id = -1;
                  const double candidate_proto_cost = compute_proto_cost_from_counts(
                    candidate_add_hist,
                    state.output_support_k_denominator_cache[static_cast<size_t>(candidate_pos)],
                    target_local_shape_prototypes,
                    &candidate_distribution,
                    &candidate_proto_id);
                  projected_proto_cost_sum += candidate_proto_cost;
                  if (candidate_proto_id >= 0 &&
                      candidate_proto_id < static_cast<int>(projected_proto_counts.size())) {
                    ++projected_proto_counts[static_cast<size_t>(candidate_proto_id)];
                  }

                  candidate_energy +=
                    strategic_local_shape_weight *
                    compute_strategic_local_shape_energy(
                      projected_proto_cost_sum,
                      projected_proto_counts,
                      static_cast<int>(selected_positions.size()) + 1);
                }

                if (candidate_energy < best_candidate_energy) {
                  best_candidate_energy = candidate_energy;
                  best_candidate_pos = candidate_pos;
                  best_candidate_add_hist = candidate_add_hist;
                  best_candidate_pair_bins = candidate_pair_bins;
                }
              }
            }

            if (best_candidate_pos < 0) {
              std::uniform_int_distribution<int> fallback_pick(
                0,
                static_cast<int>(probe_positions.size()) - 1);
              best_candidate_pos =
                probe_positions[static_cast<size_t>(fallback_pick(attempt_rng))];
              collect_candidate_pair_state(
                best_candidate_pos,
                best_candidate_add_hist,
                best_candidate_pair_bins);
            }

            if (use_strategic_local_shape) {
              for (size_t j = 0; j < best_candidate_pair_bins.size(); ++j) {
                const int k = best_candidate_pair_bins[j];
                if (k >= 0 && j < strategic_point_hist_counts.size()) {
                  const int bin = std::min(k, bin_count - 1);
                  ++strategic_point_hist_counts[j][static_cast<size_t>(bin)];
                }
              }
              strategic_point_hist_counts.push_back(best_candidate_add_hist);
            }
            selected_positions.push_back(best_candidate_pos);
            selected_mask[static_cast<size_t>(best_candidate_pos)] = 1;
            for (int k = 0; k < bin_count; ++k) {
              running_hist[static_cast<size_t>(k)] +=
                best_candidate_add_hist[static_cast<size_t>(k)];
            }
            running_pair_count +=
              static_cast<int>(selected_positions.size()) - 1;
            if (use_strategic_local_shape) {
              recompute_strategic_local_shape_state();
            }
          }

          if (static_cast<int>(selected_positions.size()) < target_count) {
            std::vector<int> remaining_positions;
            remaining_positions.reserve(all_positions.size());
            for (int pos : all_positions) {
              if (!selected_mask[static_cast<size_t>(pos)]) {
                remaining_positions.push_back(pos);
              }
            }
            std::shuffle(
              remaining_positions.begin(),
              remaining_positions.end(),
              attempt_rng);
            const int needed =
              target_count - static_cast<int>(selected_positions.size());
            const int add_count = std::min(
              needed,
              static_cast<int>(remaining_positions.size()));
            for (int i = 0; i < add_count; ++i) {
              selected_positions.push_back(
                remaining_positions[static_cast<size_t>(i)]);
            }
          }

          std::vector<int> generated_support_positions = selected_positions;
          std::vector<Eigen::Vector2d> generated_uv;
          generated_uv.reserve(generated_support_positions.size());
          for (int pos : generated_support_positions) {
            generated_uv.push_back(output_support_uv[static_cast<size_t>(pos)]);
          }

          result.support_positions = std::move(generated_support_positions);
          result.exact_error = compute_live_pcf_error(generated_uv);
          result.polish_moves = 0;
          result.valid = !result.support_positions.empty();
          return result;
        };

      std::vector<int> generated_support_positions;
      int strategic_polish_moves = 0;
      double strategic_polish_error = std::numeric_limits<double>::infinity();
      double strategic_elapsed_seconds = -1.0;
      if (gen_mode == 0) {
        std::shuffle(all_positions.begin(), all_positions.end(), rng);
        const int pick_count =
          std::min(target_count, static_cast<int>(all_positions.size()));
        generated_support_positions.assign(
          all_positions.begin(),
          all_positions.begin() + pick_count);
      } else {
        const auto strategic_start_time = std::chrono::steady_clock::now();
        std::random_device rd;
        const unsigned int base_seed = rd();
        std::vector<StrategicAttemptResult> attempt_results(
          static_cast<size_t>(strategic_restart_count));
        #pragma omp parallel for schedule(static) if (strategic_restart_count > 1)
        for (int attempt_idx = 0;
             attempt_idx < strategic_restart_count;
             ++attempt_idx) {
          const unsigned int attempt_seed =
            base_seed +
            static_cast<unsigned int>(0x9e3779b9u * static_cast<unsigned int>(attempt_idx + 1));
          attempt_results[static_cast<size_t>(attempt_idx)] =
            run_strategic_attempt(attempt_seed);
        }

        int best_attempt_idx = -1;
        for (int attempt_idx = 0;
             attempt_idx < strategic_restart_count;
             ++attempt_idx) {
          const auto& candidate =
            attempt_results[static_cast<size_t>(attempt_idx)];
          if (!candidate.valid) {
            continue;
          }
          if (best_attempt_idx < 0) {
            best_attempt_idx = attempt_idx;
            continue;
          }
          const auto& best =
            attempt_results[static_cast<size_t>(best_attempt_idx)];
          const bool candidate_finite = std::isfinite(candidate.exact_error);
          const bool best_finite = std::isfinite(best.exact_error);
          if (candidate_finite != best_finite) {
            if (candidate_finite) {
              best_attempt_idx = attempt_idx;
            }
            continue;
          }
          if (candidate_finite &&
              candidate.exact_error + 1e-9 < best.exact_error) {
            best_attempt_idx = attempt_idx;
            continue;
          }
          if ((!best_finite ||
               std::abs(candidate.exact_error - best.exact_error) <= 1e-9) &&
              candidate.support_positions.size() > best.support_positions.size()) {
            best_attempt_idx = attempt_idx;
          }
        }

        if (best_attempt_idx < 0) {
          const StrategicAttemptResult fallback_attempt =
            run_strategic_attempt(base_seed);
          generated_support_positions = fallback_attempt.support_positions;
          strategic_polish_moves = fallback_attempt.polish_moves;
          strategic_polish_error = fallback_attempt.exact_error;
        } else {
          const auto& best_attempt =
            attempt_results[static_cast<size_t>(best_attempt_idx)];
          generated_support_positions = best_attempt.support_positions;
          strategic_polish_moves = best_attempt.polish_moves;
          strategic_polish_error = best_attempt.exact_error;
        }
        strategic_elapsed_seconds =
          std::chrono::duration<double>(
            std::chrono::steady_clock::now() - strategic_start_time).count();
      }

      std::vector<Eigen::Vector2d> generated_uv;
      generated_uv.reserve(generated_support_positions.size());
      state.output_pattern_sample_indices.clear();
      state.output_pattern_sample_indices.reserve(generated_support_positions.size());
      for (int pos : generated_support_positions) {
        generated_uv.push_back(output_support_uv[static_cast<size_t>(pos)]);
      }

      state.output_pattern_sample_indices.clear();
      state.output_pattern_sample_indices.reserve(generated_support_positions.size());
      for (int pos : generated_support_positions) {
        if (pos >= 0 && pos < static_cast<int>(output_support_tri_indices.size())) {
          state.output_pattern_sample_indices.push_back(output_support_tri_indices[static_cast<size_t>(pos)]);
        }
      }

      // Sync to state
      state.output_pattern_points_uv.clear();
      state.output_pattern_points_3d.clear();
      for (const auto& uv : generated_uv) {
        state.output_pattern_points_uv.push_back(uv);
        Eigen::Vector3d lifted_3d = Eigen::Vector3d::Zero();
        if (!lift_uv_to_output_3d(uv, delaunay_helper, points_3d, points_uv, lifted_3d)) {
          lifted_3d = nearest_sample_3d(uv, points_3d, points_uv);
        }
        state.output_pattern_points_3d.push_back(lifted_3d);
      }
      state.output_pattern_dirty = true;

      // Update histogram
      update_live_hist_in_state(generated_uv);
      if (std::isfinite(strategic_polish_error)) {
        last_generated_exact_error = strategic_polish_error;
      } else {
        last_generated_exact_error = compute_live_pcf_error(generated_uv);
      }
      reset_after_generate_points = true;

      if (gen_mode == 0) {
        std::cout << "Generated " << generated_uv.size() << "/" << target_count
                  << " random triangle centers\n";
      } else {
        std::cout << "Generated " << generated_uv.size() << "/" << target_count
                  << " strategic triangle centers (strategic prefix="
                  << std::min(target_count, strategic_limit)
                  << ", restarts=" << strategic_restart_count
                  << ", probe=" << proposal_probe_limit
                  << ", eval=" << proposal_eval_limit;
        if (strategic_elapsed_seconds >= 0.0) {
          std::cout << ", elapsed=" << strategic_elapsed_seconds << "s";
        }
        if (std::isfinite(last_generated_exact_error)) {
          std::cout << ", polish-moves=" << strategic_polish_moves
                    << ", exact=" << last_generated_exact_error;
        }
        std::cout << ")\n";
      }
      return true;
    }
    return false;
  };

  GeneratedPatchBatchRunState& generated_patch_batch_run =
    root_state.generated_patch_batch_run;
  const bool generated_patch_generate_batch_current_region =
    generated_patch_batch_run.active &&
    generated_patch_batch_run.action ==
      static_cast<int>(GeneratedPatchBatchAction::GeneratePoints) &&
    generated_patch_batch_run.current_region_offset <
      generated_patch_batch_run.region_ids.size() &&
    generated_patch_batch_run.region_ids[
      generated_patch_batch_run.current_region_offset] == state.region_id;
  if (generated_patch_generate_batch_current_region &&
      !generated_patch_batch_run.current_region_started) {
    if (generated_patch_batch_run.requested_point_count > 0) {
      gen_point_count = generated_patch_batch_run.requested_point_count;
      gen_point_count_initialized = true;
    } else {
      const int batch_auto_point_count =
        (auto_generation_start_count > 0)
          ? auto_generation_start_count
          : ((estimated_target_count > 0)
              ? estimated_target_count
              : gen_point_count);
      if (batch_auto_point_count > 0) {
        gen_point_count = batch_auto_point_count;
        gen_point_count_initialized = true;
      }
    }
    generated_patch_batch_run.current_region_started = true;
    if (!generate_points_from_support()) {
      std::ostringstream status;
      status << "Generate Points could not start on region "
             << pattern_region_label(root_state, root_state.active_region_index)
             << "; continuing to the next generated patch.";
      set_generated_patch_batch_status(status.str(), true);
    }
    generated_patch_batch_run.current_region_completed = true;
  }

  if (ImGui::Button("Generate Points", ImVec2(-1, 0))) {
    const bool generated = generate_points_from_support();
    if (generated && should_broadcast_to_generated_patch_family()) {
      const std::vector<int> family_region_indices =
        active_generated_patch_family_indices();
      if (family_region_indices.size() > 1 &&
          begin_generated_patch_batch_run(
            root_state,
            state.region_id,
            GeneratedPatchBatchAction::GeneratePoints,
            gen_point_count,
            true,
            true)) {
        std::ostringstream status;
        status << "Generate Points queued across "
               << family_region_indices.size()
               << " generated exemplar patches.";
        set_generated_patch_batch_status(status.str(), false);
      }
    }
  }

  if (output_support_count > 0) {
    ImGui::Text("Support candidates: %d", output_support_count);
  } else {
    ImGui::TextDisabled(
      is_transition_region
        ? "Build the transition target to enable optimization"
        : "Compute input PCF to enable optimization");
  }

  // Neighbor-based optimization
  static bool optimize_running = false;
  static std::vector<Eigen::Vector2d> optimize_points;
  static std::vector<int> optimize_triangle_indices;
  static std::vector<Eigen::Vector2d> optimize_best_points;
  static std::vector<int> optimize_best_triangle_indices;
  static int optimize_swaps_made = 0;
  static double optimize_best_error = 0.0;
  static double global_best_error = 1e10;
  static double optimize_initial_error = 0.0;
  static double optimize_current_exact_error = std::numeric_limits<double>::infinity();
  static double optimize_current_total_error = std::numeric_limits<double>::infinity();
  static bool optimize_current_error_valid = false;
  static bool optimize_trust_strategic_start = false;
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static int no_progress_iters = 0;
  static int soft_no_progress_iters = 0;
  static int optimize_iteration = 0;
  static float uphill_tolerance_percent = 0.0f;
  static float convergence_delta_percent = 0.005f;
  static float settled_best_gap_percent = 0.25f;
  static float settled_worst_bin_tol = 0.02f;
  static int settled_sweeps_required = 4;
  static int settled_move_limit = 2;
  static float early_stop_best_gap_percent = 1.5f;
  static float early_stop_worst_bin_tol = 0.1f;
  static int early_stop_soft_no_progress_limit = 20;
  static int early_stop_move_limit = 3;
  static bool early_stop_allow_plateau = false;
  static int settled_sweeps = 0;
  static int max_iterations = 400;
  static int stagnation_patience = 60;
  static float stagnation_best_improve_eps = 1e-3f;
  static int plateau_window = 40;
  static float plateau_error_band_percent = 1.0f;
  static float plateau_worst_bin_band = 0.002f;
  static bool plateau_jitter_enabled = false;
  static int plateau_jitter_points = 4;
  static int plateau_jitter_proposals = 16;
  static int plateau_jitter_attempt_limit = 3;
  static int plateau_jitter_attempts_used = 0;
  static int optimizer_visual_update_sweeps = 4;
  static int optimizer_visual_update_min_interval_ms = 750;
  static int optimizer_sweeps_since_visual_update = 0;
  static std::chrono::steady_clock::time_point optimizer_last_visual_update_time{};
  static int repair_points_per_sweep = 8;
  static int optimization_budget_ms = 0; // 0 = no time limit
  static int global_support_proposals_per_point = 8;
  static bool adaptive_count_moves = true;
  static int adaptive_count_period = 5;
  static int adaptive_count_add_proposals = 8;
  static int adaptive_count_delete_candidates = 4;
  static int adaptive_count_window = 12;
  static float adaptive_count_accept_eps = 1e-4f;
  static float adaptive_count_delete_accept_scale = 12.0f;
  static int adaptive_count_last_move_direction = 0;
  static int adaptive_count_reversal_cooldown = 0;
  static std::vector<double> plateau_recent_errors;
  static std::vector<double> plateau_recent_worst_bins;
  static std::vector<float> optimize_point_priority_penalty;
  static std::vector<int> optimize_point_priority_cooldown;
  static std::vector<float> optimize_point_priority_boost;
  static std::vector<int> optimize_point_priority_boost_cooldown;
  static bool use_incremental_optimizer = true;
  static bool optimizer_debug_focus = false;

  struct IncrementalPCFCache {
    bool valid = false;
    int bin_count = 0;
    int raw_connectivity_bins = 0;
    int triangle_count = 0;
    int point_count = 0;
    int valid_points = 0;
    int pair_count = 0;
    int target_max_bin = -1;
    std::vector<int> global_hist_counts;
    std::vector<std::vector<int>> point_hist;
    std::vector<std::vector<int>> point_support;
    std::vector<char> has_support;
    std::vector<float> sum_distribution;
    std::vector<double> empty_penalty_per_point;
    double empty_penalty_sum = 0.0;
    std::vector<char> empty_bin_mask;
    std::vector<int> support_row_for_triangle;
    std::vector<int> point_support_rows;
    std::vector<std::vector<int>> target_raw_connectivity_prototypes;
    std::vector<float> target_raw_connectivity_proto_fractions;
    std::vector<int> point_raw_connectivity_proto_id;
    std::vector<double> point_raw_connectivity_cost;
    std::vector<int> current_raw_connectivity_proto_counts;
    double raw_connectivity_cost_sum = 0.0;
    double fast_error = 0.0;
  };
  static IncrementalPCFCache inc;

  struct TriangleGeometryCache {
    bool valid = false;
    int triangle_count = -1;
    Eigen::MatrixXd boundary_uv;
    std::vector<Eigen::Vector2d> centers;
    std::vector<char> valid_flags;
    std::vector<char> inside_flags;
  };
  static TriangleGeometryCache tri_geom_cache;

  const auto reset_optimizer_runtime_state = [&]() {
    optimize_points.clear();
    optimize_triangle_indices.clear();
    optimize_best_points.clear();
    optimize_best_triangle_indices.clear();
    optimize_swaps_made = 0;
    optimize_best_error = 0.0;
    global_best_error = 1e10;
    optimize_initial_error = 0.0;
    optimize_current_exact_error = std::numeric_limits<double>::infinity();
    optimize_current_total_error = std::numeric_limits<double>::infinity();
    optimize_current_error_valid = false;
    optimize_trust_strategic_start = false;
    no_progress_iters = 0;
    soft_no_progress_iters = 0;
    optimize_iteration = 0;
    inc.valid = false;
    tri_geom_cache.valid = false;
    tri_geom_cache.triangle_count = -1;
    tri_geom_cache.boundary_uv.resize(0, 0);
    tri_geom_cache.centers.clear();
    tri_geom_cache.valid_flags.clear();
    tri_geom_cache.inside_flags.clear();
    adaptive_count_last_move_direction = 0;
    adaptive_count_reversal_cooldown = 0;
    plateau_jitter_attempts_used = 0;
    plateau_recent_errors.clear();
    plateau_recent_worst_bins.clear();
    optimize_point_priority_penalty.clear();
    optimize_point_priority_cooldown.clear();
    optimize_point_priority_boost.clear();
    optimize_point_priority_boost_cooldown.clear();
    settled_sweeps = 0;
    optimizer_sweeps_since_visual_update = 0;
    optimizer_last_visual_update_time = std::chrono::steady_clock::time_point{};
    state.optimizer_improvements = 0;
    state.optimizer_iterations_ran = 0;
  };
  const auto sync_optimizer_point_priority_state = [&](bool clear_existing = false) {
    const size_t point_count = optimize_points.size();
    if (clear_existing ||
        optimize_point_priority_penalty.size() != point_count ||
        optimize_point_priority_cooldown.size() != point_count ||
        optimize_point_priority_boost.size() != point_count ||
        optimize_point_priority_boost_cooldown.size() != point_count) {
      optimize_point_priority_penalty.assign(point_count, 0.0f);
      optimize_point_priority_cooldown.assign(point_count, 0);
      optimize_point_priority_boost.assign(point_count, 0.0f);
      optimize_point_priority_boost_cooldown.assign(point_count, 0);
    }
  };

  static int last_runtime_region_id = -1;
  if (last_runtime_region_id != region_runtime_id) {
    optimize_running = false;
    reset_optimizer_runtime_state();
    live_target_hist.clear();
    live_target_individual_distributions.clear();
    live_target_raw_connectivity_signatures.clear();
    live_position_targets_enabled = false;
    live_transition_optimizer_targets.ready = false;
    live_transition_optimizer_targets.distribution_for_support_row.clear();
    live_transition_optimizer_targets.raw_point_hist_counts_for_support_row.clear();
    live_hist_bin_count = 0;
    live_last_worst_bin_residual = std::numeric_limits<double>::infinity();
    live_last_worst_bin_index = -1;
    gen_point_count_initialized = false;
    last_generated_exact_error = std::numeric_limits<double>::infinity();
    last_runtime_region_id = region_runtime_id;
  }

  refresh_transition_optimizer_targets();

  const auto optimization_target_count = [&]() {
    if (estimated_target_count >= 2) {
      return estimated_target_count;
    }
    if (state.baseline_graph_point_count >= 2) {
      return state.baseline_graph_point_count;
    }
    return std::max(2, gen_point_count);
  };

  const auto optimization_count_penalty = [&](int point_count) {
    const int target_count = std::max(2, optimization_target_count());
    if (point_count <= 0 || target_count <= 0) {
      return 0.0;
    }
    const int count_deadband = std::max(0, adaptive_count_window);
    const int lower_count_deadband =
      (count_deadband <= 0) ? 0 : std::max(1, count_deadband / 4);
    const int lower_preferred_count =
      std::max(2, target_count - lower_count_deadband);
    const int upper_preferred_count = target_count + count_deadband;
    if (point_count < lower_preferred_count) {
      const double gap =
        static_cast<double>(lower_preferred_count - point_count);
      const double gap_scale =
        static_cast<double>(std::max(2, lower_count_deadband + 1));
      const double normalized_gap = gap / gap_scale;
      constexpr double kBelowCountPenaltyLinear = 0.35;
      constexpr double kBelowCountPenaltyQuadratic = 0.5;
      return
        kBelowCountPenaltyLinear * gap +
        kBelowCountPenaltyQuadratic * normalized_gap * normalized_gap;
    } else if (point_count > upper_preferred_count) {
      const double gap =
        static_cast<double>(point_count - upper_preferred_count);
      const double gap_scale =
        static_cast<double>(std::max(6, count_deadband + 1));
      const double normalized_gap = gap / gap_scale;
      constexpr double kAboveCountPenaltyLinear = 0.08;
      constexpr double kAboveCountPenaltyQuadratic = 0.25;
      return
        kAboveCountPenaltyLinear * gap +
        kAboveCountPenaltyQuadratic * normalized_gap * normalized_gap;
    } else {
      return 0.0;
    }
  };

  const auto augment_error_with_count_penalty =
    [&](double base_error, int point_count) {
      if (!std::isfinite(base_error)) {
        return base_error;
      }
      return base_error + optimization_count_penalty(point_count);
    };

  constexpr double kOptimizerRawConnectivityWeight = 6.0;
  constexpr double kOptimizerRawConnectivityRankWeight = 24.0;
  const auto augment_optimizer_objective =
    [&](double exact_error,
        int point_count,
        double raw_connectivity_error) {
      const double total_error = augment_error_with_count_penalty(exact_error, point_count);
      if (!std::isfinite(total_error)) {
        return total_error;
      }
      return total_error +
        kOptimizerRawConnectivityWeight * std::max(0.0, raw_connectivity_error);
    };

  const auto compute_cached_raw_connectivity_objective = [&]() {
    if (!inc.valid ||
        inc.valid_points <= 0 ||
        inc.current_raw_connectivity_proto_counts.empty() ||
        inc.target_raw_connectivity_proto_fractions.empty()) {
      return 0.0;
    }
    double occupancy_error = 0.0;
    const int proto_count = std::min(
      static_cast<int>(inc.current_raw_connectivity_proto_counts.size()),
      static_cast<int>(inc.target_raw_connectivity_proto_fractions.size()));
    for (int p = 0; p < proto_count; ++p) {
      const double curr_frac =
        static_cast<double>(inc.current_raw_connectivity_proto_counts[static_cast<size_t>(p)]) /
        static_cast<double>(inc.valid_points);
      const double target_frac =
        static_cast<double>(inc.target_raw_connectivity_proto_fractions[static_cast<size_t>(p)]);
      const double d = curr_frac - target_frac;
      occupancy_error += d * d;
    }
    return
      (inc.raw_connectivity_cost_sum / static_cast<double>(inc.valid_points)) +
      4.0 * occupancy_error;
  };

  const auto compute_live_bin_residual_stats_from_avg_plot = [&](
    const std::vector<float>& avg_plot,
    int focus_count,
    int* out_worst_bin_idx,
    std::vector<int>* out_focus_bins) -> double {
    if (out_worst_bin_idx) {
      *out_worst_bin_idx = -1;
    }
    if (out_focus_bins) {
      out_focus_bins->clear();
    }
    const int n = std::max(live_hist_bin_count, static_cast<int>(live_target_hist.size()));
    if (n <= 0) {
      return std::numeric_limits<double>::infinity();
    }

    std::vector<std::pair<double, int>> residuals;
    residuals.reserve(static_cast<size_t>(n));
    double worst_residual = 0.0;
    int worst_bin_idx = -1;
    for (int k = 0; k < n; ++k) {
      const double p = (k < static_cast<int>(avg_plot.size()))
        ? static_cast<double>(avg_plot[static_cast<size_t>(k)])
        : 0.0;
      const double t = (k < static_cast<int>(live_target_hist.size()))
        ? static_cast<double>(live_target_hist[static_cast<size_t>(k)])
        : 0.0;
      const double residual = std::abs(p - t);
      residuals.emplace_back(residual, k);
      if (residual > worst_residual) {
        worst_residual = residual;
        worst_bin_idx = k;
      }
    }

    if (out_worst_bin_idx) {
      *out_worst_bin_idx = worst_bin_idx;
    }
    if (out_focus_bins && !residuals.empty() && focus_count > 0) {
      std::sort(
        residuals.begin(),
        residuals.end(),
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
          if (a.first == b.first) {
            return a.second < b.second;
          }
          return a.first > b.first;
        });
      const int top_n = std::min(focus_count, static_cast<int>(residuals.size()));
      for (int i = 0; i < top_n; ++i) {
        if (residuals[static_cast<size_t>(i)].first <= 1e-12) {
          break;
        }
        out_focus_bins->push_back(residuals[static_cast<size_t>(i)].second);
      }
    }
    return worst_residual;
  };

  const auto update_live_hist_in_state_from_incremental =
    [&](std::vector<float>* out_avg_plot = nullptr) -> bool {
      if (!inc.valid || inc.bin_count <= 0) {
        return false;
      }
      std::vector<float> avg_plot = average_individual_histogram(
        inc.point_hist,
        inc.point_support,
        live_hist_bin_count);
      state.output_voronoi_pcf_hist_counts = inc.global_hist_counts;
      if (static_cast<int>(state.output_voronoi_pcf_hist_counts.size()) < live_hist_bin_count) {
        state.output_voronoi_pcf_hist_counts.resize(static_cast<size_t>(live_hist_bin_count), 0);
      } else if (static_cast<int>(state.output_voronoi_pcf_hist_counts.size()) > live_hist_bin_count) {
        state.output_voronoi_pcf_hist_counts.resize(static_cast<size_t>(live_hist_bin_count));
      }
      state.output_voronoi_pcf_hist_plot = avg_plot;
      state.output_voronoi_pcf_pair_count = inc.pair_count;
      state.output_voronoi_pcf_max_k = 0;
      for (int k = 0; k < static_cast<int>(state.output_voronoi_pcf_hist_counts.size()); ++k) {
        if (state.output_voronoi_pcf_hist_counts[static_cast<size_t>(k)] > 0) {
          state.output_voronoi_pcf_max_k = k;
        }
      }
      if (out_avg_plot) {
        *out_avg_plot = std::move(avg_plot);
      }
      return true;
    };

  const auto compute_live_bin_residual_stats_from_incremental = [&](
    int focus_count,
    int* out_worst_bin_idx,
    std::vector<int>* out_focus_bins,
    std::vector<float>* out_avg_plot = nullptr) -> double {
    if (!inc.valid || inc.bin_count <= 0) {
      if (out_worst_bin_idx) {
        *out_worst_bin_idx = -1;
      }
      if (out_focus_bins) {
        out_focus_bins->clear();
      }
      if (out_avg_plot) {
        out_avg_plot->clear();
      }
      return std::numeric_limits<double>::infinity();
    }
    std::vector<float> avg_plot = average_individual_histogram(
      inc.point_hist,
      inc.point_support,
      live_hist_bin_count);
    if (out_avg_plot) {
      *out_avg_plot = avg_plot;
    }
    return compute_live_bin_residual_stats_from_avg_plot(
      avg_plot,
      focus_count,
      out_worst_bin_idx,
      out_focus_bins);
  };

  const auto flush_optimizer_visual_state =
    [&](bool force, bool can_use_incremental_for_visual) -> bool {
      const auto now = std::chrono::steady_clock::now();
      const bool time_due =
        optimizer_visual_update_min_interval_ms <= 0 ||
        optimizer_last_visual_update_time == std::chrono::steady_clock::time_point{} ||
        std::chrono::duration_cast<std::chrono::milliseconds>(
          now - optimizer_last_visual_update_time).count() >=
          static_cast<long long>(optimizer_visual_update_min_interval_ms);
      const bool sweep_due =
        optimizer_visual_update_sweeps <= 1 ||
        optimizer_sweeps_since_visual_update >= optimizer_visual_update_sweeps;
      if (!force && !(time_due || sweep_due)) {
        return false;
      }
      sync_live_points_to_state(optimize_points);
      if (!(can_use_incremental_for_visual && inc.valid &&
            update_live_hist_in_state_from_incremental(nullptr))) {
        update_live_hist_in_state(optimize_points);
      }
      optimizer_last_visual_update_time = now;
      optimizer_sweeps_since_visual_update = 0;
      return true;
    };

  const auto support_row_for_triangle_idx = [&](int tri_idx) -> int {
    if (tri_idx < 0 ||
        tri_idx >= inc.triangle_count ||
        static_cast<size_t>(tri_idx) >= inc.support_row_for_triangle.size()) {
      return -1;
    }
    return inc.support_row_for_triangle[static_cast<size_t>(tri_idx)];
  };

  const auto load_cached_support_row =
    [&](int support_row, std::vector<int>* out_support) -> bool {
      if (out_support == nullptr) {
        return false;
      }
      if (!state.output_support_denominator_cache_valid ||
          support_row < 0 ||
          support_row >= static_cast<int>(state.output_support_k_denominator_cache.size())) {
        return false;
      }
      out_support->assign(static_cast<size_t>(inc.bin_count), 0);
      const std::vector<int>& cached_row =
        state.output_support_k_denominator_cache[static_cast<size_t>(support_row)];
      const int copy_bins = std::min(inc.bin_count, static_cast<int>(cached_row.size()));
      for (int k = 0; k < copy_bins; ++k) {
        (*out_support)[static_cast<size_t>(k)] = cached_row[static_cast<size_t>(k)];
      }
      return true;
    };

  const auto compute_exact_error_from_incremental_move =
    [&](int point_idx,
        const Eigen::Vector2d& old_uv,
        const Eigen::Vector2d& new_uv,
        int new_tri,
        const std::vector<int>* old_k_cache,
        double* out_raw_connectivity_error = nullptr) -> double {
      if (!inc.valid ||
          point_idx < 0 ||
          point_idx >= inc.point_count ||
          inc.bin_count <= 0) {
        return std::numeric_limits<double>::infinity();
      }
      ExactEvalScratch& scratch = get_exact_eval_scratch();
      scratch.candidate_point_hist.resize(inc.point_hist.size());
      scratch.candidate_point_support.resize(inc.point_support.size());
      for (size_t row = 0; row < inc.point_hist.size(); ++row) {
        scratch.candidate_point_hist[row] = inc.point_hist[row];
        scratch.candidate_point_support[row] = inc.point_support[row];
      }
      std::vector<int>& moved_counts =
        scratch.candidate_point_hist[static_cast<size_t>(point_idx)];
      const int old_support_idx =
        (static_cast<size_t>(point_idx) < inc.point_support_rows.size())
          ? inc.point_support_rows[static_cast<size_t>(point_idx)]
          : -1;
      const int new_support_idx = support_row_for_triangle_idx(new_tri);
      for (int j = 0; j < inc.point_count; ++j) {
        if (j == point_idx) {
          continue;
        }
        const int j_support_idx =
          (static_cast<size_t>(j) < inc.point_support_rows.size())
            ? inc.point_support_rows[static_cast<size_t>(j)]
            : -1;
        const int old_k =
          (old_k_cache != nullptr &&
           j < static_cast<int>(old_k_cache->size()))
            ? (*old_k_cache)[static_cast<size_t>(j)]
            : ((old_support_idx >= 0 && j_support_idx >= 0)
                ? get_support_pairwise_dist(state, old_support_idx, j_support_idx)
            : delaunay_helper->count_triangles_crossed(
                old_uv,
                optimize_points[static_cast<size_t>(j)]));
        const int new_k =
          (new_support_idx >= 0 && j_support_idx >= 0)
            ? get_support_pairwise_dist(state, new_support_idx, j_support_idx)
            : delaunay_helper->count_triangles_crossed(
                new_uv,
                optimize_points[static_cast<size_t>(j)]);
        if (old_k == new_k) {
          continue;
        }
        if (old_k >= 0) {
          const int old_bin = std::min(old_k, inc.bin_count - 1);
          --moved_counts[static_cast<size_t>(old_bin)];
          --scratch.candidate_point_hist[static_cast<size_t>(j)][static_cast<size_t>(old_bin)];
        }
        if (new_k >= 0) {
          const int new_bin = std::min(new_k, inc.bin_count - 1);
          ++moved_counts[static_cast<size_t>(new_bin)];
          ++scratch.candidate_point_hist[static_cast<size_t>(j)][static_cast<size_t>(new_bin)];
        }
      }

      std::vector<int>& new_support = scratch.scratch_support_row;
      new_support.assign(static_cast<size_t>(inc.bin_count), 0);
      bool loaded_from_cache = load_cached_support_row(new_support_idx, &new_support);
      if (!loaded_from_cache) {
        for (const Eigen::Vector2d& support_uv : output_support_uv) {
          const int k = delaunay_helper->count_triangles_crossed(new_uv, support_uv);
          if (k >= 0) {
            const int bin = std::min(k, inc.bin_count - 1);
            ++new_support[static_cast<size_t>(bin)];
          }
        }
      }
      scratch.candidate_point_support[static_cast<size_t>(point_idx)] = new_support;
      const std::vector<int>* candidate_support_rows = nullptr;
      if (structured_support_rows_required()) {
        scratch.candidate_support_rows = inc.point_support_rows;
        if (point_idx >= 0 &&
            point_idx < static_cast<int>(scratch.candidate_support_rows.size())) {
          scratch.candidate_support_rows[static_cast<size_t>(point_idx)] = new_support_idx;
        }
        candidate_support_rows = &scratch.candidate_support_rows;
      }
      return compute_exact_error_from_point_data(
        scratch.candidate_point_hist,
        scratch.candidate_point_support,
        candidate_support_rows,
        out_raw_connectivity_error);
    };

  const auto compute_exact_error_from_incremental_remove =
    [&](int remove_idx,
        double* out_raw_connectivity_error = nullptr) -> double {
      if (!inc.valid ||
          remove_idx < 0 ||
          remove_idx >= inc.point_count ||
          inc.bin_count <= 0 ||
          inc.point_count < 2) {
        return std::numeric_limits<double>::infinity();
      }
      ExactEvalScratch& scratch = get_exact_eval_scratch();
      scratch.candidate_point_hist.resize(inc.point_hist.size());
      scratch.candidate_point_support.resize(inc.point_support.size());
      for (size_t row = 0; row < inc.point_hist.size(); ++row) {
        scratch.candidate_point_hist[row] = inc.point_hist[row];
        scratch.candidate_point_support[row] = inc.point_support[row];
      }
      const Eigen::Vector2d& removed_uv = optimize_points[static_cast<size_t>(remove_idx)];
      const int removed_support_idx =
        (static_cast<size_t>(remove_idx) < inc.point_support_rows.size())
          ? inc.point_support_rows[static_cast<size_t>(remove_idx)]
          : -1;
      for (int j = 0; j < inc.point_count; ++j) {
        if (j == remove_idx) {
          continue;
        }
        const int j_support_idx =
          (static_cast<size_t>(j) < inc.point_support_rows.size())
            ? inc.point_support_rows[static_cast<size_t>(j)]
            : -1;
        const int k =
          (removed_support_idx >= 0 && j_support_idx >= 0)
            ? get_support_pairwise_dist(state, removed_support_idx, j_support_idx)
            : delaunay_helper->count_triangles_crossed(
                removed_uv,
                optimize_points[static_cast<size_t>(j)]);
          if (k >= 0) {
            const int bin = std::min(k, inc.bin_count - 1);
            --scratch.candidate_point_hist[static_cast<size_t>(j)][static_cast<size_t>(bin)];
          }
        }
        scratch.candidate_point_hist.erase(scratch.candidate_point_hist.begin() + remove_idx);
        scratch.candidate_point_support.erase(scratch.candidate_point_support.begin() + remove_idx);
        const std::vector<int>* candidate_support_rows = nullptr;
        if (structured_support_rows_required()) {
          scratch.candidate_support_rows = inc.point_support_rows;
          if (remove_idx >= 0 &&
              remove_idx < static_cast<int>(scratch.candidate_support_rows.size())) {
            scratch.candidate_support_rows.erase(
              scratch.candidate_support_rows.begin() + remove_idx);
          }
          candidate_support_rows = &scratch.candidate_support_rows;
        }
        return compute_exact_error_from_point_data(
          scratch.candidate_point_hist,
          scratch.candidate_point_support,
          candidate_support_rows,
          out_raw_connectivity_error);
      };

  const auto compute_exact_error_from_incremental_add =
    [&](const Eigen::Vector2d& add_uv,
        int support_idx,
        double* out_raw_connectivity_error = nullptr) -> double {
      if (!inc.valid || inc.bin_count <= 0) {
        return std::numeric_limits<double>::infinity();
      }
      ExactEvalScratch& scratch = get_exact_eval_scratch();
      const size_t new_size = inc.point_hist.size() + 1;
      scratch.candidate_point_hist.resize(new_size);
      scratch.candidate_point_support.resize(new_size);
      for (size_t row = 0; row < inc.point_hist.size(); ++row) {
        scratch.candidate_point_hist[row] = inc.point_hist[row];
        scratch.candidate_point_support[row] = inc.point_support[row];
      }
      scratch.candidate_point_hist.back().assign(static_cast<size_t>(inc.bin_count), 0);
      const int new_idx = static_cast<int>(scratch.candidate_point_hist.size()) - 1;
      for (int j = 0; j < inc.point_count; ++j) {
        const int j_support_idx =
          (static_cast<size_t>(j) < inc.point_support_rows.size())
            ? inc.point_support_rows[static_cast<size_t>(j)]
            : -1;
        const int k =
          (support_idx >= 0 && j_support_idx >= 0)
            ? get_support_pairwise_dist(state, support_idx, j_support_idx)
            : delaunay_helper->count_triangles_crossed(
                add_uv,
                optimize_points[static_cast<size_t>(j)]);
        if (k >= 0) {
          const int bin = std::min(k, inc.bin_count - 1);
          ++scratch.candidate_point_hist[static_cast<size_t>(new_idx)][static_cast<size_t>(bin)];
          ++scratch.candidate_point_hist[static_cast<size_t>(j)][static_cast<size_t>(bin)];
        }
      }

      std::vector<int>& new_support = scratch.scratch_support_row;
      new_support.assign(static_cast<size_t>(inc.bin_count), 0);
      bool loaded_from_cache = load_cached_support_row(support_idx, &new_support);
      if (!loaded_from_cache) {
        for (const Eigen::Vector2d& support_uv : output_support_uv) {
          const int k = delaunay_helper->count_triangles_crossed(add_uv, support_uv);
            if (k >= 0) {
              const int bin = std::min(k, inc.bin_count - 1);
              ++new_support[static_cast<size_t>(bin)];
            }
          }
        }
        scratch.candidate_point_support.back() = new_support;
        const std::vector<int>* candidate_support_rows = nullptr;
        if (structured_support_rows_required()) {
          scratch.candidate_support_rows = inc.point_support_rows;
          scratch.candidate_support_rows.push_back(support_idx);
          candidate_support_rows = &scratch.candidate_support_rows;
        }
        return compute_exact_error_from_point_data(
          scratch.candidate_point_hist,
          scratch.candidate_point_support,
          candidate_support_rows,
          out_raw_connectivity_error);
      };

  const auto maybe_apply_adaptive_count_move =
    [&](int triangle_count,
        bool* io_can_use_incremental,
        double* io_current_error,
        double* io_current_exact_error,
        bool force_probe) -> bool {
      const bool can_use_incremental =
        (io_can_use_incremental != nullptr) ? *io_can_use_incremental : false;
      if (!adaptive_count_moves ||
          adaptive_count_period <= 0 ||
          optimize_iteration <= 0 ||
          (!force_probe && (optimize_iteration % adaptive_count_period) != 0) ||
          output_support_uv.empty() ||
          optimize_points.size() < 2) {
        return false;
      }

      const int support_count = static_cast<int>(output_support_uv.size());
      const int current_count = static_cast<int>(optimize_points.size());
      const int target_count = std::max(2, optimization_target_count());
      const int count_deadband = std::max(0, adaptive_count_window);
      const int lower_count_deadband =
        (count_deadband <= 0) ? 0 : std::max(1, count_deadband / 4);
      const int preferred_min_count =
        std::max(2, target_count - lower_count_deadband);
      const int preferred_max_count =
        std::min(support_count, target_count + count_deadband);
      const int min_count = 2;
      const int max_count = support_count;
      if (max_count < min_count) {
        return false;
      }
      const bool allow_delete_this_step = current_count > min_count;
      const double delete_accept_eps =
        static_cast<double>(adaptive_count_accept_eps) *
        std::max(1.0f, adaptive_count_delete_accept_scale);
      constexpr int kAdaptiveCountReversalCooldownSweeps = 6;

      double current_exact_error =
        (io_current_exact_error != nullptr) ? *io_current_exact_error
                                           : std::numeric_limits<double>::infinity();
      double current_total_error =
        (io_current_error != nullptr) ? *io_current_error
                                      : std::numeric_limits<double>::infinity();
      const bool has_incremental_cache =
        can_use_incremental &&
        inc.valid &&
        inc.point_count == current_count &&
        inc.bin_count > 0 &&
        inc.point_hist.size() == optimize_points.size() &&
        inc.point_support.size() == optimize_points.size();
      const double current_raw_connectivity_error =
        has_incremental_cache
          ? compute_cached_raw_connectivity_objective()
          : compute_raw_connectivity_objective_from_uv_points(optimize_points);
        if (!std::isfinite(current_exact_error)) {
          current_exact_error = has_incremental_cache
            ? compute_exact_error_from_point_data(
                inc.point_hist,
                inc.point_support,
                structured_support_rows_required() ? &inc.point_support_rows : nullptr)
            : compute_live_pcf_error(optimize_points);
        }
      if (!std::isfinite(current_total_error)) {
        current_total_error = augment_optimizer_objective(
          current_exact_error,
          current_count,
          current_raw_connectivity_error);
      }
      if (!std::isfinite(current_total_error)) {
        return false;
      }
      const int bin_count = has_incremental_cache ? inc.bin_count : live_hist_bin_count;
      if (bin_count <= 0) {
        return false;
      }
      const bool reversal_guard_active = adaptive_count_reversal_cooldown > 0;
      const int below_preferred_gap =
        std::max(0, preferred_min_count - current_count);
      const int above_preferred_gap =
        std::max(0, current_count - preferred_max_count);
      const double reversal_accept_eps = std::max(
        0.05,
        current_total_error * 0.005);
      const double delete_preference_bias = std::max(
        0.05,
        current_total_error * 0.0025);
      const double delete_gap_bias =
        static_cast<double>(below_preferred_gap) * 0.12;
      const double low_count_delete_bias =
        (current_count <= preferred_min_count)
          ? std::max(
              std::max(0.05, current_total_error * 0.0025),
              static_cast<double>(below_preferred_gap) * 0.18)
          : 0.0;
      const bool use_raw_connectivity_signal =
        has_incremental_cache &&
        inc.raw_connectivity_bins > 0 &&
        !inc.target_raw_connectivity_prototypes.empty() &&
        !inc.target_raw_connectivity_proto_fractions.empty();
      std::vector<float> local_distribution_scratch;
      std::vector<int> support_row_for_triangle_fallback(
        static_cast<size_t>(std::max(0, triangle_count)),
        -1);
      if (!has_incremental_cache) {
        for (size_t si = 0; si < output_support_tri_indices.size(); ++si) {
          const int tri = output_support_tri_indices[si];
          if (tri >= 0 && tri < triangle_count) {
            support_row_for_triangle_fallback[static_cast<size_t>(tri)] =
              static_cast<int>(si);
          }
        }
      }
      const auto current_support_row_for_point = [&](int point_idx) {
        if (point_idx < 0 || point_idx >= current_count) {
          return -1;
        }
        if (has_incremental_cache &&
            static_cast<size_t>(point_idx) < inc.point_support_rows.size()) {
          return inc.point_support_rows[static_cast<size_t>(point_idx)];
        }
        const int tri_idx = optimize_triangle_indices[static_cast<size_t>(point_idx)];
        if (tri_idx < 0 ||
            tri_idx >= triangle_count ||
            static_cast<size_t>(tri_idx) >= support_row_for_triangle_fallback.size()) {
          return -1;
        }
        return support_row_for_triangle_fallback[static_cast<size_t>(tri_idx)];
      };

      const auto build_support_for_add_candidate =
        [&](int support_idx, std::vector<int>* out_support) {
          if (out_support == nullptr) {
            return false;
          }
          out_support->assign(static_cast<size_t>(bin_count), 0);
          if (support_idx < 0 ||
              support_idx >= static_cast<int>(output_support_uv.size())) {
            return false;
          }
          if (state.output_support_denominator_cache_valid &&
              support_idx < static_cast<int>(state.output_support_k_denominator_cache.size())) {
            const std::vector<int>& cached =
              state.output_support_k_denominator_cache[static_cast<size_t>(support_idx)];
            const int copy_bins = std::min(bin_count, static_cast<int>(cached.size()));
            for (int k = 0; k < copy_bins; ++k) {
              (*out_support)[static_cast<size_t>(k)] = cached[static_cast<size_t>(k)];
            }
          } else {
            const Eigen::Vector2d& candidate_uv = output_support_uv[static_cast<size_t>(support_idx)];
            for (const Eigen::Vector2d& support_uv : output_support_uv) {
              const int k = delaunay_helper->count_triangles_crossed(candidate_uv, support_uv);
              if (k >= 0 && k < bin_count) {
                ++(*out_support)[static_cast<size_t>(k)];
              }
            }
          }
          return std::any_of(
            out_support->begin(),
            out_support->end(),
            [](int v) { return v > 0; });
        };

      const auto compute_local_signal =
        [&](const std::vector<int>& counts,
            const std::vector<int>& support,
            int support_row) {
          constexpr double kCountMoveShapeWeight = 0.2;
          const int eval_bins = std::min(
            static_cast<int>(counts.size()),
            static_cast<int>(support.size()));
          local_distribution_scratch.assign(static_cast<size_t>(eval_bins), 0.0f);
          for (int k = 0; k < eval_bins; ++k) {
            const int denom = support[static_cast<size_t>(k)];
            if (denom <= 0) {
              continue;
            }
            const float p =
              static_cast<float>(counts[static_cast<size_t>(k)]) /
              static_cast<float>(denom);
            local_distribution_scratch[static_cast<size_t>(k)] = p;
          }
          const std::vector<float>* target_distribution =
            active_target_distribution_for_support_row(support_row);
          if (target_distribution != nullptr) {
            const int strong_prefix_bins = std::min(
              eval_bins,
              std::max(2, std::min(6, near_field_split_for_bins(live_hist_bin_count))));
            double prefix_error = 0.0;
            double prefix_mass_error = 0.0;
            for (int k = 0; k < strong_prefix_bins; ++k) {
              const double out_v =
                (k < static_cast<int>(local_distribution_scratch.size()))
                  ? static_cast<double>(local_distribution_scratch[static_cast<size_t>(k)])
                  : 0.0;
              const double tgt_v =
                (k < static_cast<int>(target_distribution->size()))
                  ? static_cast<double>((*target_distribution)[static_cast<size_t>(k)])
                  : 0.0;
              const double d = out_v - tgt_v;
              prefix_error += d * d;
              prefix_mass_error += std::abs(d);
            }
            return
              kCountMoveShapeWeight *
              (weighted_distribution_l2(local_distribution_scratch, *target_distribution) +
               12.0 * prefix_error +
               6.0 * prefix_mass_error);
          }
          return
            kCountMoveShapeWeight *
            weighted_distribution_l2(local_distribution_scratch, live_target_hist);
        };
      const auto compute_local_raw_signal =
        [&](const std::vector<int>& counts,
            int support_row) {
          const int eval_bins =
            inc.raw_connectivity_bins > 0
              ? inc.raw_connectivity_bins
              : raw_connectivity_eval_bins_for_hist(live_hist_bin_count);
          const std::vector<int>* target_signature =
            active_target_raw_signature_for_support_row(support_row);
          if (target_signature != nullptr && eval_bins > 0) {
            return compute_raw_connectivity_cost_from_signature(
              build_raw_connectivity_signature(counts, eval_bins),
              build_raw_connectivity_signature(*target_signature, eval_bins));
          }
          return compute_raw_connectivity_cost_from_counts(
            counts,
            inc.target_raw_connectivity_prototypes,
            inc.raw_connectivity_bins,
            nullptr);
        };

      enum class CountMoveType { None, Add, Remove, RemoveBurst };
      CountMoveType best_move = CountMoveType::None;
      double best_exact_error = current_exact_error;
      double best_total_error = current_total_error;
      int best_index = -1;
      std::array<int, 2> best_remove_pair{{-1, -1}};
      int best_tri = -1;
      Eigen::Vector2d best_uv = Eigen::Vector2d::Zero();
      const auto is_better_count_move =
        [&](double candidate_exact_error,
            double candidate_total_error,
            double incumbent_exact_error,
            double incumbent_total_error,
            double eps) {
          if (!std::isfinite(candidate_exact_error) ||
              !std::isfinite(candidate_total_error)) {
            return false;
          }
          if (candidate_total_error + eps < incumbent_total_error) {
            return true;
          }
          if (std::abs(candidate_total_error - incumbent_total_error) <= eps &&
              candidate_exact_error + eps < incumbent_exact_error) {
            return true;
          }
          return false;
        };

      if (current_count < max_count) {
        const bool add_reverses_last_move =
          reversal_guard_active &&
          adaptive_count_last_move_direction < 0;
        const double add_accept_eps = add_reverses_last_move
          ? std::max(
              static_cast<double>(adaptive_count_accept_eps),
              reversal_accept_eps)
          : static_cast<double>(adaptive_count_accept_eps);
        std::vector<int> occupied_by_triangle(static_cast<size_t>(std::max(0, triangle_count)), 0);
        for (int tri_idx : optimize_triangle_indices) {
          if (tri_idx >= 0 && tri_idx < triangle_count) {
            occupied_by_triangle[static_cast<size_t>(tri_idx)] = 1;
          }
        }

        std::vector<int> add_positions;
        add_positions.reserve(output_support_tri_indices.size());
        for (size_t si = 0; si < output_support_tri_indices.size(); ++si) {
          const int tri_idx = output_support_tri_indices[si];
          if (tri_idx < 0 || tri_idx >= triangle_count) {
            continue;
          }
          if (occupied_by_triangle[static_cast<size_t>(tri_idx)] != 0) {
            continue;
          }
          add_positions.push_back(static_cast<int>(si));
        }
        if (!add_positions.empty()) {
          std::vector<std::pair<double, int>> ranked_add_positions;
          ranked_add_positions.reserve(add_positions.size());
          std::vector<int> add_hist(static_cast<size_t>(bin_count), 0);
          std::vector<int> candidate_support;
          for (int support_idx : add_positions) {
            std::fill(add_hist.begin(), add_hist.end(), 0);
            const Eigen::Vector2d& candidate_uv =
              output_support_uv[static_cast<size_t>(support_idx)];
            for (int j = 0; j < current_count; ++j) {
              const int j_support_idx = current_support_row_for_point(j);
              int k =
                (support_idx >= 0 && j_support_idx >= 0)
                  ? get_support_pairwise_dist(state, support_idx, j_support_idx)
                  : -1;
              if (k < 0) {
                k = delaunay_helper->count_triangles_crossed(
                  candidate_uv,
                  optimize_points[static_cast<size_t>(j)]);
              }
              if (k < 0 || k >= bin_count) {
                continue;
              }
              ++add_hist[static_cast<size_t>(k)];
            }
            if (!build_support_for_add_candidate(support_idx, &candidate_support)) {
              continue;
            }
            const double local_signal = compute_local_signal(
              add_hist,
              candidate_support,
              support_idx);
            const double raw_candidate_cost =
              use_raw_connectivity_signal
                ? compute_local_raw_signal(add_hist, support_idx)
                : 0.0;
            const double candidate_score =
              local_signal +
              kOptimizerRawConnectivityRankWeight * raw_candidate_cost;
            ranked_add_positions.emplace_back(candidate_score, support_idx);
          }
          std::sort(
            ranked_add_positions.begin(),
            ranked_add_positions.end(),
            [](const auto& a, const auto& b) {
              if (a.first != b.first) {
                return a.first < b.first;
              }
              return a.second < b.second;
            });
          add_positions.clear();
          add_positions.reserve(ranked_add_positions.size());
          for (const auto& entry : ranked_add_positions) {
            add_positions.push_back(entry.second);
          }
          const int eval_add_count = std::min(
            force_probe ? std::max(adaptive_count_add_proposals, adaptive_count_add_proposals * 2)
                        : adaptive_count_add_proposals,
            static_cast<int>(add_positions.size()));
          for (int c = 0; c < eval_add_count; ++c) {
            const int support_idx = add_positions[static_cast<size_t>(c)];
            std::vector<Eigen::Vector2d> test_points = optimize_points;
            test_points.push_back(output_support_uv[static_cast<size_t>(support_idx)]);
            double candidate_raw_connectivity_error = 0.0;
            const double candidate_error = can_use_incremental
              ? compute_exact_error_from_incremental_add(
                  output_support_uv[static_cast<size_t>(support_idx)],
                  support_idx,
                  &candidate_raw_connectivity_error)
              : compute_live_pcf_error(test_points, &candidate_raw_connectivity_error);
            const double candidate_total_error = augment_optimizer_objective(
              candidate_error,
              static_cast<int>(test_points.size()),
              candidate_raw_connectivity_error);
            if (is_better_count_move(
                  candidate_error,
                  candidate_total_error,
                  best_exact_error,
                  best_total_error,
                  add_accept_eps)) {
              best_exact_error = candidate_error;
              best_total_error = candidate_total_error;
              best_move = CountMoveType::Add;
              best_index = support_idx;
              best_tri = output_support_tri_indices[static_cast<size_t>(support_idx)];
              best_uv = output_support_uv[static_cast<size_t>(support_idx)];
            }
          }
        }
      }

      if (allow_delete_this_step && current_count > min_count) {
        const bool delete_reverses_last_move =
          reversal_guard_active &&
          adaptive_count_last_move_direction > 0;
        double effective_delete_accept_eps = std::max(
          delete_accept_eps,
          delete_preference_bias);
        if (delete_gap_bias > 0.0) {
          effective_delete_accept_eps = std::max(
            effective_delete_accept_eps,
            delete_gap_bias);
        }
        if (delete_reverses_last_move) {
          effective_delete_accept_eps = std::max(
            effective_delete_accept_eps,
            reversal_accept_eps);
        }
        if (low_count_delete_bias > 0.0) {
          effective_delete_accept_eps = std::max(
            effective_delete_accept_eps,
            low_count_delete_bias);
        }
        std::vector<std::pair<double, int>> ranked_delete_candidates;
        ranked_delete_candidates.reserve(static_cast<size_t>(current_count));
        for (int i = 0; i < current_count; ++i) {
          double score = 0.0;
          const int support_row = current_support_row_for_point(i);
          if (has_incremental_cache &&
              inc.has_support[static_cast<size_t>(i)] != 0) {
            score = compute_local_signal(
              inc.point_hist[static_cast<size_t>(i)],
              inc.point_support[static_cast<size_t>(i)],
              support_row);
          }
          if (has_incremental_cache &&
              i >= 0 &&
              i < static_cast<int>(inc.point_hist.size())) {
            score +=
              kOptimizerRawConnectivityRankWeight *
              compute_local_raw_signal(
                inc.point_hist[static_cast<size_t>(i)],
                support_row);
          }
          ranked_delete_candidates.emplace_back(-score, i);
        }
        std::sort(
          ranked_delete_candidates.begin(),
          ranked_delete_candidates.end(),
          [](const auto& a, const auto& b) {
            if (a.first != b.first) {
              return a.first < b.first;
            }
            return a.second < b.second;
          });
        std::vector<int> delete_candidates;
        delete_candidates.reserve(ranked_delete_candidates.size());
        for (const auto& entry : ranked_delete_candidates) {
          delete_candidates.push_back(entry.second);
        }
        if (!has_incremental_cache) {
          std::shuffle(delete_candidates.begin(), delete_candidates.end(), gen);
        }
        const int delete_candidate_bonus =
          std::min(8, std::max(0, below_preferred_gap));
        const int eval_delete_count = std::min(
          (force_probe ? std::max(adaptive_count_delete_candidates, adaptive_count_delete_candidates * 2)
                       : adaptive_count_delete_candidates) +
            delete_candidate_bonus,
          static_cast<int>(delete_candidates.size()));
        for (int c = 0; c < eval_delete_count; ++c) {
          const int remove_idx = delete_candidates[static_cast<size_t>(c)];
          std::vector<Eigen::Vector2d> test_points = optimize_points;
          test_points.erase(test_points.begin() + remove_idx);
          double candidate_raw_connectivity_error = 0.0;
          const double candidate_error = can_use_incremental
            ? compute_exact_error_from_incremental_remove(
                remove_idx,
                &candidate_raw_connectivity_error)
            : compute_live_pcf_error(test_points, &candidate_raw_connectivity_error);
          const double candidate_total_error = augment_optimizer_objective(
            candidate_error,
            static_cast<int>(test_points.size()),
            candidate_raw_connectivity_error);
          if (is_better_count_move(
                candidate_error,
                candidate_total_error,
                best_exact_error,
                best_total_error,
                effective_delete_accept_eps)) {
            best_exact_error = candidate_error;
            best_total_error = candidate_total_error;
            best_move = CountMoveType::Remove;
            best_index = remove_idx;
            best_tri = -1;
          }
        }
        if (below_preferred_gap > 0 && eval_delete_count >= 2) {
          const int burst_seed_count = std::min(
            eval_delete_count,
            std::max(2, adaptive_count_delete_candidates + std::min(4, below_preferred_gap)));
          for (int a = 0; a < burst_seed_count; ++a) {
            const int remove_idx_a = delete_candidates[static_cast<size_t>(a)];
            for (int b = a + 1; b < burst_seed_count; ++b) {
              const int remove_idx_b = delete_candidates[static_cast<size_t>(b)];
              if (remove_idx_a == remove_idx_b) {
                continue;
              }
              std::vector<Eigen::Vector2d> test_points = optimize_points;
              const int first_remove = std::max(remove_idx_a, remove_idx_b);
              const int second_remove = std::min(remove_idx_a, remove_idx_b);
              test_points.erase(test_points.begin() + first_remove);
              test_points.erase(test_points.begin() + second_remove);
              double candidate_raw_connectivity_error = 0.0;
              const double candidate_error =
                compute_live_pcf_error(test_points, &candidate_raw_connectivity_error);
              const double candidate_total_error = augment_optimizer_objective(
                candidate_error,
                static_cast<int>(test_points.size()),
                candidate_raw_connectivity_error);
              if (is_better_count_move(
                    candidate_error,
                    candidate_total_error,
                    best_exact_error,
                    best_total_error,
                    effective_delete_accept_eps)) {
                best_exact_error = candidate_error;
                best_total_error = candidate_total_error;
                best_move = CountMoveType::RemoveBurst;
                best_remove_pair = {{first_remove, second_remove}};
                best_index = second_remove;
                best_tri = -1;
              }
            }
          }
        }
      }

      if (best_move == CountMoveType::None) {
        return false;
      }

      if (best_move == CountMoveType::Remove) {
        optimize_points.erase(optimize_points.begin() + best_index);
        optimize_triangle_indices.erase(optimize_triangle_indices.begin() + best_index);
      } else if (best_move == CountMoveType::RemoveBurst) {
        for (int remove_idx : best_remove_pair) {
          if (remove_idx < 0 ||
              remove_idx >= static_cast<int>(optimize_points.size())) {
            continue;
          }
          optimize_points.erase(optimize_points.begin() + remove_idx);
          optimize_triangle_indices.erase(optimize_triangle_indices.begin() + remove_idx);
        }
      } else if (best_move == CountMoveType::Add) {
        optimize_points.push_back(best_uv);
        optimize_triangle_indices.push_back(best_tri);
      }
      adaptive_count_last_move_direction =
        (best_move == CountMoveType::Add) ? 1 : -1;
      adaptive_count_reversal_cooldown = kAdaptiveCountReversalCooldownSweeps;

      if (io_can_use_incremental) {
        *io_can_use_incremental = false;
      }
      *io_current_error = best_total_error;
      if (io_current_exact_error != nullptr) {
        *io_current_exact_error = best_exact_error;
      }

      return true;
    };

  const auto ensure_triangle_geometry_cache =
    [&](int triangle_count) -> bool {
      if (!delaunay_helper || !delaunay_helper->is_ready() || triangle_count <= 0) {
        tri_geom_cache.valid = false;
        return false;
      }
      if (state.output_boundary_uv_poly.rows() < 3 || state.output_boundary_uv_poly.cols() < 2) {
        tri_geom_cache.valid = false;
        return false;
      }
      const bool cache_is_current =
        tri_geom_cache.valid &&
        tri_geom_cache.triangle_count == triangle_count &&
        static_cast<int>(tri_geom_cache.centers.size()) == triangle_count &&
        static_cast<int>(tri_geom_cache.valid_flags.size()) == triangle_count &&
        static_cast<int>(tri_geom_cache.inside_flags.size()) == triangle_count &&
        same_boundary_polygon(tri_geom_cache.boundary_uv, state.output_boundary_uv_poly);
      if (cache_is_current) {
        return true;
      }

      tri_geom_cache.valid = false;
      tri_geom_cache.triangle_count = triangle_count;
      tri_geom_cache.boundary_uv = state.output_boundary_uv_poly;
      tri_geom_cache.centers.assign(
        static_cast<size_t>(std::max(0, triangle_count)),
        Eigen::Vector2d::Zero());
      tri_geom_cache.valid_flags.assign(static_cast<size_t>(std::max(0, triangle_count)), 0);
      tri_geom_cache.inside_flags.assign(static_cast<size_t>(std::max(0, triangle_count)), 0);
      #pragma omp parallel for schedule(static)
      for (int tri = 0; tri < triangle_count; ++tri) {
        Eigen::Vector2d center;
        if (!delaunay_helper->triangle_center(tri, center)) {
          continue;
        }
        tri_geom_cache.centers[static_cast<size_t>(tri)] = center;
        tri_geom_cache.valid_flags[static_cast<size_t>(tri)] = 1;
        tri_geom_cache.inside_flags[static_cast<size_t>(tri)] =
          point_in_or_on_polygon_for_pcf(center, state.output_boundary_uv_poly) ? 1 : 0;
      }
      tri_geom_cache.valid = true;
      return true;
    };

  if (reset_after_generate_points) {
    optimize_running = false;
    reset_optimizer_runtime_state();
  }

  ImGui::Spacing();
  ImGui::Checkbox("Adaptive count moves", &adaptive_count_moves);
  ImGui::SliderInt("Max iterations", &max_iterations, 100, 50000);
  ImGui::SliderFloat("Empty-bin penalty##pcf", &state.voronoi_pcf_empty_bin_penalty, 0.0f, 5.0f, "%.1fx");
  ImGui::TextDisabled("(weight on pairs in zero-frequency input bins)");
  ImGui::SliderInt("No change iters", &stagnation_patience, 50, 10000);
  early_stop_best_gap_percent = 1.5f;
  early_stop_worst_bin_tol = 0.1f;
  early_stop_soft_no_progress_limit = stagnation_patience;
  early_stop_move_limit = 3;
  hard_near_field_objective = true;
  hard_near_field_split_bin = 0;
  repair_points_per_sweep = 8;
  global_support_proposals_per_point = 8;
  adaptive_count_window = 12;
  adaptive_count_period = 5;
  optimization_budget_ms = 0;
  use_bin_weighting = true;
  use_incremental_optimizer = true;
  early_stop_allow_plateau = false;
  plateau_jitter_enabled = false;
  optimizer_debug_focus = false;
  ImGui::Text("Current output points: %d", static_cast<int>(state.output_pattern_points_uv.size()));
  if (std::isfinite(live_last_worst_bin_residual) && live_last_worst_bin_index >= 0) {
    ImGui::Text(
      "Worst-bin residual: %.6f at k=%d",
      live_last_worst_bin_residual,
      live_last_worst_bin_index);
  }
  ImGui::TextDisabled("Stops on settled convergence, stagnation, or max iterations");

  ImGui::Spacing();
  const bool generated_patch_optimize_batch_current_region =
    generated_patch_batch_run.active &&
    generated_patch_batch_run.action ==
      static_cast<int>(GeneratedPatchBatchAction::Optimize) &&
    generated_patch_batch_run.current_region_offset <
      generated_patch_batch_run.region_ids.size() &&
    generated_patch_batch_run.region_ids[
      generated_patch_batch_run.current_region_offset] == state.region_id;
  if (generated_patch_optimize_batch_current_region &&
      generated_patch_batch_run.current_region_started &&
      root_state.generated_patch_batch_cancel_requested) {
    optimize_running = false;
    clear_generated_patch_batch_run(root_state);
    set_generated_patch_batch_status("Generated patch batch cancelled.", false);
  }
  bool batch_requested_auto_optimize_start = false;
  if (generated_patch_optimize_batch_current_region &&
      !generated_patch_batch_run.current_region_started) {
    if (generated_patch_batch_run.requested_point_count > 0) {
      gen_point_count = generated_patch_batch_run.requested_point_count;
      gen_point_count_initialized = true;
    } else {
      const int batch_auto_point_count =
        (auto_generation_start_count > 0)
          ? auto_generation_start_count
          : ((estimated_target_count > 0)
              ? estimated_target_count
              : gen_point_count);
      if (batch_auto_point_count > 0) {
        gen_point_count = batch_auto_point_count;
        gen_point_count_initialized = true;
      }
    }
    generated_patch_batch_run.current_region_started = true;
    optimize_running = true;
    batch_requested_auto_optimize_start = true;
  }
  bool optimize_checkbox_value =
    optimize_running || state.generated_patch_batch_optimize_requested;
  const bool optimize_toggled = ImGui::Checkbox("Optimize", &optimize_checkbox_value);
  if (optimize_toggled) {
    optimize_running = optimize_checkbox_value;
  }
  if (optimize_toggled && !optimize_checkbox_value &&
      state.generated_patch_batch_optimize_requested) {
    clear_generated_patch_batch_run(root_state);
    set_generated_patch_batch_status("Generated patch batch cancelled.", false);
  }
  if ((optimize_toggled || batch_requested_auto_optimize_start) &&
      optimize_running && state.output_pattern_points_uv.empty()) {
    const bool generated = generate_points_from_support();
    if (generated) {
      // We already reset now; avoid stopping optimization on the next frame.
      reset_optimizer_runtime_state();
      reset_after_generate_points = false;
      std::cout << "No output points found; generated points and started optimization\n";
    } else {
      optimize_running = false;
      std::cout << "Unable to auto-generate points; optimization not started\n";
      if (generated_patch_optimize_batch_current_region) {
        generated_patch_batch_run.current_region_completed = true;
        set_generated_patch_batch_status(
          "Optimization skipped on the current generated patch because point generation failed.",
          true);
      }
    }
  }
  if (optimize_toggled && optimize_running &&
      should_broadcast_to_generated_patch_family()) {
    const std::vector<int> family_region_indices =
      active_generated_patch_family_indices();
    if (family_region_indices.size() > 1 &&
        begin_generated_patch_batch_run(
          root_state,
          state.region_id,
          GeneratedPatchBatchAction::Optimize,
          gen_point_count,
          true,
          false)) {
      std::ostringstream status;
      status << "Optimize queued across " << family_region_indices.size()
             << " generated exemplar patches.";
      set_generated_patch_batch_status(status.str(), false);
    }
  }
  if (optimize_running) {
    ImGui::SameLine();
    if (ImGui::Button("Stop", ImVec2(100, 0))) {
      optimize_running = false;
      if (generated_patch_optimize_batch_current_region) {
        clear_generated_patch_batch_run(root_state);
        set_generated_patch_batch_status("Generated patch batch cancelled.", false);
      }
      std::cout << "Optimization stopped by user\n";
    }
    if (!state.voronoi_pcf_hist_plot.empty()) {
      ImGui::TextDisabled("Adaptive coordinate descent...");
    }
  }

  const auto compute_fast_error_from_sum = [&](const std::vector<float>& sum_dist,
                                               int valid_points,
                                               double raw_connectivity_cost_sum,
                                               const std::vector<int>& current_raw_connectivity_proto_counts,
                                               const std::vector<float>& target_raw_connectivity_proto_fractions) -> double {
    if (valid_points <= 0) {
      return std::numeric_limits<double>::infinity();
    }
    constexpr double kFastShapeWeight = 0.2;
    constexpr double kFastRawConnectivityWeight = 18.0;
    constexpr double kFastRawConnectivityOccupancyFactor = 4.0;
    double shape_error = 0.0;
    const int bins = static_cast<int>(sum_dist.size());
    for (int k = 0; k < bins; ++k) {
      const float bin_weight = use_bin_weighting
        ? (k < bin_weight_transition ? early_bin_weight : late_bin_weight)
        : 1.0f;
      const double avg = static_cast<double>(sum_dist[static_cast<size_t>(k)]) /
                         static_cast<double>(valid_points);
      const double t = (k < static_cast<int>(live_target_hist.size()))
        ? static_cast<double>(live_target_hist[static_cast<size_t>(k)])
        : 0.0;
      const double d = avg - t;
      shape_error += d * d * bin_weight;
    }
    const bool has_raw_connectivity_term =
      !current_raw_connectivity_proto_counts.empty() &&
      !target_raw_connectivity_proto_fractions.empty();
    double raw_connectivity_term = 0.0;
    if (has_raw_connectivity_term) {
      const double avg_raw_cost =
        raw_connectivity_cost_sum / static_cast<double>(valid_points);
      double raw_proto_occupancy_error = 0.0;
      const int proto_count = std::min(
        static_cast<int>(current_raw_connectivity_proto_counts.size()),
        static_cast<int>(target_raw_connectivity_proto_fractions.size()));
      for (int p = 0; p < proto_count; ++p) {
        const double curr_frac =
          static_cast<double>(current_raw_connectivity_proto_counts[static_cast<size_t>(p)]) /
          static_cast<double>(valid_points);
        const double target_frac =
          static_cast<double>(target_raw_connectivity_proto_fractions[static_cast<size_t>(p)]);
        const double d = curr_frac - target_frac;
        raw_proto_occupancy_error += d * d;
      }
      raw_connectivity_term =
        avg_raw_cost +
        kFastRawConnectivityOccupancyFactor * raw_proto_occupancy_error;
    }
    if (!has_raw_connectivity_term) {
      return shape_error;
    }
    return kFastShapeWeight * shape_error +
           kFastRawConnectivityWeight * raw_connectivity_term;
  };

  const auto rebuild_incremental_cache = [&](const std::vector<Eigen::Vector2d>& uv_points,
                                             const std::vector<int>& tri_indices,
                                             int triangle_count) -> bool {
    inc.valid = false;
    inc.bin_count = live_hist_bin_count;
    inc.raw_connectivity_bins = raw_connectivity_eval_bins_for_hist(live_hist_bin_count);
    inc.triangle_count = triangle_count;
    inc.point_count = static_cast<int>(uv_points.size());
    inc.valid_points = 0;
    inc.pair_count = 0;
    inc.fast_error = std::numeric_limits<double>::infinity();
    inc.empty_penalty_sum = 0.0;
    inc.raw_connectivity_cost_sum = 0.0;

    if (inc.bin_count <= 0 || inc.point_count < 2 || triangle_count <= 0) {
      return false;
    }
    if (output_support_uv.empty() ||
        output_support_uv.size() != output_support_tri_indices.size()) {
      return false;
    }

    inc.target_max_bin = -1;
    for (int k = static_cast<int>(live_target_hist.size()) - 1; k >= 0; --k) {
      if (live_target_hist[static_cast<size_t>(k)] > 1e-10f) {
        inc.target_max_bin = k;
        break;
      }
    }
    inc.empty_bin_mask.assign(static_cast<size_t>(inc.bin_count), 0);
    if (inc.target_max_bin >= 0) {
      for (int k = 0; k < inc.bin_count; ++k) {
        if (k > inc.target_max_bin) {
          continue;
        }
        const double t = (k < static_cast<int>(live_target_hist.size()))
          ? static_cast<double>(live_target_hist[static_cast<size_t>(k)])
          : 0.0;
        if (t <= 1e-10) {
          inc.empty_bin_mask[static_cast<size_t>(k)] = 1;
        }
      }
    }

    inc.point_hist.assign(
      static_cast<size_t>(inc.point_count),
      std::vector<int>(static_cast<size_t>(inc.bin_count), 0));
    inc.global_hist_counts.assign(static_cast<size_t>(inc.bin_count), 0);

    inc.point_support.assign(
      static_cast<size_t>(inc.point_count),
      std::vector<int>(static_cast<size_t>(inc.bin_count), 0));
    inc.has_support.assign(static_cast<size_t>(inc.point_count), 0);
    inc.sum_distribution.assign(static_cast<size_t>(inc.bin_count), 0.0f);
    inc.empty_penalty_per_point.assign(static_cast<size_t>(inc.point_count), 0.0);
    inc.target_raw_connectivity_prototypes =
      select_target_raw_connectivity_prototypes(
        state.voronoi_pcf_raw_point_hist_counts,
        inc.raw_connectivity_bins,
        &inc.target_raw_connectivity_proto_fractions);
    inc.point_raw_connectivity_proto_id.assign(static_cast<size_t>(inc.point_count), -1);
    inc.point_raw_connectivity_cost.assign(static_cast<size_t>(inc.point_count), 0.0);
    inc.current_raw_connectivity_proto_counts.assign(
      inc.target_raw_connectivity_prototypes.size(),
      0);

    inc.support_row_for_triangle.assign(static_cast<size_t>(triangle_count), -1);
    for (size_t si = 0; si < output_support_tri_indices.size(); ++si) {
      const int tri = output_support_tri_indices[si];
      if (tri >= 0 && tri < triangle_count) {
        inc.support_row_for_triangle[static_cast<size_t>(tri)] = static_cast<int>(si);
      }
    }
    inc.point_support_rows.assign(static_cast<size_t>(inc.point_count), -1);
    for (int i = 0; i < inc.point_count; ++i) {
      const int tri_idx =
        (i < static_cast<int>(tri_indices.size()))
          ? tri_indices[static_cast<size_t>(i)]
          : -1;
      inc.point_support_rows[static_cast<size_t>(i)] =
        support_row_for_triangle_idx(tri_idx);
    }

    // Build pair histogram in deterministic serial order, using cached
    // support-site distances whenever the optimizer points coincide with
    // triangle-center support sites.
    for (int i = 0; i < inc.point_count; ++i) {
      const int row_i = inc.point_support_rows[static_cast<size_t>(i)];
      for (int j = i + 1; j < inc.point_count; ++j) {
        const int row_j = inc.point_support_rows[static_cast<size_t>(j)];
        const int k =
          (row_i >= 0 && row_j >= 0)
            ? get_support_pairwise_dist(state, row_i, row_j)
            : delaunay_helper->count_triangles_crossed(
                uv_points[static_cast<size_t>(i)],
                uv_points[static_cast<size_t>(j)]);
        if (k >= 0) {
          const int bin = std::min(k, inc.bin_count - 1);
          ++inc.point_hist[static_cast<size_t>(i)][static_cast<size_t>(bin)];
          ++inc.point_hist[static_cast<size_t>(j)][static_cast<size_t>(bin)];
          ++inc.global_hist_counts[static_cast<size_t>(bin)];
          ++inc.pair_count;
        }
      }
    }

    const bool can_use_cached_support =
      state.output_support_denominator_cache_valid &&
      state.output_support_tri_indices_cache.size() == output_support_uv.size() &&
      state.output_support_k_denominator_cache.size() == output_support_uv.size();
    const double empty_bin_weight =
      std::max(0.0, static_cast<double>(state.voronoi_pcf_empty_bin_penalty));

    for (int i = 0; i < inc.point_count; ++i) {
      bool loaded_from_cache = false;
      if (can_use_cached_support) {
        loaded_from_cache = load_cached_support_row(
          inc.point_support_rows[static_cast<size_t>(i)],
          &inc.point_support[static_cast<size_t>(i)]);
      }

      if (!loaded_from_cache) {
        for (const Eigen::Vector2d& support_uv : output_support_uv) {
          const int k = delaunay_helper->count_triangles_crossed(
            uv_points[static_cast<size_t>(i)],
            support_uv);
          if (k >= 0) {
            const int bin = std::min(k, inc.bin_count - 1);
            ++inc.point_support[static_cast<size_t>(i)][static_cast<size_t>(bin)];
          }
        }
      }

      bool has_valid_support = false;
      for (int k = 0; k < inc.bin_count; ++k) {
        if (inc.point_support[static_cast<size_t>(i)][static_cast<size_t>(k)] > 0) {
          has_valid_support = true;
          break;
        }
      }
      if (!has_valid_support) {
        continue;
      }

      inc.has_support[static_cast<size_t>(i)] = 1;
      ++inc.valid_points;

      double penalty_i = 0.0;
      for (int k = 0; k < inc.bin_count; ++k) {
        const int denom = inc.point_support[static_cast<size_t>(i)][static_cast<size_t>(k)];
        if (denom <= 0) {
          continue;
        }
        const float p =
          static_cast<float>(inc.point_hist[static_cast<size_t>(i)][static_cast<size_t>(k)]) /
          static_cast<float>(denom);
        inc.sum_distribution[static_cast<size_t>(k)] += p;
        if (inc.empty_bin_mask[static_cast<size_t>(k)] != 0) {
          const double t = (k < static_cast<int>(live_target_hist.size()))
            ? static_cast<double>(live_target_hist[static_cast<size_t>(k)])
            : 0.0;
          const double d = static_cast<double>(p) - t;
          penalty_i += d * d * empty_bin_weight;
        }
      }
      inc.empty_penalty_per_point[static_cast<size_t>(i)] = penalty_i;
      inc.empty_penalty_sum += penalty_i;
      if (!inc.target_raw_connectivity_prototypes.empty()) {
        int best_raw_proto = -1;
        const double raw_cost = compute_raw_connectivity_cost_from_counts(
          inc.point_hist[static_cast<size_t>(i)],
          inc.target_raw_connectivity_prototypes,
          inc.raw_connectivity_bins,
          &best_raw_proto);
        inc.point_raw_connectivity_proto_id[static_cast<size_t>(i)] = best_raw_proto;
        inc.point_raw_connectivity_cost[static_cast<size_t>(i)] = raw_cost;
        inc.raw_connectivity_cost_sum += raw_cost;
        if (best_raw_proto >= 0 &&
            best_raw_proto < static_cast<int>(inc.current_raw_connectivity_proto_counts.size())) {
          ++inc.current_raw_connectivity_proto_counts[static_cast<size_t>(best_raw_proto)];
        }
      }
    }

    inc.fast_error = compute_fast_error_from_sum(
      inc.sum_distribution,
      inc.valid_points,
      inc.raw_connectivity_cost_sum,
      inc.current_raw_connectivity_proto_counts,
      inc.target_raw_connectivity_proto_fractions);
    inc.valid = (inc.valid_points > 0);
    return inc.valid;
  };

  struct FastEvalScratch {
    std::vector<int> new_counts_i;
    std::vector<int> new_support;
    std::vector<float> sum_dist;
    std::vector<int> raw_proto_counts;
  };

  const auto evaluate_move_fast = [&](int point_idx,
                                      const Eigen::Vector2d& old_uv,
                                      const Eigen::Vector2d& new_uv,
                                      int new_tri,
                                      const std::vector<int>* old_k_cache,
                                      FastEvalScratch* scratch,
                                      bool* out_activates_empty_bin) -> double {
    if (out_activates_empty_bin) {
      *out_activates_empty_bin = false;
    }
    if (!inc.valid || point_idx < 0 || point_idx >= inc.point_count) {
      return std::numeric_limits<double>::infinity();
    }
    if (inc.bin_count <= 0) {
      return std::numeric_limits<double>::infinity();
    }

    const int bin_count = inc.bin_count;
    std::vector<int> local_new_counts_i;
    std::vector<int> local_new_support;
    std::vector<float> local_sum_dist;
    std::vector<int> local_raw_proto_counts;
    std::vector<int>& new_counts_i = scratch ? scratch->new_counts_i : local_new_counts_i;
    std::vector<int>& new_support = scratch ? scratch->new_support : local_new_support;
    std::vector<float>& sum_dist = scratch ? scratch->sum_dist : local_sum_dist;
    std::vector<int>& raw_proto_counts =
      scratch ? scratch->raw_proto_counts : local_raw_proto_counts;

    new_counts_i = inc.point_hist[static_cast<size_t>(point_idx)];
    sum_dist = inc.sum_distribution;
    double empty_penalty_sum = inc.empty_penalty_sum;
    const double empty_bin_weight =
      std::max(0.0, static_cast<double>(state.voronoi_pcf_empty_bin_penalty));
    const bool use_raw_connectivity =
      inc.raw_connectivity_bins > 0 &&
      !inc.target_raw_connectivity_prototypes.empty() &&
      !inc.target_raw_connectivity_proto_fractions.empty();
    double raw_connectivity_cost_sum = inc.raw_connectivity_cost_sum;
    if (use_raw_connectivity) {
      raw_proto_counts = inc.current_raw_connectivity_proto_counts;
    }

    // Pre-resolve support indices for the moved point (old and new positions)
    // so the j-loop can use O(1) pairwise-cache lookups for new_k.
    const int old_support_idx_ef =
      (static_cast<size_t>(point_idx) < optimize_triangle_indices.size() &&
       optimize_triangle_indices[static_cast<size_t>(point_idx)] >= 0 &&
       static_cast<size_t>(optimize_triangle_indices[static_cast<size_t>(point_idx)]) <
         inc.support_row_for_triangle.size())
        ? inc.support_row_for_triangle[
            static_cast<size_t>(optimize_triangle_indices[static_cast<size_t>(point_idx)])]
        : -1;
    const int new_support_idx_ef =
      (new_tri >= 0 &&
       new_tri < inc.triangle_count &&
       static_cast<size_t>(new_tri) < inc.support_row_for_triangle.size())
        ? inc.support_row_for_triangle[static_cast<size_t>(new_tri)]
        : -1;
    (void)old_support_idx_ef; // used below via old_k_cache path

    new_support.assign(static_cast<size_t>(bin_count), 0);
    bool new_has_support = false;
    if (new_tri >= 0 && new_tri < inc.triangle_count) {
      const int row = inc.support_row_for_triangle[static_cast<size_t>(new_tri)];
      if (row >= 0 &&
          row < static_cast<int>(state.output_support_k_denominator_cache.size()) &&
          state.output_support_denominator_cache_valid) {
        const std::vector<int>& cached_row =
          state.output_support_k_denominator_cache[static_cast<size_t>(row)];
        const int copy_bins = std::min(bin_count, static_cast<int>(cached_row.size()));
        for (int k = 0; k < copy_bins; ++k) {
          new_support[static_cast<size_t>(k)] = cached_row[static_cast<size_t>(k)];
        }
        for (int k = 0; k < copy_bins; ++k) {
          if (new_support[static_cast<size_t>(k)] > 0) {
            new_has_support = true;
            break;
          }
        }
      }
    }
    if (!new_has_support) {
      for (const Eigen::Vector2d& support_uv : output_support_uv) {
        const int k = delaunay_helper->count_triangles_crossed(new_uv, support_uv);
        if (k >= 0) {
          const int bin = std::min(k, bin_count - 1);
          ++new_support[static_cast<size_t>(bin)];
        }
      }
      for (int k = 0; k < bin_count; ++k) {
        if (new_support[static_cast<size_t>(k)] > 0) {
          new_has_support = true;
          break;
        }
      }
    }

    const bool old_has_support = (inc.has_support[static_cast<size_t>(point_idx)] != 0);
    int new_valid_points = inc.valid_points +
      (new_has_support ? 1 : 0) - (old_has_support ? 1 : 0);
    if (new_valid_points <= 0) {
      return std::numeric_limits<double>::infinity();
    }

    const auto update_point_bin = [&](int j, int bin, int delta) {
      if (bin < 0 || bin >= bin_count) {
        return;
      }
      if (inc.has_support[static_cast<size_t>(j)] == 0) {
        return;
      }
      const int denom = inc.point_support[static_cast<size_t>(j)][static_cast<size_t>(bin)];
      if (denom <= 0) {
        return;
      }
      const int old_count = inc.point_hist[static_cast<size_t>(j)][static_cast<size_t>(bin)];
      const int new_count = old_count + delta;
      const double old_p = static_cast<double>(old_count) / static_cast<double>(denom);
      const double new_p = static_cast<double>(new_count) / static_cast<double>(denom);
      sum_dist[static_cast<size_t>(bin)] += static_cast<float>(new_p - old_p);
      if (inc.empty_bin_mask[static_cast<size_t>(bin)] != 0) {
        const double t = (bin < static_cast<int>(live_target_hist.size()))
          ? static_cast<double>(live_target_hist[static_cast<size_t>(bin)])
          : 0.0;
        const double old_d = old_p - t;
        const double new_d = new_p - t;
        empty_penalty_sum += (new_d * new_d - old_d * old_d) * empty_bin_weight;
      }
    };

    for (int j = 0; j < inc.point_count; ++j) {
      if (j == point_idx) {
        continue;
      }
      // Resolve j's support index for pairwise-cache lookups.
      const int j_tri_ef =
        (static_cast<size_t>(j) < optimize_triangle_indices.size())
          ? optimize_triangle_indices[static_cast<size_t>(j)]
          : -1;
      const int j_support_idx_ef =
        (j_tri_ef >= 0 && static_cast<size_t>(j_tri_ef) < inc.support_row_for_triangle.size())
          ? inc.support_row_for_triangle[static_cast<size_t>(j_tri_ef)]
          : -1;

      const int old_k =
        (old_k_cache && j < static_cast<int>(old_k_cache->size()))
          ? (*old_k_cache)[static_cast<size_t>(j)]
          : delaunay_helper->count_triangles_crossed(
              old_uv,
              optimize_points[static_cast<size_t>(j)]);
      // Use pairwise cache for new_k when available (O(1) instead of traversal).
      const int new_k =
        (new_support_idx_ef >= 0 && j_support_idx_ef >= 0)
          ? get_support_pairwise_dist(state, new_support_idx_ef, j_support_idx_ef)
          : delaunay_helper->count_triangles_crossed(
              new_uv,
              optimize_points[static_cast<size_t>(j)]);
      if (old_k == new_k) {
        continue;
      }
      if (use_raw_connectivity && inc.has_support[static_cast<size_t>(j)] != 0) {
        const int old_raw_proto_id =
          inc.point_raw_connectivity_proto_id[static_cast<size_t>(j)];
        int new_raw_proto_id = old_raw_proto_id;
        const double new_raw_cost = compute_raw_connectivity_cost_from_counts_with_delta(
          inc.point_hist[static_cast<size_t>(j)],
          old_k,
          new_k,
          inc.target_raw_connectivity_prototypes,
          inc.raw_connectivity_bins,
          &new_raw_proto_id);
        raw_connectivity_cost_sum +=
          new_raw_cost - inc.point_raw_connectivity_cost[static_cast<size_t>(j)];
        if (old_raw_proto_id >= 0 &&
            old_raw_proto_id < static_cast<int>(raw_proto_counts.size())) {
          --raw_proto_counts[static_cast<size_t>(old_raw_proto_id)];
        }
        if (new_raw_proto_id >= 0 &&
            new_raw_proto_id < static_cast<int>(raw_proto_counts.size())) {
          ++raw_proto_counts[static_cast<size_t>(new_raw_proto_id)];
        }
      }
      if (old_k >= 0) {
        const int old_bin = std::min(old_k, bin_count - 1);
        --new_counts_i[static_cast<size_t>(old_bin)];
        update_point_bin(j, old_bin, -1);
      }
      if (new_k >= 0) {
        const int new_bin = std::min(new_k, bin_count - 1);
        ++new_counts_i[static_cast<size_t>(new_bin)];
        update_point_bin(j, new_bin, +1);
      }
    }

    if (old_has_support || new_has_support) {
      const std::vector<int>& old_support =
        inc.point_support[static_cast<size_t>(point_idx)];
      for (int k = 0; k < bin_count; ++k) {
        double old_p = 0.0;
        double new_p = 0.0;
        if (old_has_support) {
          const int denom = old_support[static_cast<size_t>(k)];
          if (denom > 0) {
            old_p = static_cast<double>(inc.point_hist[static_cast<size_t>(point_idx)][static_cast<size_t>(k)]) /
              static_cast<double>(denom);
          }
        }
        if (new_has_support) {
          const int denom = new_support[static_cast<size_t>(k)];
          if (denom > 0) {
            new_p = static_cast<double>(new_counts_i[static_cast<size_t>(k)]) /
              static_cast<double>(denom);
          }
        }
        sum_dist[static_cast<size_t>(k)] += static_cast<float>(new_p - old_p);
      }
    }

    double old_penalty = old_has_support
      ? inc.empty_penalty_per_point[static_cast<size_t>(point_idx)]
      : 0.0;
    double new_penalty = 0.0;
    if (new_has_support) {
      for (int k = 0; k < bin_count; ++k) {
        if (inc.empty_bin_mask[static_cast<size_t>(k)] == 0) {
          continue;
        }
        const int denom = new_support[static_cast<size_t>(k)];
        if (denom <= 0) {
          continue;
        }
        const double p = static_cast<double>(new_counts_i[static_cast<size_t>(k)]) /
          static_cast<double>(denom);
        const double t = (k < static_cast<int>(live_target_hist.size()))
          ? static_cast<double>(live_target_hist[static_cast<size_t>(k)])
          : 0.0;
        const double d = p - t;
        new_penalty += d * d * empty_bin_weight;
      }
    }
    empty_penalty_sum += (new_penalty - old_penalty);

    if (use_raw_connectivity) {
      const int old_raw_proto_id = old_has_support
        ? inc.point_raw_connectivity_proto_id[static_cast<size_t>(point_idx)]
        : -1;
      const double old_raw_cost = old_has_support
        ? inc.point_raw_connectivity_cost[static_cast<size_t>(point_idx)]
        : 0.0;
      int new_raw_proto_id = -1;
      const double new_raw_cost = compute_raw_connectivity_cost_from_counts(
        new_counts_i,
        inc.target_raw_connectivity_prototypes,
        inc.raw_connectivity_bins,
        &new_raw_proto_id);
      raw_connectivity_cost_sum += new_raw_cost - old_raw_cost;
      if (old_raw_proto_id >= 0 &&
          old_raw_proto_id < static_cast<int>(raw_proto_counts.size())) {
        --raw_proto_counts[static_cast<size_t>(old_raw_proto_id)];
      }
      if (new_raw_proto_id >= 0 &&
          new_raw_proto_id < static_cast<int>(raw_proto_counts.size())) {
        ++raw_proto_counts[static_cast<size_t>(new_raw_proto_id)];
      }
    }

    if (out_activates_empty_bin && inc.valid_points > 0 && new_valid_points > 0) {
      constexpr double kEmptyEps = 1e-12;
      for (int k = 0; k < bin_count; ++k) {
        if (inc.empty_bin_mask[static_cast<size_t>(k)] == 0) {
          continue;
        }
        const double curr_avg = static_cast<double>(inc.sum_distribution[static_cast<size_t>(k)]) /
                                static_cast<double>(inc.valid_points);
        const double next_avg = static_cast<double>(sum_dist[static_cast<size_t>(k)]) /
                                static_cast<double>(new_valid_points);
        if (curr_avg <= kEmptyEps && next_avg > kEmptyEps) {
          *out_activates_empty_bin = true;
          break;
        }
      }
    }

    return compute_fast_error_from_sum(
      sum_dist,
      new_valid_points,
      raw_connectivity_cost_sum,
      raw_proto_counts,
      inc.target_raw_connectivity_proto_fractions);
  };

  const auto apply_move_fast = [&](int point_idx,
                                   const Eigen::Vector2d& old_uv,
                                   const Eigen::Vector2d& new_uv,
                                   int new_tri) -> double {
    if (!inc.valid || point_idx < 0 || point_idx >= inc.point_count) {
      return inc.fast_error;
    }
    const int bin_count = inc.bin_count;
    std::vector<int> new_counts_i = inc.point_hist[static_cast<size_t>(point_idx)];
    const double empty_bin_weight =
      std::max(0.0, static_cast<double>(state.voronoi_pcf_empty_bin_penalty));
    const bool use_raw_connectivity =
      inc.raw_connectivity_bins > 0 &&
      !inc.target_raw_connectivity_prototypes.empty() &&
      !inc.target_raw_connectivity_proto_fractions.empty();

    std::vector<int> new_support(static_cast<size_t>(bin_count), 0);
    bool new_has_support = false;
    if (new_tri >= 0 && new_tri < inc.triangle_count) {
      const int row = inc.support_row_for_triangle[static_cast<size_t>(new_tri)];
      if (row >= 0 &&
          row < static_cast<int>(state.output_support_k_denominator_cache.size()) &&
          state.output_support_denominator_cache_valid) {
        const std::vector<int>& cached_row =
          state.output_support_k_denominator_cache[static_cast<size_t>(row)];
        const int copy_bins = std::min(bin_count, static_cast<int>(cached_row.size()));
        for (int k = 0; k < copy_bins; ++k) {
          new_support[static_cast<size_t>(k)] = cached_row[static_cast<size_t>(k)];
        }
        for (int k = 0; k < copy_bins; ++k) {
          if (new_support[static_cast<size_t>(k)] > 0) {
            new_has_support = true;
            break;
          }
        }
      }
    }
    if (!new_has_support) {
      for (const Eigen::Vector2d& support_uv : output_support_uv) {
        const int k = delaunay_helper->count_triangles_crossed(new_uv, support_uv);
        if (k >= 0) {
          const int bin = std::min(k, bin_count - 1);
          ++new_support[static_cast<size_t>(bin)];
        }
      }
      for (int k = 0; k < bin_count; ++k) {
        if (new_support[static_cast<size_t>(k)] > 0) {
          new_has_support = true;
          break;
        }
      }
    }

    const bool old_has_support = (inc.has_support[static_cast<size_t>(point_idx)] != 0);

    // Pre-resolve support indices for pairwise-cache O(1) lookups in the j-loop.
    const int old_support_idx_amf =
      (static_cast<size_t>(point_idx) < inc.point_support_rows.size())
        ? inc.point_support_rows[static_cast<size_t>(point_idx)]
        : -1;
    const int new_support_idx_amf = support_row_for_triangle_idx(new_tri);

    const auto update_point_bin = [&](int j, int bin, int delta) {
      if (bin < 0 || bin >= bin_count) {
        return;
      }
      if (inc.has_support[static_cast<size_t>(j)] == 0) {
        return;
      }
      const int denom = inc.point_support[static_cast<size_t>(j)][static_cast<size_t>(bin)];
      if (denom <= 0) {
        return;
      }
      const int old_count = inc.point_hist[static_cast<size_t>(j)][static_cast<size_t>(bin)];
      const int new_count = old_count + delta;
      const double old_p = static_cast<double>(old_count) / static_cast<double>(denom);
      const double new_p = static_cast<double>(new_count) / static_cast<double>(denom);
      inc.sum_distribution[static_cast<size_t>(bin)] += static_cast<float>(new_p - old_p);
      if (inc.empty_bin_mask[static_cast<size_t>(bin)] != 0) {
        const double t = (bin < static_cast<int>(live_target_hist.size()))
          ? static_cast<double>(live_target_hist[static_cast<size_t>(bin)])
          : 0.0;
        const double old_d = old_p - t;
        const double new_d = new_p - t;
        const double delta_penalty = (new_d * new_d - old_d * old_d) * empty_bin_weight;
        inc.empty_penalty_per_point[static_cast<size_t>(j)] += delta_penalty;
        inc.empty_penalty_sum += delta_penalty;
      }
      inc.point_hist[static_cast<size_t>(j)][static_cast<size_t>(bin)] = new_count;
    };

    for (int j = 0; j < inc.point_count; ++j) {
      if (j == point_idx) {
        continue;
      }
      // Resolve j's support index for pairwise-cache lookups.
      const int j_support_idx_amf =
        (static_cast<size_t>(j) < inc.point_support_rows.size())
          ? inc.point_support_rows[static_cast<size_t>(j)]
          : -1;

      // Use pairwise cache for O(1) lookups when both sides are known support candidates.
      const int old_k =
        (old_support_idx_amf >= 0 && j_support_idx_amf >= 0)
          ? get_support_pairwise_dist(state, old_support_idx_amf, j_support_idx_amf)
          : delaunay_helper->count_triangles_crossed(
              old_uv,
              optimize_points[static_cast<size_t>(j)]);
      const int new_k =
        (new_support_idx_amf >= 0 && j_support_idx_amf >= 0)
          ? get_support_pairwise_dist(state, new_support_idx_amf, j_support_idx_amf)
          : delaunay_helper->count_triangles_crossed(
              new_uv,
              optimize_points[static_cast<size_t>(j)]);
      if (old_k == new_k) {
        continue;
      }
      if (use_raw_connectivity && inc.has_support[static_cast<size_t>(j)] != 0) {
        const int old_raw_proto_id =
          inc.point_raw_connectivity_proto_id[static_cast<size_t>(j)];
        int new_raw_proto_id = old_raw_proto_id;
        const double new_raw_cost = compute_raw_connectivity_cost_from_counts_with_delta(
          inc.point_hist[static_cast<size_t>(j)],
          old_k,
          new_k,
          inc.target_raw_connectivity_prototypes,
          inc.raw_connectivity_bins,
          &new_raw_proto_id);
        inc.raw_connectivity_cost_sum +=
          new_raw_cost - inc.point_raw_connectivity_cost[static_cast<size_t>(j)];
        if (old_raw_proto_id >= 0 &&
            old_raw_proto_id < static_cast<int>(inc.current_raw_connectivity_proto_counts.size())) {
          --inc.current_raw_connectivity_proto_counts[static_cast<size_t>(old_raw_proto_id)];
        }
        if (new_raw_proto_id >= 0 &&
            new_raw_proto_id < static_cast<int>(inc.current_raw_connectivity_proto_counts.size())) {
          ++inc.current_raw_connectivity_proto_counts[static_cast<size_t>(new_raw_proto_id)];
        }
        inc.point_raw_connectivity_proto_id[static_cast<size_t>(j)] = new_raw_proto_id;
        inc.point_raw_connectivity_cost[static_cast<size_t>(j)] = new_raw_cost;
      }
      if (old_k >= 0) {
        const int old_bin = std::min(old_k, bin_count - 1);
        --inc.global_hist_counts[static_cast<size_t>(old_bin)];
        --inc.pair_count;
        --new_counts_i[static_cast<size_t>(old_bin)];
        update_point_bin(j, old_bin, -1);
      }
      if (new_k >= 0) {
        const int new_bin = std::min(new_k, bin_count - 1);
        ++inc.global_hist_counts[static_cast<size_t>(new_bin)];
        ++inc.pair_count;
        ++new_counts_i[static_cast<size_t>(new_bin)];
        update_point_bin(j, new_bin, +1);
      }
    }

    if (old_has_support || new_has_support) {
      const std::vector<int>& old_support =
        inc.point_support[static_cast<size_t>(point_idx)];
      for (int k = 0; k < bin_count; ++k) {
        double old_p = 0.0;
        double new_p = 0.0;
        if (old_has_support) {
          const int denom = old_support[static_cast<size_t>(k)];
          if (denom > 0) {
            old_p = static_cast<double>(inc.point_hist[static_cast<size_t>(point_idx)][static_cast<size_t>(k)]) /
              static_cast<double>(denom);
          }
        }
        if (new_has_support) {
          const int denom = new_support[static_cast<size_t>(k)];
          if (denom > 0) {
            new_p = static_cast<double>(new_counts_i[static_cast<size_t>(k)]) /
              static_cast<double>(denom);
          }
        }
        inc.sum_distribution[static_cast<size_t>(k)] += static_cast<float>(new_p - old_p);
      }
    }

    double old_penalty = old_has_support
      ? inc.empty_penalty_per_point[static_cast<size_t>(point_idx)]
      : 0.0;
    double new_penalty = 0.0;
    if (new_has_support) {
      for (int k = 0; k < bin_count; ++k) {
        if (inc.empty_bin_mask[static_cast<size_t>(k)] == 0) {
          continue;
        }
        const int denom = new_support[static_cast<size_t>(k)];
        if (denom <= 0) {
          continue;
        }
        const double p = static_cast<double>(new_counts_i[static_cast<size_t>(k)]) /
          static_cast<double>(denom);
        const double t = (k < static_cast<int>(live_target_hist.size()))
          ? static_cast<double>(live_target_hist[static_cast<size_t>(k)])
          : 0.0;
        const double d = p - t;
        new_penalty += d * d * empty_bin_weight;
      }
    }
    inc.empty_penalty_per_point[static_cast<size_t>(point_idx)] = new_penalty;
    inc.empty_penalty_sum += (new_penalty - old_penalty);

    inc.valid_points += (new_has_support ? 1 : 0) - (old_has_support ? 1 : 0);
    inc.has_support[static_cast<size_t>(point_idx)] = new_has_support ? 1 : 0;
    if (static_cast<size_t>(point_idx) < inc.point_support_rows.size()) {
      inc.point_support_rows[static_cast<size_t>(point_idx)] = new_support_idx_amf;
    }
    inc.point_support[static_cast<size_t>(point_idx)] = new_support;
    inc.point_hist[static_cast<size_t>(point_idx)] = new_counts_i;
    if (use_raw_connectivity) {
      const int old_raw_proto_id = old_has_support
        ? inc.point_raw_connectivity_proto_id[static_cast<size_t>(point_idx)]
        : -1;
      const double old_raw_cost = old_has_support
        ? inc.point_raw_connectivity_cost[static_cast<size_t>(point_idx)]
        : 0.0;
      int new_raw_proto_id = -1;
      const double new_raw_cost = compute_raw_connectivity_cost_from_counts(
        new_counts_i,
        inc.target_raw_connectivity_prototypes,
        inc.raw_connectivity_bins,
        &new_raw_proto_id);
      inc.raw_connectivity_cost_sum += new_raw_cost - old_raw_cost;
      if (old_raw_proto_id >= 0 &&
          old_raw_proto_id < static_cast<int>(inc.current_raw_connectivity_proto_counts.size())) {
        --inc.current_raw_connectivity_proto_counts[static_cast<size_t>(old_raw_proto_id)];
      }
      if (new_raw_proto_id >= 0 &&
          new_raw_proto_id < static_cast<int>(inc.current_raw_connectivity_proto_counts.size())) {
        ++inc.current_raw_connectivity_proto_counts[static_cast<size_t>(new_raw_proto_id)];
      }
      inc.point_raw_connectivity_proto_id[static_cast<size_t>(point_idx)] = new_raw_proto_id;
      inc.point_raw_connectivity_cost[static_cast<size_t>(point_idx)] = new_raw_cost;
    }

    inc.fast_error = compute_fast_error_from_sum(
      inc.sum_distribution,
      inc.valid_points,
      inc.raw_connectivity_cost_sum,
      inc.current_raw_connectivity_proto_counts,
      inc.target_raw_connectivity_proto_fractions);
    return inc.fast_error;
  };

  
  // Neighbor-based optimization loop
  if (optimize_running && delaunay_helper && delaunay_helper->is_ready()) {
    if (!state.output_pattern_points_uv.empty() && !state.voronoi_pcf_hist_plot.empty()) {
      const int triangle_count = delaunay_helper->triangle_count();
      bool can_use_incremental =
        use_incremental_optimizer &&
        state.output_support_denominator_cache_valid &&
        !output_support_uv.empty() &&
        output_support_uv.size() == output_support_tri_indices.size();
      const VoronoiOptimizerMode optimizer_mode = current_optimizer_mode();
      // Initialize optimization state if needed
      if (optimize_points.empty() ||
          optimize_triangle_indices.empty() ||
          optimize_triangle_indices.size() != optimize_points.size()) {
        optimize_points = state.output_pattern_points_uv;
        optimize_triangle_indices.clear();
        optimize_triangle_indices.resize(optimize_points.size(), -1);
        inc.valid = false;
        
        // Initialize target histogram and parameters for error computation
        live_hist_bin_count = state.voronoi_pcf_bin_count;
        live_target_hist = state.voronoi_pcf_hist_plot;
        live_target_individual_distributions = state.voronoi_pcf_individual_plots;
        live_target_raw_connectivity_signatures = state.voronoi_pcf_raw_point_hist_counts;
        state.voronoi_pcf_position_targets_enabled = false;
        live_position_targets_enabled = false;
        state.baseline_graph_point_count = optimization_target_count();
        
        // Find triangle indices for each point
        for (size_t i = 0; i < optimize_points.size(); ++i) {
          int tri_idx = -1;
          Eigen::Vector3i tri_verts(-1, -1, -1);
          if (delaunay_helper->find_containing_triangle(optimize_points[i], tri_idx, tri_verts)) {
            optimize_triangle_indices[i] = tri_idx;
          }
        }

        if (can_use_incremental) {
          if (!rebuild_incremental_cache(optimize_points, optimize_triangle_indices, triangle_count)) {
            can_use_incremental = false;
          }
        }

        double initial_raw_connectivity_error =
          can_use_incremental ? compute_cached_raw_connectivity_objective() : 0.0;
        const double initial_exact_error =
          can_use_incremental
            ? compute_exact_error_from_point_data(
                inc.point_hist,
                inc.point_support,
                structured_support_rows_required() ? &inc.point_support_rows : nullptr,
                &initial_raw_connectivity_error)
            : compute_live_pcf_error(
                optimize_points,
                &initial_raw_connectivity_error);
        last_generated_exact_error = initial_exact_error;
        optimize_best_error = augment_optimizer_objective(
          initial_exact_error,
          static_cast<int>(optimize_points.size()),
          initial_raw_connectivity_error);
        optimize_swaps_made = 0;
        global_best_error = optimize_best_error;
        optimize_current_exact_error = initial_exact_error;
        optimize_current_total_error = optimize_best_error;
        optimize_current_error_valid = true;
        optimize_trust_strategic_start =
          std::isfinite(initial_exact_error) &&
          initial_exact_error <= 120.0 * static_cast<double>(optimize_points.size());
        optimize_best_points = optimize_points;
        optimize_best_triangle_indices = optimize_triangle_indices;
        optimize_initial_error = optimize_best_error;
        state.output_voronoi_pcf_energy = optimize_best_error;
        state.output_voronoi_objective_energy = optimize_best_error;
        no_progress_iters = 0;
        optimize_iteration = 0;
        state.optimizer_iterations_ran = 0;
        if (can_use_incremental && inc.valid) {
          live_last_worst_bin_residual = compute_live_bin_residual_stats_from_incremental(
            live_worst_bin_focus_count,
            &live_last_worst_bin_index,
            nullptr,
            nullptr);
        } else {
          live_last_worst_bin_residual = compute_live_bin_residual_stats(
            optimize_points,
            live_worst_bin_focus_count,
            &live_last_worst_bin_index,
            nullptr,
            nullptr);
        }
        
        root_state.task_start_time = std::chrono::steady_clock::now();
        optimizer_last_visual_update_time = root_state.task_start_time;
        optimizer_sweeps_since_visual_update = 0;

        std::cout << "Adaptive coordinate descent started: " << optimize_points.size() 
                  << " points, mode=" << optimizer_mode_label()
                  << ", initial exact=" << initial_exact_error
                  << ", raw-connectivity=" << initial_raw_connectivity_error
                  << ", initial error=" << optimize_best_error 
                  << ", target count=" << optimization_target_count()
                  << ", worst-bin residual=" << live_last_worst_bin_residual
            << " (uphill tolerance=" << uphill_tolerance_percent << "%)\n";
      }
      sync_optimizer_point_priority_state();
      
      // === STOCHASTIC COORDINATE DESCENT ===
      // Randomize point order, accept near-improvements for faster convergence
      
      int points_moved_this_iter = 0;
      int sweep_valid_candidates = 0;
      int sweep_exact_scored_candidates = 0;
      int sweep_accepted_moves = 0;
      int sweep_relaxed_moves = 0;
      int sweep_count_changes = 0;
      bool sweep_relaxed_move_used = false;
      std::vector<char> point_priority_state_refreshed_this_sweep(
        optimize_points.size(),
        0);
      std::vector<double> point_priority_scores(
        optimize_points.size(),
        0.0);
      std::vector<int> point_local_proposal_radii(
        optimize_points.size(),
        1);
      
      // Compute current error once.
      double current_global_error = std::numeric_limits<double>::infinity();
      if (can_use_incremental) {
        if (!inc.valid ||
            inc.point_count != static_cast<int>(optimize_points.size()) ||
            inc.bin_count != live_hist_bin_count ||
            inc.triangle_count != triangle_count) {
          if (!rebuild_incremental_cache(optimize_points, optimize_triangle_indices, triangle_count)) {
            can_use_incremental = false;
          }
        }
      }
      if (can_use_incremental && optimize_current_error_valid) {
        current_global_error = optimize_current_total_error;
      } else if (can_use_incremental) {
        current_global_error = inc.fast_error;
      }
      double current_raw_connectivity_error = 0.0;
      double current_exact_error = std::numeric_limits<double>::infinity();
      if (optimize_current_error_valid) {
        current_exact_error = optimize_current_exact_error;
        current_global_error = optimize_current_total_error;
      } else if (can_use_incremental) {
        current_raw_connectivity_error = compute_cached_raw_connectivity_objective();
        current_exact_error = compute_exact_error_from_point_data(
          inc.point_hist,
          inc.point_support,
          structured_support_rows_required() ? &inc.point_support_rows : nullptr);
      } else {
        current_exact_error = compute_live_pcf_error(
          optimize_points,
          &current_raw_connectivity_error);
      }
      if (!std::isfinite(current_exact_error)) {
        optimize_running = false;
        std::cout << "Stopped: exact objective became non-finite (Took " << std::chrono::duration<double>(std::chrono::steady_clock::now() - root_state.task_start_time).count() << "s)\n";
      }
      if (!optimize_current_error_valid) {
        current_global_error = augment_optimizer_objective(
          current_exact_error,
          static_cast<int>(optimize_points.size()),
          current_raw_connectivity_error);
      }
      optimize_current_exact_error = current_exact_error;
      optimize_current_total_error = current_global_error;
      optimize_current_error_valid = true;
      const double sweep_start_error = current_global_error;
      const double uphill_tolerance_fraction =
        std::max(0.0, static_cast<double>(uphill_tolerance_percent)) / 100.0;
      
      // Track global best error.
      const double best_error_before_sweep = global_best_error;
      if (current_global_error + stagnation_best_improve_eps < global_best_error) {
        global_best_error = current_global_error;
        optimize_best_points = optimize_points;
        optimize_best_triangle_indices = optimize_triangle_indices;
      }
      
      // Acceptance threshold: allow moves up to tolerance% worse
      std::vector<int> occupancy(static_cast<size_t>(std::max(0, triangle_count)), 0);
      for (int tri_idx : optimize_triangle_indices) {
        if (tri_idx >= 0 && tri_idx < triangle_count) {
          ++occupancy[static_cast<size_t>(tri_idx)];
        }
      }

      if (!ensure_triangle_geometry_cache(triangle_count)) {
        optimize_running = false;
        std::cout << "Stopped: unable to build triangle geometry cache (Took " 
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - root_state.task_start_time).count() << "s)\n";
      }
      const std::vector<Eigen::Vector2d>& triangle_center_cache = tri_geom_cache.centers;
      const std::vector<char>& triangle_center_valid = tri_geom_cache.valid_flags;
      const std::vector<char>& triangle_center_inside = tri_geom_cache.inside_flags;

      const int early_exploration_threshold = std::clamp(adaptive_count_period, 2, 5);
      int aggressive_level = 0;
      if (no_progress_iters >= early_exploration_threshold * 2) {
        aggressive_level = 2;
      } else if (no_progress_iters >= early_exploration_threshold) {
        aggressive_level = 1;
      }
      const int effective_repair_points_per_sweep =
        (repair_points_per_sweep <= 0)
          ? static_cast<int>(optimize_points.size())
          : std::min(
              static_cast<int>(optimize_points.size()),
              repair_points_per_sweep *
                (aggressive_level >= 2 ? 3 : (aggressive_level == 1 ? 2 : 1)));
      constexpr int kExpandedSweepCadence = 10;
      const int expanded_sweep_threshold = early_exploration_threshold * 3;
      const bool use_full_sweep_search =
        no_progress_iters >= expanded_sweep_threshold &&
        ((no_progress_iters - expanded_sweep_threshold) % kExpandedSweepCadence == 0);
      const int base_global_support_proposals_per_point =
        use_full_sweep_search
          ? std::max(2, (global_support_proposals_per_point + 1) / 2)
          : std::max(1, global_support_proposals_per_point / 4);
      const int global_support_bonus =
        use_full_sweep_search
          ? (aggressive_level >= 2 ? 6 : (aggressive_level == 1 ? 3 : 0))
          : (aggressive_level >= 2 ? 2 : (aggressive_level == 1 ? 1 : 0));
      const int effective_global_support_proposals_per_point = std::min(
        16,
        base_global_support_proposals_per_point + global_support_bonus);
      const int effective_plateau_jitter_points = std::min(
        32,
        plateau_jitter_points + (aggressive_level >= 2 ? 4 : (aggressive_level == 1 ? 2 : 0)));
      const int effective_plateau_jitter_proposals = std::min(
        96,
        plateau_jitter_proposals +
          (aggressive_level >= 2 ? 16 : (aggressive_level == 1 ? 8 : 0)));
      constexpr float kMovedPointPriorityPenaltyBase = 0.08f;
      constexpr float kMovedPointPriorityPenaltyMax = 0.20f;
      constexpr float kMovedPointPriorityPenaltyDecay = 0.55f;
      constexpr int kMovedPointPriorityCooldownSweeps = 2;
      constexpr float kNearbyPointPriorityBoostBase = 0.10f;
      constexpr float kNearbyPointPriorityBoostMax = 0.24f;
      constexpr float kNearbyPointPriorityBoostDecay = 0.65f;
      constexpr int kNearbyPointPriorityBoostSweeps = 2;
      constexpr int kHighOrderProposalRadiusThreshold = 3;
      constexpr int kNearbyPriorityBoostCount = 4;
      constexpr int kPairMovePartnerCount = 3;
      constexpr int kPairMoveCandidateCountPerPoint = 2;
      const auto compute_individual_distribution_match_cost =
        [&](const std::vector<float>& out_dist,
            const std::vector<float>& tgt_dist) {
          const int m = std::min(
            static_cast<int>(tgt_dist.size()),
            live_hist_bin_count);
          const double weighted_error = weighted_distribution_l2(out_dist, tgt_dist);
          const int strong_prefix_bins = std::min(
            m,
            std::max(2, std::min(6, near_field_split_for_bins(live_hist_bin_count))));
          double prefix_error = 0.0;
          double prefix_mass_error = 0.0;
          for (int k = 0; k < strong_prefix_bins; ++k) {
            const double out_v =
              (k < static_cast<int>(out_dist.size()))
                ? static_cast<double>(out_dist[static_cast<size_t>(k)])
                : 0.0;
            const double tgt_v =
              (k < static_cast<int>(tgt_dist.size()))
                ? static_cast<double>(tgt_dist[static_cast<size_t>(k)])
                : 0.0;
            const double d = out_v - tgt_v;
            prefix_error += d * d;
            prefix_mass_error += std::abs(d);
          }
          return weighted_error +
                 12.0 * prefix_error +
                 6.0 * prefix_mass_error;
        };
      const auto find_best_target_distribution_match =
        [&](const std::vector<float>& current_distribution,
            const std::vector<float>** best_target_distribution,
            double* best_target_cost) {
          const std::vector<float>* local_best_target_distribution = nullptr;
          double local_best_target_cost = std::numeric_limits<double>::infinity();
          if (!live_target_individual_distributions.empty()) {
            for (const auto& target_distribution : live_target_individual_distributions) {
              const double cost = compute_individual_distribution_match_cost(
                current_distribution,
                target_distribution);
              if (cost < local_best_target_cost) {
                local_best_target_cost = cost;
                local_best_target_distribution = &target_distribution;
              }
            }
          }
          if (local_best_target_distribution == nullptr &&
              !live_target_hist.empty()) {
            local_best_target_distribution = &live_target_hist;
            local_best_target_cost = compute_individual_distribution_match_cost(
              current_distribution,
              live_target_hist);
          }
          if (best_target_distribution != nullptr) {
            *best_target_distribution = local_best_target_distribution;
          }
          if (best_target_cost != nullptr) {
            *best_target_cost = local_best_target_cost;
          }
          return local_best_target_distribution != nullptr;
        };
      const auto target_distribution_for_support_row =
        [&](int support_row) -> const std::vector<float>* {
          return active_target_distribution_for_support_row(support_row);
        };
      const auto find_point_target_distribution_match =
        [&](size_t point_idx,
            const std::vector<float>& current_distribution,
            const std::vector<float>** best_target_distribution,
            double* best_target_cost) {
          const int support_row =
            point_idx < inc.point_support_rows.size()
              ? inc.point_support_rows[point_idx]
              : -1;
          const std::vector<float>* support_row_target =
            target_distribution_for_support_row(support_row);
          if (support_row_target != nullptr) {
            const double support_row_cost = compute_individual_distribution_match_cost(
              current_distribution,
              *support_row_target);
            if (best_target_distribution != nullptr) {
              *best_target_distribution = support_row_target;
            }
            if (best_target_cost != nullptr) {
              *best_target_cost = support_row_cost;
            }
            return true;
          }
          return find_best_target_distribution_match(
            current_distribution,
            best_target_distribution,
            best_target_cost);
        };
      const auto target_raw_signature_for_support_row =
        [&](int support_row) -> const std::vector<int>* {
          return active_target_raw_signature_for_support_row(support_row);
        };
      const auto compute_point_raw_connectivity_priority_cost =
        [&](const std::vector<int>& point_counts,
            int support_row) {
          const std::vector<int>* target_signature =
            target_raw_signature_for_support_row(support_row);
          const int eval_bins =
            inc.raw_connectivity_bins > 0
              ? inc.raw_connectivity_bins
              : raw_connectivity_eval_bins_for_hist(live_hist_bin_count);
          if (target_signature != nullptr && eval_bins > 0) {
            return compute_raw_connectivity_cost_from_signature(
              build_raw_connectivity_signature(point_counts, eval_bins),
              build_raw_connectivity_signature(*target_signature, eval_bins));
          }
          return compute_raw_connectivity_cost_from_counts(
            point_counts,
            inc.target_raw_connectivity_prototypes,
            inc.raw_connectivity_bins,
            nullptr);
        };
      const auto compute_cluster_priority_bonus =
        [&](const std::vector<float>& current_distribution,
            const std::vector<float>& target_distribution,
            int eval_bins) {
          if (eval_bins <= 0 ||
              target_distribution.empty()) {
            return 0.0;
          }

          const int strong_prefix_bins = std::min(
            eval_bins,
            std::max(2, std::min(6, near_field_split_for_bins(live_hist_bin_count))));
          if (strong_prefix_bins <= 0) {
            return 0.0;
          }

          double target_weighted_mass = 0.0;
          double deficit_weighted_mass = 0.0;
          double excess_weighted_mass = 0.0;
          double target_peak = 0.0;
          for (int k = 0; k < strong_prefix_bins; ++k) {
            const double out_v =
              (k < static_cast<int>(current_distribution.size()))
                ? static_cast<double>(current_distribution[static_cast<size_t>(k)])
                : 0.0;
            const double tgt_v =
              (k < static_cast<int>(target_distribution.size()))
                ? static_cast<double>(target_distribution[static_cast<size_t>(k)])
                : 0.0;
            const double proximity_weight =
              1.0 +
              0.7 *
                (1.0 -
                 static_cast<double>(k) /
                   static_cast<double>(std::max(1, strong_prefix_bins - 1)));
            target_weighted_mass += proximity_weight * tgt_v;
            deficit_weighted_mass +=
              proximity_weight * std::max(0.0, tgt_v - out_v);
            excess_weighted_mass +=
              proximity_weight * std::max(0.0, out_v - tgt_v);
            target_peak = std::max(target_peak, tgt_v);
          }

          if (target_weighted_mass <= 1e-9 && target_peak <= 1e-9) {
            return 0.0;
          }

          // Missing near neighbors should matter more as the matched target row
          // becomes more strongly clustered. Extra near neighbors still matter,
          // but remain a smaller corrective signal.
          const double order_weight =
            0.4 + 0.95 * target_weighted_mass + 0.55 * target_peak;
          const double crowd_weight =
            0.25 + 0.35 * target_weighted_mass + 0.20 * target_peak;
          const double bonus =
            order_weight * deficit_weighted_mass +
            0.5 * crowd_weight * excess_weighted_mass;
          constexpr double kMaxClusterPriorityBonus = 4.0;
          return std::min(kMaxClusterPriorityBonus, bonus);
        };
      const auto compute_local_proposal_radius =
        [&](size_t point_idx) {
          if (!can_use_incremental ||
              !inc.valid) {
            return 1;
          }
          if (point_idx >= inc.point_hist.size() ||
              point_idx >= inc.point_support.size()) {
            return 1;
          }
          if (point_idx >= inc.has_support.size() ||
              inc.has_support[point_idx] == 0) {
            return 2;
          }
          const int eval_bins = std::min(
            static_cast<int>(inc.point_hist[point_idx].size()),
            static_cast<int>(inc.point_support[point_idx].size()));
          if (eval_bins <= 0) {
            return 1;
          }

          std::vector<float> current_distribution(
            static_cast<size_t>(eval_bins),
            0.0f);
          for (int k = 0; k < eval_bins; ++k) {
            const int denom = inc.point_support[point_idx][static_cast<size_t>(k)];
            if (denom <= 0) {
              continue;
            }
            current_distribution[static_cast<size_t>(k)] =
              static_cast<float>(
                static_cast<double>(inc.point_hist[point_idx][static_cast<size_t>(k)]) /
                static_cast<double>(denom));
          }

          const std::vector<float>* best_target_distribution = nullptr;
      if (!find_point_target_distribution_match(
        point_idx,
                current_distribution,
                &best_target_distribution,
                nullptr) ||
              best_target_distribution == nullptr) {
            return 1;
          }

          return infer_local_proposal_radius_from_distributions(
            current_distribution,
            *best_target_distribution,
            eval_bins,
            live_hist_bin_count);
        };
      const auto compute_incremental_local_exact_budget =
        [&](int local_count, int local_radius) {
          if (local_count <= 0) {
            return 0;
          }
          if (local_count <= 8) {
            return local_count;
          }
          int budget = 6 + 2 * std::max(0, local_radius - 1);
          if (use_full_sweep_search) {
            budget += 2;
          }
          if (aggressive_level >= 1) {
            ++budget;
          }
          if (aggressive_level >= 2) {
            budget += 2;
          }
          return std::min(local_count, std::max(6, budget));
        };
      // Create randomized order and widen to full sweeps after a few dead iterations.
      std::vector<size_t> point_order;
      point_order.reserve(optimize_points.size());
      if (optimize_running) {
        const int requested_repair_points =
          use_full_sweep_search
            ? std::min(
                static_cast<int>(optimize_points.size()),
                std::max(
                  effective_repair_points_per_sweep,
                  effective_repair_points_per_sweep *
                    (aggressive_level >= 2 ? 3 : 2)))
            : effective_repair_points_per_sweep;
        if (can_use_incremental &&
            inc.valid &&
            inc.point_count == static_cast<int>(optimize_points.size()) &&
            inc.point_support.size() == optimize_points.size() &&
            inc.point_hist.size() == optimize_points.size()) {
          std::vector<std::pair<double, size_t>> ranked_points;
          ranked_points.reserve(optimize_points.size());
          std::vector<float> point_distribution_scratch;
          for (size_t i = 0; i < optimize_points.size(); ++i) {
            const int tri_idx = optimize_triangle_indices[i];
            if (tri_idx < 0) {
              continue;
            }
            point_local_proposal_radii[i] = compute_local_proposal_radius(i);
            double score = 0.0;
            if (inc.has_support[i] == 0) {
              score = std::numeric_limits<double>::infinity();
            } else {
              constexpr double kRepairRankingShapeWeight = 0.25;
              constexpr double kRepairRankingClusterPriorityWeight = 0.9;
              const int support_row =
                i < inc.point_support_rows.size()
                  ? inc.point_support_rows[i]
                  : -1;
              const int eval_bins = std::min(
                static_cast<int>(inc.point_hist[i].size()),
                static_cast<int>(inc.point_support[i].size()));
              point_distribution_scratch.assign(static_cast<size_t>(eval_bins), 0.0f);
              for (int k = 0; k < eval_bins; ++k) {
                const int denom = inc.point_support[i][static_cast<size_t>(k)];
                if (denom <= 0) {
                  continue;
                }
                point_distribution_scratch[static_cast<size_t>(k)] =
                  static_cast<float>(inc.point_hist[i][static_cast<size_t>(k)]) /
                  static_cast<float>(denom);
              }
              const std::vector<float>* best_target_distribution = nullptr;
              double point_target_cost = weighted_distribution_l2(
                point_distribution_scratch,
                live_target_hist);
              if (find_point_target_distribution_match(
                    i,
                    point_distribution_scratch,
                    &best_target_distribution,
                    &point_target_cost)) {
                score = kRepairRankingShapeWeight * point_target_cost;
              } else {
                score =
                  kRepairRankingShapeWeight *
                  weighted_distribution_l2(point_distribution_scratch, live_target_hist);
              }
              if (i < inc.point_hist.size()) {
                score +=
                  kOptimizerRawConnectivityRankWeight *
                  compute_point_raw_connectivity_priority_cost(
                    inc.point_hist[i],
                    support_row);
              }
              if (best_target_distribution != nullptr) {
                score +=
                  kRepairRankingClusterPriorityWeight *
                  compute_cluster_priority_bonus(
                    point_distribution_scratch,
                    *best_target_distribution,
                    eval_bins);
              }
              if (i < optimize_point_priority_penalty.size() &&
                  i < optimize_point_priority_cooldown.size() &&
                  optimize_point_priority_cooldown[i] > 0) {
                const double penalty =
                  std::clamp(
                    static_cast<double>(optimize_point_priority_penalty[i]),
                    0.0,
                    static_cast<double>(kMovedPointPriorityPenaltyMax));
                score *= (1.0 - penalty);
              }
              if (i < optimize_point_priority_boost.size() &&
                  i < optimize_point_priority_boost_cooldown.size() &&
                  optimize_point_priority_boost_cooldown[i] > 0) {
                const double boost =
                  std::clamp(
                    static_cast<double>(optimize_point_priority_boost[i]),
                    0.0,
                    static_cast<double>(kNearbyPointPriorityBoostMax));
                score *= (1.0 + boost);
              }
            }
            point_priority_scores[i] = score;
            ranked_points.emplace_back(score, i);
          }
          const size_t keep_count = static_cast<size_t>(
            std::max(1, std::min(requested_repair_points, static_cast<int>(ranked_points.size()))));
          if (keep_count < ranked_points.size()) {
            std::partial_sort(
              ranked_points.begin(),
              ranked_points.begin() + static_cast<std::ptrdiff_t>(keep_count),
              ranked_points.end(),
              [](const auto& a, const auto& b) {
                if (a.first != b.first) {
                  return a.first > b.first;
                }
                return a.second < b.second;
              });
            ranked_points.resize(keep_count);
          } else {
            std::sort(
              ranked_points.begin(),
              ranked_points.end(),
              [](const auto& a, const auto& b) {
                if (a.first != b.first) {
                  return a.first > b.first;
                }
                return a.second < b.second;
              });
          }
          point_order.reserve(ranked_points.size());
          for (const auto& entry : ranked_points) {
            point_order.push_back(entry.second);
          }
        } else {
          point_order.resize(optimize_points.size());
          std::iota(point_order.begin(), point_order.end(), 0);
          std::shuffle(point_order.begin(), point_order.end(), gen);
          if (requested_repair_points < static_cast<int>(point_order.size())) {
            point_order.resize(static_cast<size_t>(requested_repair_points));
          }
        }
        std::shuffle(point_order.begin(), point_order.end(), gen);
      }

      // Iterate through all points in random order
      for (size_t idx = 0; idx < point_order.size(); ++idx) {
        size_t i = point_order[idx];
        const int tri_idx = optimize_triangle_indices[i];
        if (tri_idx < 0) continue;

        // Hybrid proposal pool: local ring proposals + random global support triangles.
        const int local_proposal_radius =
          compute_local_proposal_radius(i);
        std::vector<int> proposal_tris;
        proposal_tris.reserve(
          static_cast<size_t>(
            6 * std::max(1, local_proposal_radius) +
            std::max(0, effective_global_support_proposals_per_point)));
        const auto append_unique_tri = [&](int tri_candidate) {
          if (tri_candidate < 0) {
            return;
          }
          if (std::find(proposal_tris.begin(), proposal_tris.end(), tri_candidate) == proposal_tris.end()) {
            proposal_tris.push_back(tri_candidate);
          }
        };
        std::vector<int> frontier;
        std::vector<int> next_frontier;
        frontier.reserve(8);
        next_frontier.reserve(16);
        frontier.push_back(tri_idx);
        for (int hop = 0; hop < local_proposal_radius; ++hop) {
          next_frontier.clear();
          for (int frontier_tri : frontier) {
            std::array<int, 3> neighbors;
            delaunay_helper->get_triangle_neighbors(frontier_tri, neighbors);
            for (int n : neighbors) {
              if (n < 0) {
                continue;
              }
              const size_t proposal_count_before = proposal_tris.size();
              append_unique_tri(n);
              if (proposal_tris.size() != proposal_count_before) {
                next_frontier.push_back(n);
              }
            }
          }
          if (next_frontier.empty()) {
            break;
          }
          frontier.swap(next_frontier);
        }
        const size_t local_proposal_prefix = proposal_tris.size();
        if (effective_global_support_proposals_per_point > 0 &&
            !output_support_tri_indices.empty() &&
            output_support_tri_indices.size() == output_support_uv.size()) {
          std::uniform_int_distribution<int> support_pick(
            0,
            static_cast<int>(output_support_tri_indices.size()) - 1);
          for (int gp = 0; gp < effective_global_support_proposals_per_point; ++gp) {
            append_unique_tri(output_support_tri_indices[static_cast<size_t>(support_pick(gen))]);
          }
        }
        // Precompute old crossed-bin indices for this source point once.
        // Use the pairwise support cache when available (O(1) per pair instead of traversal).
        std::vector<int> old_k_cache;
        if (can_use_incremental) {
          old_k_cache.assign(static_cast<size_t>(inc.point_count), -1);
          const Eigen::Vector2d& src_uv = optimize_points[i];
          const int src_tri = optimize_triangle_indices[i];
          const int src_support_idx =
            (src_tri >= 0 &&
             static_cast<size_t>(src_tri) < inc.support_row_for_triangle.size())
              ? inc.support_row_for_triangle[static_cast<size_t>(src_tri)]
              : -1;
          #pragma omp parallel for schedule(static)
          for (int j = 0; j < inc.point_count; ++j) {
            if (j == static_cast<int>(i)) {
              continue;
            }
            const int j_tri = optimize_triangle_indices[static_cast<size_t>(j)];
            const int j_support_idx =
              (j_tri >= 0 &&
               static_cast<size_t>(j_tri) < inc.support_row_for_triangle.size())
                ? inc.support_row_for_triangle[static_cast<size_t>(j_tri)]
                : -1;
            if (src_support_idx >= 0 && j_support_idx >= 0) {
              old_k_cache[static_cast<size_t>(j)] =
                get_support_pairwise_dist(state, src_support_idx, j_support_idx);
            } else {
              old_k_cache[static_cast<size_t>(j)] =
                delaunay_helper->count_triangles_crossed(
                  src_uv,
                  optimize_points[static_cast<size_t>(j)]);
            }
          }
        }
        
        // Rank local ring proposals with the fast surrogate, then exact-score
        // only the best few locals before trying a small shuffled global tail.
        double best_exact_move_error = current_exact_error;
        double best_total_move_error = current_global_error;
        int best_neighbor = -1;
        bool use_pair_move = false;
        int pair_partner_idx = -1;
        int pair_partner_neighbor = -1;
        int pair_partner_local_radius = 1;
        bool best_move_uses_relaxed_acceptance = false;
        std::vector<size_t> ranked_candidate_indices;
        ranked_candidate_indices.reserve(proposal_tris.size());
        std::vector<size_t> ranked_local_candidate_indices;
        std::vector<size_t> local_candidate_indices;
        std::vector<size_t> global_candidate_indices;
        ranked_local_candidate_indices.reserve(local_candidate_indices.size());
        local_candidate_indices.reserve(local_proposal_prefix);
        global_candidate_indices.reserve(
          proposal_tris.size() > local_proposal_prefix
            ? proposal_tris.size() - local_proposal_prefix
            : 0);
        for (size_t p = 0; p < proposal_tris.size(); ++p) {
          const int neighbor_tri = proposal_tris[p];
          if (neighbor_tri != tri_idx &&
              neighbor_tri >= 0 &&
              neighbor_tri < triangle_count &&
              occupancy[static_cast<size_t>(neighbor_tri)] > 0) {
            continue;
          }
          if (neighbor_tri < 0 || neighbor_tri >= triangle_count) {
            continue;
          }
          if (triangle_center_valid[static_cast<size_t>(neighbor_tri)] == 0) {
            continue;
          }
          if (triangle_center_inside[static_cast<size_t>(neighbor_tri)] == 0) {
            continue;
          }
          if (p < local_proposal_prefix) {
            local_candidate_indices.push_back(p);
          } else {
            global_candidate_indices.push_back(p);
          }
        }
        sweep_valid_candidates +=
          static_cast<int>(local_candidate_indices.size() + global_candidate_indices.size());
        const auto boost_nearby_bad_points = [&](size_t source_idx, int source_local_radius) {
          if (!can_use_incremental ||
              !inc.valid ||
              source_local_radius < kHighOrderProposalRadiusThreshold ||
              source_idx >= optimize_triangle_indices.size() ||
              source_idx >= inc.point_support_rows.size()) {
            return;
          }
          const int source_support_idx =
            inc.point_support_rows[static_cast<size_t>(source_idx)];
          if (source_support_idx < 0) {
            return;
          }
          const int nearby_dist_limit = std::max(2, source_local_radius + 1);
          std::vector<std::tuple<double, size_t, int>> nearby_bad_points;
          nearby_bad_points.reserve(optimize_points.size());
          for (size_t candidate_idx = 0; candidate_idx < optimize_points.size(); ++candidate_idx) {
            if (candidate_idx == source_idx ||
                candidate_idx >= optimize_triangle_indices.size() ||
                optimize_triangle_indices[candidate_idx] < 0 ||
                candidate_idx >= point_local_proposal_radii.size() ||
                point_local_proposal_radii[candidate_idx] < 2 ||
                candidate_idx >= point_priority_scores.size()) {
              continue;
            }
            const double candidate_score = point_priority_scores[candidate_idx];
            if (!std::isfinite(candidate_score) || candidate_score <= 1e-9) {
              continue;
            }
            const int candidate_support_idx =
              (candidate_idx < inc.point_support_rows.size())
                ? inc.point_support_rows[candidate_idx]
                : -1;
            if (candidate_support_idx < 0) {
              continue;
            }
            const int support_dist =
              get_support_pairwise_dist(state, source_support_idx, candidate_support_idx);
            if (support_dist <= 0 || support_dist > nearby_dist_limit) {
              continue;
            }
            const double rank =
              candidate_score / (1.0 + 0.45 * static_cast<double>(support_dist));
            nearby_bad_points.emplace_back(rank, candidate_idx, support_dist);
          }
          std::sort(
            nearby_bad_points.begin(),
            nearby_bad_points.end(),
            [](const auto& a, const auto& b) {
              if (std::get<0>(a) != std::get<0>(b)) {
                return std::get<0>(a) > std::get<0>(b);
              }
              return std::get<1>(a) < std::get<1>(b);
            });
          const int boost_count = std::min(
            kNearbyPriorityBoostCount,
            static_cast<int>(nearby_bad_points.size()));
          for (int boost_idx = 0; boost_idx < boost_count; ++boost_idx) {
            const size_t candidate_idx =
              std::get<1>(nearby_bad_points[static_cast<size_t>(boost_idx)]);
            const int support_dist =
              std::get<2>(nearby_bad_points[static_cast<size_t>(boost_idx)]);
            if (candidate_idx >= optimize_point_priority_boost.size() ||
                candidate_idx >= optimize_point_priority_boost_cooldown.size()) {
              continue;
            }
            const double dist_factor =
              1.0 / (1.0 + 0.35 * std::max(0, support_dist - 1));
            const float boost = std::clamp(
              static_cast<float>(
                (kNearbyPointPriorityBoostBase +
                 0.03f * static_cast<float>(std::max(0, source_local_radius - 3))) *
                dist_factor),
              kNearbyPointPriorityBoostBase * 0.6f,
              kNearbyPointPriorityBoostMax);
            optimize_point_priority_boost[candidate_idx] =
              std::max(optimize_point_priority_boost[candidate_idx], boost);
            optimize_point_priority_boost_cooldown[candidate_idx] =
              std::max(
                optimize_point_priority_boost_cooldown[candidate_idx],
                kNearbyPointPriorityBoostSweeps);
            if (candidate_idx < point_priority_state_refreshed_this_sweep.size()) {
              point_priority_state_refreshed_this_sweep[candidate_idx] = 1;
            }
          }
        };
        int local_exact_budget = static_cast<int>(local_candidate_indices.size());
        if (can_use_incremental) {
          std::vector<std::pair<double, size_t>> ranked_local_candidates;
          ranked_local_candidates.reserve(local_candidate_indices.size());
          FastEvalScratch fast_eval_scratch;
          for (size_t p : local_candidate_indices) {
            const int neighbor_tri = proposal_tris[p];
            const Eigen::Vector2d& neighbor_center =
              triangle_center_cache[static_cast<size_t>(neighbor_tri)];
            const double fast_error = evaluate_move_fast(
              static_cast<int>(i),
              optimize_points[i],
              neighbor_center,
              neighbor_tri,
              &old_k_cache,
              &fast_eval_scratch,
              nullptr);
            ranked_local_candidates.emplace_back(
              std::isfinite(fast_error)
                ? fast_error
                : std::numeric_limits<double>::infinity(),
              p);
          }
          local_exact_budget = compute_incremental_local_exact_budget(
            static_cast<int>(ranked_local_candidates.size()),
            local_proposal_radius);
          if (local_exact_budget < static_cast<int>(ranked_local_candidates.size())) {
            std::partial_sort(
              ranked_local_candidates.begin(),
              ranked_local_candidates.begin() + static_cast<std::ptrdiff_t>(local_exact_budget),
              ranked_local_candidates.end(),
              [](const auto& a, const auto& b) {
                if (a.first != b.first) {
                  return a.first < b.first;
                }
                return a.second < b.second;
              });
            ranked_local_candidates.resize(static_cast<size_t>(local_exact_budget));
          } else {
            std::sort(
              ranked_local_candidates.begin(),
              ranked_local_candidates.end(),
              [](const auto& a, const auto& b) {
                if (a.first != b.first) {
                  return a.first < b.first;
                }
                return a.second < b.second;
              });
          }
          for (const auto& entry : ranked_local_candidates) {
            ranked_local_candidate_indices.push_back(entry.second);
          }
        } else {
          ranked_local_candidate_indices.insert(
            ranked_local_candidate_indices.end(),
            local_candidate_indices.begin(),
            local_candidate_indices.end());
        }
        ranked_candidate_indices.insert(
          ranked_candidate_indices.end(),
          ranked_local_candidate_indices.begin(),
          ranked_local_candidate_indices.end());
        std::shuffle(global_candidate_indices.begin(), global_candidate_indices.end(), gen);
        ranked_candidate_indices.insert(
          ranked_candidate_indices.end(),
          global_candidate_indices.begin(),
          global_candidate_indices.end());

        int exact_check_budget = static_cast<int>(ranked_candidate_indices.size());
        if (can_use_incremental) {
          const int total_candidates = static_cast<int>(ranked_candidate_indices.size());
          int extra_global_budget = use_full_sweep_search ? 3 : 1;
          if (aggressive_level >= 1) {
            extra_global_budget = use_full_sweep_search ? 4 : 2;
          }
          if (aggressive_level >= 2) {
            extra_global_budget = use_full_sweep_search ? 6 : 3;
          }
          exact_check_budget = std::min(
            total_candidates,
            local_exact_budget + extra_global_budget);
        } else {
          const int total_candidates = static_cast<int>(ranked_candidate_indices.size());
          const int local_count = static_cast<int>(local_candidate_indices.size());
          exact_check_budget = std::min(total_candidates, std::max(local_count, 8));
        }
        sweep_exact_scored_candidates += exact_check_budget;
        if (exact_check_budget > 0) {
          std::vector<double> exact_candidate_errors(
            static_cast<size_t>(exact_check_budget),
            std::numeric_limits<double>::infinity());
          std::vector<double> total_candidate_errors(
            static_cast<size_t>(exact_check_budget),
            std::numeric_limits<double>::infinity());
          std::vector<int> exact_candidate_neighbors(
            static_cast<size_t>(exact_check_budget),
            -1);
#if defined(_OPENMP)
          #pragma omp parallel for schedule(static) if (exact_check_budget > 1 && (can_use_incremental || current_optimizer_mode() == VoronoiOptimizerMode::StructuredTargets))
#endif
          for (int rank = 0; rank < exact_check_budget; ++rank) {
            const size_t p = ranked_candidate_indices[static_cast<size_t>(rank)];
            const int neighbor_tri = proposal_tris[p];
            const Eigen::Vector2d& neighbor_center =
              triangle_center_cache[static_cast<size_t>(neighbor_tri)];
            double candidate_raw_connectivity_error = 0.0;
            const double candidate_exact_error = can_use_incremental
              ? compute_exact_error_from_incremental_move(
                  static_cast<int>(i),
                  optimize_points[i],
                  neighbor_center,
                  neighbor_tri,
                  &old_k_cache,
                  &candidate_raw_connectivity_error)
              : compute_live_pcf_error(
                  [&]() {
                    std::vector<Eigen::Vector2d> test_points = optimize_points;
                    test_points[i] = neighbor_center;
                    return test_points;
                  }(),
                  &candidate_raw_connectivity_error);
            if (!std::isfinite(candidate_exact_error)) {
              continue;
            }
            const double candidate_total_error = augment_optimizer_objective(
              candidate_exact_error,
              static_cast<int>(optimize_points.size()),
              candidate_raw_connectivity_error);
            exact_candidate_errors[static_cast<size_t>(rank)] = candidate_exact_error;
            total_candidate_errors[static_cast<size_t>(rank)] = candidate_total_error;
            exact_candidate_neighbors[static_cast<size_t>(rank)] = neighbor_tri;
          }
          const double exact_eps = static_cast<double>(stagnation_best_improve_eps);
          const bool allow_plateau_walk =
            !sweep_relaxed_move_used &&
            no_progress_iters >= early_exploration_threshold * 2;
          const double relaxed_improve_eps = std::max(1e-6, exact_eps * 0.5);
          const double uphill_total_limit =
            current_global_error * (1.0 + uphill_tolerance_fraction);
          double best_relaxed_exact_move_error = std::numeric_limits<double>::infinity();
          double best_relaxed_total_move_error = std::numeric_limits<double>::infinity();
          int best_relaxed_neighbor = -1;
          for (int rank = 0; rank < exact_check_budget; ++rank) {
            const double candidate_exact_error =
              exact_candidate_errors[static_cast<size_t>(rank)];
            if (!std::isfinite(candidate_exact_error)) {
              continue;
            }
            const double candidate_total_error =
              total_candidate_errors[static_cast<size_t>(rank)];
            const int neighbor_tri =
              exact_candidate_neighbors[static_cast<size_t>(rank)];
            if (candidate_total_error + exact_eps < best_total_move_error ||
                (std::abs(candidate_total_error - best_total_move_error) <= exact_eps &&
                 candidate_exact_error + exact_eps < best_exact_move_error)) {
              best_exact_move_error = candidate_exact_error;
              best_total_move_error = candidate_total_error;
              best_neighbor = neighbor_tri;
              best_move_uses_relaxed_acceptance = false;
            }
            const bool plateau_candidate =
              allow_plateau_walk &&
              candidate_total_error + relaxed_improve_eps < current_global_error;
            const bool uphill_candidate =
              uphill_tolerance_fraction > 0.0 &&
              candidate_total_error <= uphill_total_limit;
            if ((plateau_candidate || uphill_candidate) &&
                (candidate_total_error + exact_eps < best_relaxed_total_move_error ||
                 (std::abs(candidate_total_error - best_relaxed_total_move_error) <= exact_eps &&
                  candidate_exact_error + exact_eps < best_relaxed_exact_move_error))) {
              best_relaxed_exact_move_error = candidate_exact_error;
              best_relaxed_total_move_error = candidate_total_error;
              best_relaxed_neighbor = neighbor_tri;
            }
          }
          if (best_neighbor < 0 &&
              best_relaxed_neighbor >= 0 &&
              !sweep_relaxed_move_used) {
            best_exact_move_error = best_relaxed_exact_move_error;
            best_total_move_error = best_relaxed_total_move_error;
            best_neighbor = best_relaxed_neighbor;
            best_move_uses_relaxed_acceptance = true;
            sweep_relaxed_move_used = true;
            ++sweep_relaxed_moves;
          }
        }
        if (best_neighbor < 0 &&
            can_use_incremental &&
            inc.valid &&
            local_proposal_radius >= kHighOrderProposalRadiusThreshold &&
            aggressive_level >= 1 &&
            !ranked_local_candidate_indices.empty()) {
          const double exact_eps = static_cast<double>(stagnation_best_improve_eps);
          std::vector<std::pair<double, size_t>> nearby_partner_candidates;
          nearby_partner_candidates.reserve(optimize_points.size());
          const int partner_distance_limit = std::max(3, local_proposal_radius + 1);
          for (size_t partner_idx = 0; partner_idx < optimize_points.size(); ++partner_idx) {
            if (partner_idx == i ||
                partner_idx >= optimize_triangle_indices.size() ||
                optimize_triangle_indices[partner_idx] < 0 ||
                partner_idx >= point_priority_scores.size() ||
                partner_idx >= point_local_proposal_radii.size() ||
                point_local_proposal_radii[partner_idx] < 2) {
              continue;
            }
            const double partner_score = point_priority_scores[partner_idx];
            if (!std::isfinite(partner_score) || partner_score <= 1e-9) {
              continue;
            }
            const int pair_dist =
              (partner_idx < old_k_cache.size())
                ? old_k_cache[partner_idx]
                : -1;
            if (pair_dist <= 0 || pair_dist > partner_distance_limit) {
              continue;
            }
            const double rank =
              partner_score / (1.0 + 0.40 * static_cast<double>(pair_dist));
            nearby_partner_candidates.emplace_back(rank, partner_idx);
          }
          std::sort(
            nearby_partner_candidates.begin(),
            nearby_partner_candidates.end(),
            [](const auto& a, const auto& b) {
              if (a.first != b.first) {
                return a.first > b.first;
              }
              return a.second < b.second;
            });
          if (static_cast<int>(nearby_partner_candidates.size()) > kPairMovePartnerCount) {
            nearby_partner_candidates.resize(static_cast<size_t>(kPairMovePartnerCount));
          }
          const int anchor_candidate_limit = std::min(
            kPairMoveCandidateCountPerPoint,
            static_cast<int>(ranked_local_candidate_indices.size()));
          for (const auto& partner_entry : nearby_partner_candidates) {
            const size_t partner_idx = partner_entry.second;
            const int partner_tri = optimize_triangle_indices[partner_idx];
            if (partner_tri < 0) {
              continue;
            }
            std::vector<int> partner_old_k_cache(
              static_cast<size_t>(inc.point_count),
              -1);
            const Eigen::Vector2d& partner_uv = optimize_points[partner_idx];
            const int partner_support_idx =
              (partner_idx < inc.point_support_rows.size())
                ? inc.point_support_rows[partner_idx]
                : -1;
            for (int jj = 0; jj < inc.point_count; ++jj) {
              if (jj == static_cast<int>(partner_idx)) {
                continue;
              }
              const int jj_support_idx =
                (static_cast<size_t>(jj) < inc.point_support_rows.size())
                  ? inc.point_support_rows[static_cast<size_t>(jj)]
                  : -1;
              if (partner_support_idx >= 0 && jj_support_idx >= 0) {
                partner_old_k_cache[static_cast<size_t>(jj)] =
                  get_support_pairwise_dist(state, partner_support_idx, jj_support_idx);
              } else {
                partner_old_k_cache[static_cast<size_t>(jj)] =
                  delaunay_helper->count_triangles_crossed(
                    partner_uv,
                    optimize_points[static_cast<size_t>(jj)]);
              }
            }

            const int partner_local_radius = std::max(
              1,
              point_local_proposal_radii[partner_idx]);
            std::vector<int> partner_proposal_tris;
            partner_proposal_tris.reserve(
              static_cast<size_t>(6 * std::max(1, partner_local_radius) + 8));
            const auto append_partner_tri = [&](int tri_candidate) {
              if (tri_candidate < 0) {
                return;
              }
              if (std::find(
                    partner_proposal_tris.begin(),
                    partner_proposal_tris.end(),
                    tri_candidate) == partner_proposal_tris.end()) {
                partner_proposal_tris.push_back(tri_candidate);
              }
            };
            std::vector<int> partner_frontier;
            std::vector<int> partner_next_frontier;
            partner_frontier.reserve(8);
            partner_next_frontier.reserve(16);
            partner_frontier.push_back(partner_tri);
            for (int hop = 0; hop < partner_local_radius; ++hop) {
              partner_next_frontier.clear();
              for (int frontier_tri : partner_frontier) {
                std::array<int, 3> partner_neighbors;
                delaunay_helper->get_triangle_neighbors(frontier_tri, partner_neighbors);
                for (int n : partner_neighbors) {
                  if (n < 0) {
                    continue;
                  }
                  const size_t before = partner_proposal_tris.size();
                  append_partner_tri(n);
                  if (partner_proposal_tris.size() != before) {
                    partner_next_frontier.push_back(n);
                  }
                }
              }
              if (partner_next_frontier.empty()) {
                break;
              }
              partner_frontier.swap(partner_next_frontier);
            }
            append_partner_tri(tri_idx);
            std::array<int, 3> anchor_neighbors_for_pair;
            delaunay_helper->get_triangle_neighbors(tri_idx, anchor_neighbors_for_pair);
            for (int n : anchor_neighbors_for_pair) {
              append_partner_tri(n);
            }

            std::vector<size_t> partner_candidate_indices;
            partner_candidate_indices.reserve(partner_proposal_tris.size());
            for (size_t pp = 0; pp < partner_proposal_tris.size(); ++pp) {
              const int neighbor_tri = partner_proposal_tris[pp];
              if (neighbor_tri < 0 || neighbor_tri >= triangle_count) {
                continue;
              }
              if (triangle_center_valid[static_cast<size_t>(neighbor_tri)] == 0 ||
                  triangle_center_inside[static_cast<size_t>(neighbor_tri)] == 0) {
                continue;
              }
              if (neighbor_tri != partner_tri &&
                  neighbor_tri != tri_idx &&
                  neighbor_tri < static_cast<int>(occupancy.size()) &&
                  occupancy[static_cast<size_t>(neighbor_tri)] > 0) {
                continue;
              }
              partner_candidate_indices.push_back(pp);
            }
            if (partner_candidate_indices.empty()) {
              continue;
            }

            FastEvalScratch partner_fast_eval_scratch;
            std::vector<std::pair<double, size_t>> ranked_partner_candidates;
            ranked_partner_candidates.reserve(partner_candidate_indices.size());
            for (size_t pp : partner_candidate_indices) {
              const int neighbor_tri = partner_proposal_tris[pp];
              const Eigen::Vector2d& neighbor_center =
                triangle_center_cache[static_cast<size_t>(neighbor_tri)];
              const double fast_error = evaluate_move_fast(
                static_cast<int>(partner_idx),
                optimize_points[partner_idx],
                neighbor_center,
                neighbor_tri,
                &partner_old_k_cache,
                &partner_fast_eval_scratch,
                nullptr);
              ranked_partner_candidates.emplace_back(
                std::isfinite(fast_error)
                  ? fast_error
                  : std::numeric_limits<double>::infinity(),
                pp);
            }
            std::sort(
              ranked_partner_candidates.begin(),
              ranked_partner_candidates.end(),
              [](const auto& a, const auto& b) {
                if (a.first != b.first) {
                  return a.first < b.first;
                }
                return a.second < b.second;
              });
            if (static_cast<int>(ranked_partner_candidates.size()) >
                kPairMoveCandidateCountPerPoint) {
              ranked_partner_candidates.resize(
                static_cast<size_t>(kPairMoveCandidateCountPerPoint));
            }

            for (int anchor_rank = 0; anchor_rank < anchor_candidate_limit; ++anchor_rank) {
              const size_t anchor_candidate_idx =
                ranked_local_candidate_indices[static_cast<size_t>(anchor_rank)];
              const int anchor_neighbor_tri =
                proposal_tris[anchor_candidate_idx];
              if (anchor_neighbor_tri < 0 ||
                  anchor_neighbor_tri >= triangle_count) {
                continue;
              }
              for (const auto& partner_candidate : ranked_partner_candidates) {
                const int partner_neighbor_tri =
                  partner_proposal_tris[partner_candidate.second];
                if (partner_neighbor_tri < 0 ||
                    partner_neighbor_tri >= triangle_count ||
                    anchor_neighbor_tri == partner_neighbor_tri) {
                  continue;
                }
                const auto occupied_by_other = [&](int candidate_tri) {
                  if (candidate_tri < 0 ||
                      candidate_tri >= static_cast<int>(occupancy.size())) {
                    return 0;
                  }
                  int occupied = occupancy[static_cast<size_t>(candidate_tri)];
                  if (candidate_tri == tri_idx) {
                    --occupied;
                  }
                  if (candidate_tri == partner_tri) {
                    --occupied;
                  }
                  return occupied;
                };
                if (occupied_by_other(anchor_neighbor_tri) > 0 ||
                    occupied_by_other(partner_neighbor_tri) > 0) {
                  continue;
                }
                std::vector<Eigen::Vector2d> test_points = optimize_points;
                test_points[i] =
                  triangle_center_cache[static_cast<size_t>(anchor_neighbor_tri)];
                test_points[partner_idx] =
                  triangle_center_cache[static_cast<size_t>(partner_neighbor_tri)];
                double candidate_raw_connectivity_error = 0.0;
                const double candidate_exact_error =
                  compute_live_pcf_error(
                    test_points,
                    &candidate_raw_connectivity_error);
                if (!std::isfinite(candidate_exact_error)) {
                  continue;
                }
                const double candidate_total_error = augment_optimizer_objective(
                  candidate_exact_error,
                  static_cast<int>(test_points.size()),
                  candidate_raw_connectivity_error);
                if (candidate_total_error + exact_eps < best_total_move_error ||
                    (std::abs(candidate_total_error - best_total_move_error) <= exact_eps &&
                     candidate_exact_error + exact_eps < best_exact_move_error)) {
                  best_exact_move_error = candidate_exact_error;
                  best_total_move_error = candidate_total_error;
                  best_neighbor = anchor_neighbor_tri;
                  use_pair_move = true;
                  pair_partner_idx = static_cast<int>(partner_idx);
                  pair_partner_neighbor = partner_neighbor_tri;
                  pair_partner_local_radius = partner_local_radius;
                  best_move_uses_relaxed_acceptance = false;
                }
              }
            }
          }
        }
        
        // Apply move if we found an acceptable neighbor
        if (best_neighbor >= 0) {
          if (best_neighbor >= 0 && best_neighbor < triangle_count &&
              triangle_center_valid[static_cast<size_t>(best_neighbor)] != 0) {
            const Eigen::Vector2d& new_center =
              triangle_center_cache[static_cast<size_t>(best_neighbor)];
            Eigen::Vector2d old_center = optimize_points[i];
            const int old_tri = optimize_triangle_indices[i];
            const double exact_before_move = current_exact_error;
            const double total_before_move = current_global_error;
            if (!can_use_incremental || live_position_targets_enabled || use_pair_move) {
              std::vector<Eigen::Vector2d> validated_points = optimize_points;
              validated_points[i] = new_center;
              if (use_pair_move &&
                  pair_partner_idx >= 0 &&
                  pair_partner_idx < static_cast<int>(validated_points.size()) &&
                  pair_partner_neighbor >= 0 &&
                  pair_partner_neighbor < triangle_count &&
                  triangle_center_valid[static_cast<size_t>(pair_partner_neighbor)] != 0) {
                validated_points[static_cast<size_t>(pair_partner_idx)] =
                  triangle_center_cache[static_cast<size_t>(pair_partner_neighbor)];
              }
              double validated_raw_connectivity_error = 0.0;
              const double validated_exact_error = compute_live_pcf_error(
                validated_points,
                &validated_raw_connectivity_error);
              const double validated_total_error = augment_optimizer_objective(
                validated_exact_error,
                static_cast<int>(validated_points.size()),
                validated_raw_connectivity_error);
              const double exact_eps = static_cast<double>(stagnation_best_improve_eps);
              const double uphill_total_limit =
                current_global_error * (1.0 + uphill_tolerance_fraction);
              const bool validated_improves =
                validated_total_error + exact_eps < current_global_error ||
                (std::abs(validated_total_error - current_global_error) <= exact_eps &&
                 validated_exact_error + exact_eps < current_exact_error);
              const bool validated_uphill =
                best_move_uses_relaxed_acceptance &&
                uphill_tolerance_fraction > 0.0 &&
                validated_total_error <= uphill_total_limit;
              if (!std::isfinite(validated_exact_error) ||
                  !(validated_improves || validated_uphill)) {
                if (best_move_uses_relaxed_acceptance && sweep_relaxed_moves > 0) {
                  --sweep_relaxed_moves;
                }
                best_neighbor = -1;
                use_pair_move = false;
                pair_partner_idx = -1;
                pair_partner_neighbor = -1;
              } else {
                best_exact_move_error = validated_exact_error;
                best_total_move_error = validated_total_error;
              }
            }
            if (best_neighbor < 0) {
              continue;
            }
            if (can_use_incremental) {
              (void)apply_move_fast(
                static_cast<int>(i),
                old_center,
                new_center,
                best_neighbor);
            }
            optimize_points[i] = new_center;
            optimize_triangle_indices[i] = best_neighbor;
            if (old_tri != best_neighbor &&
                old_tri >= 0 && old_tri < triangle_count &&
                best_neighbor >= 0 && best_neighbor < triangle_count) {
              occupancy[static_cast<size_t>(old_tri)] =
                std::max(0, occupancy[static_cast<size_t>(old_tri)] - 1);
              ++occupancy[static_cast<size_t>(best_neighbor)];
            }
            if (use_pair_move &&
                pair_partner_idx >= 0 &&
                pair_partner_idx < static_cast<int>(optimize_points.size()) &&
                pair_partner_neighbor >= 0 &&
                pair_partner_neighbor < triangle_count &&
                triangle_center_valid[static_cast<size_t>(pair_partner_neighbor)] != 0) {
              const Eigen::Vector2d partner_old_center =
                optimize_points[static_cast<size_t>(pair_partner_idx)];
              const int partner_old_tri =
                optimize_triangle_indices[static_cast<size_t>(pair_partner_idx)];
              const Eigen::Vector2d& partner_new_center =
                triangle_center_cache[static_cast<size_t>(pair_partner_neighbor)];
              if (can_use_incremental) {
                (void)apply_move_fast(
                  pair_partner_idx,
                  partner_old_center,
                  partner_new_center,
                  pair_partner_neighbor);
              }
              optimize_points[static_cast<size_t>(pair_partner_idx)] = partner_new_center;
              optimize_triangle_indices[static_cast<size_t>(pair_partner_idx)] =
                pair_partner_neighbor;
              if (partner_old_tri != pair_partner_neighbor &&
                  partner_old_tri >= 0 && partner_old_tri < triangle_count &&
                  pair_partner_neighbor >= 0 &&
                  pair_partner_neighbor < triangle_count) {
                occupancy[static_cast<size_t>(partner_old_tri)] =
                  std::max(0, occupancy[static_cast<size_t>(partner_old_tri)] - 1);
                ++occupancy[static_cast<size_t>(pair_partner_neighbor)];
              }
            }
            current_exact_error = best_exact_move_error;
            current_global_error = best_total_move_error;  // Update running exact objective
            optimize_current_exact_error = current_exact_error;
            optimize_current_total_error = current_global_error;
            optimize_current_error_valid = true;
            const double move_gain = total_before_move - current_global_error;
            if (move_gain > static_cast<double>(stagnation_best_improve_eps) &&
                i < optimize_point_priority_penalty.size() &&
                i < optimize_point_priority_cooldown.size()) {
              const double relative_gain =
                move_gain / std::max(1e-6, std::abs(total_before_move));
              const float penalty = std::clamp(
                kMovedPointPriorityPenaltyBase +
                  static_cast<float>(0.5 * std::max(0.0, relative_gain)),
                kMovedPointPriorityPenaltyBase,
                kMovedPointPriorityPenaltyMax);
              optimize_point_priority_penalty[i] =
                std::max(optimize_point_priority_penalty[i], penalty);
              optimize_point_priority_cooldown[i] =
                std::max(optimize_point_priority_cooldown[i], kMovedPointPriorityCooldownSweeps);
              if (i < point_priority_state_refreshed_this_sweep.size()) {
                point_priority_state_refreshed_this_sweep[i] = 1;
              }
              boost_nearby_bad_points(i, local_proposal_radius);
              if (use_pair_move &&
                  pair_partner_idx >= 0 &&
                  static_cast<size_t>(pair_partner_idx) < optimize_point_priority_penalty.size() &&
                  static_cast<size_t>(pair_partner_idx) < optimize_point_priority_cooldown.size()) {
                optimize_point_priority_penalty[static_cast<size_t>(pair_partner_idx)] =
                  std::max(optimize_point_priority_penalty[static_cast<size_t>(pair_partner_idx)], penalty);
                optimize_point_priority_cooldown[static_cast<size_t>(pair_partner_idx)] =
                  std::max(
                    optimize_point_priority_cooldown[static_cast<size_t>(pair_partner_idx)],
                    kMovedPointPriorityCooldownSweeps);
                if (static_cast<size_t>(pair_partner_idx) <
                    point_priority_state_refreshed_this_sweep.size()) {
                  point_priority_state_refreshed_this_sweep[static_cast<size_t>(pair_partner_idx)] = 1;
                }
                boost_nearby_bad_points(
                  static_cast<size_t>(pair_partner_idx),
                  pair_partner_local_radius);
              }
            }
            points_moved_this_iter += use_pair_move ? 2 : 1;
            ++sweep_accepted_moves;
            if (current_global_error + stagnation_best_improve_eps < global_best_error) {
              global_best_error = current_global_error;
              optimize_best_points = optimize_points;
              optimize_best_triangle_indices = optimize_triangle_indices;
            }

          }
        }

      }

      const bool stagnation_count_probe =
        no_progress_iters >= early_exploration_threshold;
      const bool no_move_count_probe =
        (sweep_accepted_moves == 0 &&
         sweep_relaxed_moves == 0 &&
         points_moved_this_iter == 0);
      bool count_changed = false;
      const bool allow_regular_count_moves =
        !optimize_trust_strategic_start || stagnation_count_probe;
      if (allow_regular_count_moves) {
        count_changed = maybe_apply_adaptive_count_move(
          triangle_count,
          &can_use_incremental,
          &current_global_error,
          &current_exact_error,
          stagnation_count_probe);
      }
      if (!count_changed && no_move_count_probe) {
        const bool fallback_count_changed = maybe_apply_adaptive_count_move(
          triangle_count,
          &can_use_incremental,
          &current_global_error,
          &current_exact_error,
          true);
        if (fallback_count_changed) {
          count_changed = true;
        }
      }
      if (count_changed) {
        ++sweep_count_changes;
        sync_optimizer_point_priority_state(true);
      }
      if (!count_changed && adaptive_count_reversal_cooldown > 0) {
        --adaptive_count_reversal_cooldown;
      }
      if (count_changed) {
        optimize_current_exact_error = current_exact_error;
        optimize_current_total_error = current_global_error;
        optimize_current_error_valid = true;
        if (use_incremental_optimizer &&
            state.output_support_denominator_cache_valid &&
            !output_support_uv.empty() &&
            output_support_uv.size() == output_support_tri_indices.size()) {
          if (!rebuild_incremental_cache(optimize_points, optimize_triangle_indices, triangle_count)) {
            can_use_incremental = false;
          } else {
            can_use_incremental = true;
          }
        }
        points_moved_this_iter = std::max(points_moved_this_iter, 1);
      }

      if (!count_changed) {
        sync_optimizer_point_priority_state();
        const size_t tracked_points = std::min(
          std::min(
            optimize_point_priority_penalty.size(),
            optimize_point_priority_cooldown.size()),
          std::min(
            optimize_point_priority_boost.size(),
            optimize_point_priority_boost_cooldown.size()));
        for (size_t point_idx = 0; point_idx < tracked_points; ++point_idx) {
          if (point_idx < point_priority_state_refreshed_this_sweep.size() &&
              point_priority_state_refreshed_this_sweep[point_idx] != 0) {
            continue;
          }
          if (optimize_point_priority_cooldown[point_idx] <= 0) {
            optimize_point_priority_penalty[point_idx] = 0.0f;
          } else {
            --optimize_point_priority_cooldown[point_idx];
            if (optimize_point_priority_cooldown[point_idx] <= 0) {
              optimize_point_priority_penalty[point_idx] = 0.0f;
            } else {
              optimize_point_priority_penalty[point_idx] *=
                kMovedPointPriorityPenaltyDecay;
            }
          }
          if (optimize_point_priority_boost_cooldown[point_idx] <= 0) {
            optimize_point_priority_boost[point_idx] = 0.0f;
          } else {
            --optimize_point_priority_boost_cooldown[point_idx];
            if (optimize_point_priority_boost_cooldown[point_idx] <= 0) {
              optimize_point_priority_boost[point_idx] = 0.0f;
            } else {
              optimize_point_priority_boost[point_idx] *=
                kNearbyPointPriorityBoostDecay;
            }
          }
        }
      }

      
      // Update state after full sweep through all points
      optimize_iteration++;
      state.optimizer_iterations_ran = optimize_iteration;
      if (points_moved_this_iter > 0) {
        optimize_best_error = current_global_error;
        if (optimize_best_error + stagnation_best_improve_eps < global_best_error) {
          global_best_error = optimize_best_error;
          optimize_best_points = optimize_points;
          optimize_best_triangle_indices = optimize_triangle_indices;
        }

        const double rel_improve = (sweep_start_error > 1e-12)
          ? ((sweep_start_error - optimize_best_error) / sweep_start_error)
          : 0.0;
        const double rel_improve_percent = rel_improve * 100.0;
        if (rel_improve_percent >= static_cast<double>(convergence_delta_percent)) {
          no_progress_iters = 0;
        } else {
          no_progress_iters++;
        }

        optimize_swaps_made++;
        ++optimizer_sweeps_since_visual_update;
        
        flush_optimizer_visual_state(false, can_use_incremental);
        state.output_voronoi_pcf_energy = optimize_best_error;
        state.optimizer_improvements = optimize_swaps_made;
        if (can_use_incremental && inc.valid) {
          live_last_worst_bin_residual = compute_live_bin_residual_stats_from_incremental(
            live_worst_bin_focus_count,
            &live_last_worst_bin_index,
            nullptr,
            nullptr);
        } else {
          live_last_worst_bin_residual = compute_live_bin_residual_stats(
            optimize_points,
            live_worst_bin_focus_count,
            &live_last_worst_bin_index,
            nullptr,
            nullptr);
        }
        
      } else {
        no_progress_iters++;
        if (can_use_incremental && inc.valid) {
          live_last_worst_bin_residual = compute_live_bin_residual_stats_from_incremental(
            live_worst_bin_focus_count,
            &live_last_worst_bin_index,
            nullptr,
            nullptr);
        } else {
          live_last_worst_bin_residual = compute_live_bin_residual_stats(
            optimize_points,
            live_worst_bin_focus_count,
            &live_last_worst_bin_index,
            nullptr,
            nullptr);
        }
      }

      double best_progress_percent = 0.0;
      if (std::isfinite(best_error_before_sweep) &&
          std::isfinite(global_best_error) &&
          global_best_error + static_cast<double>(stagnation_best_improve_eps) <
            best_error_before_sweep) {
        const double best_ref = std::max(1e-6, std::abs(best_error_before_sweep));
        best_progress_percent =
          100.0 * (best_error_before_sweep - global_best_error) / best_ref;
      }
      const double meaningful_progress_percent = std::max(
        0.5,
        static_cast<double>(settled_best_gap_percent) * 2.0);
      if (best_progress_percent >= meaningful_progress_percent) {
        soft_no_progress_iters = 0;
      } else {
        ++soft_no_progress_iters;
      }

      double plateau_error_range = std::numeric_limits<double>::infinity();
      double plateau_worst_range = std::numeric_limits<double>::infinity();
      bool plateau_detected = false;
      if (plateau_window > 1 &&
          std::isfinite(current_global_error) &&
          std::isfinite(live_last_worst_bin_residual)) {
        plateau_recent_errors.push_back(current_global_error);
        plateau_recent_worst_bins.push_back(live_last_worst_bin_residual);
        const size_t max_plateau_samples = static_cast<size_t>(std::max(2, plateau_window));
        if (plateau_recent_errors.size() > max_plateau_samples) {
          plateau_recent_errors.erase(plateau_recent_errors.begin());
        }
        if (plateau_recent_worst_bins.size() > max_plateau_samples) {
          plateau_recent_worst_bins.erase(plateau_recent_worst_bins.begin());
        }
        if (plateau_recent_errors.size() >= max_plateau_samples &&
            plateau_recent_worst_bins.size() >= max_plateau_samples) {
          const auto error_minmax = std::minmax_element(
            plateau_recent_errors.begin(),
            plateau_recent_errors.end());
          const auto worst_minmax = std::minmax_element(
            plateau_recent_worst_bins.begin(),
            plateau_recent_worst_bins.end());
          plateau_error_range = *error_minmax.second - *error_minmax.first;
          plateau_worst_range = *worst_minmax.second - *worst_minmax.first;
          const double error_ref =
            std::max(1e-6, std::abs(current_global_error));
          const double allowed_error_range = std::max(
            static_cast<double>(stagnation_best_improve_eps),
            error_ref * static_cast<double>(plateau_error_band_percent) / 100.0);
          const double allowed_worst_range = std::max(
            5e-4,
            static_cast<double>(plateau_worst_bin_band));
          plateau_detected =
            (plateau_error_range <= allowed_error_range) &&
            (plateau_worst_range <= allowed_worst_range);
        }
      } else {
        plateau_recent_errors.clear();
        plateau_recent_worst_bins.clear();
      }

      if (!count_changed && plateau_detected) {
        const bool plateau_count_changed = maybe_apply_adaptive_count_move(
          triangle_count,
          &can_use_incremental,
          &current_global_error,
          &current_exact_error,
          true);
        if (plateau_count_changed) {
          count_changed = true;
          ++sweep_count_changes;
          sync_optimizer_point_priority_state(true);
          points_moved_this_iter = std::max(points_moved_this_iter, 1);
          no_progress_iters = 0;
          settled_sweeps = 0;
          plateau_detected = false;
          plateau_recent_errors.clear();
          plateau_recent_worst_bins.clear();
          if (use_incremental_optimizer &&
              state.output_support_denominator_cache_valid &&
              !output_support_uv.empty() &&
              output_support_uv.size() == output_support_tri_indices.size()) {
            if (!rebuild_incremental_cache(optimize_points, optimize_triangle_indices, triangle_count)) {
              can_use_incremental = false;
            } else {
              can_use_incremental = true;
            }
          }
          if (current_global_error + stagnation_best_improve_eps < global_best_error) {
            global_best_error = current_global_error;
            optimize_best_points = optimize_points;
            optimize_best_triangle_indices = optimize_triangle_indices;
          }
        }
      }

      constexpr int kPlateauClusterJitterCadenceSweeps = 4;
      const bool plateau_has_high_order_pressure =
        std::any_of(
          point_local_proposal_radii.begin(),
          point_local_proposal_radii.end(),
          [&](int proposal_radius) {
            return proposal_radius >= kHighOrderProposalRadiusThreshold;
          });
      const bool plateau_jitter_due =
        (soft_no_progress_iters >= std::max(6, early_exploration_threshold * 2) &&
         (soft_no_progress_iters % kPlateauClusterJitterCadenceSweeps == 0));
      bool plateau_jitter_applied = false;
      if (plateau_detected &&
          plateau_jitter_enabled &&
          plateau_jitter_attempts_used < plateau_jitter_attempt_limit &&
          optimize_running &&
          !point_order.empty() &&
          plateau_has_high_order_pressure &&
          plateau_jitter_due) {
        std::vector<size_t> plateau_jitter_seed_order;
        plateau_jitter_seed_order.reserve(point_order.size());
        if (!point_priority_scores.empty()) {
          std::vector<std::pair<double, size_t>> scored_plateau_seeds;
          scored_plateau_seeds.reserve(optimize_points.size());
          for (size_t point_idx = 0; point_idx < optimize_points.size(); ++point_idx) {
            if (point_idx >= optimize_triangle_indices.size() ||
                optimize_triangle_indices[point_idx] < 0 ||
                point_idx >= point_local_proposal_radii.size() ||
                point_local_proposal_radii[point_idx] < kHighOrderProposalRadiusThreshold ||
                point_idx >= point_priority_scores.size()) {
              continue;
            }
            const double score = point_priority_scores[point_idx];
            if (!std::isfinite(score)) {
              continue;
            }
            scored_plateau_seeds.emplace_back(score, point_idx);
          }
          std::sort(
            scored_plateau_seeds.begin(),
            scored_plateau_seeds.end(),
            [](const auto& a, const auto& b) {
              if (a.first != b.first) {
                return a.first > b.first;
              }
              return a.second < b.second;
            });
          for (const auto& entry : scored_plateau_seeds) {
            plateau_jitter_seed_order.push_back(entry.second);
          }
        }
        if (plateau_jitter_seed_order.empty()) {
          plateau_jitter_seed_order = point_order;
        }
        const int jitter_target_count = std::min(
          effective_plateau_jitter_points,
          static_cast<int>(plateau_jitter_seed_order.size()));
        int jitter_moves = 0;
        std::vector<Eigen::Vector2d> jitter_candidate_points = optimize_points;
        std::vector<int> jitter_candidate_triangles = optimize_triangle_indices;
        std::vector<int> jitter_candidate_occupancy = occupancy;
        double jitter_candidate_exact_error = current_exact_error;
        double jitter_candidate_total_error = current_global_error;

        for (int jitter_idx = 0; jitter_idx < jitter_target_count; ++jitter_idx) {
          const size_t i = plateau_jitter_seed_order[static_cast<size_t>(jitter_idx)];
          const int tri_idx = jitter_candidate_triangles[i];
          if (tri_idx < 0) {
            continue;
          }

          const int jitter_local_radius =
            (i < point_local_proposal_radii.size())
              ? std::max(1, point_local_proposal_radii[i])
              : 1;
          const int jitter_support_proposal_budget =
            std::max(4, effective_plateau_jitter_proposals / 3);
          std::vector<int> proposal_tris;
          proposal_tris.reserve(
            static_cast<size_t>(
              6 * std::max(1, jitter_local_radius) +
              std::max(0, jitter_support_proposal_budget) +
              8));
          const auto append_unique_tri = [&](int tri_candidate) {
            if (tri_candidate < 0) {
              return;
            }
            if (std::find(proposal_tris.begin(), proposal_tris.end(), tri_candidate) == proposal_tris.end()) {
              proposal_tris.push_back(tri_candidate);
            }
          };

          std::vector<int> jitter_frontier;
          std::vector<int> jitter_next_frontier;
          jitter_frontier.reserve(8);
          jitter_next_frontier.reserve(16);
          jitter_frontier.push_back(tri_idx);
          for (int hop = 0; hop < jitter_local_radius; ++hop) {
            jitter_next_frontier.clear();
            for (int frontier_tri : jitter_frontier) {
              std::array<int, 3> neighbors;
              delaunay_helper->get_triangle_neighbors(frontier_tri, neighbors);
              for (int n : neighbors) {
                if (n < 0) {
                  continue;
                }
                const size_t before = proposal_tris.size();
                append_unique_tri(n);
                if (proposal_tris.size() != before) {
                  jitter_next_frontier.push_back(n);
                }
              }
            }
            if (jitter_next_frontier.empty()) {
              break;
            }
            jitter_frontier.swap(jitter_next_frontier);
          }
          if (can_use_incremental &&
              inc.valid &&
              i < point_local_proposal_radii.size() &&
              point_local_proposal_radii[i] >= kHighOrderProposalRadiusThreshold &&
              i < inc.point_support_rows.size()) {
            const int jitter_support_idx = inc.point_support_rows[i];
            if (jitter_support_idx >= 0) {
              std::vector<std::tuple<double, size_t, int>> nearby_cluster_points;
              nearby_cluster_points.reserve(optimize_points.size());
              const int cluster_dist_limit = std::max(2, jitter_local_radius + 1);
              for (size_t candidate_idx = 0; candidate_idx < optimize_points.size(); ++candidate_idx) {
                if (candidate_idx == i ||
                    candidate_idx >= jitter_candidate_triangles.size() ||
                    jitter_candidate_triangles[candidate_idx] < 0 ||
                    candidate_idx >= point_priority_scores.size() ||
                    candidate_idx >= point_local_proposal_radii.size() ||
                    point_local_proposal_radii[candidate_idx] < 2 ||
                    candidate_idx >= inc.point_support_rows.size()) {
                  continue;
                }
                const double candidate_score = point_priority_scores[candidate_idx];
                if (!std::isfinite(candidate_score) || candidate_score <= 1e-9) {
                  continue;
                }
                const int candidate_support_idx = inc.point_support_rows[candidate_idx];
                if (candidate_support_idx < 0) {
                  continue;
                }
                const int support_dist =
                  get_support_pairwise_dist(state, jitter_support_idx, candidate_support_idx);
                if (support_dist <= 0 || support_dist > cluster_dist_limit) {
                  continue;
                }
                const double rank =
                  candidate_score / (1.0 + 0.45 * static_cast<double>(support_dist));
                nearby_cluster_points.emplace_back(rank, candidate_idx, support_dist);
              }
              std::sort(
                nearby_cluster_points.begin(),
                nearby_cluster_points.end(),
                [](const auto& a, const auto& b) {
                  if (std::get<0>(a) != std::get<0>(b)) {
                    return std::get<0>(a) > std::get<0>(b);
                  }
                  return std::get<1>(a) < std::get<1>(b);
                });
              const int nearby_seed_count = std::min(
                2,
                static_cast<int>(nearby_cluster_points.size()));
              for (int nearby_idx = 0; nearby_idx < nearby_seed_count; ++nearby_idx) {
                const size_t cluster_point_idx =
                  std::get<1>(nearby_cluster_points[static_cast<size_t>(nearby_idx)]);
                const int cluster_tri =
                  jitter_candidate_triangles[cluster_point_idx];
                append_unique_tri(cluster_tri);
                std::array<int, 3> cluster_neighbors;
                delaunay_helper->get_triangle_neighbors(cluster_tri, cluster_neighbors);
                for (int n : cluster_neighbors) {
                  append_unique_tri(n);
                }
              }
            }
          }
          if (!output_support_tri_indices.empty() &&
              output_support_tri_indices.size() == output_support_uv.size()) {
            std::uniform_int_distribution<int> support_pick(
              0,
              static_cast<int>(output_support_tri_indices.size()) - 1);

            // Gradient-informed proposals: bias toward positions that fill under-represented bins.
            // Use the pairwise cache (when available) to score candidates efficiently.
            // Allocate half of the jitter budget to gradient-informed proposals (the other half
            // remains random). The 50% split keeps diversity while steering toward the deficit bins.
            const int gradient_jitter_proposals =
              (state.output_support_pairwise_cache_valid && inc.valid && inc.valid_points > 0)
                ? std::min(jitter_support_proposal_budget / 2,
                           static_cast<int>(output_support_tri_indices.size()))
                : 0;
            if (gradient_jitter_proposals > 0) {
              int under_bin = -1;
              double max_deficit = 0.0;
              for (int bk = 0; bk < inc.bin_count; ++bk) {
                const double avg_bk =
                  static_cast<double>(inc.sum_distribution[static_cast<size_t>(bk)]) /
                  inc.valid_points;
                const double tgt_bk =
                  (bk < static_cast<int>(live_target_hist.size()))
                    ? static_cast<double>(live_target_hist[static_cast<size_t>(bk)])
                    : 0.0;
                const double deficit = tgt_bk - avg_bk;
                if (deficit > max_deficit) {
                  max_deficit = deficit;
                  under_bin = bk;
                }
              }
              if (under_bin >= 0) {
                constexpr int kGradientJitterPoolSize = 200;
                const int pool_size =
                  std::min(kGradientJitterPoolSize,
                           static_cast<int>(output_support_tri_indices.size()));
                std::vector<std::pair<int, int>> scored_cands;
                scored_cands.reserve(static_cast<size_t>(pool_size));
                const int cur_support_jitter =
                  (tri_idx >= 0 &&
                   static_cast<size_t>(tri_idx) < inc.support_row_for_triangle.size())
                    ? inc.support_row_for_triangle[static_cast<size_t>(tri_idx)]
                    : -1;
                (void)cur_support_jitter;
                for (int sp = 0; sp < pool_size; ++sp) {
                  const int cand_pos = support_pick(gen);
                  const int cand_tri =
                    output_support_tri_indices[static_cast<size_t>(cand_pos)];
                  if (cand_tri < 0 ||
                      static_cast<size_t>(cand_tri) >= jitter_candidate_occupancy.size() ||
                      jitter_candidate_occupancy[static_cast<size_t>(cand_tri)] > 0 ||
                      triangle_center_valid[static_cast<size_t>(cand_tri)] == 0 ||
                      triangle_center_inside[static_cast<size_t>(cand_tri)] == 0) {
                    continue;
                  }
                  const int cand_support =
                    (static_cast<size_t>(cand_tri) < inc.support_row_for_triangle.size())
                      ? inc.support_row_for_triangle[static_cast<size_t>(cand_tri)]
                      : -1;
                  if (cand_support < 0) {
                    continue;
                  }
                  int count = 0;
                  for (int ji = 0; ji < inc.point_count; ++ji) {
                    if (ji == static_cast<int>(i)) {
                      continue;
                    }
                    const int ji_tri =
                      jitter_candidate_triangles[static_cast<size_t>(ji)];
                    const int ji_support =
                      (ji_tri >= 0 &&
                       static_cast<size_t>(ji_tri) < inc.support_row_for_triangle.size())
                        ? inc.support_row_for_triangle[static_cast<size_t>(ji_tri)]
                        : -1;
                    if (ji_support >= 0 &&
                        get_support_pairwise_dist(state, cand_support, ji_support) ==
                          under_bin) {
                      ++count;
                    }
                  }
                  if (count > 0) {
                    scored_cands.emplace_back(count, cand_tri);
                  }
                }
                std::sort(scored_cands.begin(), scored_cands.end(), std::greater<>());
                const int add_count =
                  std::min(gradient_jitter_proposals, static_cast<int>(scored_cands.size()));
                for (int gi = 0; gi < add_count; ++gi) {
                  append_unique_tri(scored_cands[static_cast<size_t>(gi)].second);
                }
              }
            }

            for (int gp = 0; gp < jitter_support_proposal_budget; ++gp) {
              append_unique_tri(output_support_tri_indices[static_cast<size_t>(support_pick(gen))]);
            }
          }

          double best_jitter_exact_error = jitter_candidate_exact_error;
          double best_jitter_total_error = jitter_candidate_total_error;
          int best_jitter_neighbor = -1;
          for (int neighbor_tri : proposal_tris) {
            if (neighbor_tri == tri_idx) {
              continue;
            }
            if (neighbor_tri < 0 || neighbor_tri >= triangle_count) {
              continue;
            }
            if (neighbor_tri < static_cast<int>(jitter_candidate_occupancy.size()) &&
                jitter_candidate_occupancy[static_cast<size_t>(neighbor_tri)] > 0) {
              continue;
            }
            if (triangle_center_valid[static_cast<size_t>(neighbor_tri)] == 0 ||
                triangle_center_inside[static_cast<size_t>(neighbor_tri)] == 0) {
              continue;
            }

            const Eigen::Vector2d& neighbor_center =
              triangle_center_cache[static_cast<size_t>(neighbor_tri)];
            double candidate_raw_connectivity_error = 0.0;
            std::vector<Eigen::Vector2d> test_points = jitter_candidate_points;
            test_points[i] = neighbor_center;
            const double candidate_exact_error = compute_live_pcf_error(
              test_points,
              &candidate_raw_connectivity_error);
            if (!std::isfinite(candidate_exact_error)) {
              continue;
            }
            const double candidate_total_error = augment_optimizer_objective(
              candidate_exact_error,
              static_cast<int>(test_points.size()),
              candidate_raw_connectivity_error);
            const double exact_eps = static_cast<double>(stagnation_best_improve_eps);
            if (candidate_total_error + exact_eps < best_jitter_total_error ||
                (std::abs(candidate_total_error - best_jitter_total_error) <= exact_eps &&
                 candidate_exact_error + exact_eps < best_jitter_exact_error)) {
              best_jitter_exact_error = candidate_exact_error;
              best_jitter_total_error = candidate_total_error;
              best_jitter_neighbor = neighbor_tri;
            }
          }

          if (best_jitter_neighbor >= 0) {
            const int old_tri = jitter_candidate_triangles[i];
            jitter_candidate_points[i] =
              triangle_center_cache[static_cast<size_t>(best_jitter_neighbor)];
            jitter_candidate_triangles[i] = best_jitter_neighbor;
            if (old_tri >= 0 && old_tri < triangle_count) {
              jitter_candidate_occupancy[static_cast<size_t>(old_tri)] =
                std::max(0, jitter_candidate_occupancy[static_cast<size_t>(old_tri)] - 1);
            }
            if (best_jitter_neighbor >= 0 && best_jitter_neighbor < triangle_count) {
              ++jitter_candidate_occupancy[static_cast<size_t>(best_jitter_neighbor)];
            }
            jitter_candidate_exact_error = best_jitter_exact_error;
            jitter_candidate_total_error = best_jitter_total_error;
            ++jitter_moves;
          }
        }

        if (jitter_moves > 0) {
          ++plateau_jitter_attempts_used;
          double jitter_raw_connectivity_error = 0.0;
          const double jitter_exact_error =
            compute_live_pcf_error(jitter_candidate_points, &jitter_raw_connectivity_error);
          const double jitter_total_error = augment_optimizer_objective(
            jitter_exact_error,
            static_cast<int>(jitter_candidate_points.size()),
            jitter_raw_connectivity_error);
          const double exact_eps = static_cast<double>(stagnation_best_improve_eps);
          const bool jitter_improves_exact =
            std::isfinite(jitter_exact_error) &&
            (jitter_total_error + exact_eps < current_global_error ||
             (std::abs(jitter_total_error - current_global_error) <= exact_eps &&
              jitter_exact_error + exact_eps < current_exact_error));
          if (jitter_improves_exact) {
            plateau_jitter_applied = true;
            optimize_points = std::move(jitter_candidate_points);
            optimize_triangle_indices = std::move(jitter_candidate_triangles);
            occupancy = std::move(jitter_candidate_occupancy);
            sync_optimizer_point_priority_state(true);
            current_exact_error = jitter_exact_error;
            current_global_error = jitter_total_error;
            optimize_current_exact_error = current_exact_error;
            optimize_current_total_error = current_global_error;
            optimize_current_error_valid = true;
            if (use_incremental_optimizer &&
                state.output_support_denominator_cache_valid &&
                !output_support_uv.empty() &&
                output_support_uv.size() == output_support_tri_indices.size()) {
              if (!rebuild_incremental_cache(optimize_points, optimize_triangle_indices, triangle_count)) {
                can_use_incremental = false;
              } else {
                can_use_incremental = true;
              }
            }
            no_progress_iters = 0;
            settled_sweeps = 0;
            plateau_recent_errors.clear();
            plateau_recent_worst_bins.clear();
            optimize_best_error = current_global_error;
            if (optimize_best_error + stagnation_best_improve_eps < global_best_error) {
              global_best_error = optimize_best_error;
              optimize_best_points = optimize_points;
              optimize_best_triangle_indices = optimize_triangle_indices;
            }
            ++optimizer_sweeps_since_visual_update;
            flush_optimizer_visual_state(false, can_use_incremental);
            state.output_voronoi_pcf_energy = optimize_best_error;
            if (can_use_incremental && inc.valid) {
              live_last_worst_bin_residual = compute_live_bin_residual_stats_from_incremental(
                live_worst_bin_focus_count,
                &live_last_worst_bin_index,
                nullptr,
                nullptr);
            } else {
              live_last_worst_bin_residual = compute_live_bin_residual_stats(
                optimize_points,
                live_worst_bin_focus_count,
                &live_last_worst_bin_index,
                nullptr,
                nullptr);
            }
            std::cout << "Plateau jitter applied: moved " << jitter_moves
                      << " points (attempt " << plateau_jitter_attempts_used
                      << "/" << plateau_jitter_attempt_limit
                      << ", error=" << optimize_best_error
                      << ", worst-bin=" << live_last_worst_bin_residual << ")\n";
          } else {
            std::cout << "Plateau jitter attempt produced no exact improvement (attempt "
                      << plateau_jitter_attempts_used
                      << "/" << plateau_jitter_attempt_limit
                      << ", candidate error=" << jitter_total_error << ")\n";
          }
        }
      }

      double settled_best_gap = std::numeric_limits<double>::infinity();
      bool settled_convergence_reached = false;
      bool early_convergence_reached = false;
      if (std::isfinite(current_global_error) &&
          std::isfinite(global_best_error) &&
          std::isfinite(live_last_worst_bin_residual)) {
        const double best_ref = std::max(1e-6, std::abs(global_best_error));
        settled_best_gap =
          100.0 * std::abs(current_global_error - global_best_error) / best_ref;
        const bool within_best_gap =
          settled_best_gap <= static_cast<double>(settled_best_gap_percent);
        const bool within_worst_tol =
          live_last_worst_bin_residual <= static_cast<double>(settled_worst_bin_tol);
        const bool low_activity =
          points_moved_this_iter <= std::max(0, settled_move_limit) &&
          !count_changed;
        if (within_best_gap && within_worst_tol && low_activity) {
          ++settled_sweeps;
        } else {
          settled_sweeps = 0;
        }
        settled_convergence_reached =
          settled_sweeps >= std::max(1, settled_sweeps_required);

        const bool relaxed_low_activity =
          points_moved_this_iter <= std::max(0, early_stop_move_limit) &&
          !count_changed;
        const bool relaxed_tail =
          (early_stop_allow_plateau && plateau_detected) ||
          soft_no_progress_iters >= std::max(1, early_stop_soft_no_progress_limit);
        early_convergence_reached =
          relaxed_tail &&
          relaxed_low_activity &&
          settled_best_gap <= static_cast<double>(early_stop_best_gap_percent) &&
          live_last_worst_bin_residual <= static_cast<double>(early_stop_worst_bin_tol);
      } else {
        settled_sweeps = 0;
      }

      // Automatic stopping conditions
      if (settled_convergence_reached) {
        optimize_running = false;
        std::cout << "Stopped: settled convergence reached (error="
                  << current_global_error
                  << ", best-gap=" << settled_best_gap << "%, worst-bin="
                  << live_last_worst_bin_residual
                  << ", sweeps=" << settled_sweeps << ", Took " 
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - root_state.task_start_time).count() << "s)\n";
      } else if (early_convergence_reached) {
        optimize_running = false;
        std::cout << "Stopped: early convergence reached (error="
                  << current_global_error
                  << ", best-gap=" << settled_best_gap << "%/" << early_stop_best_gap_percent
                  << "%, worst-bin=" << live_last_worst_bin_residual
                  << "/" << early_stop_worst_bin_tol
                  << ", soft-no-progress=" << soft_no_progress_iters
                  << "/" << early_stop_soft_no_progress_limit << ", Took "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - root_state.task_start_time).count() << "s)\n";
      } else if (plateau_detected && !plateau_jitter_applied) {
        plateau_recent_errors.clear();
        plateau_recent_worst_bins.clear();
        settled_sweeps = 0;
        if (plateau_jitter_enabled &&
            plateau_jitter_attempts_used >= plateau_jitter_attempt_limit) {
          std::cout << "Plateau detected but jitter budget exhausted; continuing until stagnation or max iterations "
                    << "(error range=" << plateau_error_range
                    << ", worst-bin range=" << plateau_worst_range << ")\n";
        }
      } else if (no_progress_iters >= stagnation_patience) {
        optimize_running = false;
        std::cout << "Stopped: stagnation detected for " << no_progress_iters
                  << " iterations (best error=" << global_best_error << ", Took " 
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - root_state.task_start_time).count() << "s)\n";
      } else if (optimize_iteration >= max_iterations) {
        optimize_running = false;
        std::cout << "Stopped: max iterations reached (" << optimize_iteration
                  << ", best error=" << global_best_error << ", Took " 
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - root_state.task_start_time).count() << "s)\n";
      } else if (optimization_budget_ms > 0) {
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - root_state.task_start_time).count();
        if (elapsed_ms >= static_cast<long long>(optimization_budget_ms)) {
          optimize_running = false;
          std::cout << "Stopped: time budget " << optimization_budget_ms
                    << "ms reached (elapsed=" << elapsed_ms
                    << "ms, iter=" << optimize_iteration
                    << ", best error=" << global_best_error << ")\n";
        }
      }
    } else {
      optimize_running = false;
    }
  }
  if (generated_patch_optimize_batch_current_region &&
      generated_patch_batch_run.current_region_started &&
      !optimize_running) {
    generated_patch_batch_run.current_region_completed = true;
  }
  // Note: optimize_points persists after optimization stops, so pattern remains visible
  
  // Status
  if (state.voronoi_pcf_ready) {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextUnformatted(
      region_is_transition(state) ? "Transition Target:" : "Input Pattern:");
    if (region_is_transition(state)) {
      ImGui::BulletText("Effective reference points: %d", state.voronoi_pcf_points_inside);
      ImGui::BulletText(
        "Blended local distributions: %zu",
        state.voronoi_pcf_individual_plots.size());
    } else {
      ImGui::BulletText("Points inside boundary: %d", state.voronoi_pcf_points_inside);
      ImGui::BulletText("In-range pair count: %d", state.voronoi_pcf_pair_count);
    }

    if (!state.voronoi_pcf_hist_plot.empty()) {
      float shared_max = 1e-6f;
      for (float v : state.voronoi_pcf_hist_plot) {
        shared_max = std::max(shared_max, v);
      }
      if (!state.output_voronoi_pcf_hist_plot.empty()) {
        for (float v : state.output_voronoi_pcf_hist_plot) {
          shared_max = std::max(shared_max, v);
        }
      }
      shared_max *= 1.05f;
      draw_distribution_plot(
        "##pcf_input_hist_plot",
        region_is_transition(state)
          ? "Transition avg local distribution"
          : "Input avg local distribution",
        state.voronoi_pcf_hist_plot,
        shared_max);
    }
  }
  
  if (state.output_voronoi_pcf_ready) {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextUnformatted("Output Pattern:");
    ImGui::BulletText("Points: %zu", state.output_pattern_points_uv.size());
    ImGui::BulletText("Pair count: %d", state.output_voronoi_pcf_pair_count);
    ImGui::BulletText("Optimizer objective: %.8g", state.output_voronoi_pcf_energy);
    ImGui::BulletText("Iterations: %d (improvements: %d)", 
                      state.optimizer_iterations_ran, 
                      state.optimizer_improvements);

    if (!state.output_voronoi_pcf_hist_plot.empty()) {
      float shared_max = 1e-6f;
      for (float v : state.output_voronoi_pcf_hist_plot) {
        shared_max = std::max(shared_max, v);
      }
      if (!state.voronoi_pcf_hist_plot.empty()) {
        for (float v : state.voronoi_pcf_hist_plot) {
          shared_max = std::max(shared_max, v);
        }
      }
      shared_max *= 1.05f;
      draw_distribution_plot(
        "##pcf_output_hist_plot",
        "Output avg local distribution",
        state.output_voronoi_pcf_hist_plot,
        shared_max);
    }
  }

  if (region_is_exemplar(state)) {
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextUnformatted("Plot export:");
    ImGui::TextDisabled(
      "Uses output export directory: %s",
      root_state.output_pattern_export_dir);
    if (ImGui::Button("Export Current Input/Output Plots", ImVec2(-1, 0))) {
      const std::string region_label =
        pattern_region_label(root_state, root_state.active_region_index);
      if (save_exemplar_region_hist_plots(
            std::string(root_state.output_pattern_export_dir),
            state,
            region_label,
            root_state.active_region_index,
            state.plot_export_status)) {
        state.plot_export_status_is_error = false;
      } else {
        state.plot_export_status_is_error = true;
      }
    }
    if (!state.plot_export_status.empty()) {
      const ImVec4 color = state.plot_export_status_is_error
        ? ImVec4(1.0f, 0.4f, 0.4f, 1.0f)
        : ImVec4(0.5f, 0.9f, 0.5f, 1.0f);
      ImGui::TextColored(color, "%s", state.plot_export_status.c_str());
    }
  }
}
