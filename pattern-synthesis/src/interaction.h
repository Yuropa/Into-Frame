#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "sampling.h"

namespace polyscope {
class SurfaceMesh;
class PointCloud;
class CurveNetwork;
class SurfaceFaceColorQuantity;
class SurfaceFaceScalarQuantity;
}

class DelaunayTraversalHelper;

constexpr int kInitialPatternRegionCount = 2;
constexpr int kPatternClassCount = 2;
constexpr int kTwoClassPairChannelCount = 3;

enum class PatternRegionMode : int {
  Exemplar = 0,
  // Legacy serialized values. New interaction state is normalized to Exemplar.
  Transition = 1,
  TwoClass = 2,
};

enum class GeneratedPatchBatchAction : int {
  None = 0,
  GeneratePoints = 1,
  Optimize = 2,
};

struct GeneratedPatchBatchRunState {
  bool active = false;
  int family_id = -1;
  int initiating_region_id = -1;
  int action = static_cast<int>(GeneratedPatchBatchAction::None);
  std::vector<int> region_ids;
  size_t current_region_offset = 0;
  bool current_region_started = false;
  bool current_region_completed = false;
  int requested_point_count = -1;
};

struct PatternRegionState {
  int region_id = -1;
  int region_mode = static_cast<int>(PatternRegionMode::Exemplar);
  int transition_source_a_region_id = -1;
  int transition_source_b_region_id = -1;
  int generated_patch_family_id = -1;
  int generated_patch_source_region_id = -1;
  int generated_patch_index = -1;
  int generated_patch_support_gap_steps = 0;
  bool generated_patch_batch_optimize_requested = false;
  bool enable_input_selection = false;
  bool enable_input_paint = false;
  bool enable_output_paint = false;
  float face_paint_brush_radius = 0.02f;
  double input_boundary_margin = 0.02;

  bool selected_dirty = false;
  bool input_boundary_dirty = false;
  bool input_boundary_pending_edits = false;
  bool output_boundary_dirty = false;
  bool output_boundary_pending_edits = false;

  std::vector<Eigen::Vector3d> pattern_points_3d;
  std::vector<Eigen::Vector2d> pattern_points_uv;
  std::vector<Eigen::Vector2d> pattern_processing_uv;
  std::vector<int> pattern_points_delaunay_triangle;
  std::vector<Eigen::Vector3i> pattern_points_delaunay_vertices;
  std::vector<int> pattern_point_class_ids;
  int active_pattern_class_id = 0;
  int last_crossed_triangle_count = -1;
  std::vector<int> voronoi_pcf_hist_counts;
  std::vector<float> voronoi_pcf_hist_plot;
  std::vector<std::vector<float>> voronoi_pcf_individual_plots;
  std::vector<float> voronoi_pcf_avg_shell_counts;
  std::vector<std::vector<int>> voronoi_pcf_raw_point_hist_counts;
  std::array<std::vector<int>, kTwoClassPairChannelCount> two_class_voronoi_pcf_hist_counts;
  std::array<std::vector<float>, kTwoClassPairChannelCount> two_class_voronoi_pcf_hist_plot;
  std::array<std::vector<std::vector<float>>, kTwoClassPairChannelCount> two_class_voronoi_pcf_individual_plots;
  std::array<std::vector<float>, kTwoClassPairChannelCount> two_class_voronoi_pcf_hist_min_plot;
  std::array<std::vector<float>, kTwoClassPairChannelCount> two_class_voronoi_pcf_hist_max_plot;
  std::array<int, kPatternClassCount> two_class_voronoi_pcf_points_inside = {0, 0};
  std::array<int, kTwoClassPairChannelCount> two_class_voronoi_pcf_pair_count = {0, 0, 0};
  std::array<int, kPatternClassCount> two_class_target_output_counts = {0, 0};
  std::array<bool, kTwoClassPairChannelCount> two_class_pair_channel_enabled = {true, true, true};
  std::array<float, kTwoClassPairChannelCount> two_class_pair_channel_weights = {1.0f, 1.0f, 1.25f};
  int two_class_anchor_class_id = 0;
  bool two_class_sequential_dependency_enabled = true;
  bool two_class_optimize_anchor_points = false;
  bool two_class_anchor_normal_stage_active = false;
  int two_class_locked_anchor_class_id = 0;
  std::vector<Eigen::Vector2d> two_class_locked_anchor_points_uv;
  std::vector<int> two_class_locked_anchor_sample_indices;
  bool two_class_envelope_penalty_enabled = false;
  float two_class_envelope_penalty_weight = 0.35f;
  bool voronoi_pcf_position_targets_enabled = false;
  int voronoi_pcf_max_k = 0;
  int voronoi_pcf_points_inside = 0;
  int voronoi_pcf_pair_count = 0;
  bool voronoi_pcf_ready = false;
  int voronoi_pcf_bin_count = 48;
  float voronoi_pcf_empty_bin_penalty = 5.f;  // penalty weight for pairs in empty input bins
  std::vector<int> output_pattern_sample_indices;
  std::vector<Eigen::Vector3d> output_pattern_points_3d;
  std::vector<Eigen::Vector2d> output_pattern_points_uv;
  std::vector<int> output_pattern_class_ids;
  std::vector<int> output_voronoi_pcf_hist_counts;
  std::vector<float> output_voronoi_pcf_hist_plot;
  std::array<std::vector<int>, kTwoClassPairChannelCount> two_class_output_voronoi_pcf_hist_counts;
  std::array<std::vector<float>, kTwoClassPairChannelCount> two_class_output_voronoi_pcf_hist_plot;
  std::array<int, kPatternClassCount> two_class_output_counts = {0, 0};
  std::array<int, kTwoClassPairChannelCount> two_class_output_voronoi_pcf_pair_count = {0, 0, 0};
  int output_voronoi_pcf_max_k = 0;
  int output_voronoi_pcf_pair_count = 0;
  bool output_voronoi_pcf_ready = false;
  double output_voronoi_pcf_energy = 0.0;
  double output_voronoi_objective_energy = 0.0;
  int baseline_graph_point_count = 32;
  int baseline_graph_target_offset_percent = 0;
  int baseline_graph_mode = 0; // 0=random, 1=clustered, 2=regular
  
  // Clustering parameters (interactive tuning)
  float clustered_sigma_multiplier = 0.4f;    // multiplier for median distance
  float clustered_max_k_multiplier = 0.8f;    // multiplier for percentile
  int clustered_percentile_choice = 75;       // 25, 50, or 75
  float clustered_num_centers_multiplier = 0.35f; // multiplier for sqrt(target_count)
  int clustered_num_centers_max = 20;         // maximum number of cluster centers
  
  int optimizer_improvements = 0;
  int optimizer_iterations_ran = 0;
  bool output_pattern_dirty = false;

  // Cached denominator support counts for output-boundary triangle centers.
  // Indexed as [support_point_index][k_bin].
  bool output_support_denominator_cache_valid = false;
  int output_support_denominator_cache_bin_count = 0;
  int output_support_denominator_cache_triangle_count = -1;
  Eigen::MatrixXd output_support_denominator_cache_boundary_uv;
  std::vector<Eigen::Vector2d> output_support_uv_cache;
  std::vector<int> output_support_tri_indices_cache;
  std::vector<std::vector<int>> output_support_k_denominator_cache;

  // Compact triangular pairwise hop-distance matrix for output support candidates.
  // For pair (sj > si): distances[sj*(sj-1)/2 + si] = hop count (int16_t, -1 = invalid).
  // Built as a side-effect of ensure_output_support_denominator_cache.
  bool output_support_pairwise_cache_valid = false;
  std::vector<int16_t> output_support_pairwise_distances;

  std::vector<int> selected_sample_indices;
  std::vector<int> input_painted_face_indices;
  std::vector<int> painted_face_indices;
  std::vector<int> input_reference_indices;

  Eigen::MatrixXd input_boundary_uv;
  Eigen::MatrixXd input_boundary_3d;
  Eigen::MatrixXd output_boundary_uv_poly;
  Eigen::MatrixXd output_boundary_3d_poly;
  Eigen::MatrixXd output_boundary_preview_uv_poly;
  Eigen::MatrixXd output_boundary_preview_3d_poly;
  std::vector<Eigen::Vector2d> generated_patch_interface_segment_uv_starts;
  std::vector<Eigen::Vector2d> generated_patch_interface_segment_uv_ends;
  Eigen::Vector2d uv_display_offset = Eigen::Vector2d::Zero();

  polyscope::PointCloud* face_centers = nullptr;
  polyscope::PointCloud* selected_samples_3d = nullptr;
  polyscope::PointCloud* selected_samples_uv = nullptr;
  polyscope::PointCloud* selected_samples_3d_class_1 = nullptr;
  polyscope::PointCloud* selected_samples_uv_class_1 = nullptr;
  polyscope::PointCloud* output_pattern_3d = nullptr;
  polyscope::PointCloud* output_pattern_uv = nullptr;
  polyscope::PointCloud* output_pattern_3d_class_1 = nullptr;
  polyscope::PointCloud* output_pattern_uv_class_1 = nullptr;
  polyscope::PointCloud* input_reference_3d = nullptr;
  polyscope::PointCloud* input_reference_uv = nullptr;
  polyscope::CurveNetwork* input_boundary_curve_3d = nullptr;
  polyscope::CurveNetwork* input_boundary_curve = nullptr;
  polyscope::CurveNetwork* output_boundary_curve_3d = nullptr;
  polyscope::CurveNetwork* output_boundary_curve_uv = nullptr;
  polyscope::SurfaceFaceScalarQuantity* output_boundary_surface = nullptr;
  polyscope::SurfaceFaceScalarQuantity* output_boundary_uv = nullptr;
  polyscope::SurfaceFaceScalarQuantity* input_boundary_surface = nullptr;
  polyscope::SurfaceFaceScalarQuantity* input_boundary_uv_faces = nullptr;

  int last_painted_face = -1;
  int last_painted_input_face = -1;

  char pattern_file_path[512] = "pattern_exemplar.txt";
  char display_name[64] = "";
  std::string pattern_file_status;
  bool pattern_file_status_is_error = false;
  std::string plot_export_status;
  bool plot_export_status_is_error = false;
};

struct InteractionState {
  std::chrono::steady_clock::time_point task_start_time;
  int active_region_index = 0;
  int next_region_id = 0;
  int next_generated_patch_family_id = 0;

  polyscope::PointCloud* face_centers = nullptr;
  std::vector<PatternRegionState> regions =
    std::vector<PatternRegionState>(kInitialPatternRegionCount);
  bool generated_patch_batch_mode_enabled = false;
  GeneratedPatchBatchRunState generated_patch_batch_run;
  bool generated_patch_batch_cancel_requested = false;
  std::string generated_patch_batch_status;
  bool generated_patch_batch_status_is_error = false;
  bool whole_model_patch_preview_active = false;
  int whole_model_patch_preview_source_region_id = -1;
  std::vector<std::vector<int>> whole_model_patch_preview_face_partitions;
  polyscope::SurfaceFaceColorQuantity* whole_model_patch_preview_surface = nullptr;
  polyscope::SurfaceFaceColorQuantity* whole_model_patch_preview_uv = nullptr;
  char output_pattern_export_dir[512] = "output_patterns";
  std::string output_pattern_export_status;
  bool output_pattern_export_status_is_error = false;
};

inline int region_count(const InteractionState& state) {
  return std::max(1, static_cast<int>(state.regions.size()));
}

inline bool region_is_transition(const PatternRegionState& region) {
  (void)region;
  return false;
}

inline bool region_is_two_class(const PatternRegionState& region) {
  (void)region;
  return false;
}

inline bool region_is_exemplar(const PatternRegionState& region) {
  (void)region;
  return true;
}

inline bool region_is_generated_patch_exemplar(const PatternRegionState& region) {
  return region_is_exemplar(region) && region.generated_patch_family_id >= 0;
}

inline bool region_uses_exemplar_input(const PatternRegionState& region) {
  (void)region;
  return true;
}

inline int sanitize_pattern_class_id(int class_id) {
  (void)class_id;
  return 0;
}

inline const char* pattern_region_mode_label(const PatternRegionState& region) {
  if (region_is_transition(region)) {
    return "transition";
  }
  if (region_is_two_class(region)) {
    return "2-class";
  }
  return "exemplar";
}

inline int clamp_region_index(const InteractionState& state, int index) {
  return std::clamp(index, 0, region_count(state) - 1);
}

inline PatternRegionState& region_state(InteractionState& state, int index) {
  return state.regions[static_cast<size_t>(clamp_region_index(state, index))];
}

inline const PatternRegionState& region_state(const InteractionState& state, int index) {
  return state.regions[static_cast<size_t>(clamp_region_index(state, index))];
}

inline PatternRegionState& active_region(InteractionState& state) {
  return region_state(state, state.active_region_index);
}

inline const PatternRegionState& active_region(const InteractionState& state) {
  return region_state(state, state.active_region_index);
}

inline int region_index_from_id(const InteractionState& state, int region_id) {
  for (int region_index = 0; region_index < region_count(state); ++region_index) {
    if (region_state(state, region_index).region_id == region_id) {
      return region_index;
    }
  }
  return -1;
}

inline PatternRegionState* find_region_by_id(InteractionState& state, int region_id) {
  const int region_index = region_index_from_id(state, region_id);
  if (region_index < 0) {
    return nullptr;
  }
  return &region_state(state, region_index);
}

inline const PatternRegionState* find_region_by_id(const InteractionState& state, int region_id) {
  const int region_index = region_index_from_id(state, region_id);
  if (region_index < 0) {
    return nullptr;
  }
  return &region_state(state, region_index);
}

inline std::string default_pattern_region_label(int index) {
  if (index >= 0 && index < 26) {
    return std::string(1, static_cast<char>('A' + index));
  }
  return "R" + std::to_string(index + 1);
}

inline std::string pattern_region_label(const PatternRegionState& region, int index) {
  if (region.display_name[0] != '\0') {
    return std::string(region.display_name);
  }
  return default_pattern_region_label(index);
}

inline std::string pattern_region_label(const InteractionState& state, int index) {
  return pattern_region_label(region_state(state, index), index);
}

std::vector<int> generated_patch_family_region_indices(
  const InteractionState& state,
  int family_id);

void set_generated_patch_family_optimize_requested(
  InteractionState& state,
  int family_id,
  bool requested);

void clear_generated_patch_batch_run(
  InteractionState& state,
  bool clear_optimize_requested = true);

bool begin_generated_patch_batch_run(
  InteractionState& state,
  int initiating_region_id,
  GeneratedPatchBatchAction action,
  int requested_point_count,
  bool current_region_started,
  bool current_region_completed);

void init_interaction(
  InteractionState& state,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXi& F);

void reset_interaction_for_new_model(
  InteractionState& state,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh);

void handle_interaction_input(
  InteractionState& state,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F,
  const Eigen::MatrixXd& sample_points_3d,
  const Eigen::MatrixXd& sample_points_uv,
  const DelaunayTraversalHelper* delaunay_helper);

void build_interaction_ui(
  InteractionState& state,
  polyscope::SurfaceMesh* surfaceMesh,
  polyscope::SurfaceMesh* uvMesh,
  const std::vector<DartSample>& samples_uv,
  const Eigen::MatrixXd& V,
  const Eigen::MatrixXd& points_3d,
  const Eigen::MatrixXd& points_uv,
  const Eigen::MatrixXd& UV,
  const Eigen::MatrixXi& F,
  const DelaunayTraversalHelper* delaunay_helper,
  const Eigen::VectorXd* face_distortion = nullptr);
