#include "synthesis_core.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

// ============================================================
// Geometry helpers
// ============================================================

// Ray-casting point-in-polygon test. Returns true if uv is strictly inside.
static bool point_in_polygon(
    const Eigen::Vector2d& uv,
    const Eigen::MatrixXd& poly) {
    const int n = static_cast<int>(poly.rows());
    if (n < 3) return false;
    bool inside = false;
    int j = n - 1;
    for (int i = 0; i < n; ++i) {
        const double xi = poly(i, 0), yi = poly(i, 1);
        const double xj = poly(j, 0), yj = poly(j, 1);
        if (((yi > uv.y()) != (yj > uv.y())) &&
            (uv.x() < (xj - xi) * (uv.y() - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
        j = i;
    }
    return inside;
}

// Collect triangle centers within boundary polygon.
// Returns UV positions and their triangle indices.
static void collect_candidates(
    const DelaunayTraversalHelper& delaunay,
    const Eigen::MatrixXd& boundary_uv,
    std::vector<Eigen::Vector2d>& out_uv,
    std::vector<int>& out_tri_indices) {
    out_uv.clear();
    out_tri_indices.clear();
    const int n_tri = delaunay.triangle_count();
    out_uv.reserve(static_cast<size_t>(n_tri));
    out_tri_indices.reserve(static_cast<size_t>(n_tri));
    for (int t = 0; t < n_tri; ++t) {
        Eigen::Vector2d center;
        if (!delaunay.triangle_center(t, center)) continue;
        if (!point_in_polygon(center, boundary_uv)) continue;
        out_uv.push_back(center);
        out_tri_indices.push_back(t);
    }
}

// ============================================================
// PCF computation
// Mirrors build_pair_hist_and_average_individual_plot from voronoi-pcf.cpp
// but takes plain vectors, no InteractionState.
// ============================================================

struct PCFData {
    std::vector<float> avg_plot;               // normalized average PCF histogram
    std::vector<std::vector<int>> point_hist;  // per-point pair counts [n][bin_count]
    std::vector<std::vector<int>> point_support; // per-point support counts [n][bin_count]
    std::vector<int> global_hist;
    int valid_points = 0;
    int bin_count = 0;
};

// Compute support counts: for each point in uv_points, how many support_points
// are at each hop distance? (Used as denominator for normalization.)
static void compute_support_counts(
    const std::vector<Eigen::Vector2d>& uv_points,
    const std::vector<Eigen::Vector2d>& support_points,
    const DelaunayTraversalHelper& delaunay,
    int bin_count,
    std::vector<std::vector<int>>& out_support) {
    const int n = static_cast<int>(uv_points.size());
    const int s = static_cast<int>(support_points.size());
    out_support.assign(static_cast<size_t>(n),
                       std::vector<int>(static_cast<size_t>(bin_count), 0));
    for (int i = 0; i < n; ++i) {
        for (int si = 0; si < s; ++si) {
            const int k = delaunay.count_triangles_crossed(
                uv_points[static_cast<size_t>(i)],
                support_points[static_cast<size_t>(si)]);
            if (k >= 0) {
                const int bin = std::min(k, bin_count - 1);
                ++out_support[static_cast<size_t>(i)][static_cast<size_t>(bin)];
            }
        }
    }
}

static PCFData compute_pcf(
    const std::vector<Eigen::Vector2d>& uv_points,
    const std::vector<Eigen::Vector2d>& support_points,
    const DelaunayTraversalHelper& delaunay,
    int bin_count) {
    PCFData result;
    result.bin_count = bin_count;
    const int n = static_cast<int>(uv_points.size());
    if (n < 2 || bin_count <= 0 || !delaunay.is_ready()) return result;

    result.global_hist.assign(static_cast<size_t>(bin_count), 0);
    result.point_hist.assign(static_cast<size_t>(n),
                             std::vector<int>(static_cast<size_t>(bin_count), 0));

    // Pairwise hop distances → populate point_hist and global_hist.
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const int k = delaunay.count_triangles_crossed(
                uv_points[static_cast<size_t>(i)],
                uv_points[static_cast<size_t>(j)]);
            if (k < 0) continue;
            const int bin = std::min(k, bin_count - 1);
            ++result.global_hist[static_cast<size_t>(bin)];
            ++result.point_hist[static_cast<size_t>(i)][static_cast<size_t>(bin)];
            ++result.point_hist[static_cast<size_t>(j)][static_cast<size_t>(bin)];
        }
    }

    // Support counts.
    const std::vector<Eigen::Vector2d>& supp =
        support_points.empty() ? uv_points : support_points;
    compute_support_counts(uv_points, supp, delaunay, bin_count, result.point_support);

    // Average individual PCF plot.
    result.avg_plot.assign(static_cast<size_t>(bin_count), 0.0f);
    int valid = 0;
    for (int i = 0; i < n; ++i) {
        bool has_support = false;
        for (int k = 0; k < bin_count; ++k) {
            if (result.point_support[static_cast<size_t>(i)][static_cast<size_t>(k)] > 0) {
                has_support = true;
                break;
            }
        }
        if (!has_support) continue;
        ++valid;
        for (int k = 0; k < bin_count; ++k) {
            const int denom =
                result.point_support[static_cast<size_t>(i)][static_cast<size_t>(k)];
            if (denom <= 0) continue;
            result.avg_plot[static_cast<size_t>(k)] +=
                static_cast<float>(result.point_hist[static_cast<size_t>(i)][static_cast<size_t>(k)]) /
                static_cast<float>(denom);
        }
    }
    if (valid > 0) {
        const float inv = 1.0f / static_cast<float>(valid);
        for (float& v : result.avg_plot) v *= inv;
    }
    result.valid_points = valid;
    return result;
}

// Recompute avg_plot from current point_hist and point_support.
static std::vector<float> avg_plot_from_pcf(const PCFData& pcf) {
    const int n = static_cast<int>(pcf.point_hist.size());
    std::vector<float> avg(static_cast<size_t>(pcf.bin_count), 0.0f);
    int valid = 0;
    for (int i = 0; i < n; ++i) {
        bool has_support = false;
        for (int k = 0; k < pcf.bin_count; ++k) {
            if (pcf.point_support[static_cast<size_t>(i)][static_cast<size_t>(k)] > 0) {
                has_support = true; break;
            }
        }
        if (!has_support) continue;
        ++valid;
        for (int k = 0; k < pcf.bin_count; ++k) {
            const int denom = pcf.point_support[static_cast<size_t>(i)][static_cast<size_t>(k)];
            if (denom <= 0) continue;
            avg[static_cast<size_t>(k)] +=
                static_cast<float>(pcf.point_hist[static_cast<size_t>(i)][static_cast<size_t>(k)]) /
                static_cast<float>(denom);
        }
    }
    if (valid > 0) {
        const float inv = 1.0f / static_cast<float>(valid);
        for (float& v : avg) v *= inv;
    }
    return avg;
}

// ============================================================
// Energy function — mirrors weighted_distribution_l2 from voronoi-pcf.cpp
// ============================================================

static double energy_l2(
    const std::vector<float>& current,
    const std::vector<float>& target) {
    const size_t n = std::max(current.size(), target.size());
    if (n == 0) return 0.0;

    constexpr double kZeroBinTol = 1e-12;
    constexpr double kZeroBinQuadraticPenalty = 80.0;
    constexpr double kZeroBinLeakagePenalty = 400.0;
    const size_t strong_prefix = std::min(n, std::max<size_t>(3, n / 4));
    const size_t early_span = std::max<size_t>(strong_prefix, n / 2);

    double energy = 0.0;
    double cdf_cur = 0.0, cdf_tgt = 0.0;
    double zero_leakage = 0.0;

    for (size_t i = 0; i < n; ++i) {
        const double a = (i < current.size()) ? current[i] : 0.0;
        const double b = (i < target.size()) ? target[i] : 0.0;
        const double d = a - b;
        double early_focus = 0.75;
        if (i < strong_prefix) {
            const double t = static_cast<double>(i) /
                             static_cast<double>(std::max<size_t>(1, strong_prefix));
            early_focus = 12.0 - 7.0 * t;
        } else if (i < early_span) {
            const double t =
                static_cast<double>(i - strong_prefix) /
                static_cast<double>(std::max<size_t>(1, early_span - strong_prefix));
            early_focus = 5.0 - 3.0 * t;
        }
        const double w = early_focus * (1.0 + 4.0 * b);
        energy += w * d * d;
        if (b <= kZeroBinTol) {
            zero_leakage += a;
            energy += kZeroBinQuadraticPenalty * a * a;
        }
        cdf_cur += a;
        cdf_tgt += b;
        const double cdf_d = cdf_cur - cdf_tgt;
        const double cdf_w = (i < strong_prefix) ? 1.5 : ((i < early_span) ? 0.6 : 0.2);
        energy += cdf_w * cdf_d * cdf_d;
    }
    energy += kZeroBinLeakagePenalty * zero_leakage * zero_leakage;
    return energy;
}

// ============================================================
// Initial placement — farthest-point greedy on Delaunay graph
// ============================================================

static std::vector<int> place_farthest_point(
    const std::vector<Eigen::Vector2d>& candidates,
    const DelaunayTraversalHelper& delaunay,
    int target_count,
    std::mt19937& rng) {
    const int m = static_cast<int>(candidates.size());
    if (m == 0 || target_count <= 0) return {};
    target_count = std::min(target_count, m);

    // Precompute pairwise hop distances (O(m^2) — done once at initialization).
    // Use int16_t to keep memory reasonable; hop counts > 32767 are clamped.
    const size_t pair_size = static_cast<size_t>(m) * static_cast<size_t>(m - 1) / 2;
    std::vector<int16_t> dists(pair_size, -1);
    const auto pair_idx = [m](int i, int j) -> size_t {
        if (i > j) std::swap(i, j);
        return static_cast<size_t>(j) * static_cast<size_t>(j - 1) / 2 +
               static_cast<size_t>(i);
    };
    const auto get_dist = [&](int i, int j) -> double {
        if (i == j) return 0.0;
        const int16_t d = dists[pair_idx(i, j)];
        return (d < 0) ? std::numeric_limits<double>::infinity()
                       : static_cast<double>(d);
    };

    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < m; ++j) {
            const int k = delaunay.count_triangles_crossed(
                candidates[static_cast<size_t>(i)],
                candidates[static_cast<size_t>(j)]);
            dists[pair_idx(i, j)] =
                static_cast<int16_t>(std::clamp(k, -1, 32767));
        }
    }

    // Greedy farthest-point: pick seed randomly, then always pick the
    // candidate farthest from the already-selected set.
    std::vector<int> selected;
    selected.reserve(static_cast<size_t>(target_count));
    std::uniform_int_distribution<int> pick_seed(0, m - 1);
    selected.push_back(pick_seed(rng));

    std::vector<double> min_dist(static_cast<size_t>(m),
                                 std::numeric_limits<double>::infinity());
    for (int i = 0; i < m; ++i) {
        min_dist[static_cast<size_t>(i)] = get_dist(selected[0], i);
    }

    while (static_cast<int>(selected.size()) < target_count) {
        int best = -1;
        double best_d = -1.0;
        for (int i = 0; i < m; ++i) {
            if (min_dist[static_cast<size_t>(i)] > best_d) {
                best_d = min_dist[static_cast<size_t>(i)];
                best = i;
            }
        }
        if (best < 0) break;
        selected.push_back(best);
        for (int i = 0; i < m; ++i) {
            const double d = get_dist(best, i);
            if (d < min_dist[static_cast<size_t>(i)])
                min_dist[static_cast<size_t>(i)] = d;
        }
    }
    return selected;
}

// ============================================================
// Incremental PCF state for the optimizer
// ============================================================

struct IncState {
    // point_hist[i][k]    = number of other output points at hop k from point i
    // point_support[i][k] = number of candidate triangles at hop k from point i
    //                       (precomputed per triangle → looked up by tri index)
    std::vector<std::vector<int>> point_hist;
    std::vector<std::vector<int>> point_support;
    std::vector<int> global_hist;
    int bin_count = 0;
    int n_points = 0;

    // Keyed by candidate triangle index: support_by_tri[tri_idx][k]
    std::vector<std::vector<int>> support_by_candidate; // indexed by candidate pos
    // candidate_for_output[i] = which candidate index output point i occupies
    std::vector<int> candidate_for_output;
};

static IncState build_inc_state(
    const std::vector<int>& selected,              // indices into candidates
    const std::vector<Eigen::Vector2d>& candidates,
    const std::vector<std::vector<int>>& cand_support, // precomputed per-candidate
    const DelaunayTraversalHelper& delaunay,
    int bin_count) {
    IncState inc;
    inc.bin_count = bin_count;
    const int n = static_cast<int>(selected.size());
    inc.n_points = n;
    inc.support_by_candidate = cand_support;
    inc.candidate_for_output = selected;

    inc.global_hist.assign(static_cast<size_t>(bin_count), 0);
    inc.point_hist.assign(static_cast<size_t>(n),
                          std::vector<int>(static_cast<size_t>(bin_count), 0));
    inc.point_support.resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        inc.point_support[static_cast<size_t>(i)] =
            cand_support[static_cast<size_t>(selected[static_cast<size_t>(i)])];
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const int k = delaunay.count_triangles_crossed(
                candidates[static_cast<size_t>(selected[static_cast<size_t>(i)])],
                candidates[static_cast<size_t>(selected[static_cast<size_t>(j)])]);
            if (k < 0) continue;
            const int bin = std::min(k, bin_count - 1);
            ++inc.global_hist[static_cast<size_t>(bin)];
            ++inc.point_hist[static_cast<size_t>(i)][static_cast<size_t>(bin)];
            ++inc.point_hist[static_cast<size_t>(j)][static_cast<size_t>(bin)];
        }
    }
    return inc;
}

static std::vector<float> avg_plot_from_inc(const IncState& inc) {
    const int n = inc.n_points;
    std::vector<float> avg(static_cast<size_t>(inc.bin_count), 0.0f);
    int valid = 0;
    for (int i = 0; i < n; ++i) {
        bool has_supp = false;
        for (int k = 0; k < inc.bin_count; ++k) {
            if (inc.point_support[static_cast<size_t>(i)][static_cast<size_t>(k)] > 0) {
                has_supp = true; break;
            }
        }
        if (!has_supp) continue;
        ++valid;
        for (int k = 0; k < inc.bin_count; ++k) {
            const int denom =
                inc.point_support[static_cast<size_t>(i)][static_cast<size_t>(k)];
            if (denom <= 0) continue;
            avg[static_cast<size_t>(k)] +=
                static_cast<float>(inc.point_hist[static_cast<size_t>(i)][static_cast<size_t>(k)]) /
                static_cast<float>(denom);
        }
    }
    if (valid > 0) {
        const float inv = 1.0f / static_cast<float>(valid);
        for (float& v : avg) v *= inv;
    }
    return avg;
}

// Incremental move: move output point `point_idx` to `new_cand`.
// Saves enough state to undo the move.
struct MoveRecord {
    int point_idx;
    int old_cand;
    int new_cand;
    std::vector<int> old_hist_i;                   // saved point_hist[point_idx]
    std::vector<int> old_support_i;                // saved point_support[point_idx]
    std::vector<int> affected_j;
    std::vector<std::vector<int>> old_hist_j;      // saved point_hist[j] for each affected j
    std::vector<int> saved_global_hist;            // saved global_hist
};

static void apply_move(
    IncState& inc,
    const std::vector<Eigen::Vector2d>& candidates,
    const DelaunayTraversalHelper& delaunay,
    int point_idx,
    int new_cand,
    MoveRecord& rec) {
    const int old_cand = inc.candidate_for_output[static_cast<size_t>(point_idx)];
    rec.point_idx = point_idx;
    rec.old_cand = old_cand;
    rec.new_cand = new_cand;

    // Snapshot everything that will change so undo_move is a pure restore.
    rec.old_hist_i = inc.point_hist[static_cast<size_t>(point_idx)];
    rec.old_support_i = inc.point_support[static_cast<size_t>(point_idx)];
    rec.saved_global_hist = inc.global_hist;
    rec.affected_j.clear();
    rec.old_hist_j.clear();

    const Eigen::Vector2d& old_uv = candidates[static_cast<size_t>(old_cand)];
    const Eigen::Vector2d& new_uv = candidates[static_cast<size_t>(new_cand)];

    const int n = inc.n_points;
    for (int j = 0; j < n; ++j) {
        if (j == point_idx) continue;
        const Eigen::Vector2d& j_uv =
            candidates[static_cast<size_t>(inc.candidate_for_output[static_cast<size_t>(j)])];
        const int old_k = delaunay.count_triangles_crossed(j_uv, old_uv);
        const int new_k = delaunay.count_triangles_crossed(j_uv, new_uv);
        if (old_k == new_k) continue;

        rec.affected_j.push_back(j);
        rec.old_hist_j.push_back(inc.point_hist[static_cast<size_t>(j)]);

        if (old_k >= 0) {
            const int ob = std::min(old_k, inc.bin_count - 1);
            --inc.global_hist[static_cast<size_t>(ob)];
            --inc.point_hist[static_cast<size_t>(point_idx)][static_cast<size_t>(ob)];
            --inc.point_hist[static_cast<size_t>(j)][static_cast<size_t>(ob)];
        }
        if (new_k >= 0) {
            const int nb = std::min(new_k, inc.bin_count - 1);
            ++inc.global_hist[static_cast<size_t>(nb)];
            ++inc.point_hist[static_cast<size_t>(point_idx)][static_cast<size_t>(nb)];
            ++inc.point_hist[static_cast<size_t>(j)][static_cast<size_t>(nb)];
        }
    }

    inc.candidate_for_output[static_cast<size_t>(point_idx)] = new_cand;
    inc.point_support[static_cast<size_t>(point_idx)] =
        inc.support_by_candidate[static_cast<size_t>(new_cand)];
}

static void undo_move(IncState& inc, const MoveRecord& rec) {
    inc.candidate_for_output[static_cast<size_t>(rec.point_idx)] = rec.old_cand;
    inc.point_support[static_cast<size_t>(rec.point_idx)] = rec.old_support_i;
    inc.point_hist[static_cast<size_t>(rec.point_idx)] = rec.old_hist_i;
    inc.global_hist = rec.saved_global_hist;
    for (size_t ai = 0; ai < rec.affected_j.size(); ++ai) {
        inc.point_hist[static_cast<size_t>(rec.affected_j[ai])] = rec.old_hist_j[ai];
    }
}

// ============================================================
// Candidate BFS neighborhood expansion
// ============================================================

// Returns candidate indices (into the candidates array) that are within
// `radius` triangle-adjacency hops from the triangle at candidates[cand_idx].
// Uses triangle adjacency from the Delaunay mesh.
static std::vector<int> neighbors_within_radius(
    const DelaunayTraversalHelper& delaunay,
    const std::vector<int>& candidate_tri_indices, // tri index for each candidate pos
    const std::vector<int>& tri_to_cand,           // tri_index → candidate index (-1 if none)
    int center_cand,
    int radius) {
    std::vector<int> result;
    const int center_tri = candidate_tri_indices[static_cast<size_t>(center_cand)];
    if (center_tri < 0) return result;

    std::unordered_set<int> visited_tri;
    std::queue<std::pair<int, int>> q; // (tri_idx, depth)
    q.push({center_tri, 0});
    visited_tri.insert(center_tri);

    while (!q.empty()) {
        const auto [tri, depth] = q.front(); q.pop();
        const int cand = tri_to_cand[static_cast<size_t>(tri)];
        if (cand >= 0 && cand != center_cand) {
            result.push_back(cand);
        }
        if (depth >= radius) continue;
        std::array<int, 3> nbrs;
        delaunay.get_triangle_neighbors(tri, nbrs);
        for (int nbr : nbrs) {
            if (nbr >= 0 && visited_tri.find(nbr) == visited_tri.end()) {
                visited_tri.insert(nbr);
                q.push({nbr, depth + 1});
            }
        }
    }
    return result;
}

// ============================================================
// Main synthesis function
// ============================================================

SynthesisResult synthesize_pattern(
    const DelaunayTraversalHelper& delaunay,
    const std::vector<Eigen::Vector2d>& exemplar_uv,
    const Eigen::MatrixXd& input_boundary_uv,
    const Eigen::MatrixXd& output_boundary_uv,
    int n_points,
    int bin_count,
    int max_iters,
    int seed) {
    SynthesisResult result;
    result.final_energy = std::numeric_limits<double>::infinity();
    result.iterations_ran = 0;

    if (!delaunay.is_ready() || exemplar_uv.empty() ||
        input_boundary_uv.rows() < 3 || output_boundary_uv.rows() < 3 ||
        bin_count <= 0) {
        return result;
    }

    std::mt19937 rng(static_cast<uint32_t>(seed));

    // ----------------------------------------------------------
    // 1. Filter exemplar points to those inside input boundary.
    // ----------------------------------------------------------
    std::vector<Eigen::Vector2d> in_exemplar;
    in_exemplar.reserve(exemplar_uv.size());
    for (const auto& uv : exemplar_uv) {
        if (point_in_polygon(uv, input_boundary_uv)) in_exemplar.push_back(uv);
    }
    if (static_cast<int>(in_exemplar.size()) < 2) {
        std::cerr << "synthesis_pattern: fewer than 2 exemplar points inside input boundary\n";
        return result;
    }

    // ----------------------------------------------------------
    // 2. Collect candidate positions in output boundary.
    // ----------------------------------------------------------
    std::vector<Eigen::Vector2d> out_candidates;
    std::vector<int> out_candidate_tri;
    collect_candidates(delaunay, output_boundary_uv, out_candidates, out_candidate_tri);

    // Collect candidate positions in input boundary too (for exemplar support).
    std::vector<Eigen::Vector2d> in_candidates;
    std::vector<int> in_candidate_tri;
    collect_candidates(delaunay, input_boundary_uv, in_candidates, in_candidate_tri);

    if (static_cast<int>(out_candidates.size()) < 2) {
        std::cerr << "synthesis_pattern: too few candidate positions in output boundary\n";
        return result;
    }

    // ----------------------------------------------------------
    // 3. Compute exemplar PCF.
    // ----------------------------------------------------------
    const PCFData exemplar_pcf = compute_pcf(in_exemplar, in_candidates, delaunay, bin_count);
    if (exemplar_pcf.valid_points < 2) {
        std::cerr << "synthesis_pattern: exemplar PCF computation failed\n";
        return result;
    }
    result.exemplar_pcf = exemplar_pcf.avg_plot;

    // ----------------------------------------------------------
    // 4. Infer target point count from density if not specified.
    // ----------------------------------------------------------
    const int m_out = static_cast<int>(out_candidates.size());
    const int m_in  = static_cast<int>(in_candidates.size());
    if (n_points <= 0 && m_in > 0) {
        const double density =
            static_cast<double>(in_exemplar.size()) / static_cast<double>(m_in);
        n_points = static_cast<int>(std::llround(density * static_cast<double>(m_out)));
    }
    n_points = std::max(2, std::min(n_points, m_out));

    std::cout << "synthesize_pattern: exemplar_pts=" << in_exemplar.size()
              << " input_support=" << m_in
              << " output_support=" << m_out
              << " target_n=" << n_points
              << " bins=" << bin_count << "\n";

    // ----------------------------------------------------------
    // 5. Precompute support counts per output candidate.
    //    support_by_cand[c][k] = number of candidates at hop k from candidate c.
    // ----------------------------------------------------------
    std::vector<std::vector<int>> support_by_cand(
        static_cast<size_t>(m_out),
        std::vector<int>(static_cast<size_t>(bin_count), 0));
    for (int i = 0; i < m_out; ++i) {
        for (int j = 0; j < m_out; ++j) {
            const int k = delaunay.count_triangles_crossed(
                out_candidates[static_cast<size_t>(i)],
                out_candidates[static_cast<size_t>(j)]);
            if (k >= 0) {
                const int bin = std::min(k, bin_count - 1);
                ++support_by_cand[static_cast<size_t>(i)][static_cast<size_t>(bin)];
            }
        }
    }

    // Reverse map: triangle index → candidate index.
    const int n_tri = delaunay.triangle_count();
    std::vector<int> tri_to_cand(static_cast<size_t>(std::max(0, n_tri)), -1);
    for (int c = 0; c < m_out; ++c) {
        const int t = out_candidate_tri[static_cast<size_t>(c)];
        if (t >= 0 && t < n_tri) tri_to_cand[static_cast<size_t>(t)] = c;
    }

    // ----------------------------------------------------------
    // 6. Farthest-point initial placement.
    // ----------------------------------------------------------
    std::vector<int> selected = place_farthest_point(out_candidates, delaunay, n_points, rng);
    if (static_cast<int>(selected.size()) < 2) {
        std::cerr << "synthesis_pattern: initial placement failed\n";
        return result;
    }

    // ----------------------------------------------------------
    // 7. Build incremental PCF state.
    // ----------------------------------------------------------
    IncState inc = build_inc_state(
        selected, out_candidates, support_by_cand, delaunay, bin_count);

    // ----------------------------------------------------------
    // 8. Stochastic coordinate descent.
    // ----------------------------------------------------------
    // Track which candidate positions are occupied to prevent collisions.
    std::vector<bool> occupied(static_cast<size_t>(m_out), false);
    for (int s : selected) occupied[static_cast<size_t>(s)] = true;

    const int n_out = static_cast<int>(selected.size());
    std::vector<int> point_order(static_cast<size_t>(n_out));
    std::iota(point_order.begin(), point_order.end(), 0);

    std::vector<float> current_avg = avg_plot_from_inc(inc);
    double current_energy = energy_l2(current_avg, result.exemplar_pcf);

    double best_energy = current_energy;
    std::vector<int> best_selected = inc.candidate_for_output;

    constexpr int kLocalRadius = 3;
    constexpr int kGlobalProposals = 4;

    for (int iter = 0; iter < max_iters; ++iter) {
        std::shuffle(point_order.begin(), point_order.end(), rng);
        int moves_accepted = 0;

        for (int pi : point_order) {
            const int cur_cand = inc.candidate_for_output[static_cast<size_t>(pi)];

            // Build candidate moves: local neighborhood + random global.
            std::vector<int> proposals = neighbors_within_radius(
                delaunay, out_candidate_tri, tri_to_cand, cur_cand, kLocalRadius);

            // Add a few random global candidates.
            std::uniform_int_distribution<int> rand_cand(0, m_out - 1);
            for (int g = 0; g < kGlobalProposals; ++g) {
                proposals.push_back(rand_cand(rng));
            }

            // Evaluate each proposal.
            MoveRecord rec;
            for (int cand : proposals) {
                if (cand == cur_cand) continue;
                if (occupied[static_cast<size_t>(cand)]) continue;

                apply_move(inc, out_candidates, delaunay, pi, cand, rec);
                const std::vector<float> new_avg = avg_plot_from_inc(inc);
                const double new_energy = energy_l2(new_avg, result.exemplar_pcf);

                if (new_energy < current_energy) {
                    occupied[static_cast<size_t>(cur_cand)] = false;
                    occupied[static_cast<size_t>(cand)] = true;
                    current_energy = new_energy;
                    current_avg = new_avg;
                    ++moves_accepted;
                    if (new_energy < best_energy) {
                        best_energy = new_energy;
                        best_selected = inc.candidate_for_output;
                    }
                    break; // accept first improving move, move to next point
                } else {
                    undo_move(inc, rec);
                }
            }
        }

        ++result.iterations_ran;

        if (moves_accepted == 0) {
            // No improvement this sweep — done.
            break;
        }
    }

    std::cout << "synthesize_pattern: finished after " << result.iterations_ran
              << " iterations, energy " << current_energy << " -> " << best_energy << "\n";

    // ----------------------------------------------------------
    // 9. Restore best configuration and extract output.
    // ----------------------------------------------------------
    result.final_energy = best_energy;
    result.output_uv.reserve(static_cast<size_t>(n_out));
    for (int c : best_selected) {
        result.output_uv.push_back(out_candidates[static_cast<size_t>(c)]);
    }

    // Recompute final output PCF for diagnostics.
    {
        IncState best_inc = build_inc_state(
            best_selected, out_candidates, support_by_cand, delaunay, bin_count);
        result.output_pcf = avg_plot_from_inc(best_inc);
    }

    return result;
}
