#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <igl/boundary_loop.h>
#include <igl/lscm.h>

// Existing library headers (polyscope-free)
#include "sampling.h"
#include "lloyd_relaxation.h"
#include "synthesis_core.h"

namespace py = pybind11;

// ============================================================
// LSCM parameterization — re-implemented without polyscope.
// Logic mirrors parameterization.cpp but strips all rendering.
// ============================================================

namespace {

struct UvQuality {
    bool valid = false;
    int degenerate_face_count = std::numeric_limits<int>::max();
    double min_positive_face_area = 0.0;
    double total_face_area = 0.0;
};

UvQuality evaluate_uv_quality(const Eigen::MatrixXd& UV, const Eigen::MatrixXi& F) {
    UvQuality q;
    if (UV.rows() == 0 || UV.cols() < 2 || F.rows() == 0 || F.cols() < 3) return q;
    q.valid = true;
    q.degenerate_face_count = 0;
    double min_pos = std::numeric_limits<double>::infinity();
    for (int f = 0; f < F.rows(); ++f) {
        const int i0 = F(f, 0), i1 = F(f, 1), i2 = F(f, 2);
        if (i0 < 0 || i1 < 0 || i2 < 0 ||
            i0 >= UV.rows() || i1 >= UV.rows() || i2 >= UV.rows()) {
            q.valid = false; return q;
        }
        const Eigen::Vector2d u0 = UV.row(i0).head<2>();
        const Eigen::Vector2d u1 = UV.row(i1).head<2>();
        const Eigen::Vector2d u2 = UV.row(i2).head<2>();
        if (!u0.allFinite() || !u1.allFinite() || !u2.allFinite()) { q.valid = false; return q; }
        const double area = 0.5 * std::abs(
            (u1.x() - u0.x()) * (u2.y() - u0.y()) -
            (u1.y() - u0.y()) * (u2.x() - u0.x()));
        if (!std::isfinite(area)) { q.valid = false; return q; }
        q.total_face_area += area;
        if (area < 1e-12) ++q.degenerate_face_count;
        else if (area < min_pos) min_pos = area;
    }
    q.min_positive_face_area = std::isfinite(min_pos) ? min_pos : 0.0;
    return q;
}

bool uv_quality_better(const UvQuality& a, const UvQuality& b) {
    if (!a.valid) return false;
    if (!b.valid) return true;
    if (a.degenerate_face_count != b.degenerate_face_count)
        return a.degenerate_face_count < b.degenerate_face_count;
    if (a.min_positive_face_area != b.min_positive_face_area)
        return a.min_positive_face_area > b.min_positive_face_area;
    return a.total_face_area > b.total_face_area;
}

bool build_lscm_constraints(
    const Eigen::MatrixXd& V, int va, int vb,
    Eigen::VectorXi& out_b, Eigen::MatrixXd& out_bc) {
    if (va < 0 || vb < 0 || va == vb || va >= V.rows() || vb >= V.rows()) return false;
    double pin_spacing = 1.0;
    if (V.cols() >= 3) {
        pin_spacing = (V.row(vb).head<3>() - V.row(va).head<3>()).norm();
        if (!std::isfinite(pin_spacing) || pin_spacing <= 1e-8) pin_spacing = 1.0;
    }
    out_b.resize(2);
    out_b << va, vb;
    out_bc.resize(2, 2);
    out_bc << 0.0, 0.0, pin_spacing, 0.0;
    return true;
}

std::vector<std::pair<int, int>> build_lscm_candidates(
    const Eigen::MatrixXd& V, const Eigen::VectorXi& loop) {
    std::vector<std::pair<int, int>> pairs;
    std::set<std::pair<int, int>> seen;
    if (loop.size() < 2) return pairs;

    auto add = [&](int a, int b) {
        if (a < 0 || b < 0 || a == b) return;
        auto key = std::make_pair(std::min(a, b), std::max(a, b));
        if (!seen.insert(key).second) return;
        pairs.emplace_back(a, b);
    };

    const int n = static_cast<int>(loop.size());
    for (int k = 0; k < 8; ++k)
        add(loop[(k * n) / 8], loop[((k * n) / 8 + n / 2) % n]);

    // Add the geometrically farthest pair
    if (V.cols() >= 3) {
        double best = -1.0;
        int fa = -1, fb = -1;
        for (int i = 0; i < n; ++i) {
            const int vi = loop[i];
            if (vi < 0 || vi >= V.rows()) continue;
            for (int j = i + 1; j < n; ++j) {
                const int vj = loop[j];
                if (vj < 0 || vj >= V.rows()) continue;
                const double d = (V.row(vj).head<3>() - V.row(vi).head<3>()).squaredNorm();
                if (d > best) { best = d; fa = vi; fb = vj; }
            }
        }
        add(fa, fb);
    }
    return pairs;
}

} // namespace

// ============================================================
// Core API functions
// ============================================================

// Compute LSCM UV parameterization. Open meshes only.
static Eigen::MatrixXd api_parameterize(
    const Eigen::MatrixXd& V, const Eigen::MatrixXi& F) {
    Eigen::VectorXi boundary;
    igl::boundary_loop(F, boundary);
    if (boundary.size() == 0)
        throw std::runtime_error("Mesh has no boundary — LSCM requires an open mesh");

    Eigen::MatrixXd best_uv;
    UvQuality best_quality;

    for (const auto& [va, vb] : build_lscm_candidates(V, boundary)) {
        Eigen::VectorXi pinned;
        Eigen::MatrixXd pinned_uv;
        if (!build_lscm_constraints(V, va, vb, pinned, pinned_uv)) continue;
        Eigen::MatrixXd candidate_uv;
        if (!igl::lscm(V, F, pinned, pinned_uv, candidate_uv)) continue;
        const UvQuality q = evaluate_uv_quality(candidate_uv, F);
        if (uv_quality_better(q, best_quality)) {
            best_uv = candidate_uv;
            best_quality = q;
        }
    }

    if (best_uv.rows() == 0) {
        if (!igl::lscm(V, F, best_uv))
            throw std::runtime_error("LSCM parameterization failed");
    }
    return best_uv;
}

// Per-vertex area distortion (3D area / UV area ratio).
static Eigen::VectorXd api_compute_distortion(
    const Eigen::MatrixXd& V, const Eigen::MatrixXd& UV, const Eigen::MatrixXi& F) {
    const int nV = static_cast<int>(V.rows());
    Eigen::VectorXd area3d = Eigen::VectorXd::Zero(nV);
    Eigen::VectorXd areaUV = Eigen::VectorXd::Zero(nV);

    for (int f = 0; f < F.rows(); ++f) {
        const int i0 = F(f, 0), i1 = F(f, 1), i2 = F(f, 2);
        const Eigen::Vector3d p0 = V.row(i0), p1 = V.row(i1), p2 = V.row(i2);
        const double a3d = 0.5 * ((p1 - p0).cross(p2 - p0)).norm();
        const Eigen::Vector2d u0 = UV.row(i0).head<2>();
        const Eigen::Vector2d u1 = UV.row(i1).head<2>();
        const Eigen::Vector2d u2 = UV.row(i2).head<2>();
        const double auv = 0.5 * std::abs(
            (u1.x() - u0.x()) * (u2.y() - u0.y()) -
            (u1.y() - u0.y()) * (u2.x() - u0.x()));
        area3d[i0] += a3d; area3d[i1] += a3d; area3d[i2] += a3d;
        areaUV[i0] += auv; areaUV[i1] += auv; areaUV[i2] += auv;
    }

    Eigen::VectorXd distortion(nV);
    for (int v = 0; v < nV; ++v)
        distortion[v] = (areaUV[v] > 1e-14) ? area3d[v] / areaUV[v] : 0.0;
    return distortion;
}

// Sample points on the surface. Returns (uv_points, tri_idx, bary_coords).
static std::tuple<Eigen::MatrixXd, Eigen::VectorXi, Eigen::MatrixXd> api_sample(
    const Eigen::MatrixXd& UV, const Eigen::MatrixXi& F,
    int n, int seed, bool uniform_uv) {
    const std::vector<DartSample> samples = uniform_uv
        ? sample_random_points_uniform_uv(UV, F, n, seed)
        : sample_random_points(UV, F, n, seed);

    const int m = static_cast<int>(samples.size());
    Eigen::MatrixXd uv_out(m, 2);
    Eigen::VectorXi tri_idx(m);
    Eigen::MatrixXd bary(m, 3);
    for (int i = 0; i < m; ++i) {
        uv_out.row(i) = samples[i].uv.transpose();
        tri_idx[i] = samples[i].tri_idx;
        bary.row(i) = samples[i].bary.transpose();
    }
    return {uv_out, tri_idx, bary};
}

// Lloyd relaxation for well-distributed samples.
static Eigen::MatrixXd api_relax(
    const Eigen::MatrixXd& points_uv, const Eigen::MatrixXd& UV,
    const Eigen::MatrixXi& F, const Eigen::VectorXd& distortion,
    int max_iters, double convergence, int grid_size) {
    return lloyd_relaxation(
        points_uv, UV, F, distortion,
        max_iters, convergence, grid_size,
        nullptr, nullptr, nullptr, nullptr, nullptr);
}

// Project UV sample positions back to 3D using barycentric coordinates.
static Eigen::MatrixXd api_project_3d(
    const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
    const Eigen::VectorXi& tri_idx, const Eigen::MatrixXd& bary) {
    const int n = static_cast<int>(tri_idx.size());
    Eigen::MatrixXd out(n, 3);
    for (int i = 0; i < n; ++i) {
        const int tri = tri_idx[i];
        out.row(i) = bary(i, 0) * V.row(F(tri, 0)) +
                     bary(i, 1) * V.row(F(tri, 1)) +
                     bary(i, 2) * V.row(F(tri, 2));
    }
    return out;
}

// All-in-one: parameterize, sample, relax, and project in a single call.
static py::dict api_distribute(
    const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
    int n_points, int seed, int relax_iters, bool uniform_uv) {
    Eigen::MatrixXd UV = api_parameterize(V, F);
    Eigen::VectorXd distortion = api_compute_distortion(V, UV, F);

    auto [uv_points, tri_idx, bary] = api_sample(UV, F, n_points, seed, uniform_uv);

    if (relax_iters > 0) {
        uv_points = api_relax(uv_points, UV, F, distortion, relax_iters, 1e-4, 64);

        // Rebuild tri_idx and bary after relaxation since UV positions changed.
        UniformGridTriangleFinder finder(UV, F, 64);
        const int m = static_cast<int>(uv_points.rows());
        for (int i = 0; i < m; ++i) {
            DartSample s;
            if (build_sample_from_uv(uv_points.row(i).head<2>(), finder, s)) {
                tri_idx[i] = s.tri_idx;
                bary.row(i) = s.bary.transpose();
            }
        }
    }

    Eigen::MatrixXd points_3d = api_project_3d(V, F, tri_idx, bary);

    py::dict result;
    result["points_3d"]  = points_3d;
    result["points_uv"]  = uv_points;
    result["uv"]         = UV;
    result["distortion"] = distortion;
    return result;
}

// Project arbitrary UV positions to 3D by finding the containing triangle
// in the mesh UV layout. Points outside the mesh land at origin.
static Eigen::MatrixXd api_project_uv_to_3d(
    const Eigen::MatrixXd& V, const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& UV, const Eigen::MatrixXd& points_uv) {
    UniformGridTriangleFinder finder(UV, F, 64);
    const int n = static_cast<int>(points_uv.rows());
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n, 3);
    for (int i = 0; i < n; ++i) {
        DartSample s;
        if (build_sample_from_uv(points_uv.row(i).head<2>(), finder, s) &&
            s.tri_idx >= 0 && s.tri_idx < F.rows()) {
            out.row(i) =
                s.bary[0] * V.row(F(s.tri_idx, 0)).head<3>() +
                s.bary[1] * V.row(F(s.tri_idx, 1)).head<3>() +
                s.bary[2] * V.row(F(s.tri_idx, 2)).head<3>();
        }
    }
    return out;
}

// ============================================================
// Exemplar-based pattern synthesis
// ============================================================

// Build a DelaunayTraversalHelper from UV coords + mesh faces (uses boundary loop).
// Returns the helper by value wrapped in shared_ptr so Python can cache it.
static std::shared_ptr<DelaunayTraversalHelper> api_build_delaunay(
    const Eigen::MatrixXd& UV, const Eigen::MatrixXi& F) {
    Eigen::VectorXi boundary;
    igl::boundary_loop(F, boundary);

    Eigen::MatrixXd boundary_uv(boundary.size(), 2);
    for (int i = 0; i < static_cast<int>(boundary.size()); ++i) {
        boundary_uv.row(i) = UV.row(boundary[i]).head<2>();
    }

    auto helper = std::make_shared<DelaunayTraversalHelper>();
    if (!helper->build(UV.leftCols(2), boundary_uv)) {
        throw std::runtime_error("DelaunayTraversalHelper build failed");
    }
    return helper;
}

// All-in-one synthesis:
// - Parameterize mesh (if UV not provided) and build Delaunay
// - Compute exemplar PCF from exemplar_uv inside input_boundary_uv
// - Place and optimize output points inside output_boundary_uv
static py::dict api_synthesize(
    const Eigen::MatrixXd& UV,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& exemplar_uv,       // (N,2) exemplar points in UV space
    const Eigen::MatrixXd& input_boundary_uv,  // (K,2) polygon bounding exemplar region
    const Eigen::MatrixXd& output_boundary_uv, // (M,2) polygon bounding output region
    int n_points,
    int bin_count,
    int max_iters,
    int seed) {

    if (UV.rows() == 0 || UV.cols() < 2)
        throw std::invalid_argument("UV must be (n_vertices, 2+)");
    if (F.rows() == 0 || F.cols() < 3)
        throw std::invalid_argument("F must be (n_faces, 3)");
    if (exemplar_uv.rows() < 2 || exemplar_uv.cols() < 2)
        throw std::invalid_argument("exemplar_uv must have at least 2 rows and 2 columns");
    if (input_boundary_uv.rows() < 3 || input_boundary_uv.cols() < 2)
        throw std::invalid_argument("input_boundary_uv must be a polygon with >= 3 vertices");
    if (output_boundary_uv.rows() < 3 || output_boundary_uv.cols() < 2)
        throw std::invalid_argument("output_boundary_uv must be a polygon with >= 3 vertices");

    // Build DelaunayTraversalHelper over the full UV domain.
    auto helper = api_build_delaunay(UV, F);

    // Convert exemplar_uv rows to std::vector<Eigen::Vector2d>.
    const int n_ex = static_cast<int>(exemplar_uv.rows());
    std::vector<Eigen::Vector2d> ex_pts(static_cast<size_t>(n_ex));
    for (int i = 0; i < n_ex; ++i) ex_pts[static_cast<size_t>(i)] = exemplar_uv.row(i).head<2>();

    SynthesisResult res = synthesize_pattern(
        *helper, ex_pts,
        input_boundary_uv, output_boundary_uv,
        n_points, bin_count, max_iters, seed);

    const int n_out = static_cast<int>(res.output_uv.size());
    Eigen::MatrixXd out_uv(n_out, 2);
    for (int i = 0; i < n_out; ++i) out_uv.row(i) = res.output_uv[static_cast<size_t>(i)].transpose();

    const int bins = static_cast<int>(res.exemplar_pcf.size());
    Eigen::VectorXd ex_pcf(bins), out_pcf_vec(static_cast<int>(res.output_pcf.size()));
    for (int k = 0; k < bins; ++k) ex_pcf[k] = static_cast<double>(res.exemplar_pcf[static_cast<size_t>(k)]);
    for (int k = 0; k < static_cast<int>(res.output_pcf.size()); ++k)
        out_pcf_vec[k] = static_cast<double>(res.output_pcf[static_cast<size_t>(k)]);

    py::dict d;
    d["output_uv"]    = out_uv;
    d["exemplar_pcf"] = ex_pcf;
    d["output_pcf"]   = out_pcf_vec;
    d["energy"]       = res.final_energy;
    d["iterations"]   = res.iterations_ran;
    return d;
}

// ============================================================
// Module definition
// ============================================================

PYBIND11_MODULE(_pattern_synthesis, m) {
    m.doc() = "Distribute point patterns over 3D surface meshes";

    m.def("parameterize", &api_parameterize,
        py::arg("V"), py::arg("F"),
        R"(Compute LSCM UV parameterization for an open mesh.

Args:
    V: (n_vertices, 3) float64 vertex positions
    F: (n_faces, 3) int32 triangle indices

Returns:
    UV: (n_vertices, 2) float64 UV coordinates)");

    m.def("compute_distortion", &api_compute_distortion,
        py::arg("V"), py::arg("UV"), py::arg("F"),
        R"(Per-vertex area distortion (3D area / UV area ratio).

Args:
    V:  (n_vertices, 3) vertex positions
    UV: (n_vertices, 2) UV coordinates
    F:  (n_faces, 3) triangle indices

Returns:
    distortion: (n_vertices,) float64)");

    m.def("sample", &api_sample,
        py::arg("UV"), py::arg("F"), py::arg("n"),
        py::arg("seed") = 42, py::arg("uniform_uv") = true,
        R"(Sample n points on the mesh surface.

Args:
    UV:         (n_vertices, 2) UV coordinates
    F:          (n_faces, 3) triangle indices
    n:          number of samples
    seed:       random seed (default 42)
    uniform_uv: weight by UV area for uniform UV coverage (default True)

Returns:
    uv_points: (n, 2) UV positions of samples
    tri_idx:   (n,) triangle index for each sample
    bary:      (n, 3) barycentric coordinates)");

    m.def("relax", &api_relax,
        py::arg("points_uv"), py::arg("UV"), py::arg("F"), py::arg("distortion"),
        py::arg("max_iters") = 50, py::arg("convergence") = 1e-4,
        py::arg("grid_size") = 64,
        R"(Lloyd relaxation for a well-distributed point set.

Moves points toward Voronoi centroids, weighted by surface distortion,
so samples spread evenly across the 3D surface rather than the UV domain.

Args:
    points_uv:   (n, 2) current UV positions
    UV:          (n_vertices, 2) mesh UV coordinates
    F:           (n_faces, 3) triangle indices
    distortion:  (n_vertices,) per-vertex area distortion from compute_distortion
    max_iters:   maximum Lloyd iterations (default 50)
    convergence: stop when max displacement falls below this (default 1e-4)
    grid_size:   acceleration grid resolution (default 64)

Returns:
    relaxed_uv: (n, 2) updated UV positions)");

    m.def("project_3d", &api_project_3d,
        py::arg("V"), py::arg("F"), py::arg("tri_idx"), py::arg("bary"),
        R"(Project UV sample positions to 3D using barycentric coordinates.

Args:
    V:       (n_vertices, 3) vertex positions
    F:       (n_faces, 3) triangle indices
    tri_idx: (n,) triangle index for each sample
    bary:    (n, 3) barycentric coordinates

Returns:
    points_3d: (n, 3) world-space positions)");

    m.def("project_uv_to_3d", &api_project_uv_to_3d,
        py::arg("V"), py::arg("F"), py::arg("UV"), py::arg("points_uv"),
        R"(Project UV positions to 3D by locating the containing triangle.

Unlike project_3d, this does not require pre-computed triangle indices —
it looks up the triangle from the UV position directly.

Args:
    V:         (n_vertices, 3) vertex positions
    F:         (n_faces, 3) triangle indices
    UV:        (n_vertices, 2) mesh UV coordinates
    points_uv: (n, 2) UV positions to project

Returns:
    points_3d: (n, 3) world-space positions (zero for points outside mesh))");

    m.def("distribute", &api_distribute,
        py::arg("V"), py::arg("F"),
        py::arg("n_points") = 5000, py::arg("seed") = 42,
        py::arg("relax_iters") = 50, py::arg("uniform_uv") = true,
        R"(Parameterize, sample, and relax in a single call.

Args:
    V:           (n_vertices, 3) vertex positions
    F:           (n_faces, 3) triangle indices
    n_points:    number of output points (default 5000)
    seed:        random seed (default 42)
    relax_iters: Lloyd relaxation iterations, 0 to skip (default 50)
    uniform_uv:  weight sampling by UV area (default True)

Returns:
    dict with keys:
        points_3d:  (n, 3) world-space point positions
        points_uv:  (n, 2) UV positions
        uv:         (n_vertices, 2) full UV parameterization
        distortion: (n_vertices,) per-vertex area distortion)");

    m.def("build_delaunay", &api_build_delaunay,
        py::arg("UV"), py::arg("F"),
        R"(Build a Delaunay traversal helper for the UV domain.

Use this when you want to call synthesize() multiple times on the same mesh
without rebuilding the Delaunay triangulation each time.

Args:
    UV: (n_vertices, 2+) UV coordinates
    F:  (n_faces, 3) triangle indices

Returns:
    DelaunayTraversalHelper object (opaque handle))");

    py::class_<DelaunayTraversalHelper, std::shared_ptr<DelaunayTraversalHelper>>(
        m, "DelaunayTraversalHelper")
        .def("is_ready", &DelaunayTraversalHelper::is_ready)
        .def("triangle_count", &DelaunayTraversalHelper::triangle_count);

    m.def("synthesize", &api_synthesize,
        py::arg("UV"),
        py::arg("F"),
        py::arg("exemplar_uv"),
        py::arg("input_boundary_uv"),
        py::arg("output_boundary_uv"),
        py::arg("n_points") = -1,
        py::arg("bin_count") = 32,
        py::arg("max_iters") = 400,
        py::arg("seed") = 42,
        R"(Exemplar-based point pattern synthesis via PCF matching.

Given a reference set of points (exemplar) and their bounding region on the
mesh UV domain, synthesizes a new point set in the output region whose
pair-correlation function (measured via Delaunay hop distances) matches the
exemplar's PCF as closely as possible.

Args:
    UV:                  (n_vertices, 2) mesh UV coordinates from parameterize()
    F:                   (n_faces, 3) triangle indices
    exemplar_uv:         (N, 2) exemplar point positions in UV space
    input_boundary_uv:   (K, 2) polygon enclosing the exemplar region in UV space
    output_boundary_uv:  (M, 2) polygon enclosing the output region in UV space
    n_points:            target output point count; -1 = infer from exemplar density
    bin_count:           PCF histogram bins (default 32)
    max_iters:           optimizer sweep iterations (default 400)
    seed:                random seed (default 42)

Returns:
    dict with keys:
        output_uv:    (n_out, 2) synthesized point positions in UV space
        exemplar_pcf: (bin_count,) normalized exemplar pair-correlation histogram
        output_pcf:   (bin_count,) normalized output pair-correlation histogram
        energy:       final L2 distance between output and exemplar PCF
        iterations:   number of optimizer sweeps run)");
}
