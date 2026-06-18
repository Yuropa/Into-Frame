#include "lloyd_relaxation.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include <igl/boundary_loop.h>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Constrained_triangulation_face_base_2.h>
#include <CGAL/Triangulation_data_structure_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Voronoi_diagram_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/intersections.h>
#include <CGAL/number_utils.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_2<int, K> Vb;
typedef CGAL::Constrained_triangulation_face_base_2<K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb> TDS;
typedef CGAL::Exact_intersections_tag Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, TDS, Itag> CDT;

typedef CGAL::Delaunay_triangulation_adaptation_traits_2<CDT> AT;
typedef CGAL::Voronoi_diagram_2<CDT, AT> Voronoi_diagram;

typedef CGAL::Polygon_2<K> Polygon_2;
typedef CGAL::Polygon_with_holes_2<K> Polygon_with_holes_2;
typedef K::Point_2 Point_2;

UniformGridTriangleFinder::UniformGridTriangleFinder(
  const Eigen::MatrixXd &UV_,
  const Eigen::MatrixXi &F_,
  int grid_size_)
  : UV(UV_), F(F_), grid_size(std::max(1, grid_size_)) {
  uv_min = UV.colwise().minCoeff();
  uv_max = UV.colwise().maxCoeff();

  grid.resize(grid_size * grid_size);

  const double dx = uv_max.x() - uv_min.x();
  const double dy = uv_max.y() - uv_min.y();
  if (dx <= 0.0 || dy <= 0.0) {
    return;
  }

  for (int f = 0; f < F.rows(); ++f) {
    Eigen::Vector2d u0 = UV.row(F(f, 0)).head<2>();
    Eigen::Vector2d u1 = UV.row(F(f, 1)).head<2>();
    Eigen::Vector2d u2 = UV.row(F(f, 2)).head<2>();

    double min_ux = std::min({u0.x(), u1.x(), u2.x()});
    double max_ux = std::max({u0.x(), u1.x(), u2.x()});
    double min_uy = std::min({u0.y(), u1.y(), u2.y()});
    double max_uy = std::max({u0.y(), u1.y(), u2.y()});

    int min_x = std::max(0, static_cast<int>(std::floor((min_ux - uv_min.x()) / dx * grid_size)));
    int max_x = std::min(grid_size - 1, static_cast<int>(std::ceil((max_ux - uv_min.x()) / dx * grid_size)));
    int min_y = std::max(0, static_cast<int>(std::floor((min_uy - uv_min.y()) / dy * grid_size)));
    int max_y = std::min(grid_size - 1, static_cast<int>(std::ceil((max_uy - uv_min.y()) / dy * grid_size)));

    for (int y = min_y; y <= max_y; ++y) {
      for (int x = min_x; x <= max_x; ++x) {
        grid[y * grid_size + x].push_back(f);
      }
    }
  }
}

int UniformGridTriangleFinder::find_containing_triangle(const Eigen::Vector2d &uv) const {
  const double dx = uv_max.x() - uv_min.x();
  const double dy = uv_max.y() - uv_min.y();
  if (dx <= 0.0 || dy <= 0.0) {
    return -1;
  }

  int gx = static_cast<int>((uv.x() - uv_min.x()) / dx * grid_size);
  int gy = static_cast<int>((uv.y() - uv_min.y()) / dy * grid_size);

  if (gx < 0 || gx >= grid_size || gy < 0 || gy >= grid_size) {
    return -1;
  }

  const auto &candidates = grid[gy * grid_size + gx];
  for (int tri_idx : candidates) {
    if (is_point_in_triangle_uv(uv, tri_idx)) {
      return tri_idx;
    }
  }

  return -1;
}

Eigen::Vector3d UniformGridTriangleFinder::compute_barycentric_coords_uv(
  const Eigen::Vector2d &p,
  int tri_idx) const {
  Eigen::Vector2d u0 = UV.row(F(tri_idx, 0)).head<2>();
  Eigen::Vector2d u1 = UV.row(F(tri_idx, 1)).head<2>();
  Eigen::Vector2d u2 = UV.row(F(tri_idx, 2)).head<2>();

  Eigen::Vector2d v = p - u0;
  Eigen::Vector2d e1 = u1 - u0;
  Eigen::Vector2d e2 = u2 - u0;

  double det = e1.x() * e2.y() - e1.y() * e2.x();
  if (std::abs(det) < 1e-14) {
    return Eigen::Vector3d(-1.0, -1.0, -1.0);
  }

  double inv00 =  e2.y() / det;
  double inv01 = -e2.x() / det;
  double inv10 = -e1.y() / det;
  double inv11 =  e1.x() / det;

  double b1 = inv00 * v.x() + inv01 * v.y();
  double b2 = inv10 * v.x() + inv11 * v.y();
  double b0 = 1.0 - b1 - b2;

  return Eigen::Vector3d(b0, b1, b2);
}

bool UniformGridTriangleFinder::is_point_in_triangle_uv(
  const Eigen::Vector2d &p,
  int tri_idx) const {
  Eigen::Vector2d u0 = UV.row(F(tri_idx, 0)).head<2>();
  Eigen::Vector2d u1 = UV.row(F(tri_idx, 1)).head<2>();
  Eigen::Vector2d u2 = UV.row(F(tri_idx, 2)).head<2>();

  Eigen::Vector2d v = p - u0;
  Eigen::Vector2d e1 = u1 - u0;
  Eigen::Vector2d e2 = u2 - u0;

  double det = e1.x() * e2.y() - e1.y() * e2.x();
  if (std::abs(det) < 1e-14) {
    return false;
  }

  double inv00 =  e2.y() / det;
  double inv01 = -e2.x() / det;
  double inv10 = -e1.y() / det;
  double inv11 =  e1.x() / det;

  double b1 = inv00 * v.x() + inv01 * v.y();
  double b2 = inv10 * v.x() + inv11 * v.y();
  double b0 = 1.0 - b1 - b2;

  constexpr double eps = -1e-8;
  return (b0 >= eps && b1 >= eps && b2 >= eps);
}

Eigen::MatrixXd dart_samples_to_uv_points(const std::vector<DartSample> &samples) {
  Eigen::MatrixXd points(samples.size(), 2);
  for (size_t i = 0; i < samples.size(); ++i) {
    points.row(i) = samples[i].uv.transpose();
  }
  return points;
}

bool build_sample_from_uv(
  const Eigen::Vector2d &uv,
  const UniformGridTriangleFinder &finder,
  DartSample &out_sample) {
  int tri_idx = finder.find_containing_triangle(uv);
  if (tri_idx < 0) {
    return false;
  }

  Eigen::Vector3d bary = finder.compute_barycentric_coords_uv(uv, tri_idx);
  constexpr double kBaryTolerance = -1e-8;
  if (bary.x() < kBaryTolerance || bary.y() < kBaryTolerance || bary.z() < kBaryTolerance) {
    return false;
  }

  out_sample.uv = uv;
  out_sample.tri_idx = tri_idx;
  out_sample.bary = bary;
  return true;
}

std::vector<std::vector<int>> build_constrained_delaunay_adjacency(
  const Eigen::MatrixXd& points_uv,
  const Eigen::MatrixXd& boundary_uv) {
  const int n = static_cast<int>(points_uv.rows());
  std::vector<std::vector<int>> adjacency(static_cast<size_t>(std::max(0, n)));
  if (n < 2 || points_uv.cols() < 2) {
    return adjacency;
  }

  CDT cdt;

  if (boundary_uv.rows() >= 3 && boundary_uv.cols() >= 2) {
    const int m = static_cast<int>(boundary_uv.rows());
    std::vector<CDT::Vertex_handle> boundary_handles(static_cast<size_t>(m));
    for (int i = 0; i < m; ++i) {
      boundary_handles[static_cast<size_t>(i)] =
        cdt.insert(Point_2(boundary_uv(i, 0), boundary_uv(i, 1)));
      boundary_handles[static_cast<size_t>(i)]->info() = -1;
    }
    for (int i = 0; i < m; ++i) {
      const int j = (i + 1) % m;
      cdt.insert_constraint(
        boundary_handles[static_cast<size_t>(i)],
        boundary_handles[static_cast<size_t>(j)]);
    }
  }

  for (int i = 0; i < n; ++i) {
    auto vh = cdt.insert(Point_2(points_uv(i, 0), points_uv(i, 1)));
    vh->info() = i;
  }

  std::unordered_set<unsigned long long> edge_set;
  edge_set.reserve(static_cast<size_t>(n) * 3);

  for (auto edge = cdt.finite_edges_begin(); edge != cdt.finite_edges_end(); ++edge) {
    auto face = edge->first;
    const int idx = edge->second;
    int i = face->vertex((idx + 1) % 3)->info();
    int j = face->vertex((idx + 2) % 3)->info();

    if (i < 0 || j < 0 || i >= n || j >= n || i == j) {
      continue;
    }
    if (i > j) {
      std::swap(i, j);
    }

    const unsigned long long key =
      (static_cast<unsigned long long>(static_cast<unsigned int>(i)) << 32) |
      static_cast<unsigned int>(j);
    if (!edge_set.insert(key).second) {
      continue;
    }

    adjacency[static_cast<size_t>(i)].push_back(j);
    adjacency[static_cast<size_t>(j)].push_back(i);
  }

  return adjacency;
}

namespace {

double cross2(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
  return a.x() * b.y() - a.y() * b.x();
}

} // namespace

bool DelaunayTraversalHelper::build(
  const Eigen::MatrixXd& points_uv,
  const Eigen::MatrixXd& boundary_uv) {
  faces_.clear();
  grid_faces_.clear();
  grid_res_ = 0;
  bbox_min_.setZero();
  bbox_max_.setZero();

  const int n = static_cast<int>(points_uv.rows());
  if (n < 3 || points_uv.cols() < 2) {
    return false;
  }

  CDT cdt;

  if (boundary_uv.rows() >= 3 && boundary_uv.cols() >= 2) {
    const int m = static_cast<int>(boundary_uv.rows());
    std::vector<CDT::Vertex_handle> boundary_handles(static_cast<size_t>(m));
    for (int i = 0; i < m; ++i) {
      boundary_handles[static_cast<size_t>(i)] =
        cdt.insert(Point_2(boundary_uv(i, 0), boundary_uv(i, 1)));
      boundary_handles[static_cast<size_t>(i)]->info() = -1;
    }
    for (int i = 0; i < m; ++i) {
      const int j = (i + 1) % m;
      cdt.insert_constraint(
        boundary_handles[static_cast<size_t>(i)],
        boundary_handles[static_cast<size_t>(j)]);
    }
  }

  for (int i = 0; i < n; ++i) {
    auto vh = cdt.insert(Point_2(points_uv(i, 0), points_uv(i, 1)));
    vh->info() = i;
  }

  std::vector<CDT::Face_handle> face_handles;
  face_handles.reserve(static_cast<size_t>(cdt.number_of_faces()));
  std::unordered_map<const void*, int> face_index_map;
  face_index_map.reserve(static_cast<size_t>(cdt.number_of_faces()));

  for (auto fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); ++fit) {
    CDT::Face_handle fh = fit;
    const int i0 = fh->vertex(0)->info();
    const int i1 = fh->vertex(1)->info();
    const int i2 = fh->vertex(2)->info();
    if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= n || i1 >= n || i2 >= n) {
      continue;
    }

    FaceData data;
    data.vertex_indices = Eigen::Vector3i(i0, i1, i2);
    data.tri_points.row(0) = points_uv.row(i0).head<2>();
    data.tri_points.row(1) = points_uv.row(i1).head<2>();
    data.tri_points.row(2) = points_uv.row(i2).head<2>();
    data.p0 = data.tri_points.row(0).transpose();

    Eigen::Matrix2d basis;
    basis.col(0) = data.tri_points.row(1).transpose() - data.p0;
    basis.col(1) = data.tri_points.row(2).transpose() - data.p0;
    const double det = basis.determinant();
    if (std::abs(det) > 1e-14) {
      data.inv_basis = basis.inverse();
      data.has_inv_basis = true;
    }

    const int face_idx = static_cast<int>(faces_.size());
    faces_.push_back(data);
    face_handles.push_back(fh);
    face_index_map.emplace(static_cast<const void*>(&*fh), face_idx);
  }

  if (faces_.empty()) {
    return false;
  }

  for (size_t fi = 0; fi < face_handles.size(); ++fi) {
    CDT::Face_handle fh = face_handles[fi];
    for (int e = 0; e < 3; ++e) {
      CDT::Face_handle nh = fh->neighbor(e);
      const auto it = face_index_map.find(static_cast<const void*>(&*nh));
      faces_[fi].neighbors[static_cast<size_t>(e)] =
        (it != face_index_map.end()) ? it->second : -1;
    }
  }

  bbox_min_ = points_uv.leftCols<2>().colwise().minCoeff().transpose();
  bbox_max_ = points_uv.leftCols<2>().colwise().maxCoeff().transpose();
  const Eigen::Vector2d span = bbox_max_ - bbox_min_;
  if (span.x() <= 1e-14 || span.y() <= 1e-14) {
    return true;
  }

  grid_res_ = std::clamp(
    static_cast<int>(std::sqrt(static_cast<double>(faces_.size())) * 2.0),
    8,
    256);
  grid_faces_.assign(static_cast<size_t>(grid_res_ * grid_res_), {});

  const auto coord_to_cell = [&](double value, double minv, double axis_span) -> int {
    const double u = (value - minv) / axis_span;
    const int cell = static_cast<int>(std::floor(u * static_cast<double>(grid_res_)));
    return std::clamp(cell, 0, grid_res_ - 1);
  };

  for (int fi = 0; fi < static_cast<int>(faces_.size()); ++fi) {
    const auto& face = faces_[static_cast<size_t>(fi)];
    const double min_x = face.tri_points.col(0).minCoeff();
    const double max_x = face.tri_points.col(0).maxCoeff();
    const double min_y = face.tri_points.col(1).minCoeff();
    const double max_y = face.tri_points.col(1).maxCoeff();

    const int gx0 = coord_to_cell(min_x, bbox_min_.x(), span.x());
    const int gx1 = coord_to_cell(max_x, bbox_min_.x(), span.x());
    const int gy0 = coord_to_cell(min_y, bbox_min_.y(), span.y());
    const int gy1 = coord_to_cell(max_y, bbox_min_.y(), span.y());

    for (int gy = gy0; gy <= gy1; ++gy) {
      for (int gx = gx0; gx <= gx1; ++gx) {
        grid_faces_[static_cast<size_t>(gy * grid_res_ + gx)].push_back(fi);
      }
    }
  }

  return true;
}

bool DelaunayTraversalHelper::is_ready() const {
  return !faces_.empty();
}

int DelaunayTraversalHelper::triangle_count() const {
  return static_cast<int>(faces_.size());
}

bool DelaunayTraversalHelper::triangle_center(
  int triangle_idx,
  Eigen::Vector2d& out_center_uv) const {
  if (triangle_idx < 0 || triangle_idx >= static_cast<int>(faces_.size())) {
    out_center_uv.setZero();
    return false;
  }
  const FaceData& face = faces_[static_cast<size_t>(triangle_idx)];
  out_center_uv =
    (face.tri_points.row(0).transpose() +
     face.tri_points.row(1).transpose() +
     face.tri_points.row(2).transpose()) / 3.0;
  return true;
}

void DelaunayTraversalHelper::triangle_adjacency_edges(
  std::vector<Eigen::Vector2i>& out_edges) const {
  out_edges.clear();
  out_edges.reserve(faces_.size() * 2);
  for (int i = 0; i < static_cast<int>(faces_.size()); ++i) {
    const FaceData& face = faces_[static_cast<size_t>(i)];
    for (int e = 0; e < 3; ++e) {
      const int j = face.neighbors[static_cast<size_t>(e)];
      if (j > i) {
        out_edges.emplace_back(i, j);
      }
    }
  }
}

void DelaunayTraversalHelper::get_triangle_neighbors(
  int triangle_idx,
  std::array<int, 3>& out_neighbors) const {
  out_neighbors = { -1, -1, -1 };
  if (triangle_idx < 0 || triangle_idx >= static_cast<int>(faces_.size())) {
    return;
  }
  out_neighbors = faces_[static_cast<size_t>(triangle_idx)].neighbors;
}

bool DelaunayTraversalHelper::point_in_face(
  const FaceData& face,
  const Eigen::Vector2d& uv) const {
  constexpr double kEps = 1e-10;
  if (face.has_inv_basis) {
    const Eigen::Vector2d b12 = face.inv_basis * (uv - face.p0);
    const double b1 = b12.x();
    const double b2 = b12.y();
    const double b0 = 1.0 - b1 - b2;
    return b0 >= -kEps && b1 >= -kEps && b2 >= -kEps;
  }

  const Eigen::Vector2d p0 = face.tri_points.row(0).transpose();
  const Eigen::Vector2d p1 = face.tri_points.row(1).transpose();
  const Eigen::Vector2d p2 = face.tri_points.row(2).transpose();
  const double c0 = cross2(p1 - p0, uv - p0);
  const double c1 = cross2(p2 - p1, uv - p1);
  const double c2 = cross2(p0 - p2, uv - p2);
  const bool nonneg = (c0 >= -kEps && c1 >= -kEps && c2 >= -kEps);
  const bool nonpos = (c0 <= kEps && c1 <= kEps && c2 <= kEps);
  return nonneg || nonpos;
}

int DelaunayTraversalHelper::locate_triangle_index(const Eigen::Vector2d& uv) const {
  if (faces_.empty()) {
    return -1;
  }

  const Eigen::Vector2d span = bbox_max_ - bbox_min_;
  if (grid_res_ > 0 && span.x() > 1e-14 && span.y() > 1e-14 &&
      uv.x() >= bbox_min_.x() - 1e-12 && uv.x() <= bbox_max_.x() + 1e-12 &&
      uv.y() >= bbox_min_.y() - 1e-12 && uv.y() <= bbox_max_.y() + 1e-12) {
    const int gx = std::clamp(
      static_cast<int>(std::floor((uv.x() - bbox_min_.x()) / span.x() * grid_res_)),
      0,
      grid_res_ - 1);
    const int gy = std::clamp(
      static_cast<int>(std::floor((uv.y() - bbox_min_.y()) / span.y() * grid_res_)),
      0,
      grid_res_ - 1);
    const auto& candidates = grid_faces_[static_cast<size_t>(gy * grid_res_ + gx)];
    for (int fi : candidates) {
      if (fi >= 0 && fi < static_cast<int>(faces_.size()) &&
          point_in_face(faces_[static_cast<size_t>(fi)], uv)) {
        return fi;
      }
    }
  }

  for (int fi = 0; fi < static_cast<int>(faces_.size()); ++fi) {
    if (point_in_face(faces_[static_cast<size_t>(fi)], uv)) {
      return fi;
    }
  }
  return -1;
}

bool DelaunayTraversalHelper::find_containing_triangle(
  const Eigen::Vector2d& uv,
  int& out_triangle_idx,
  Eigen::Vector3i& out_triangle_vertices) const {
  const int tri_idx = locate_triangle_index(uv);
  if (tri_idx < 0) {
    return false;
  }
  out_triangle_idx = tri_idx;
  out_triangle_vertices = faces_[static_cast<size_t>(tri_idx)].vertex_indices;
  return true;
}

int DelaunayTraversalHelper::count_triangles_crossed(
  const Eigen::Vector2d& start_uv,
  const Eigen::Vector2d& end_uv) const {
  if (faces_.empty()) {
    return -1;
  }

  const Eigen::Vector2d seg = end_uv - start_uv;
  if (seg.squaredNorm() <= 1e-20) {
    return (locate_triangle_index(start_uv) >= 0) ? 0 : -1;
  }

  int current_face = locate_triangle_index(start_uv);
  if (current_face < 0) {
    const Eigen::Vector2d nudged = start_uv + 1e-8 * seg;
    current_face = locate_triangle_index(nudged);
  }
  if (current_face < 0) {
    return -1;
  }

  if (point_in_face(faces_[static_cast<size_t>(current_face)], end_uv)) {
    return 0;
  }

  constexpr double kTStepEps = 1e-9;
  double t_cur = 0.0;
  int visited_faces = 1;
  int prev_face = -1;
  const int max_steps = std::max(16, static_cast<int>(faces_.size()) * 3);

  for (int step = 0; step < max_steps && t_cur < 1.0 - kTStepEps; ++step) {
    const FaceData& face = faces_[static_cast<size_t>(current_face)];

    double best_t = std::numeric_limits<double>::infinity();
    int best_next_face = -1;

    for (int e = 0; e < 3; ++e) {
      const int i = (e + 1) % 3;
      const int j = (e + 2) % 3;
      const Eigen::Vector2d edge_a = face.tri_points.row(i).transpose();
      const Eigen::Vector2d edge_b = face.tri_points.row(j).transpose();
      const Eigen::Vector2d edge = edge_b - edge_a;

      const double det = cross2(seg, edge);
      if (std::abs(det) <= 1e-14) {
        continue;
      }

      const Eigen::Vector2d r = edge_a - start_uv;
      const double t = cross2(r, edge) / det;
      const double s = cross2(r, seg) / det;
      if (!(t > t_cur + kTStepEps && t <= 1.0 + kTStepEps)) {
        continue;
      }
      if (s < -kTStepEps || s > 1.0 + kTStepEps) {
        continue;
      }

      const int next_face = face.neighbors[static_cast<size_t>(e)];
      if (next_face == prev_face && t > t_cur + 1e-6) {
        continue;
      }
      if (t < best_t) {
        best_t = t;
        best_next_face = next_face;
      }
    }

    if (!std::isfinite(best_t) || best_t >= 1.0 - kTStepEps || best_next_face < 0) {
      break;
    }

    prev_face = current_face;
    current_face = best_next_face;
    ++visited_faces;
    t_cur = std::min(1.0, best_t + kTStepEps);

    if (point_in_face(faces_[static_cast<size_t>(current_face)], end_uv)) {
      break;
    }
  }

  if (point_in_face(faces_[static_cast<size_t>(current_face)], end_uv)) {
    return std::max(0, visited_faces - 1);
  }
  return -1;
}

static Eigen::Vector2d polygon_centroid_geometric(const std::vector<Eigen::Vector2d> &pts) {
  if (pts.size() < 3) {
    return pts.empty() ? Eigen::Vector2d::Zero() : pts.front();
  }

  double area = 0.0;
  Eigen::Vector2d centroid(0.0, 0.0);
  for (size_t i = 0; i < pts.size(); ++i) {
    const Eigen::Vector2d &p0 = pts[i];
    const Eigen::Vector2d &p1 = pts[(i + 1) % pts.size()];
    double cross = p0.x() * p1.y() - p1.x() * p0.y();
    area += cross;
    centroid.x() += (p0.x() + p1.x()) * cross;
    centroid.y() += (p0.y() + p1.y()) * cross;
  }

  if (std::abs(area) < 1e-12) {
    Eigen::Vector2d avg = Eigen::Vector2d::Zero();
    for (const auto &p : pts) {
      avg += p;
    }
    return avg / static_cast<double>(pts.size());
  }

  area *= 0.5;
  centroid /= (6.0 * area);
  return centroid;
}

static Eigen::Vector2d polygon_centroid(
  const std::vector<Eigen::Vector2d> &pts,
  const UniformGridTriangleFinder *uv_finder,
  const Eigen::VectorXd &vertex_distortion,
  const Eigen::MatrixXi &F) {
  constexpr int kTargetSamples = 50;
  constexpr int kMaxAttempts = 1000;
  constexpr double kDampingExponent = 2.0;
  constexpr double kMinWeight = 1e-10;
  constexpr double kMinDistortion = 1e-8;

  if (pts.size() < 3) {
    return pts.empty() ? Eigen::Vector2d::Zero() : pts[0];
  }

  if (!uv_finder || vertex_distortion.size() == 0) {
    return polygon_centroid_geometric(pts);
  }

  Eigen::Vector2d bbox_min = pts[0];
  Eigen::Vector2d bbox_max = pts[0];
  for (const auto &pt : pts) {
    bbox_min = bbox_min.cwiseMin(pt);
    bbox_max = bbox_max.cwiseMax(pt);
  }

  Polygon_2 cell_poly;
  for (const auto &pt : pts) {
    cell_poly.push_back(Point_2(pt.x(), pt.y()));
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> x_dist(bbox_min.x(), bbox_max.x());
  std::uniform_real_distribution<double> y_dist(bbox_min.y(), bbox_max.y());

  double total_weight = 0.0;
  Eigen::Vector2d weighted_sum = Eigen::Vector2d::Zero();
  int samples_inside = 0;

  for (int attempt = 0; attempt < kMaxAttempts && samples_inside < kTargetSamples; ++attempt) {
    const double x = x_dist(gen);
    const double y = y_dist(gen);
    const Point_2 sample_pt(x, y);

    if (cell_poly.bounded_side(sample_pt) != CGAL::ON_BOUNDED_SIDE) {
      continue;
    }

    const Eigen::Vector2d sample_uv(x, y);
    const int tri_idx = uv_finder->find_containing_triangle(sample_uv);
    if (tri_idx < 0 || tri_idx >= F.rows()) {
      continue;
    }

    const Eigen::Vector3d bary = uv_finder->compute_barycentric_coords_uv(sample_uv, tri_idx);
    constexpr double kBaryTolerance = -1e-8;
    if (bary.x() < kBaryTolerance || bary.y() < kBaryTolerance || bary.z() < kBaryTolerance) {
      continue;
    }

    const double s = bary.x() * vertex_distortion[F(tri_idx, 0)] +
                     bary.y() * vertex_distortion[F(tri_idx, 1)] +
                     bary.z() * vertex_distortion[F(tri_idx, 2)];

    const double rho = std::pow(std::max(s, kMinDistortion), kDampingExponent);
    if (!std::isfinite(rho) || rho < kMinWeight) {
      continue;
    }

    weighted_sum += rho * sample_uv;
    total_weight += rho;
    ++samples_inside;
  }

  if (samples_inside >= 10 && total_weight > kMinWeight) {
    return weighted_sum / total_weight;
  }

  return polygon_centroid_geometric(pts);
}

Eigen::MatrixXd lloyd_relaxation(
  const Eigen::MatrixXd &points_uv,
  const Eigen::MatrixXd &UV,
  const Eigen::MatrixXi &F,
  const Eigen::VectorXd &vertex_distortion,
  int max_iters,
  double convergence_threshold,
  int grid_size,
  const UniformGridTriangleFinder *finder_override,
  double *out_last_max_displacement,
  double *out_last_min_displacement,
  double *out_last_avg_displacement,
  int *out_iters_run) {
  if (points_uv.rows() == 0 || points_uv.cols() < 2) {
    return points_uv;
  }
  if (UV.rows() == 0 || UV.cols() < 2 || F.rows() == 0) {
    return points_uv;
  }

  Eigen::MatrixXd points = points_uv;
  Eigen::VectorXi uv_boundary;
  igl::boundary_loop(F, uv_boundary);

  if (uv_boundary.size() < 3) {
    return points;
  }

  Polygon_2 boundary_polygon;
  for (int i = 0; i < uv_boundary.size(); ++i) {
    Eigen::Vector2d pt = UV.row(uv_boundary[i]).head<2>();
    boundary_polygon.push_back(Point_2(pt.x(), pt.y()));
  }

  std::unique_ptr<UniformGridTriangleFinder> finder_storage;
  const UniformGridTriangleFinder *finder = finder_override;
  if (!finder) {
    finder_storage = std::make_unique<UniformGridTriangleFinder>(UV, F, grid_size);
    finder = finder_storage.get();
  }

  const int n = static_cast<int>(points.rows());
  double prev_displacement = std::numeric_limits<double>::max();

  std::vector<std::pair<Point_2, int>> sites;
  sites.reserve(uv_boundary.size() + n);

  for (int iter = 0; iter < max_iters; ++iter) {
    Eigen::MatrixXd old_points = points;

    sites.clear();
    sites.reserve(uv_boundary.size() + n);
    for (int i = 0; i < uv_boundary.size(); ++i) {
      Eigen::Vector2d pt = UV.row(uv_boundary[i]).head<2>();
      sites.emplace_back(Point_2(pt.x(), pt.y()), -1);
    }
    for (int i = 0; i < n; ++i) {
      sites.emplace_back(Point_2(points(i, 0), points(i, 1)), i);
    }

    CDT cdt;
    cdt.insert(sites.begin(), sites.end());
    cdt.insert_constraints(boundary_polygon.edges_begin(), boundary_polygon.edges_end());

    Voronoi_diagram vd(cdt);

    for (auto fit = vd.faces_begin(); fit != vd.faces_end(); ++fit) {
      auto site_handle = fit->dual();
      if (cdt.is_infinite(site_handle)) {
        continue;
      }

      int site_idx = site_handle->info();
      if (site_idx < 0 || site_idx >= n) {
        continue;
      }

      std::vector<Point_2> cell_vertices;
      cell_vertices.reserve(10);

      auto ccb = fit->ccb();
      auto he_start = ccb;
      auto he = he_start;

      bool has_unbounded = false;
      do {
        if (he->has_target()) {
          cell_vertices.push_back(he->target()->point());
        } else {
          has_unbounded = true;
          break;
        }
        ++he;
      } while (he != he_start);

      if (has_unbounded || cell_vertices.size() < 3) {
        continue;
      }

      Polygon_2 cell_poly(cell_vertices.begin(), cell_vertices.end());
      std::list<Polygon_with_holes_2> clipped;
      CGAL::intersection(boundary_polygon, cell_poly, std::back_inserter(clipped));

      if (clipped.empty()) {
        continue;
      }

      std::vector<Eigen::Vector2d> cell_pts;
      for (auto vit = clipped.front().outer_boundary().vertices_begin();
           vit != clipped.front().outer_boundary().vertices_end(); ++vit) {
        cell_pts.emplace_back(CGAL::to_double(vit->x()), CGAL::to_double(vit->y()));
      }

      if (cell_pts.size() < 3) {
        continue;
      }

      Eigen::Vector2d centroid =  polygon_centroid(cell_pts, finder, vertex_distortion, F);
      Point_2 query(centroid.x(), centroid.y());
      if (boundary_polygon.bounded_side(query) == CGAL::ON_BOUNDED_SIDE) {
        points.row(site_idx).head<2>() = centroid;
      }
    }

    Eigen::VectorXd displacements = (points - old_points).rowwise().norm();
    double max_displacement = displacements.maxCoeff();
    double min_displacement = displacements.minCoeff();
    double avg_displacement = displacements.mean();

    if (out_last_max_displacement) {
      *out_last_max_displacement = max_displacement;
    }
    if (out_last_min_displacement) {
      *out_last_min_displacement = min_displacement;
    }
    if (out_last_avg_displacement) {
      *out_last_avg_displacement = avg_displacement;
    }
    if (out_iters_run) {
      *out_iters_run = iter + 1;
    }

    if (max_displacement < convergence_threshold) {
      break;
    }

    if (iter > 10 && std::abs(prev_displacement - max_displacement) < 1e-8) {
      break;
    }

    prev_displacement = max_displacement;
  }

  return points;
}
