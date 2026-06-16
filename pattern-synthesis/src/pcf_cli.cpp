#include "lloyd_relaxation.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Core>

// Input (stdin):
//   <bin_count>
//   <n_points>
//   <u0> <v0>
//   <u1> <v1>
//   ...
//
// Output (stdout): JSON
//   {"hist":[...],"pair_count":<N>,"n_points":<N>,"bin_count":<N>}

int main() {
    int bin_count = 0;
    int n = 0;
    if (!(std::cin >> bin_count >> n)) {
        std::cerr << "pcf_cli: failed to read header\n";
        return 1;
    }

    if (n < 2 || bin_count < 1) {
        std::cout << "{\"hist\":[],\"pair_count\":0,\"n_points\":" << n
                  << ",\"bin_count\":" << bin_count << "}\n";
        return 0;
    }

    Eigen::MatrixXd points_uv(n, 2);
    for (int i = 0; i < n; ++i) {
        double u = 0.0;
        double v = 0.0;
        if (!(std::cin >> u >> v)) {
            std::cerr << "pcf_cli: failed to read point " << i << "\n";
            return 1;
        }
        points_uv(i, 0) = u;
        points_uv(i, 1) = v;
    }

    // Unconstrained Delaunay (empty boundary)
    Eigen::MatrixXd boundary_uv(0, 2);
    DelaunayTraversalHelper delaunay;
    if (!delaunay.build(points_uv, boundary_uv)) {
        std::cerr << "pcf_cli: Delaunay build failed\n";
        return 1;
    }

    std::vector<int> hist(static_cast<size_t>(bin_count), 0);
    int pair_count = 0;
    for (int i = 0; i < n; ++i) {
        const Eigen::Vector2d pi(points_uv(i, 0), points_uv(i, 1));
        for (int j = i + 1; j < n; ++j) {
            const Eigen::Vector2d pj(points_uv(j, 0), points_uv(j, 1));
            const int k = delaunay.count_triangles_crossed(pi, pj);
            if (k < 0) {
                continue;
            }
            const int bin_idx = std::min(k, bin_count - 1);
            ++hist[static_cast<size_t>(bin_idx)];
            ++pair_count;
        }
    }

    std::cout << "{\"hist\":[";
    for (int i = 0; i < bin_count; ++i) {
        if (i > 0) {
            std::cout << ",";
        }
        std::cout << hist[static_cast<size_t>(i)];
    }
    std::cout << "],\"pair_count\":" << pair_count
              << ",\"n_points\":" << n
              << ",\"bin_count\":" << bin_count << "}\n";

    return 0;
}
