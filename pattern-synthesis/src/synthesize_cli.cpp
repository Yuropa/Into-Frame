#include "lloyd_relaxation.h"
#include "synthesis_core.h"

#include <iostream>
#include <vector>

#include <Eigen/Core>

// Input (stdin):
//   <bin_count> <n_points> <max_iters> <seed>
//   <n_domain>
//   <x0> <z0>          -- background points; synthesize_pattern's candidate positions
//   ...                   are the Delaunay triangle centers of this point set, so its
//                          local density sets the spatial resolution of both the
//                          exemplar PCF measurement and the synthesized output. The
//                          caller is expected to supply locally-dense clusters around
//                          the input and output boundaries rather than one big uniform
//                          grid, so resolution isn't wasted on the gap between them.
//   <n_exemplar>
//   <x0> <z0>          × n_exemplar
//   <n_input_boundary>
//   <x0> <z0>          × n_input_boundary
//   <n_output_boundary>
//   <x0> <z0>          × n_output_boundary
//
// n_points <= 0 infers the target count from exemplar density (see synthesize_pattern).
//
// Output (stdout): JSON
//   {"output_points":[[x,z],...],"energy":<float>,"iterations":<int>,"n_points":<int>}

namespace {

bool read_points(std::istream& in, int count, Eigen::MatrixXd& out) {
    out.resize(count, 2);
    for (int i = 0; i < count; ++i) {
        if (!(in >> out(i, 0) >> out(i, 1))) {
            return false;
        }
    }
    return true;
}

void print_empty_result() {
    std::cout << "{\"output_points\":[],\"energy\":0.0,\"iterations\":0,\"n_points\":0}\n";
}

}  // namespace

int main() {
    int bin_count = 0;
    int n_points = 0;
    int max_iters = 0;
    int seed = 0;

    if (!(std::cin >> bin_count >> n_points >> max_iters >> seed)) {
        std::cerr << "synthesize_cli: failed to read header\n";
        return 1;
    }

    int n_domain = 0;
    if (!(std::cin >> n_domain)) {
        std::cerr << "synthesize_cli: failed to read domain point count\n";
        return 1;
    }
    Eigen::MatrixXd domain_points;
    if (!read_points(std::cin, n_domain, domain_points)) {
        std::cerr << "synthesize_cli: failed to read domain points\n";
        return 1;
    }

    int n_exemplar = 0;
    if (!(std::cin >> n_exemplar)) {
        std::cerr << "synthesize_cli: failed to read exemplar count\n";
        return 1;
    }
    Eigen::MatrixXd exemplar_mat;
    if (!read_points(std::cin, n_exemplar, exemplar_mat)) {
        std::cerr << "synthesize_cli: failed to read exemplar points\n";
        return 1;
    }

    int n_input_boundary = 0;
    if (!(std::cin >> n_input_boundary)) {
        std::cerr << "synthesize_cli: failed to read input boundary count\n";
        return 1;
    }
    Eigen::MatrixXd input_boundary_uv;
    if (!read_points(std::cin, n_input_boundary, input_boundary_uv)) {
        std::cerr << "synthesize_cli: failed to read input boundary\n";
        return 1;
    }

    int n_output_boundary = 0;
    if (!(std::cin >> n_output_boundary)) {
        std::cerr << "synthesize_cli: failed to read output boundary count\n";
        return 1;
    }
    Eigen::MatrixXd output_boundary_uv;
    if (!read_points(std::cin, n_output_boundary, output_boundary_uv)) {
        std::cerr << "synthesize_cli: failed to read output boundary\n";
        return 1;
    }

    if (n_domain < 3 || n_exemplar < 2 || n_input_boundary < 3 || n_output_boundary < 3 ||
        bin_count < 1) {
        print_empty_result();
        return 0;
    }

    // Unconstrained Delaunay (empty boundary), same convention as pcf_cli.
    Eigen::MatrixXd empty_boundary(0, 2);
    DelaunayTraversalHelper delaunay;
    if (!delaunay.build(domain_points, empty_boundary)) {
        std::cerr << "synthesize_cli: Delaunay build failed\n";
        return 1;
    }

    std::vector<Eigen::Vector2d> exemplar_uv(static_cast<size_t>(n_exemplar));
    for (int i = 0; i < n_exemplar; ++i) {
        exemplar_uv[static_cast<size_t>(i)] = exemplar_mat.row(i).transpose();
    }

    const SynthesisResult result = synthesize_pattern(
        delaunay, exemplar_uv, input_boundary_uv, output_boundary_uv,
        n_points, bin_count, max_iters, seed);

    if (result.output_uv.empty()) {
        // synthesize_pattern leaves final_energy as +inf on failure, which isn't valid
        // JSON — report the empty-result shape instead.
        print_empty_result();
        return 0;
    }

    std::cout << "{\"output_points\":[";
    for (size_t i = 0; i < result.output_uv.size(); ++i) {
        if (i > 0) {
            std::cout << ",";
        }
        std::cout << "[" << result.output_uv[i].x() << "," << result.output_uv[i].y() << "]";
    }
    std::cout << "],\"energy\":" << result.final_energy
               << ",\"iterations\":" << result.iterations_ran
               << ",\"n_points\":" << result.output_uv.size() << "}\n";

    return 0;
}
