// Where does assembling a flow BDM problem go, including the ADS matrices?
//
// Two costs, reported together because they are paid together:
//
//   model.build()   the flow system A and its rhs -- exokal's stabilized BDM
//                   flux product on every cell, plus the divergence
//   bdm_complex()   C, G, Pi_rt, Pi_nd -- what hypre's ADS is handed
//
// Per-cell microseconds are printed beside the totals: every term here is
// cell-local, so every column must be FLAT against mesh size. A column that
// rises is something rebuilding a mesh-wide structure per cell, which turns the
// whole assembly quadratic. That is the number to read, not the total.
//
// NOT A UNIT TEST -- built on request, `benchmark` label:
//
//     cmake -B build-hypre -DMIMETIKA_BUILD_TESTS=ON -DMIMETIKA_BUILD_BENCHMARKS=ON
//     ctest --test-dir build-hypre -L benchmark
//
// With no arguments it asserts that per-cell cost does not run away over the
// ladder. With arguments it is a plain measurement: test_bdm_assembly 2 4 6 8

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "mimetika/linear_solver/bdm_complex.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/flow_model.hpp"

using graphos::Index;
using mimetika::FlowModel;
using mimetika::mesh::Family;
using mimetika::solver::BdmComplexCost;

namespace {

struct Row {
  long long cells{0};
  double build{0.0}, complex_total{0.0};
  BdmComplexCost cost;
};

Row one(int n) {
  const int dim = 3;
  const exokal::Mesh mesh = mimetika::mesh::box({n, n, n}, dim, Family::simplex, {1.0, 1.0, 1.0});
  FlowModel model(mesh, dim, 1.0, FlowModel::Realization::stabilized_bdm);
  for (const Index f : mimetika::boundary_facets(mesh.topology(), dim)) {
    model.flow().template emplace<mimetika::PressureBC>(std::vector<Index>{f},
                                                       exokal::centroid(mesh, dim - 1, f)[0]);
  }

  Row r;
  r.cells = mesh.count(dim);
  const auto t0 = std::chrono::steady_clock::now();
  model.build();
  r.build = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  const auto cx = mimetika::solver::bdm_complex(mesh, dim, &r.cost);
  (void)cx;
  r.complex_total = r.cost.total;
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> ladder;
  for (int i = 1; i < argc; ++i) ladder.push_back(std::atoi(argv[i]));
  const bool checking = argc == 1;
  if (ladder.empty()) ladder = {2, 4, 6, 8};

  std::printf("# flow stabilized_bdm assembly, and the matrices hypre-ADS needs\n");
  std::printf("# totals in seconds; the per-cell columns are microseconds and should be FLAT\n");
  std::printf("# %4s %8s | %8s %8s | %8s %8s %8s %8s %8s | %9s %9s\n", "n", "cells", "build_s",
              "cx_s", "curl", "flux", "fit", "scatter", "reduce", "us/cell", "us/cell");
  std::printf("# %4s %8s | %8s %8s | %8s %8s %8s %8s %8s | %9s %9s\n", "", "", "", "", "us/cell",
              "us/cell", "us/cell", "us/cell", "total", "build", "complex");

  std::vector<double> per_cell_curl, per_cell_build;
  for (const int n : ladder) {
    const Row r = one(n);
    const double c = double(r.cells);
    std::printf("  %4d %8lld | %8.2f %8.2f | %8.1f %8.1f %8.1f %8.1f %8.1f | %9.1f %9.1f\n", n,
                r.cells, r.build, r.complex_total, 1e6 * r.cost.curl / c, 1e6 * r.cost.flux / c,
                1e6 * r.cost.fit / c, 1e6 * r.cost.scatter / c, 1e6 * r.cost.reduce / c,
                1e6 * r.build / c, 1e6 * r.complex_total / c);
    std::fflush(stdout);
    per_cell_curl.push_back(1e6 * r.cost.curl / c);
    per_cell_build.push_back(1e6 * r.build / c);
  }

  if (!checking) return 0;
  int failures = 0;
  const auto flat = [&](const char* what, const std::vector<double>& v, double factor) {
    if (v.size() < 2) return;
    if (v.back() > factor * v.front()) {
      std::printf("  FAIL %s: %.1f -> %.1f us/cell over the ladder (allowed %.1fx)\n", what,
                  v.front(), v.back(), factor);
      ++failures;
    }
  };
  // EVERY TERM IS CELL-LOCAL, SO EVERY TERM MUST STAY FLAT.
  //
  // The degree-2 curl was not, once: exokal's mimetic_curl rebuilt
  // graphos::incidence(3,1) and (1,0) on every call, O(mesh) per cell, and it
  // ran 333 -> 1253 us/cell over 48 -> 10368 cells. Sharing them through a
  // MimeticCurlCache made it flat at ~26. The bound is what guards that.
  flat("model.build", per_cell_build, 2.0);
  flat("mimetic_curl", per_cell_curl, 1.6);
  std::printf(failures ? "  [FAIL] %d\n" : "  [PASS]\n", failures);
  return failures ? 1 : 0;
}
