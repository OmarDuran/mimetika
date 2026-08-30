// Darcy flow, stabilized_bdm, preconditioned by ADS acting DIRECTLY on the BDM
// flux block -- no facet-constant subspace, no two-level cycle.
//
// ADS is given the degree-2 complex instead of the lowest-order one:
//
//     P3 nodal --G--> N2E2 circulation --C--> BDM flux
//
// with Pi_rt and Pi_nd through HYPRE_ADSSetInterpolations, because the
// coordinate construction ADS uses by default assumes one unknown a facet.
// C, Pi_rt and Pi_nd come from exokal; G is built in bdm_complex.hpp and is
// topological on the edges.
//
// B.C = 0 IS THE ALIGNMENT TEST. B is the model's own discrete divergence, so
// it is stated in the model's flux dofs. If C's rows were in another order or
// another basis, div.curl would not vanish. It is checked before the solve
// because a misaligned C does not fail, it just preconditions badly.
//
// NOT A UNIT TEST. It solves a ladder of meshes -- seconds, not milliseconds --
// so it is built on request and carries the `benchmark` label:
//
//     cmake -B build-hypre -DMIMETIKA_USE_HYPRE=ON -DMIMETIKA_BUILD_BENCHMARKS=ON
//     ctest --test-dir build-hypre -L benchmark
//
// With no arguments it runs the default ladder and ASSERTS the two properties
// the direct path exists for: the count stays bounded under refinement, and
// div.curl vanishes. With arguments it is a plain measurement --
//
//     test_bdm_ads_direct 2 4 6 8 --contrast=1e6
//     test_bdm_ads_direct --two-level 3 4 6

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <map>
#include <string>
#include <vector>

#include "mimetika/linear_solver/bdm_complex.hpp"
#include "mimetika/linear_solver/fields.hpp"
#include "mimetika/linear_solver/hypre.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/flow_model.hpp"

using graphos::Index;
using mimetika::FlowModel;
using mimetika::mesh::Family;
using mimetika::solver::Sparse;
using mimetika::solver::SpaceNorm;

namespace {

SpaceNorm::Incidence to_inc(const Sparse& s) {
  SpaceNorm::Incidence i;
  i.rows = s.rows;
  i.cols = s.cols;
  i.row = s.row;
  i.col = s.col;
  i.value = s.value;
  return i;
}

struct Run {
  std::size_t cells{0}, dofs{0}, flux{0};
  int iterations{0};
  double setup{0.0}, solve{0.0}, error{0.0}, bc{0.0}, bc_ref{0.0}, assemble{0.0};
  std::size_t nnz_c{0}, nnz_g{0}, nnz_pi{0};
  int n_circ{0}, n_nodal{0};
  std::string reason;
};

Run one(int n, const mimetika::solver::HypreSolver::Options& opts, double contrast,
        bool two_level) {
  const int dim = 3;
  const exokal::Mesh mesh = mimetika::mesh::box({n, n, n}, dim, Family::simplex, {1.0, 1.0, 1.0});
  FlowModel model(mesh, dim, 1.0, FlowModel::Realization::stabilized_bdm);

  if (contrast != 1.0) {
    std::vector<double> k(static_cast<std::size_t>(mesh.count(dim)), 1.0);
    for (Index e = 0; e < mesh.count(dim); ++e) {
      const auto c = exokal::centroid(mesh, dim, e);
      const int par = int(std::floor(c[0] * n)) + int(std::floor(c[1] * n)) + int(std::floor(c[2] * n));
      k[static_cast<std::size_t>(e)] = (par % 2 == 0) ? 1.0 : contrast;
    }
    model.set_permeability(k);
  }

  std::vector<Index> facets;
  std::vector<double> values;
  for (const Index f : mimetika::boundary_facets(mesh.topology(), dim)) {
    facets.push_back(f);
    values.push_back(exokal::centroid(mesh, dim - 1, f)[0]);
  }
  for (std::size_t i = 0; i < facets.size(); ++i) {
    model.flow().template emplace<mimetika::PressureBC>(std::vector<Index>{facets[i]}, values[i]);
  }
  model.build();

  const auto blocks = mimetika::solver::field_blocks(model.simulation().epoch());
  const auto n_cells = static_cast<std::size_t>(mesh.count(dim));

  auto lo = mesh.point(0), hi = mesh.point(0);
  for (Index v = 1; v < mesh.count(0); ++v) {
    const auto& x = mesh.point(v);
    for (std::size_t d = 0; d < 3; ++d) {
      lo[d] = std::min(lo[d], x[d]);
      hi[d] = std::max(hi[d], x[d]);
    }
  }
  double diag2 = 0.0;
  for (std::size_t d = 0; d < 3; ++d) diag2 += (hi[d] - lo[d]) * (hi[d] - lo[d]);
  const double scale = model.mobility() / diag2;

  SpaceNorm norm;
  std::vector<int> flux;
  for (const Index g : blocks[0].indices()) flux.push_back(static_cast<int>(g));
  norm.factors.push_back(flux);

  const std::vector<double> k_cell = model.norm_permeability();
  std::vector<int> rest;
  std::vector<double> l2;
  for (std::size_t f = 1; f < blocks.size(); ++f) {
    const std::size_t comps = blocks[f].size() / n_cells;
    std::size_t kk = 0;
    for (const Index g : blocks[f].indices()) {
      rest.push_back(static_cast<int>(g));
      const std::size_t cell = kk / comps;
      const double k_e = k_cell.empty() ? 1.0 : k_cell[cell];
      l2.push_back(scale * k_e * exokal::measure(mesh, dim, static_cast<Index>(cell)));
      ++kk;
    }
  }
  norm.factors.push_back(rest);
  norm.l2_weight.push_back(l2);

  // ---- the degree-2 complex, in place of the lowest-order incidences -------
  const auto t_cx = std::chrono::steady_clock::now();
  const auto cx = mimetika::solver::bdm_complex(mesh, dim);
  const double assemble_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_cx).count();
  norm.space_dim = 3;
  if (two_level) {
    // THE BASELINE: the lowest-order complex, and the BDM block reached through
    // its facet-constant subspace. Same problem, same norm, same tolerance --
    // the only difference is which complex ADS is given.
    const graphos::Complex& t = mesh.topology();
    const auto inc = [](auto b, int rows, int cols) {
      SpaceNorm::Incidence o;
      o.rows = rows;
      o.cols = cols;
      for (Index r = 0; r < rows; ++r)
        for (auto k = b.offsets[static_cast<std::size_t>(r)];
             k < b.offsets[static_cast<std::size_t>(r) + 1]; ++k) {
          o.row.push_back(static_cast<int>(r));
          o.col.push_back(static_cast<int>(b.indices[static_cast<std::size_t>(k)]));
          o.value.push_back(static_cast<double>(b.signs[static_cast<std::size_t>(k)]));
        }
      return o;
    };
    norm.discrete_gradient = inc(t.boundary(1), t.count(1), t.count(0));
    norm.discrete_curl = inc(t.boundary(2), t.count(2), t.count(1));
    SpaceNorm::Incidence inj;
    inj.rows = static_cast<int>(model.system().n);
    inj.cols = static_cast<int>(t.count(2));
    for (Index f = 0; f < t.count(2); ++f) {
      inj.row.push_back(flux[static_cast<std::size_t>(3 * int(f))]);
      inj.col.push_back(static_cast<int>(f));
      inj.value.push_back(1.0);
    }
    norm.lowest_order = inj;
    norm.lowest_order_components = 1;
  } else {
    norm.discrete_gradient = to_inc(cx.grad);
    norm.discrete_curl = to_inc(cx.curl);
    norm.rt_interpolation = to_inc(cx.pi_rt);
    norm.nd_interpolation = to_inc(cx.pi_nd);
  }

  // coordinates of the P3 nodal dofs: vertices, edge midpoints, facet centroids
  const graphos::Complex& top = mesh.topology();
  const int nV = int(top.count(0)), nE = int(top.count(1)), nF = int(top.count(2));
  const graphos::Adjacency e2v = graphos::incidence(top, 1, 0);
  norm.vertex_coordinates.assign(static_cast<std::size_t>(two_level ? nV : cx.n_nodal) * 3, 0.0);
  for (Index v = 0; v < nV; ++v) {
    const auto& p = mesh.point(v);
    for (int d = 0; d < 3; ++d) norm.vertex_coordinates[static_cast<std::size_t>(v) * 3 + d] = p[d];
  }
  for (Index e = 0; !two_level && e < nE; ++e) {
    const auto b = static_cast<std::size_t>(e2v.offsets[static_cast<std::size_t>(e)]);
    const auto pa = mesh.point(e2v.indices[b]), ph = mesh.point(e2v.indices[b + 1]);
    for (int m = 0; m < 2; ++m)
      for (int d = 0; d < 3; ++d)
        norm.vertex_coordinates[static_cast<std::size_t>(nV + 2 * int(e) + m) * 3 + d] =
            0.5 * (pa[d] + ph[d]);
  }
  for (Index f = 0; !two_level && f < nF; ++f) {
    const auto c = exokal::centroid(mesh, 2, f);
    for (int d = 0; d < 3; ++d)
      norm.vertex_coordinates[static_cast<std::size_t>(nV + 2 * nE + int(f)) * 3 + d] = c[d];
  }

  Run r;
  r.assemble = assemble_s;
  r.nnz_c = cx.curl.value.size();
  r.nnz_g = cx.grad.value.size();
  r.nnz_pi = cx.pi_rt.value.size() + cx.pi_nd.value.size();
  r.n_circ = cx.n_circ;
  r.n_nodal = cx.n_nodal;
  r.cells = n_cells;
  r.dofs = model.simulation().n_dofs();
  r.flux = flux.size();

  // ---- B.C = 0 : the model's divergence against the complex's curl --------
  {
    std::vector<int> pos(model.system().n, -1);
    for (std::size_t i = 0; i < flux.size(); ++i) pos[static_cast<std::size_t>(flux[i])] = int(i);
    std::vector<int> is_p(model.system().n, 0);
    for (const int g : rest) is_p[static_cast<std::size_t>(g)] = 1;
    // B rows are pressure unknowns, columns the flux block
    std::map<int, std::map<int, double>> B;
    const auto& S = model.system();
    for (std::size_t k = 0; k < S.value.size(); ++k) {
      const int rr = int(S.row[k]), cc = int(S.col[k]);
      if (!is_p[static_cast<std::size_t>(rr)]) continue;
      if (pos[static_cast<std::size_t>(cc)] < 0) continue;
      B[rr][pos[static_cast<std::size_t>(cc)]] += S.value[k];
    }
    std::vector<std::vector<std::pair<int, double>>> Ccol(static_cast<std::size_t>(cx.n_circ));
    for (std::size_t k = 0; k < cx.curl.value.size(); ++k)
      Ccol[static_cast<std::size_t>(cx.curl.col[k])].push_back({cx.curl.row[k], cx.curl.value[k]});
    std::vector<double> col(flux.size(), 0.0);
    for (int j = 0; j < cx.n_circ; ++j) {
      for (const auto& [rr, v] : Ccol[static_cast<std::size_t>(j)]) col[static_cast<std::size_t>(rr)] = v;
      for (const auto& [rr, brow] : B) {
        double s = 0.0, mag = 0.0;
        for (const auto& [cc, v] : brow) {
          s += v * col[static_cast<std::size_t>(cc)];
          mag += std::fabs(v * col[static_cast<std::size_t>(cc)]);
        }
        r.bc = std::max(r.bc, std::fabs(s));
        r.bc_ref = std::max(r.bc_ref, mag);
      }
      for (const auto& [rr, v] : Ccol[static_cast<std::size_t>(j)]) col[static_cast<std::size_t>(rr)] = 0.0;
    }
  }

  mimetika::solver::HypreSolver hypre;
  std::vector<double> x;
  const auto rep = hypre.solve(model.system(), model.rhs(), x, norm, opts);
  model.accept(x);

  r.iterations = rep.iterations;
  r.setup = rep.setup_seconds;
  r.solve = rep.solve_seconds;
  r.reason = rep.reason;
  for (Index e = 0; e < mesh.count(dim); ++e) {
    r.error =
        std::max(r.error, std::abs(model.cell_pressure(e) - exokal::centroid(mesh, dim, e)[0]));
  }
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> ladder;
  double contrast = 1.0;
  bool two_level = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a.rfind("--contrast=", 0) == 0) contrast = std::atof(a.c_str() + 11);
    else if (a == "--two-level") two_level = true;
    else ladder.push_back(std::atoi(argv[i]));
  }
  const bool checking = argc == 1;
  if (ladder.empty()) ladder = {2, 3, 4, 6};

  mimetika::solver::HypreSolver::Options opts;
  opts.rtol = 1e-10;
  opts.max_iterations = 500;
  if (const char* v = std::getenv("MIMETIKA_RTOL")) opts.rtol = std::atof(v);
  if (const char* v = std::getenv("MIMETIKA_ADS_CYCLE")) opts.ads_cycle_type = std::atoi(v);
  if (const char* v = std::getenv("MIMETIKA_BLOCK_ITS")) opts.block_iterations = std::atoi(v);
  if (const char* v = std::getenv("MIMETIKA_PRINT")) opts.print_level = std::atoi(v);

  std::printf("# stabilized_bdm, %s. contrast %.0e, cycle %d, rtol %.0e\n",
              two_level ? "two-level over the facet-constant subspace"
                        : "ADS on the degree-2 complex",
              contrast, opts.ads_cycle_type, opts.rtol);
  std::printf("# %4s %8s %8s %8s %8s %6s %8s %8s %8s %9s %9s\n", "n", "flux", "circ", "nodal",
              "asm_s", "its", "setup_s", "solve_s", "ms/it", "nnz/row C", "nnz/row G");
  int first = 0, last = 0, failures = 0;
  for (const int n : ladder) {
    const Run r = one(n, opts, contrast, two_level);
    std::printf("  %4d %8zu %8d %8d %8.2f %6d %8.2f %8.2f %8.1f %9.1f %9.1f\n", n, r.flux,
                r.n_circ, r.n_nodal, r.assemble, r.iterations, r.setup, r.solve,
                r.iterations ? 1000.0 * r.solve / r.iterations : 0.0,
                r.flux ? double(r.nnz_c) / double(r.flux) : 0.0,
                r.n_circ ? double(r.nnz_g) / double(r.n_circ) : 0.0);
    std::fflush(stdout);
    if (!checking) continue;
    // div.curl = 0 against the model's OWN divergence: a C in the wrong order
    // or the wrong basis does not fail, it merely preconditions badly
    if (!(r.bc < 1e-8 * std::max(r.bc_ref, 1.0))) {
      std::printf("  FAIL n=%d: |B.C| = %.3e against %.3e\n", n, r.bc, r.bc_ref);
      ++failures;
    }
    if (r.reason != "CONVERGED_RTOL") {
      std::printf("  FAIL n=%d: %s\n", n, r.reason.c_str());
      ++failures;
    }
    if (!first) first = r.iterations;
    last = r.iterations;
  }
  if (checking) {
    // h-INDEPENDENCE, stated as a bound rather than a slope: over this ladder
    // the count went 21 -> 30, so twice the coarsest is loose enough to be a
    // regression guard and tight enough to catch the count starting to grow.
    if (first && last > 2 * first) {
      std::printf("  FAIL: %d -> %d iterations over the ladder\n", first, last);
      ++failures;
    }
    std::printf(failures ? "  [FAIL] %d\n" : "  [PASS]\n", failures);
  }
  return failures ? 1 : 0;
}
