// Darcy flow preconditioned by hypre's ADS, called directly.
//
// The same Riesz map the PETSc path builds -- P = diag(M + B^T W^-1 B, W) --
// with the first block handed to HYPRE_ADS rather than to PCHYPRE. What that
// buys is the auxiliary hierarchies' strength thresholds, which PETSc
// registers and never queries.
//
// The datum is affine, so the field lies in the lowest-order space exactly and
// ||Pi_0(p - p_h)|| IS the solver's error: on this problem a converged run
// returns ~1e-10 and a run that merely satisfied a stopping test does not.
//
// Usage: mimetika-hypre-flow [n ...]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mimetika/linear_solver/fields.hpp"
#include "mimetika/linear_solver/hypre.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/flow_model.hpp"

using graphos::Index;
using mimetika::FlowModel;
using mimetika::mesh::Family;
using mimetika::solver::SpaceNorm;

namespace {

// The norm, as attach_norm builds it for flow: the flux is the first factor,
// the pressure carries plain L2, and W is the Schur scale of the divergence
// constraint -- an unscaled incidence row, so the cell measure.
SpaceNorm flow_norm(const FlowModel& model, const exokal::Mesh& mesh, int dim) {
  const auto blocks = mimetika::solver::field_blocks(model.simulation().epoch());
  if (blocks.size() < 2) throw std::runtime_error("flow: fewer than two factors");
  const auto n_cells = static_cast<std::size_t>(mesh.count(dim));

  // L is the box diagonal: W carries 1/L^2 so the two factors are compared in
  // the same units whatever the mesh is written in.
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
  flux.reserve(blocks[0].size());
  for (const Index g : blocks[0].indices()) flux.push_back(static_cast<int>(g));
  norm.factors.push_back(std::move(flux));

  // W CARRIES K. It stands for the Schur complement B star_K^-1 B^T of the
  // divergence constraint, and star_K^-1 scales with K, so W does too. The
  // measure alone names the norm of a HOMOGENEOUS coefficient only: with K
  // jumping across facets the pairing the map is built from is no longer the
  // pairing the operator has, and the count pays the inf-sup constant of the
  // mismatch -- measured, 17 iterations become the 500-step cap at 1e6.
  const std::vector<double> k_cell = model.norm_permeability();
  if (!k_cell.empty() && k_cell.size() != n_cells) {
    throw std::runtime_error("flow: the coefficient does not cover the cells");
  }
  std::vector<int> rest;
  std::vector<double> l2;
  for (std::size_t f = 1; f < blocks.size(); ++f) {
    const std::size_t components = blocks[f].size() / n_cells;
    std::size_t k = 0;
    for (const Index g : blocks[f].indices()) {
      rest.push_back(static_cast<int>(g));
      const std::size_t cell = k / components;
      const double k_e = k_cell.empty() ? 1.0 : k_cell[cell];
      l2.push_back(scale * k_e * exokal::measure(mesh, dim, static_cast<Index>(cell)));
      ++k;
    }
  }
  norm.factors.push_back(std::move(rest));
  norm.l2_weight.push_back(std::move(l2));

  // the complex's own boundary operators, which is all ADS is told besides the
  // vertex coordinates
  const graphos::Complex& topo = mesh.topology();
  const auto incidence = [&](auto b, int rows, int cols) {
    SpaceNorm::Incidence out;
    out.rows = rows;
    out.cols = cols;
    for (Index r = 0; r < rows; ++r) {
      for (auto k = b.offsets[static_cast<std::size_t>(r)];
           k < b.offsets[static_cast<std::size_t>(r) + 1]; ++k) {
        out.row.push_back(static_cast<int>(r));
        out.col.push_back(static_cast<int>(b.indices[static_cast<std::size_t>(k)]));
        out.value.push_back(static_cast<double>(b.signs[static_cast<std::size_t>(k)]));
      }
    }
    return out;
  };
  norm.discrete_gradient = incidence(topo.boundary(1), topo.count(1), topo.count(0));
  norm.discrete_curl = incidence(topo.boundary(2), topo.count(2), topo.count(1));
  norm.space_dim = 3;
  norm.vertex_coordinates.reserve(static_cast<std::size_t>(topo.count(0)) * 3);
  for (Index v = 0; v < topo.count(0); ++v) {
    const auto& x = mesh.point(v);
    norm.vertex_coordinates.insert(norm.vertex_coordinates.end(), {x[0], x[1], x[2]});
  }
  return norm;
}

struct Run {
  std::size_t cells{0}, dofs{0};
  int iterations{0};
  double setup{0.0}, solve{0.0}, error{0.0};
  std::string reason;
};

Run one(int n, const mimetika::solver::HypreSolver::Options& opts) {
  const int dim = 3;
  const exokal::Mesh mesh = mimetika::mesh::box({n, n, n}, dim, Family::cartesian, {1.0, 1.0, 1.0});
  FlowModel model(mesh, dim, 1.0, FlowModel::Realization::stabilized_rt);

  // p = x on the boundary: affine, so the discrete answer is its interpolant
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

  const SpaceNorm norm = flow_norm(model, mesh, dim);
  mimetika::solver::HypreSolver hypre;
  std::vector<double> x;
  const auto rep = hypre.solve(model.system(), model.rhs(), x, norm, opts);
  model.accept(x);

  Run r;
  r.cells = static_cast<std::size_t>(mesh.count(dim));
  r.dofs = model.simulation().n_dofs();
  r.iterations = rep.iterations;
  r.setup = rep.setup_seconds;
  r.solve = rep.solve_seconds;
  r.reason = rep.reason;
  for (Index e = 0; e < mesh.count(dim); ++e) {
    r.error = std::max(r.error, std::abs(model.cell_pressure(e) - exokal::centroid(mesh, dim, e)[0]));
  }
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> ladder;
  for (int i = 1; i < argc; ++i) ladder.push_back(std::atoi(argv[i]));
  if (ladder.empty()) ladder = {4, 8, 12, 16};

  mimetika::solver::HypreSolver::Options opts;
  opts.rtol = 1e-8;
  opts.max_iterations = 500;
  // The knobs PETSc registers and never queries, reachable here. Read from the
  // environment so a sweep needs no rebuild.
  if (const char* v = std::getenv("MIMETIKA_ADS_CYCLE")) opts.ads_cycle_type = std::atoi(v);
  if (const char* v = std::getenv("MIMETIKA_AMG_THETA")) opts.amg_theta = std::atof(v);
  if (const char* v = std::getenv("MIMETIKA_AMS_THETA")) opts.ams_theta = std::atof(v);

  std::printf("# hypre %s, ADS cycle %d, amg_theta %.2f, ams_theta %.2f\n",
              HYPRE_RELEASE_VERSION, opts.ads_cycle_type, opts.amg_theta, opts.ams_theta);
  std::printf("# %5s %8s %10s %6s %9s %9s %11s  %s\n", "n", "cells", "dofs", "its", "setup_s",
              "solve_s", "max|e|", "reason");
  for (const int n : ladder) {
    const Run r = one(n, opts);
    std::printf("  %5d %8zu %10zu %6d %9.2f %9.2f %11.2e  %s\n", n, r.cells, r.dofs, r.iterations,
                r.setup, r.solve, r.error, r.reason.c_str());
    std::fflush(stdout);
  }
  return 0;
}
