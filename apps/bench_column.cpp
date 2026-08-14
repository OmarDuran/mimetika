// ASSEMBLY WALL TIME AGAINST CELL COUNT, on the consolidation column itself.
//
// The column is the problem that will be run, so it is the problem to profile:
// one cell across, n tall, with the four boundary conditions Terzaghi needs and
// the same model the driver builds. What is timed is every phase that stands
// between a mesh and a system:
//
//     mesh          the polyhedral complex
//     stress ops    the per-cell mimetic stress operators
//     flux hodge    the per-cell mimetic flux operators
//     numbering     the product space and the epoch
//     constraints   the strong imposition, which now measures its own scales
//     tangent       one assembly of the Jacobian
//     residual      one assembly of the residual
//
// Reported per cell as well as in total, because a phase that is linear and a
// phase that is not look identical in a table of totals and completely
// different in a table of rates. Anything that grows per cell is the thing to
// fix; anything flat is already paying only what the mesh costs it.

#include <array>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "exokal/constitutive/coefficient.hpp"
#include "exokal/hodge/flux_operators.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/model/boundary.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/model/compositions/poroelasticity.hpp"

using namespace mimetika;
using graphos::Index;

namespace {

exokal::Mesh column(int n, double h, double width) {
  std::vector<exokal::Mesh::Point> pts;
  for (int k = 0; k <= n; ++k) {
    for (int j = 0; j <= 1; ++j) {
      for (int i = 0; i <= 1; ++i) pts.push_back({i * width, j * width, k * h / n});
    }
  }
  static constexpr int faces[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4},
                                      {1, 3, 7, 5}, {3, 2, 6, 7}, {2, 0, 4, 6}};
  std::vector<std::vector<std::vector<Index>>> cells;
  for (int k = 0; k < n; ++k) {
    std::vector<std::vector<Index>> cell;
    for (const auto& f : faces) {
      std::vector<Index> cyc;
      for (const int l : f) {
        cyc.push_back(static_cast<Index>(((k + ((l >> 2) & 1)) * 2 + ((l >> 1) & 1)) * 2 + (l & 1)));
      }
      cell.push_back(std::move(cyc));
    }
    cells.push_back(std::move(cell));
  }
  return exokal::Mesh::from_polyhedra(std::move(pts), cells);
}

using Clock = std::chrono::steady_clock;

double since(Clock::time_point& t) {
  const auto now = Clock::now();
  const std::chrono::duration<double> dt = now - t;
  t = now;
  return dt.count();
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> sizes;
  for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
  if (sizes.empty()) sizes = {10, 20, 40, 80, 160, 320, 640, 1280, 2560};

  const double h = 1.0, width = 1.0, mu = 1.0, lam = 1.0, load = 1.0;
  const double alpha = 1.0, inv_M = 0.0, perm = 1.0;
  const double nu = lam / (2.0 * (lam + mu));
  const double inv_mod = (1.0 - 2.0 * nu) / (2.0 * mu * (1.0 - 2.0 * nu + 3.0 * nu));
  const double storage = 3.0 * alpha * alpha * inv_mod + inv_M;
  const double dt = 1.0e-4 * h * h / (perm / (inv_M + alpha * alpha / (lam + 2.0 * mu)));

  // the subsystems, in the order the product space numbers them: the stress
  // and flux blocks live on facets and the rest on cells, which is what makes
  // the first two grow with the facet count and the last three with the cell
  // count -- different slopes in the same table
  static const char* kFields[] = {"s_0", "u_0", "g_0", "q_0", "p_0"};
  std::printf("%7s %8s %8s %8s %8s %8s %9s %11s | %8s %8s %8s %8s %8s %8s %8s %9s\n", "cells", "s", "u",
              "g", "q", "p", "total", "nnz", "mesh", "select", "stress", "flux", "number",
              "constr", "tangent", "TOTAL");
  std::printf("%7s %8s %8s %8s %8s %8s %9s %11s | %8s %8s %8s %8s %8s %8s %8s %9s\n", "", "dofs",
              "dofs", "dofs", "dofs", "dofs", "dofs", "", "ms", "ms", "ms", "ms", "ms", "ms", "ms",
              "ms");
  std::vector<std::array<double, 9>> per_cell;

  for (const int n : sizes) {
    auto t = Clock::now();
    const exokal::Mesh m = column(n, h, width);
    const graphos::Complex& c = m.topology();
    const double t_mesh = since(t);

    // ONE de Rham SELECTION FOR BOTH PRODUCTS. The stress operators and the flux
  // Hodge each need a de Rham product on every cell, with different material
  // tensors but the SAME reconstruction space; the selection is the expensive
  // half and it does not depend on the material.
    const exokal::hodge::DeRhamGeometryCache geo = exokal::hodge::DeRhamGeometryCache::build(m);
    const double t_select = since(t);

    const exokal::hodge::StressOperators ops = exokal::hodge::StressOperators::build(
        m, 3, mu, lam, exokal::hodge::StressOperators::Realization::derham_afw, &geo);
    const double t_stress = since(t);

    const exokal::hodge::FluxOperators hodge = exokal::hodge::FluxOperators::build(
        m, exokal::constitutive::Coefficient::uniform(perm),
        exokal::hodge::FluxOperators::Realization::derham_bdm, &geo);
    const double t_flux = since(t);

    exokal::forms::TermContext ctx;
    ctx.provide("stress_operators", ops);
    ctx.provide("flux_operators", hodge);
    physics::ModelOptions o;
    o.mobility = dt;
    o.storage = storage;
    o.volumetric_compliance = inv_mod;
    o.biot = alpha;
    Simulation sim(physics::Catalogue::instance().build("consolidation", o),
                   {StratumSpec{"ambient", &c, 3, 0}}, ctx);
    const auto& sp = sim.epoch().stratum(0).space();
    const double t_number = since(t);

    std::vector<Index> confined = FacetSelector::where(m, 3, FacetSelector::at(2, 0.0));
    for (const double v : {0.0, width}) {
      for (const int axis : {0, 1}) {
        for (const Index f : FacetSelector::where(m, 3, FacetSelector::at(axis, v))) {
          confined.push_back(f);
        }
      }
    }
    impose_traction(sim.constraints(), sp, "s_0", 3, m,
                       FacetSelector::where(m, 3, FacetSelector::at(2, h)),
                       {0, 0, 0, 0, 0, 0, 0, 0, -load});
    impose_free_slip(sim.constraints(), sp, "s_0", 3, m, confined);
    impose_normal_flux(sim.constraints(), sp, "q_0", 3, m, confined);
    sim.freeze_constraints();
    const double t_constr = since(t);

    exokal::forms::TripletSink jac(sim.n_dofs());
    sim.jacobian(jac);
    const double t_tangent = since(t);

    const auto cells = static_cast<double>(c.count(3));
    const double total =
        t_mesh + t_select + t_stress + t_flux + t_number + t_constr + t_tangent;
    std::printf("%7.0f", cells);
    for (const char* f : kFields) {
      std::printf(" %8zu", static_cast<std::size_t>(sp.map(sp.index_of(f)).size()));
    }
    std::printf(" %9zu %11zu | %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %9.2f\n", sim.n_dofs(),
                jac.nnz(), t_mesh * 1e3, t_select * 1e3, t_stress * 1e3, t_flux * 1e3,
                t_number * 1e3, t_constr * 1e3, t_tangent * 1e3, total * 1e3);
    per_cell.push_back({cells, t_mesh, t_select, t_stress, t_flux, t_number, t_constr,
                        t_tangent, total});
  }

  std::printf("\nper cell (microseconds) -- a flat column is linear in the cell count\n");
  std::printf("%8s %9s %9s %9s %9s %9s %9s %9s %9s\n", "cells", "mesh", "select", "stress",
              "flux", "number", "constr", "tangent", "TOTAL");
  for (const auto& row : per_cell) {
    std::printf("%8.0f", row[0]);
    for (std::size_t k = 1; k < row.size(); ++k) std::printf(" %9.3f", row[k] / row[0] * 1e6);
    std::printf("\n");
  }
  return 0;
}
