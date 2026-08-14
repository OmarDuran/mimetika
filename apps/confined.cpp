// CONFINED UNIAXIAL COMPRESSION, which has no freedom left to get wrong.
//
// A column on a roller base, rollers on all four sides, a uniform compressive
// traction sigma_0 on top. Zero lateral strain is then forced by geometry
// alone, so elasticity gives the whole answer in closed form:
//
//     sigma_xx = sigma_yy = lam/(lam + 2 mu) sigma_zz
//     eps_zz   = sigma_zz / K_oed,     K_oed = lam + 2 mu
//     u_z(z)   = eps_zz z
//
// No time stepping, no pore pressure, no storage: if this is wrong then
// nothing built on top of it can be right, and the consolidation column's
// failure mode is unreadable until this one is clean.
//
// The rollers are the whole content of the test. In Hellinger-Reissner the
// vanishing SHEAR traction is essential and gets pinned, while the vanishing
// NORMAL DISPLACEMENT is natural: it is enforced by leaving the facet's normal
// traction FREE, so that its own equation reads u_n = 0 weakly. Pin that dof
// by mistake and the condition is not merely lost, it is replaced by its
// opposite -- a free lateral face -- and the column expands sideways.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "exokal/constitutive/coefficient.hpp"
#include "exokal/hodge/flux_hodge.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/model/boundary.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/models/poroelasticity.hpp"
#include "mimetika/solver/petsc.hpp"

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

}  // namespace

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 4;
  // UNDRAINED CONFINED COMPRESSION, the second closed form. Seal every facet
  // and step once from the unloaded state: no fluid can move, so eps = 0 and
  //
  //     sigma = -sigma_0 I,   p = sigma_0/alpha,   u = 0
  //
  // exactly, for an incompressible fluid (1/M = 0). It is the strongest test
  // of the poroelastic coupling there is, because the mechanics answer is
  // already known to be exact and every remaining degree of freedom is fixed.
  const bool undrained = argc > 2 && std::string(argv[2]) == "undrained";
  const double h = 1.0, width = 1.0, mu = 1.0, lam = 1.0, load = 1.0;
  const double alpha = 1.0, inv_M = 0.0, perm = 1.0, dt = 1.0e-6;
  const double nu_ = lam / (2.0 * (lam + mu));
  const double inv_mod = (1.0 - 2.0 * nu_) / (2.0 * mu * (1.0 - 2.0 * nu_ + 3.0 * nu_));
  const double storage = 3.0 * alpha * alpha * inv_mod + inv_M;

  const exokal::Mesh m = column(n, h, width);
  const graphos::Complex& c = m.topology();
  const exokal::hodge::StressOperators ops = exokal::hodge::StressOperators::build(m, mu, lam);
  exokal::forms::TermContext ctx;
  ctx.provide("stress_operators", ops);

  const exokal::hodge::FluxHodge hodge = exokal::hodge::FluxHodge::build(
      m, exokal::constitutive::Coefficient::uniform(perm),
      exokal::hodge::FluxHodge::Realization::derham);
  if (undrained) ctx.provide("flux_hodge", hodge);
  physics::ModelOptions o;
  o.mobility = dt;
  o.storage = storage;
  o.volumetric_compliance = inv_mod;
  o.biot = alpha;
  Simulation sim(physics::Catalogue::instance().build(
                     undrained ? "consolidation" : "linear_elasticity", o),
                 {StratumSpec{"ambient", &c, 3, 0}}, ctx);
  const auto& sp = sim.epoch().stratum(0).space();

  std::vector<Index> confined = FacetSelector::where(m, 3, FacetSelector::at(2, 0.0));
  for (const double v : {0.0, width}) {
    for (const int axis : {0, 1}) {
      for (const Index f : FacetSelector::where(m, 3, FacetSelector::at(axis, v))) {
        confined.push_back(f);
      }
    }
  }
  pin_facet_traction(sim.constraints(), sp, "s_0", 3, m,
                     FacetSelector::where(m, 3, FacetSelector::at(2, h)),
                     {0, 0, 0, 0, 0, 0, 0, 0, -load});
  pin_facet_roller(sim.constraints(), sp, "s_0", 3, m, confined);
  if (undrained) {
    // sealed EVERYWHERE, the top included: nothing drains, so nothing moves
    std::vector<Index> all;
    for (const Index f : boundary_facets(c, 3)) all.push_back(f);
    pin_facets(sim.constraints(), sp, "q_0", 3, all);
  }
  sim.freeze_constraints();

  exokal::forms::TripletSink jac(sim.n_dofs());
  sim.jacobian(jac);
  const auto A = solver::SparseSystem::from(jac);
  std::vector<double> b(sim.n_dofs(), 0.0), x;
  for (std::size_t d = 0; d < sim.n_dofs(); ++d) {
    // the constrained equation is written with the scale of the row it
    // replaced, so its datum carries the same factor
    if (sim.constraints().pinned(d)) {
      b[d] = sim.constraints().scale_at(d) * sim.constraints().value_at(d);
    }
  }
  solver::PetscSolver petsc;
  const auto rep = petsc.solve(A, b, x);
  if (!rep.converged) {
    std::printf("did not converge: %s\n", rep.reason.c_str());
    return 1;
  }

  const double K_oed = lam + 2.0 * mu;
  const double sxx_exact = undrained ? -load : lam / K_oed * (-load);
  const double ezz_exact = undrained ? 0.0 : -load / K_oed;
  std::printf("confined column, %d cells, %zu dofs, %zu pinned\n", n, sim.n_dofs(),
              sim.constraints().size());
  std::printf("  exact: sigma_xx = %+.6f   eps_zz = %+.6f\n", sxx_exact, ezz_exact);

  const auto& ms = sp.map(sp.index_of("s_0"));
  const auto& mus = sp.map(sp.index_of("u_0"));
  const auto s_off = static_cast<std::size_t>(sp.offset(sp.index_of("s_0")));
  const auto u_off = static_cast<std::size_t>(sp.offset(sp.index_of("u_0")));

  double worst_s = 0.0;
  for (const Index f : FacetSelector::where(m, 3, FacetSelector::at(0, 0.0))) {
    const auto av = exokal::face_area_vector(m, f);
    const double area = std::sqrt(av[0] * av[0] + av[1] * av[1] + av[2] * av[2]);
    // the facet's own outward-independent normal traction: the constant moment
    // of component x, divided by the measure it was integrated against, and by
    // the canonical normal's sign so a flipped facet does not read as a flipped
    // stress
    const double sgn = av[0] / area;
    const double sxx = x[s_off + static_cast<std::size_t>(ms.global(2, f, 0, 0))] / (area * sgn);
    worst_s = std::max(worst_s, std::abs(sxx - sxx_exact));
  }
  std::printf("  computed sigma_xx: worst |error| over the x=0 facets = %.3e\n", worst_s);

  // u is stored as the cell integral and with the opposite sign of the
  // convention the closed form is written in, so both are undone here
  double worst_u = 0.0;
  for (Index e = 0; e < c.count(3); ++e) {
    const double vol = std::abs(exokal::signed_volume(m, e));
    const double z = exokal::centroid(m, 3, e)[2];
    const double uz = -x[u_off + static_cast<std::size_t>(mus.global(3, e, 0, 2))] / vol;
    worst_u = std::max(worst_u, std::abs(uz - ezz_exact * z));
  }
  std::printf("  computed u_z:      worst |error| over the cells      = %.3e\n", worst_u);
  if (undrained) {
    const double p_exact = load / alpha;
    const auto p_off = static_cast<std::size_t>(sp.offset(sp.index_of("p_0")));
    double worst_p = 0.0;
    for (Index e = 0; e < c.count(3); ++e) {
      worst_p = std::max(worst_p, std::abs(x[p_off + static_cast<std::size_t>(e)] - p_exact));
    }
    std::printf("  exact: p = %+.6f\n  computed p:        worst |error| over the cells      = %.3e\n",
                p_exact, worst_p);
  }
  return 0;
}
