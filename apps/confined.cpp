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
#include "mimetika/model/compositions/poroelasticity.hpp"
#include "mimetika/solver/petsc.hpp"

using namespace mimetika;
using graphos::Index;

namespace {

// A ROTATION, so the same problem can be posed on a mesh with no facet aligned
// to an axis. Nothing about confined compression changes under it -- the
// undrained pressure is a scalar -- so any answer that moves is a boundary
// condition that was secretly about the axes rather than about the facets.
struct Rot {
  double a[3][3];
  static Rot of(double yaw, double pitch) {
    const double cy = std::cos(yaw), sy = std::sin(yaw);
    const double cp = std::cos(pitch), sp = std::sin(pitch);
    Rot r{{{cy, -sy * cp, sy * sp}, {sy, cy * cp, -cy * sp}, {0.0, sp, cp}}};
    return r;
  }
  exokal::Mesh::Point operator()(const exokal::Mesh::Point& x) const {
    return {a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
            a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
            a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2]};
  }
  // R S R^T, the applied stress carried into the rotated frame
  std::array<double, 9> conjugate(const std::array<double, 9>& s) const {
    std::array<double, 9> out{};
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        double v = 0.0;
        for (int k = 0; k < 3; ++k) {
          for (int l = 0; l < 3; ++l) v += a[i][k] * s[k * 3 + l] * a[j][l];
        }
        out[i * 3 + j] = v;
      }
    }
    return out;
  }
};

exokal::Mesh column(int n, double h, double width, const Rot* rot = nullptr) {
  std::vector<exokal::Mesh::Point> pts;
  for (int k = 0; k <= n; ++k) {
    for (int j = 0; j <= 1; ++j) {
      for (int i = 0; i <= 1; ++i) {
        const exokal::Mesh::Point x{i * width, j * width, k * h / n};
        pts.push_back(rot != nullptr ? (*rot)(x) : x);
      }
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
  const bool rotated = argc > 3 && std::string(argv[3]) == "rotated";
  const Rot rot = Rot::of(0.7, 0.4);
  const double h = 1.0, width = 1.0, mu = 1.0, lam = 1.0, load = 1.0;
  const double alpha = 1.0, inv_M = 0.0, perm = 1.0, dt = 1.0e-6;
  const double nu_ = lam / (2.0 * (lam + mu));
  const double inv_mod = (1.0 - 2.0 * nu_) / (2.0 * mu * (1.0 - 2.0 * nu_ + 3.0 * nu_));
  const double storage = 3.0 * alpha * alpha * inv_mod + inv_M;

  const exokal::Mesh m = column(n, h, width, rotated ? &rot : nullptr);
  const graphos::Complex& c = m.topology();
  // ONE de Rham SELECTION FOR BOTH PRODUCTS. The stress operators and the flux
  // Hodge each need a de Rham product on every cell, with different material
  // tensors but the SAME reconstruction space; the selection is the expensive
  // half and it does not depend on the material.
  const exokal::hodge::DeRhamGeometryCache geo = exokal::hodge::DeRhamGeometryCache::build(m);
  const exokal::hodge::StressOperators ops = exokal::hodge::StressOperators::build(
      m, 3, mu, lam, exokal::hodge::StressOperators::Realization::derham, &geo);
  exokal::forms::TermContext ctx;
  ctx.provide("stress_operators", ops);

  const exokal::hodge::FluxHodge hodge = exokal::hodge::FluxHodge::build(
      m, exokal::constitutive::Coefficient::uniform(perm),
      exokal::hodge::FluxHodge::Realization::derham, &geo);
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

  // THE LOADED FACET AND THE REST, selected by where they are in the ORIGINAL
  // geometry and then carried through the rotation, so the two runs pose
  // literally the same problem on two different meshes.
  const exokal::Mesh::Point up = rotated ? rot(exokal::Mesh::Point{0.0, 0.0, 1.0})
                                         : exokal::Mesh::Point{0.0, 0.0, 1.0};
  const auto height_of = [&](const exokal::Mesh::Point& x) {
    return x[0] * up[0] + x[1] * up[1] + x[2] * up[2];
  };
  std::vector<Index> loaded, confined;
  for (const Index f : boundary_facets(c, 3)) {
    (std::abs(height_of(exokal::centroid(m, 2, f)) - h) < 1e-9 ? loaded : confined).push_back(f);
  }
  const std::array<double, 9> applied{0, 0, 0, 0, 0, 0, 0, 0, -load};
  impose_traction(sim.constraints(), sp, "s_0", 3, m, loaded,
                  rotated ? rot.conjugate(applied) : applied);
  impose_free_slip(sim.constraints(), sp, "s_0", 3, m, confined);
  if (undrained) {
    // sealed EVERYWHERE, the top included: nothing drains, so nothing moves
    std::vector<Index> all;
    for (const Index f : boundary_facets(c, 3)) all.push_back(f);
    impose_normal_flux(sim.constraints(), sp, "q_0", 3, m, all);
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
      b[d] = sim.constraints().scale_at(d) * sim.constraints().rhs_at(d);
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
