// TERZAGHI'S CONSOLIDATION COLUMN, against Coussy Sect. 5.2.2.
//
// A saturated poroelastic layer on a rigid impervious base, drained at the
// top and loaded there at t = 0 by a step compressive traction. The fluid
// cannot escape instantaneously so it carries the whole load at first; it
// then drains through the surface and the load transfers to the skeleton.
//
// This problem is the right first benchmark because every boundary condition
// appears exactly once and the answer is known in closed form:
//
//     an applied traction on the loaded face   essential, on the stress
//     rollers on the sides and base            essential, on the stress
//     a drained face                           natural, and homogeneous
//     sealed faces                             essential, on the flux
//
// Get one wrong and the column stops being one-dimensional -- it bulges, or
// drains from the wrong face, or never reaches the right settlement. None of
// those is subtle once the profile is plotted against the analytic curve,
// which is exactly why this is the benchmark to port first.
//
// THE STEP SYSTEM carries no time-stepping machinery. Backward Euler over a
// step is the steady system with the flux rescaled to q~ = dt q, which is the
// Darcy term with its mobility set to dt; the old state enters only through
// the right-hand side. So what is assembled here is the composition, and what
// makes it transient is the storage package and one number.

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

// A column: one cell across, `n` cells tall, height h.
exokal::Mesh column(int n, double h, double width) {
  const auto vid = [](int i, int j, int k, int nz) {
    (void)nz;
    return static_cast<Index>((k * 2 + j) * 2 + i);
  };
  std::vector<exokal::Mesh::Point> pts;
  for (int k = 0; k <= n; ++k) {
    for (int j = 0; j <= 1; ++j) {
      for (int i = 0; i <= 1; ++i) {
        pts.push_back({i * width, j * width, k * h / n});
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
      for (const int l : f) cyc.push_back(vid(l & 1, (l >> 1) & 1, k + ((l >> 2) & 1), n));
      cell.push_back(std::move(cyc));
    }
    cells.push_back(std::move(cell));
  }
  return exokal::Mesh::from_polyhedra(std::move(pts), cells);
}

}  // namespace

int main(int argc, char** argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 20;
  const std::string out = argc > 2 ? argv[2] : "terzaghi.txt";

  // Coussy's constants, in the normalization the closed form uses
  const double h = 1.0;         // column height
  const double width = 1.0;
  const double mu = 1.0;        // shear modulus
  const double lam = 1.0;       // first Lame parameter
  const double alpha = 1.0;     // Biot coefficient
  const double inv_M = 0.0;     // incompressible FLUID; the skeleton still stores
  const double perm = 1.0;      // k / viscosity
  const double load = 1.0;      // sigma_0, compressive and positive

  // The two poroelastic coefficients, both built from the skeleton's
  // volumetric compliance. Terzaghi's incompressible constituents kill 1/M
  // but NOT the skeleton's own storage, which is what makes the column
  // consolidate at all.
  const double nu = lam / (2.0 * (lam + mu));                    // 3D Poisson
  const double inv_mod = (1.0 - 2.0 * nu) / (2.0 * mu * (1.0 - 2.0 * nu + 3.0 * nu));
  const double storage = 3.0 * alpha * alpha * inv_mod + inv_M;  // S

  const exokal::Mesh m = column(n, h, width);
  const graphos::Complex& c = m.topology();

  const exokal::hodge::StressOperators ops = exokal::hodge::StressOperators::build(m, mu, lam);
  std::printf("column: %lld cells, %zu stabilized\n", (long long)c.count(3), ops.n_stabilized());

  // the oedometric modulus, which sets the consolidation coefficient
  // THE CONSOLIDATION COEFFICIENT, derived from the uniaxial-strain condition
  // rather than taken from memory. Equilibrium along a roller-confined column
  // gives sigma_zz constant, so
  //
  //     eps_zz = (sigma_zz + alpha p) / K_oed   =>   d eps_zz/dt = (alpha/K_oed) dp/dt
  //
  // and the fluid content zeta = alpha eps_zz + p/M diffuses with
  //
  //     c_v = (k/mu_f) / S_u,      S_u = 1/M + alpha^2 / K_oed
  //
  // S_u is the UNIAXIAL storage and is NOT the coefficient the discrete term
  // carries: that one is the sigma-form S = 1/M + alpha^2/K, pairing with
  // tr(sigma) rather than with the strain, and the two are equal only when the
  // bulk and oedometric moduli are. Using the discrete coefficient here makes
  // the column consolidate on the wrong clock while every profile still has
  // exactly the right shape — which is what the closed form is for, and what
  // no amount of operator checking would have caught.
  const double K_oed = lam + 2.0 * mu;
  const double c_v = perm / (inv_M + alpha * alpha / K_oed);
  std::printf("K_oed = %.6f, c_v = %.6f, nu = %.6f\n", K_oed, c_v, nu);
  std::printf("volumetric compliance = %.6f, storage S = %.6f\n", inv_mod, storage);

  const double dt = (argc > 3 ? std::atof(argv[3]) : 1.0e-4) * h * h / c_v;
  const exokal::hodge::FluxHodge hodge = exokal::hodge::FluxHodge::build(
      m, exokal::constitutive::Coefficient::uniform(perm),
      exokal::hodge::FluxHodge::Realization::derham);

  exokal::forms::TermContext ctx;
  ctx.provide("stress_operators", ops);
  ctx.provide("flux_hodge", hodge);

  physics::ModelOptions o;
  o.mobility = dt;                     // the flux is q~ = dt q over the step
  o.storage = storage;
  o.volumetric_compliance = inv_mod;
  o.biot = alpha;
  Simulation sim(physics::Catalogue::instance().build("consolidation", o),
                 {StratumSpec{"ambient", &c, 3, 0}}, ctx);
  const auto& sp = sim.epoch().stratum(0).space();

  // ---- the four boundary conditions ------------------------------------
  const auto base = FacetSelector::where(m, 3, FacetSelector::at(2, 0.0));
  const auto top = FacetSelector::where(m, 3, FacetSelector::at(2, h));
  std::vector<Index> sides;
  for (const double v : {0.0, width}) {
    for (const int axis : {0, 1}) {
      for (const Index f : FacetSelector::where(m, 3, FacetSelector::at(axis, v))) {
        sides.push_back(f);
      }
    }
  }

  // the loaded face: a uniaxial compressive stress, formed against each
  // facet's canonical normal so the sign is never the caller's problem
  pin_facet_traction(sim.constraints(), sp, "s_0", 3, m, top,
                     {0, 0, 0, 0, 0, 0, 0, 0, -load});
  // rollers on EVERYTHING ELSE -- base and sides alike. Each facet pins the
  // two components tangential to its own normal, which is what a roller is;
  // pinning a fixed component everywhere under-constrains the sides and lets
  // the column bulge.
  std::vector<Index> confined = base;
  confined.insert(confined.end(), sides.begin(), sides.end());
  pin_facet_roller(sim.constraints(), sp, "s_0", 3, m, confined);
  // sealed on the same set; the drained top needs no condition at all,
  // because a drained face at zero pressure is the homogeneous natural one
  pin_facets(sim.constraints(), sp, "q_0", 3, confined);
  sim.freeze_constraints();
  std::printf("pinned %zu of %zu dofs\n", sim.constraints().size(), sim.n_dofs());

  std::printf("dt = %.3e, time factor per step = %.3e\n", dt, c_v * dt / (h * h));

  // ---- the step system, assembled once ----------------------------------
  // Everything here is linear, so the tangent IS the operator and it does not
  // move from step to step. The old state enters only through the right-hand
  // side, through the very blocks that make the balance transient.
  exokal::forms::TripletSink jac(sim.n_dofs());
  sim.jacobian(jac);
  const auto A = solver::SparseSystem::from(jac);

  const std::size_t p_off = static_cast<std::size_t>(sp.offset(sp.index_of("p_0")));
  const std::size_t p_end = p_off + static_cast<std::size_t>(sp.map(sp.index_of("p_0")).size());
  const std::size_t s_off = static_cast<std::size_t>(sp.offset(sp.index_of("s_0")));
  const std::size_t s_end = s_off + static_cast<std::size_t>(sp.map(sp.index_of("s_0")).size());

  // the storage blocks: the pressure row against the stress and against
  // itself. Applying THESE to the old state is the accumulation, and taking
  // them from the assembled operator rather than re-deriving them is what
  // keeps the step consistent with the system it steps.
  std::vector<std::size_t> acc;
  for (std::size_t k = 0; k < A.nnz(); ++k) {
    const auto r = static_cast<std::size_t>(A.row[k]);
    const auto c2 = static_cast<std::size_t>(A.col[k]);
    if (r >= p_off && r < p_end && ((c2 >= s_off && c2 < s_end) || (c2 >= p_off && c2 < p_end))) {
      acc.push_back(k);
    }
  }

  solver::PetscSolver petsc;
  // THE COLUMN STARTS UNLOADED, and the accumulation must say so. Freezing the
  // constraints writes the applied traction into the state, which is right for
  // a residual but wrong as a PREVIOUS state: the load is a step applied at
  // t = 0+, so at t = 0- the fluid content is zero and the first step's
  // right-hand side carries nothing. Seeding the march from the frozen state
  // instead accumulates coupling . sigma from a load that had not been applied,
  // and the column responds to a fraction of it -- an undrained plateau below
  // p_0 that then decays on the right clock with the right shape, which is
  // exactly the failure the closed form exists to expose.
  std::vector<double> y(sim.n_dofs(), 0.0), y_new, b(sim.n_dofs(), 0.0);

  // the time factors Coussy's figure is drawn at
  const std::vector<double> factors = {0.001, 0.01, 0.1, 0.25, 0.5, 1.0};
  std::size_t next = 0;
  std::FILE* f = std::fopen(out.c_str(), "w");
  // THE UNDRAINED PRESSURE, p_0 = alpha sigma_0 / (K_oed S_u), which is what
  // the closed form normalizes by. It is not imposed anywhere: the first step
  // from the unloaded state enforces the undrained condition on its own, so
  // p/p_0 at t -> 0+ is a prediction of the scheme rather than an input.
  const double S_u = inv_M + alpha * alpha / K_oed;
  const double p0 = alpha * load / (K_oed * S_u);
  std::fprintf(f, "# terzaghi: n=%d h=%g c_v=%g p0=%.17g dt=%.17g\n# factor zbar pbar\n", n, h,
               c_v, p0, dt);

  const int steps = static_cast<int>(std::ceil(factors.back() * h * h / (c_v * dt)));
  for (int step = 1; step <= steps && next < factors.size(); ++step) {
    std::fill(b.begin(), b.end(), 0.0);
    for (const std::size_t k : acc) {
      b[static_cast<std::size_t>(A.row[k])] += A.value[k] * y[static_cast<std::size_t>(A.col[k])];
    }
    for (std::size_t d = 0; d < sim.n_dofs(); ++d) {
      // the constrained equation is written with the scale of the row it
    // replaced, so its datum carries the same factor
    if (sim.constraints().pinned(d)) {
      b[d] = sim.constraints().scale_at(d) * sim.constraints().value_at(d);
    }
    }
    const auto rep = petsc.solve(A, b, y_new);
    if (!rep.converged) {
      std::printf("step %d did not converge: %s\n", step, rep.reason.c_str());
      return 1;
    }
    y = y_new;

    // THE MECHANICS, AGAINST WHAT UNIAXIAL STRAIN REQUIRES. Confined
    // compression has no freedom left: sigma_xx = lam/(lam+2mu) sigma_zz once
    // the pore pressure has gone, and eps_zz = sigma_zz/K_oed. Reading them off
    // the state separates a broken confinement from a broken coupling, which
    // the pressure profile alone cannot do.
    if (step == 1 || step == steps) {
      const auto& ms = sp.map(sp.index_of("s_0"));
      const auto& mu_ = sp.map(sp.index_of("u_0"));
      const Index side = FacetSelector::where(m, 3, FacetSelector::at(0, 0.0))[0];
      const double area = std::sqrt([&] {
        const auto av = exokal::face_area_vector(m, side);
        return av[0] * av[0] + av[1] * av[1] + av[2] * av[2];
      }());
      const double sxx = y[s_off + static_cast<std::size_t>(ms.global(2, side, 0, 0))] / area;
      const double uz =
          y[static_cast<std::size_t>(sp.offset(sp.index_of("u_0")) + mu_.global(3, 0, 0, 2))];
      std::printf("  step %-6d sigma_xx = %+.6f  u_z(base cell) = %+.6f  p(base) = %+.6f\n", step,
                  sxx, uz, y[p_off]);
    }

    const double T = c_v * dt * step / (h * h);
    if (T + 1e-12 >= factors[next]) {
      // the pressure profile against depth from the DRAINED surface
      for (Index e = 0; e < c.count(3); ++e) {
        const auto z = exokal::centroid(m, 3, e)[2];
        std::fprintf(f, "%g %g %.17g\n", factors[next], (h - z) / h,
                     y[p_off + static_cast<std::size_t>(e)] / p0);
      }
      std::printf("  T = %-6g  p(top cell) = %+.6f  p(bottom cell) = %+.6f\n", factors[next],
                  y[p_off + static_cast<std::size_t>(c.count(3) - 1)], y[p_off]);
      ++next;
    }
  }
  std::fclose(f);
  std::printf("wrote %s\n", out.c_str());
  (void)alpha;
  return 0;
}
