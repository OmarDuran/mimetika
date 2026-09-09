#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "../mimetika_test.hpp"
#include "exokal/geometry/embedding.hpp"
#include "exokal/geometry/quadrature.hpp"
#include "graphos/core/incidence.hpp"
#include "mimetika/linear_solver/petsc.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/boundary.hpp"
#include "mimetika/model/cauchy_mechanics_model.hpp"

// The constant stress stabilized_vem reports, on a manufactured linear state.
//
// Pi_E sigma_h is the L^2 projection of the discrete stress onto P_0(E; S),
// assembled from the facet traction moments alone. Only the part of the dofs
// that survives the constant projection enters; the linear normal and rotation
// moments go to the stabilizer. It is read here through the model accessor the
// examples call, so the test covers the dof gathering -- slot order, outward
// orientation -- and not only the projector.
//
// The manufactured state is the confined gravity column, and it has to be
// confined: sigma_zz = -rho g (H - z) on its own is not a stress state here. It
// gives eps_xx = -a sigma_zz / 2mu, hence u_x depending on z, hence
// eps_xz != 0, contradicting sigma_xz = 0. Carrying the lateral stress
//
//     sigma_xx = sigma_yy = K_0 sigma_zz,   K_0 = lam / (lam + 2 mu)
//
// makes eps_xx = eps_yy = 0 exactly -- a is the compliance's trace coefficient
// and a(2K_0 + 1) = K_0 -- so u = (0, 0, u_z) with u_z quadratic, div sigma is
// the constant (0, 0, rho g), and the body force is f = (0, 0, -rho g).
//
// nu = 0.25, i.e. lam = mu: two-field strong symmetry locks as nu -> 1/2, so
// the incompressible regime is out of scope for this test by construction.
//
// Measured over h, h/2, h/4: first order in L^2, the coarsest pair still
// pre-asymptotic at 0.87. The exact stress is continuous, so its traction jump
// across an interior facet vanishes and the recovered jump is the defect;
// Pi_E is constant a cell, so that defect is O(h) and not round-off.

using graphos::Index;
using mimetika::CauchyMechanicsModel;
using mimetika::ElasticMaterial;
using mimetika::mesh::Family;

namespace {

constexpr double kMu = 1.0, kLam = 1.0;  // nu = 1/4
constexpr double kRhoG = 2.0;            // rho g
constexpr double kHeight = 2.0;          // H, the column's height

double confinement() { return kLam / (kLam + 2.0 * kMu); }

// sigma_exact(x), row-major 3x3
std::array<double, 9> exact_stress(const exokal::Mesh::Point& x) {
  const double szz = -kRhoG * (kHeight - x[2]);
  const double sxx = confinement() * szz;
  return {sxx, 0.0, 0.0, 0.0, sxx, 0.0, 0.0, 0.0, szz};
}

// u_exact = (0, 0, rho g z (z - 2H) / (2 (lam + 2 mu)))
std::array<double, 3> exact_displacement(const exokal::Mesh::Point& x) {
  const double z = x[2];
  return {0.0, 0.0, kRhoG * z * (z - 2.0 * kHeight) / (2.0 * (kLam + 2.0 * kMu))};
}

// grad u_exact, row-major: only d u_z / dz is nonzero
std::array<double, 9> exact_displacement_gradient(const exokal::Mesh::Point& x) {
  std::array<double, 9> g{};
  g[8] = kRhoG * (x[2] - kHeight) / (kLam + 2.0 * kMu);
  return g;
}

double frobenius2(const std::array<double, 9>& a, const std::array<double, 9>& b) {
  double s = 0.0;
  for (std::size_t k = 0; k < 9; ++k) s += (a[k] - b[k]) * (a[k] - b[k]);
  return s;
}

struct Errors {
  double e_pi{0.0};   // ||sigma - Pi_E sigma_h||_{L2(Omega)}
  double osc_pi{0.0};  // RMS traction-jump defect, interior facets
  double h{0.0};
  std::size_t cells{0};
};

Errors solve_and_measure(int n) {
  // a column of height H on an unstructured tet mesh, n x n x 2n
  const exokal::Mesh mesh =
      mimetika::mesh::box({n, n, 2 * n}, 3, Family::simplex, {1.0, 1.0, kHeight});
  const auto n_cells = static_cast<std::size_t>(mesh.topology().count(3));

  CauchyMechanicsModel model(mesh, 3, ElasticMaterial{kMu, kLam},
                             CauchyMechanicsModel::Realization::stabilized_vem,
                             CauchyMechanicsModel::Formulation::strong_symmetry);

  // f = -div sigma_exact, the same constant in every cell
  std::vector<double> f(3 * n_cells, 0.0);
  for (std::size_t e = 0; e < n_cells; ++e) f[3 * e + 2] = -kRhoG;
  model.set_body_force(std::move(f));

  // u on the whole boundary, as the affine datum the operator takes exactly.
  // u_exact is quadratic, so this is its Taylor expansion about the cofacet's
  // centroid -- O(h^2) on the facet, an order below the stress error measured.
  for (const Index face : mimetika::boundary_facets(mesh.topology(), 3)) {
    const auto xE = exokal::centroid(mesh, 3, mimetika::cofacet_of(mesh, 3, face));
    model.prescribe_displacement({face}, exact_displacement(xE),
                                 exact_displacement_gradient(xE));
  }
  model.build();

  // MUMPS, not the default SuperLU: measured 0.96 s against 57.32 s on the
  // finest level here, and the whole test 3 s against 65 s.
  mimetika::solver::SolverOptions so;
  so.factorization = "mumps";
  mimetika::solver::PetscSolver petsc(so);
  std::vector<double> x;
  const auto rep = petsc.solve(model.system(), model.rhs(), x);
  if (!rep.converged) throw std::runtime_error("gravity column: " + rep.reason);
  model.accept(std::move(x));

  Errors out;
  out.cells = n_cells;
  out.h = kHeight / static_cast<double>(2 * n);

  // ---- the field error, by cell quadrature --------------------------------
  std::vector<std::array<double, 9>> pi_of(n_cells);
  for (std::size_t e = 0; e < n_cells; ++e) {
    const auto cell = static_cast<Index>(e);
    pi_of[e] = model.cell_stress_projection(cell);

    const exokal::QuadratureRule qr = exokal::cell_quadrature(mesh, 3, cell, 4);
    for (std::size_t p = 0; p < qr.size(); ++p) {
      out.e_pi += qr.weights[p] * frobenius2(exact_stress(qr.points[p]), pi_of[e]);
    }
  }
  out.e_pi = std::sqrt(out.e_pi);

  // ---- the traction-jump defect on interior facets ------------------------
  const graphos::CoboundaryOperator cob = graphos::coboundary(mesh.topology(), 2);
  const auto n_faces = static_cast<std::size_t>(mesh.topology().count(2));
  std::size_t interior = 0;
  for (std::size_t face = 0; face < n_faces; ++face) {
    const auto b = static_cast<std::size_t>(cob.offsets[face]);
    const auto e2 = static_cast<std::size_t>(cob.offsets[face + 1]);
    if (e2 - b != 2) continue;
    const auto lhs = static_cast<std::size_t>(cob.indices[b]);
    const auto rhs = static_cast<std::size_t>(cob.indices[b + 1]);
    const mimetika::FacetFrame fr =
        mimetika::FacetFrame::of(mesh, 3, static_cast<Index>(lhs), static_cast<Index>(face));

    for (std::size_t i = 0; i < 3; ++i) {
      double t = 0.0;
      for (std::size_t j = 0; j < 3; ++j) {
        t += (pi_of[lhs][i * 3 + j] - pi_of[rhs][i * 3 + j]) * fr.normal[j];
      }
      out.osc_pi += t * t;
    }
    ++interior;
  }
  if (interior > 0) out.osc_pi = std::sqrt(out.osc_pi / static_cast<double>(interior));
  return out;
}

double rate(double coarse, double fine) {
  return std::log(coarse / fine) / std::log(2.0);
}

}  // namespace

MIMETIKA_TEST(the_projected_constant_stress_converges_on_a_gravity_column) {
  std::vector<Errors> e;
  for (const int n : {2, 4, 8}) e.push_back(solve_and_measure(n));

  for (const Errors& r : e) {
    std::printf("  h = %.4f  %6zu cells   E_Pi = %.4e   Osc_Pi = %.2e\n", r.h, r.cells,
                r.e_pi, r.osc_pi);
    // constant a cell against a linear exact stress: the jump is O(h)
    CHECK(r.osc_pi > 1e-3);
  }

  for (std::size_t k = 0; k + 1 < e.size(); ++k) {
    const double p_pi = rate(e[k].e_pi, e[k + 1].e_pi);
    std::printf("  rate  p(E_Pi) = %.2f\n", p_pi);
    // first order, the coarsest pair still pre-asymptotic at 0.87
    CHECK(p_pi >= 0.8);
    CHECK(e[k + 1].osc_pi < e[k].osc_pi);
  }
  // asymptotically first order
  CHECK(rate(e[e.size() - 2].e_pi, e[e.size() - 1].e_pi) >= 1.0);
}

MIMETIKA_TEST_MAIN()
