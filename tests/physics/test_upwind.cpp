#include <cmath>
#include <vector>

#include "exokal/ad/dual.hpp"
#include "mimetika/physics/constitutive/upwind.hpp"
#include "../mimetika_test.hpp"

using mimetika::physics::constitutive::density_weight;
using mimetika::physics::constitutive::face_total_mobility;
using mimetika::physics::constitutive::hybrid_pair;
using mimetika::physics::constitutive::pair_mobility;
using mimetika::physics::constitutive::upwind;

namespace {

bool near(double a, double b, double tol = 1e-12) { return std::abs(a - b) <= tol; }

double U(const std::vector<double>& L, const std::vector<double>& R, const std::vector<double>& rho,
         int a, int b, double nu) {
  return hybrid_pair(std::span<const double>(L), std::span<const double>(R),
                     std::span<const double>(rho), a, b, nu);
}

}  // namespace

// the weight is a three-valued selection, with the pair members as fixed
// points and the tie taking exactly one half
MIMETIKA_TEST(the_density_weight_selects_the_nearer_member) {
  // rho_a = 1000, rho_b = 200; separator at 600
  CHECK(near(density_weight(900.0, 1000.0, 200.0), 1.0));  // nearer a
  CHECK(near(density_weight(300.0, 1000.0, 200.0), 0.0));  // nearer b
  CHECK(near(density_weight(600.0, 1000.0, 200.0), 0.5));  // exactly at the tie

  // the pair members are fixed points, whichever way round
  CHECK(near(density_weight(1000.0, 1000.0, 200.0), 1.0));
  CHECK(near(density_weight(200.0, 1000.0, 200.0), 0.0));
  CHECK(near(density_weight(1000.0, 200.0, 1000.0), 0.0));
  CHECK(near(density_weight(200.0, 200.0, 1000.0), 1.0));

  // and it is frozen: a double in, a double out, nothing to differentiate
  static_assert(std::is_same_v<decltype(density_weight(1.0, 2.0, 3.0)), double>);
}

// Consistency: two cells sharing a state give back the continuous pair
// mobility f_a f_b lambda
MIMETIKA_TEST(the_operator_is_consistent) {
  const std::vector<double> m = {0.4, 1.1, 0.25, 0.7};
  const std::vector<double> rho = {1000.0, 200.0, 640.0, 850.0};
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      if (a == b) continue;
      for (const double nu : {1.0, -1.0, 0.0}) {
        CHECK(near(U(m, m, rho, a, b, nu), pair_mobility(std::span<const double>(m), a, b)));
      }
    }
  }
}

// Pair antisymmetry: U_ab(nu) = U_ba(-nu). The physical consequence is that the
// buoyant fluxes summed over ordered pairs cancel, so the component fluxes add
// up to the total flux with nothing left over.
MIMETIKA_TEST(pairs_are_antisymmetric_so_the_component_fluxes_close) {
  const std::vector<double> L = {0.4, 1.1, 0.25, 0.7};
  const std::vector<double> R = {0.9, 0.2, 1.4, 0.05};
  const std::vector<double> rho = {1000.0, 200.0, 640.0, 850.0};

  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      if (a == b) continue;
      for (const double nu : {0.7, -0.3}) {
        CHECK(near(U(L, R, rho, a, b, nu), U(L, R, rho, b, a, -nu), 1e-12));
      }
    }
  }

  // the pairwise directions are antisymmetric too, so the total buoyant
  // contribution over all ordered pairs vanishes identically
  double total = 0.0;
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      if (a == b) continue;
      const double nu = rho[static_cast<std::size_t>(b)] - rho[static_cast<std::size_t>(a)];
      total += U(L, R, rho, a, b, nu) * nu;
    }
  }
  CHECK(near(total, 0.0, 1e-12));
}

// Reduction consistency: as a passive density approaches a pair member,
// merging the two reproduces the (N-1)-phase operator exactly
MIMETIKA_TEST(coalescing_phases_reduce_to_the_merged_operator) {
  //          a       b      g (coalescing with a)
  const std::vector<double> L = {0.4, 1.1, 0.3};
  const std::vector<double> R = {0.9, 0.2, 0.5};
  const std::vector<double> rho = {1000.0, 200.0, 1000.0};  // g == a exactly
  const double nu = 0.8;

  // the merged two-phase system: a and g become one phase
  const std::vector<double> Lm = {L[0] + L[2], L[1]};
  const std::vector<double> Rm = {R[0] + R[2], R[1]};
  const std::vector<double> rhom = {1000.0, 200.0};

  const double merged = U(Lm, Rm, rhom, 0, 1, nu);
  const double summed = U(L, R, rho, 0, 1, nu) + U(L, R, rho, 2, 1, nu);
  CHECK(near(summed, merged, 1e-12));

  // the face total mobility agrees as well, phase g having joined a's side
  CHECK(near(face_total_mobility(std::span<const double>(L), std::span<const double>(R),
                                 std::span<const double>(rho), 0, 1, nu),
             face_total_mobility(std::span<const double>(Lm), std::span<const double>(Rm),
                                 std::span<const double>(rhom), 0, 1, nu)));
}

// Permutation invariance: no ordering of the phases is encoded, so
// relabelling leaves every pair value unchanged
MIMETIKA_TEST(the_construction_encodes_no_phase_ordering) {
  const std::vector<double> L = {0.4, 1.1, 0.25, 0.7};
  const std::vector<double> R = {0.9, 0.2, 1.4, 0.05};
  const std::vector<double> rho = {1000.0, 200.0, 640.0, 850.0};
  const int perm[4] = {2, 0, 3, 1};  // an arbitrary relabelling

  std::vector<double> Lp(4), Rp(4), rp(4);
  for (int i = 0; i < 4; ++i) {
    Lp[static_cast<std::size_t>(perm[i])] = L[static_cast<std::size_t>(i)];
    Rp[static_cast<std::size_t>(perm[i])] = R[static_cast<std::size_t>(i)];
    rp[static_cast<std::size_t>(perm[i])] = rho[static_cast<std::size_t>(i)];
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      if (a == b) continue;
      CHECK(near(U(L, R, rho, a, b, 0.6), U(Lp, Rp, rp, perm[a], perm[b], 0.6), 1e-12));
    }
  }
}

// Monotonicity: raising the active mobility at the expense of a single
// donor, at fixed cell total, never decreases the operator — for every
// donor and every pair
MIMETIKA_TEST(the_operator_is_monotone_under_constrained_variation) {
  std::vector<double> L = {0.4, 1.1, 0.25, 0.7};
  const std::vector<double> R = {0.9, 0.2, 1.4, 0.05};
  const std::vector<double> rho = {1000.0, 200.0, 640.0, 850.0};
  const double eps = 1e-7;

  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      if (a == b) continue;
      const double base = U(L, R, rho, a, b, 1.0);  // nu > 0: a upwinds at L
      for (int donor = 0; donor < 4; ++donor) {
        if (donor == a) continue;
        std::vector<double> Lv = L;  // raise a, lower the donor, total fixed
        Lv[static_cast<std::size_t>(a)] += eps;
        Lv[static_cast<std::size_t>(donor)] -= eps;
        const double slope = (U(Lv, R, rho, a, b, 1.0) - base) / eps;
        CHECK(slope >= -1e-6);
      }
    }
  }
}

// a pair vanishes when either member is absent from its upstream side —
// the property the interface directionality argument rests on
MIMETIKA_TEST(an_absent_member_kills_the_pair) {
  std::vector<double> L = {0.0, 1.1, 0.25};  // phase 0 absent at L
  const std::vector<double> R = {0.9, 0.2, 1.4};
  const std::vector<double> rho = {1000.0, 200.0, 640.0};

  CHECK(near(U(L, R, rho, 0, 1, 1.0), 0.0));   // 0 upwinds at L, where it is absent
  CHECK(U(L, R, rho, 0, 1, -1.0) > 0.0);       // but not from the other side
  CHECK(near(U(L, R, rho, 1, 0, -1.0), 0.0));  // the mirrored statement
}

// the operator differentiates: it is evaluated with dual scalars inside a
// residual, while the density weight stays frozen alongside
MIMETIKA_TEST(the_operator_carries_dual_scalars) {
  using D = exokal::ad::Dual<>;
  const std::vector<double> l = {0.4, 1.1, 0.25};
  const std::vector<double> r = {0.9, 0.2, 1.4};
  const std::vector<double> rho = {1000.0, 200.0, 640.0};

  std::vector<D> L, R;
  for (std::size_t i = 0; i < 3; ++i) L.push_back(D::variable(l[i], i, 3));
  for (const double v : r) R.push_back(D(v));

  const D u = hybrid_pair(std::span<const D>(L), std::span<const D>(R),
                          std::span<const double>(rho), 0, 1, 1.0);
  CHECK(near(u.value(), U(l, r, rho, 0, 1, 1.0), 1e-12));

  // the derivative matches a finite difference in the upstream mobility
  std::vector<double> lp = l;
  lp[0] += 1e-7;
  CHECK(near(u.d(0), (U(lp, r, rho, 0, 1, 1.0) - U(l, r, rho, 0, 1, 1.0)) / 1e-7, 1e-5));
}

MIMETIKA_TEST_MAIN()
