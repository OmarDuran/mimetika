#pragma once

#include <cmath>
#include <span>
#include <stdexcept>

// The hybrid upwind operator, on the mobilities alone.
//
// It lives here rather than inside a term because three places use it and must
// not drift apart: the mesh facets, the mixed-dimensional interfaces, and the
// energy equation. It is defined on the mobility simplex and knows nothing of
// where its arguments came from — no saturations, no phase equilibrium, no
// geometry.
//
// For an active pair (a, b) advected along a signed direction ν,
//
//     U_ab = upw(f_a, ν) · upw(f_b, −ν) · Λ_ab
//     Λ_ab = upw(m_a, ν) + upw(m_b, −ν)
//            + Σ_{g ∉ {a,b}} [ w_g upw(m_g, ν) + (1 − w_g) upw(m_g, −ν) ]
//
// with f the fractional flows m/Σm. The pair members are advected
// counter-currently, and each passive phase joins the side of the member it is
// nearer in density.
//
// THE DENSITY WEIGHT IS FROZEN, AND ITS TYPE SAYS SO. w_g depends on the
// densities alone, so it is constant under saturation variation and
// carries no Jacobian — it is re-evaluated BETWEEN nonlinear iterations,
// never differentiated within one. That is why it takes and returns plain
// doubles while the mobilities carry whatever scalar the caller chose:
// the signature makes it impossible to let it into the derivative path by
// accident.
//
// Properties this construction has, each of them tested:
//   consistency        both cells sharing a state gives f_a f_b lambda
//   pair antisymmetry  U_ab(nu) = U_ba(-nu), hence the component fluxes
//                      sum to the total flux with nothing left over
//   reduction          a passive density coalescing with a pair member
//                      reproduces the merged (N-1)-phase operator exactly
//   permutation        no ordering of the phases is encoded anywhere
//   monotonicity       raising a mobility at the expense of one donor,
//                      at fixed cell total, never decreases U

namespace mimetika::physics::constitutive {

// The upwind selection: the upstream value along nu.
template <class T>
T upwind(const T& left, const T& right, double nu) {
  return nu >= 0.0 ? left : right;
}

// The phase density weight: the Heaviside of the pairwise density
// contrast, taking the value 1/2 exactly at the tie so that the reduction
// under coalescence is continuous. Unity on a's side of the arithmetic
// mean separator, zero on b's; the pair members are fixed points.
//
// Deliberately double in and double out: this quantity must not be
// differentiated.
inline double density_weight(double rho_g, double rho_a, double rho_b) {
  const double s = (rho_a - rho_b) * (2.0 * rho_g - rho_a - rho_b);
  if (s > 0.0) return 1.0;
  if (s < 0.0) return 0.0;
  return 0.5;
}

// The face total mobility of a pair: the two active mobilities upwinded
// counter-currently, plus each passive phase taken from the side of the
// member it is nearer in density.
template <class T>
T face_total_mobility(std::span<const T> mob_left, std::span<const T> mob_right,
                      std::span<const double> rho, int a, int b, double nu) {
  const std::size_t n = mob_left.size();
  const auto& up = nu >= 0.0 ? mob_left : mob_right;
  const auto& down = nu >= 0.0 ? mob_right : mob_left;
  T lam = up[static_cast<std::size_t>(a)] + down[static_cast<std::size_t>(b)];
  for (std::size_t g = 0; g < n; ++g) {
    if (g == static_cast<std::size_t>(a) || g == static_cast<std::size_t>(b)) continue;
    const double w =
        density_weight(rho[g], rho[static_cast<std::size_t>(a)], rho[static_cast<std::size_t>(b)]);
    lam = lam + w * up[g] + (1.0 - w) * down[g];
  }
  return lam;
}

// U_ab: the face instance of the pair mobility f_a f_b lambda.
template <class T>
T hybrid_pair(std::span<const T> mob_left, std::span<const T> mob_right,
              std::span<const double> rho, int a, int b, double nu) {
  const std::size_t n = mob_left.size();
  if (mob_right.size() != n || rho.size() != n) {
    throw std::invalid_argument("hybrid_pair: mobility/density counts disagree");
  }
  if (a == b || a < 0 || b < 0 || static_cast<std::size_t>(a) >= n ||
      static_cast<std::size_t>(b) >= n) {
    throw std::invalid_argument("hybrid_pair: the pair must be two distinct phases");
  }
  const auto& up = nu >= 0.0 ? mob_left : mob_right;
  const auto& down = nu >= 0.0 ? mob_right : mob_left;

  T tot_up{}, tot_down{};
  for (std::size_t i = 0; i < n; ++i) {
    tot_up = tot_up + up[i];
    tot_down = tot_down + down[i];
  }
  // the barycentric coordinates of each upstream cell, so both lie in [0,1]
  const T fa = up[static_cast<std::size_t>(a)] / tot_up;
  const T fb = down[static_cast<std::size_t>(b)] / tot_down;
  return fa * fb * face_total_mobility(mob_left, mob_right, rho, a, b, nu);
}

// The continuous pair mobility f_a f_b lambda, for the consistency check.
template <class T>
T pair_mobility(std::span<const T> mob, int a, int b) {
  T tot{};
  for (const T& m : mob) tot = tot + m;
  return mob[static_cast<std::size_t>(a)] * mob[static_cast<std::size_t>(b)] / tot;
}

}  // namespace mimetika::physics::constitutive
