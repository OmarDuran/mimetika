#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Preconditioned MINRES (Paige & Saunders 1975), in the form Elman, Silvester &
// Wathen give as Algorithm 6.1.
//
// The method a symmetric INDEFINITE system takes. A mixed saddle point is
// symmetric only once the sign of one block is fixed -- the Darcy term writes
// the divergence as +D in the pressure row and -D^T in the flux row, so the
// assembled operator is [[M, -D^T], [D, 0]] and it is the PRESSURE EQUATION
// that must be negated to make it [[M, -D^T], [-D, 0]]. That is a property of
// the operator handed in, not of this file, and a caller that forgets it gets a
// method converging to the wrong thing rather than an error: MINRES minimizes
// over a Krylov space built from a symmetric Lanczos recurrence, and the
// recurrence is silent about the assumption.
//
// The operator and the preconditioner arrive as callables, so nothing here
// knows whether the action is assembled, matrix-free, or a composition of both.
// B must be symmetric POSITIVE DEFINITE -- the Lanczos vectors are orthogonal in
// its inner product, and (B^-1 v, v) is what the recurrence takes square roots
// of. A preconditioner that is merely symmetric makes that quantity negative and
// is refused here rather than allowed to produce a number.
//
// The residual reported is ||b - Ax||_{B^-1}, relative to its own first value:
// that is the quantity MINRES minimizes, and it is available from the recurrence
// for nothing. It is NOT ||b - Ax||_2, and the two differ by the conditioning of
// B -- a stopping test read as if it were the Euclidean residual is the
// preconditioned-residual trap.

namespace mimetika::solver {

struct MinresOptions {
  double rtol{1e-10};
  int max_iterations{500};
};

struct MinresReport {
  bool converged{false};
  int iterations{0};
  double residual{0.0};  // ||b - Ax||_{B^-1} over its first value
  std::string reason;
};

namespace minres_detail {

inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

}  // namespace minres_detail

// apply_a(v, y): y = A v.  apply_b(v, y): y = B^-1 v.  x is the initial guess
// on entry and the solution on exit; a wrong size is taken as the zero guess.
template <class ApplyA, class ApplyB>
MinresReport minres(ApplyA&& apply_a, ApplyB&& apply_b, const std::vector<double>& b,
                    std::vector<double>& x, const MinresOptions& opt = {}) {
  using minres_detail::dot;
  const std::size_t n = b.size();
  if (x.size() != n) x.assign(n, 0.0);

  std::vector<double> v_prev(n, 0.0), v(n, 0.0), v_next(n, 0.0);
  std::vector<double> z(n, 0.0), z_next(n, 0.0), az(n, 0.0);
  std::vector<double> w_prev(n, 0.0), w(n, 0.0), w_next(n, 0.0);

  apply_a(x, az);
  for (std::size_t i = 0; i < n; ++i) v[i] = b[i] - az[i];
  apply_b(v, z);

  double gamma2 = dot(z, v);
  if (gamma2 < 0.0) {
    throw std::invalid_argument(
        "minres: (B^-1 r, r) is negative, so the preconditioner is not positive definite");
  }
  double gamma = std::sqrt(gamma2);
  const double gamma0 = gamma;
  MinresReport rep;
  if (!(gamma0 > 0.0)) {
    rep.converged = true;
    rep.reason = "the right-hand side is already satisfied";
    return rep;
  }

  double gamma_prev = 1.0, eta = gamma;
  double s_prev = 0.0, s = 0.0, c_prev = 1.0, c = 1.0;

  for (int it = 1; it <= opt.max_iterations; ++it) {
    for (std::size_t i = 0; i < n; ++i) z[i] /= gamma;
    apply_a(z, az);
    const double delta = dot(az, z);
    for (std::size_t i = 0; i < n; ++i) {
      v_next[i] = az[i] - (delta / gamma) * v[i] - (gamma / gamma_prev) * v_prev[i];
    }
    apply_b(v_next, z_next);
    const double g2 = dot(z_next, v_next);
    if (g2 < 0.0) {
      throw std::invalid_argument(
          "minres: the preconditioner lost positive definiteness during the recurrence");
    }
    const double gamma_next = std::sqrt(g2);

    const double a0 = c * delta - c_prev * s * gamma;
    const double a1 = std::sqrt(a0 * a0 + gamma_next * gamma_next);
    const double a2 = s * delta + c_prev * c * gamma;
    const double a3 = s_prev * gamma;
    if (!(a1 > 0.0)) {
      rep.iterations = it;
      rep.residual = std::fabs(eta) / gamma0;
      rep.reason = "the Lanczos recurrence broke down";
      return rep;
    }
    const double c_next = a0 / a1, s_next = gamma_next / a1;

    for (std::size_t i = 0; i < n; ++i) {
      w_next[i] = (z[i] - a3 * w_prev[i] - a2 * w[i]) / a1;
      x[i] += c_next * eta * w_next[i];
    }
    eta = -s_next * eta;

    rep.iterations = it;
    rep.residual = std::fabs(eta) / gamma0;
    if (rep.residual <= opt.rtol) {
      rep.converged = true;
      rep.reason = "CONVERGED_RTOL";
      return rep;
    }

    // one step of the recurrence: everything shifts and nothing is copied
    std::swap(v_prev, v);
    std::swap(v, v_next);
    std::swap(z, z_next);
    std::swap(w_prev, w);
    std::swap(w, w_next);
    gamma_prev = gamma;
    gamma = gamma_next;
    s_prev = s;
    s = s_next;
    c_prev = c;
    c = c_next;
  }
  rep.reason = "DIVERGED_ITS";
  return rep;
}

}  // namespace mimetika::solver
