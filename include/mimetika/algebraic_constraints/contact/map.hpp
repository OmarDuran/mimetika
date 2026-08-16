#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "exokal/numerics/dense.hpp"
#include "mimetika/algebraic_constraints/contact/laws.hpp"

// THE CONTACT PROBLEM AS A NONLINEAR ALGEBRAIC FUNCTION y = CD(x).
//
// Contact is a fixed-point problem in the contact traction, and nothing more:
//
//     y = CD(x),   CD(x) = P(x + r g(x)),   g(x) = J z,   A(x) z = b(x)
//
// where A(x) is the mechanics system with the fracture traction degrees of
// freedom PINNED to x. One evaluation is: pin, solve, read the gap, project.
// The solution of x = CD(x) is the converged contact state.
//
// WHY THIS IS THE RIGHT SEAM. Everything here is algebra: a linear solve, two
// linear maps, an index set and a projection. There is no mesh, no material, no
// boundary condition and no model object -- those sit behind the Mechanics
// interface below. That matters three ways:
//
//   * THE MECHANICS IS INTERCHANGEABLE. Whatever supplies (A, b) -- the mixed
//     elasticity of CauchyElasticityModel, or the poromechanics of
//     PoroelasticModel with its pore-pressure coupling on the right-hand side --
//     is invisible here, so adding a boundary condition upstream needs no
//     change at all in the contact code. This is the whole of what makes the
//     driver plug into both.
//   * THE ITERATION IS INTERCHANGEABLE. CD is just a function, so the relaxed
//     Picard iteration in fixed_point() can be replaced by Newton or Anderson
//     acceleration without touching the map.
//   * IT IS TESTABLE WITHOUT A MESH. Feed CD any x and check y: contraction,
//     the fixed point and the projection can each be checked on stub mechanics,
//     separately from the discretization.
//
// WHAT x CONTAINS. x and y are always the same object: the contact traction at
// the ENFORCEMENT POINTS, in the facet frame, normal component first --
// (t_n, t_t) in the plane, (t_n, t_t1, t_t2) in space -- under the convention
// t_n < 0 in compression, g_n > 0 open. The space is therefore identical for
// every law; what changes is the subset of it CD can return.

namespace mimetika::contact {

// ---------------------------------------------------------------------------
// WHAT CONTACT NEEDS FROM A MECHANICS PROBLEM, and nothing more.
//
// Three operations, all linear-algebraic. A model implements this and gains
// contact; contact never learns what a model is. The pinned solve is the only
// one that touches a global system, and condensation below removes even that
// from the iteration.
class ContactMechanics {
 public:
  virtual ~ContactMechanics() = default;

  virtual std::size_t n_points() const = 0;
  virtual int dim() const = 0;
  virtual std::size_t n_dofs() const = 0;

  // THE AFFINE SOLUTION OPERATOR S : m -> z(m): the solution of the mechanics
  // on the affine subspace { z : sigma|_F = m }, where F indexes the fracture
  // traction moments. An ESSENTIAL condition, since in the mixed form the
  // traction IS a degree of freedom, so prescribing it is a Dirichlet condition
  // on the unknown rather than a penalty.
  //
  //     A_CC z_C = b_C - A_CF m ,   z_F = m ,
  //
  // and A_CC does not depend on m, which is what makes S affine and lets an
  // implementation factorize once and back-substitute thereafter.
  virtual void solution_operator(const std::vector<double>& moments,
                                 std::vector<double>& z) const = 0;

  // facet-frame values at the enforcement points -> traction moments, the
  // linear map the pinning consumes
  virtual void to_moments(const std::vector<Vec3>& x, std::vector<double>& moments) const = 0;

  // THE GAP, from a solution vector. It is the RESIDUAL of the replaced fault
  // rows, -(row . z - b_f), not `J z` alone: reading J z by itself imposes a
  // spurious jump equal to b_f's coefficients -- the Biot pore-coupling term,
  // for one, which is exactly what appears when this is a poromechanics
  // problem. An implementation that forgets it is wrong only when coupled,
  // which is the worst way to be wrong.
  virtual void gap(const std::vector<double>& z, std::vector<Vec3>& g) const = 0;
};

// ---------------------------------------------------------------------------

// WHAT THE AUGMENTATION MULTIPLIES: the total normal gap, but the tangential
// INCREMENT.
//
// The two components are not treated alike, and the asymmetry is physical. The
// normal condition g_n >= 0 is a statement about the ABSOLUTE gap, so the normal
// term is driven by the total jump. Coulomb friction instead opposes the slip
// RATE -- eq. (2e) of Frigo et al. (2025) reads g_T . t_T = tau_max |g_T| with
// g_T a rate -- which a quasi-static scheme discretizes as the backward
// increment g_T,n - g_T,n-1.
//
// Driving the tangential part with the total jump instead is equivalent only
// while the loading is monotone and proportional: the first step from rest, or
// any path along a fixed direction. As soon as the slip direction rotates or
// reverses, the total jump still points along the accumulated path and the
// traction lags the direction it should oppose.
inline Vec3 driving_gap(const Vec3& gap, const Vec3* g_prev, int dim) {
  Vec3 out = gap;
  if (g_prev != nullptr) {
    for (int k = 1; k < dim; ++k) {
      out[static_cast<std::size_t>(k)] -= (*g_prev)[static_cast<std::size_t>(k)];
    }
  }
  return out;
}

struct MapEvaluation {
  std::vector<Vec3> value;       // y = CD(x)
  std::vector<Vec3> gap;         // g(x) at the enforcement points
  std::vector<State> internal;   // law state after the projection
  std::vector<double> solution;  // the raw solution vector z of the pinned system
};

// ---------------------------------------------------------------------------

class ContactMap {
 public:
  // the map holds the law by pointer as the driver does: a temporary dangles
  ContactMap(const ContactMechanics&, ContactLaw&&, std::vector<double>) = delete;

  ContactMap(const ContactMechanics& mechanics, const ContactLaw& law,
             std::vector<double> augmentation)
      : mechanics_(&mechanics), law_(&law), augmentation_(std::move(augmentation)) {
    if (augmentation_.size() != mechanics.n_points()) {
      throw std::invalid_argument("ContactMap: one augmentation per enforcement point");
    }
    for (const double r : augmentation_) {
      if (!(r > 0.0)) throw std::invalid_argument("ContactMap: the augmentation must be positive");
    }
  }

  std::size_t n_points() const { return mechanics_->n_points(); }
  int dim() const { return mechanics_->dim(); }
  const ContactLaw& law() const { return *law_; }
  const std::vector<double>& augmentation() const { return augmentation_; }

  // THE IN-SITU TRACTION at the enforcement points, if the unknown is an
  // INCREMENT.
  //
  // A contact law constrains the TOTAL traction: Signorini says the total
  // normal traction is compressive, not that some increment is. When only an
  // increment is solved for -- a depletion response on top of an in-situ state
  // -- the law must still be shown the total, or a unilateral condition will
  // read a tensile increment on a firmly closed fault as opening. The prestress
  // is added before the projection and removed after, so x stays the
  // incremental unknown the mechanics constrains while the law sees physical
  // reality.
  void set_prestress(std::vector<Vec3> p) {
    if (p.size() != n_points()) throw std::invalid_argument("ContactMap: prestress size");
    prestress_ = std::move(p);
  }
  bool has_prestress() const { return !prestress_.empty(); }
  const std::vector<Vec3>& prestress() const { return prestress_; }

  std::vector<Vec3> initial_guess() const { return std::vector<Vec3>(n_points()); }
  std::vector<State> initial_state() const { return std::vector<State>(n_points()); }

  // ONE EVALUATION: pin, solve, read the gap, project.
  MapEvaluation evaluate(const std::vector<Vec3>& x, const std::vector<State>* internal = nullptr,
                         const std::vector<Vec3>* g_prev = nullptr, double dt = 0.0) const {
    if (x.size() != n_points()) throw std::invalid_argument("ContactMap: x size");
    const int d = dim();

    std::vector<double> moments;
    mechanics_->to_moments(x, moments);

    MapEvaluation out;
    mechanics_->solution_operator(moments, out.solution);
    mechanics_->gap(out.solution, out.gap);

    out.internal = internal != nullptr ? *internal : initial_state();
    out.value.resize(n_points());
    for (std::size_t p = 0; p < n_points(); ++p) {
      const Vec3* gp = g_prev != nullptr ? &(*g_prev)[p] : nullptr;
      const Vec3 drive = driving_gap(out.gap[p], gp, d);

      Vec3 trial;
      for (int k = 0; k < d; ++k) {
        const auto kk = static_cast<std::size_t>(k);
        const double offset = has_prestress() ? prestress_[p][kk] : 0.0;
        trial[kk] = x[p][kk] + offset + augmentation_[p] * drive[kk];
      }
      Vec3 y = law_->project(trial, out.internal[p], d, &out.gap[p], gp, dt);
      // back to the incremental unknown the mechanics constrains
      if (has_prestress()) {
        for (int k = 0; k < d; ++k) {
          y[static_cast<std::size_t>(k)] -= prestress_[p][static_cast<std::size_t>(k)];
        }
      }
      out.value[p] = y;
    }
    return out;
  }

  // ||CD(x) - x||_inf, the quantity the iteration drives to zero
  double residual(const std::vector<Vec3>& x, const std::vector<State>* internal = nullptr,
                  const std::vector<Vec3>* g_prev = nullptr, double dt = 0.0) const {
    const MapEvaluation e = evaluate(x, internal, g_prev, dt);
    double worst = 0.0;
    for (std::size_t p = 0; p < n_points(); ++p) {
      for (int k = 0; k < dim(); ++k) {
        const auto kk = static_cast<std::size_t>(k);
        worst = std::max(worst, std::abs(e.value[p][kk] - x[p][kk]));
      }
    }
    return worst;
  }

 private:
  const ContactMechanics* mechanics_;
  const ContactLaw* law_;
  std::vector<double> augmentation_;
  std::vector<Vec3> prestress_;
};

// ---------------------------------------------------------------------------

struct FixedPointResult {
  std::vector<Vec3> x;
  std::vector<Vec3> gap;
  std::vector<State> internal;
  std::vector<double> solution;
  int iterations{0};
  double change{0.0};
  bool converged{false};
};

struct FixedPointOptions {
  double relaxation{0.5};
  double tolerance{1e-10};
  int max_iterations{200};
};

// -- the condensed map, and Newton on it -----------------------------------------

// THE NONLINEAR SYSTEM IS SMALL: n_points * dim unknowns, a handful per fracture
// facet. Evaluating it through a global solve every iteration is backwards, and
// two facts remove the need to.
//
// The constrained matrix DOES NOT DEPEND ON x -- pinning zeroes the same rows and
// columns whatever the pinned values are, so only the right-hand side moves --
// and that dependence is AFFINE:
//
//     b(x) = b_0 + B W x ,   z(x) = A^{-1} b(x) ,
//     g(x) = g_0 + Ghat x ,  Ghat = J A^{-1} B W .
//
// So one factorization and n_points * dim + 1 back-substitutions give a small
// dense Ghat, after which CD is a matvec and a projection and the iteration
// touches the global system not at all.
struct CondensedMap {
  std::vector<Vec3> g0;      // the gap at x = 0
  std::vector<double> ghat;  // (n*d) x (n*d), row major: dg/dx
  std::size_t n{0};
  int d{0};

  // g(x) = g_0 + Ghat x
  std::vector<Vec3> gap_at(const std::vector<Vec3>& x) const {
    std::vector<Vec3> g = g0;
    for (std::size_t i = 0; i < n; ++i) {
      for (int a = 0; a < d; ++a) {
        const std::size_t row = i * static_cast<std::size_t>(d) + static_cast<std::size_t>(a);
        double acc = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
          for (int b = 0; b < d; ++b) {
            const std::size_t col = j * static_cast<std::size_t>(d) + static_cast<std::size_t>(b);
            acc += ghat[row * n * static_cast<std::size_t>(d) + col] *
                   x[j][static_cast<std::size_t>(b)];
          }
        }
        g[i][static_cast<std::size_t>(a)] += acc;
      }
    }
    return g;
  }
};

inline CondensedMap condense(const ContactMechanics& mech) {
  CondensedMap c;
  c.n = mech.n_points();
  c.d = mech.dim();
  const std::size_t size = c.n * static_cast<std::size_t>(c.d);

  std::vector<Vec3> x(c.n);
  std::vector<double> moments, z;
  mech.to_moments(x, moments);
  mech.solution_operator(moments, z);
  mech.gap(z, c.g0);

  c.ghat.assign(size * size, 0.0);
  std::vector<Vec3> gj;
  for (std::size_t j = 0; j < c.n; ++j) {
    for (int b = 0; b < c.d; ++b) {
      std::vector<Vec3> unit(c.n);
      unit[j][static_cast<std::size_t>(b)] = 1.0;
      mech.to_moments(unit, moments);
      mech.solution_operator(moments, z);
      mech.gap(z, gj);
      const std::size_t col = j * static_cast<std::size_t>(c.d) + static_cast<std::size_t>(b);
      for (std::size_t i = 0; i < c.n; ++i) {
        for (int a = 0; a < c.d; ++a) {
          const std::size_t row = i * static_cast<std::size_t>(c.d) + static_cast<std::size_t>(a);
          c.ghat[row * size + col] =
              gj[i][static_cast<std::size_t>(a)] - c.g0[i][static_cast<std::size_t>(a)];
        }
      }
    }
  }
  return c;
}

// SEMISMOOTH NEWTON ON F(x) = CD(x) - x = 0.
//
// Picard is a good solver only when CD is a CONTRACTION, which needs the
// augmentation to match the fracture compliance AND that compliance to be close
// to diagonal. Neither holds for a fault that cuts the domain: Ghat is dense,
// every facet feels every other, and no scalar r makes I + r Ghat a contraction.
// Rescaling r cannot fix a spectral radius problem caused by off-diagonal
// coupling -- which is why the displaced-fault benchmark diverges under Picard
// while its slip profile is already right.
//
// Newton does not care. With
//
//     F(x) = P(x + r (g_0 + Ghat x)) - x ,
//     J    = T (I + r Ghat) - I ,          T = dP/dt ,
//
// the step is a dense solve of size n_points * dim -- small, which is the whole
// point of condensing. For an AFFINE law (a frictionless fault) the residual is
// linear and this converges in a SINGLE iteration.
//
// T IS THE CONSISTENT TANGENT, and it is the law's own AD tangent: the same
// projection body re-run on exokal's Local. A hand-differentiated T would have
// to agree with the branch the projection took at this very trial, which is the
// thing that silently drifts.
inline FixedPointResult newton(const ContactMap& map, const CondensedMap& cond,
                               const FixedPointOptions& opt = {},
                               const std::vector<Vec3>* x0 = nullptr,
                               const std::vector<State>* internal = nullptr,
                               const std::vector<Vec3>* g_prev = nullptr, double dt = 0.0) {
  namespace num = exokal::numerics;
  FixedPointResult res;
  res.x = x0 != nullptr ? *x0 : map.initial_guess();
  std::vector<State> state = internal != nullptr ? *internal : map.initial_state();
  const int d = map.dim();
  const std::size_t n = map.n_points();
  const std::size_t size = n * static_cast<std::size_t>(d);
  const std::vector<double>& r = map.augmentation();

  // d(trial)/dx = I + r Ghat, fixed across the iteration
  num::Dense trial_jacobian(size, size);
  for (std::size_t i = 0; i < size; ++i) {
    for (std::size_t j = 0; j < size; ++j) {
      trial_jacobian(i, j) = r[i / static_cast<std::size_t>(d)] * cond.ghat[i * size + j];
    }
    trial_jacobian(i, i) += 1.0;
  }

  double change = std::numeric_limits<double>::infinity();
  for (int it = 1; it <= opt.max_iterations; ++it) {
    const std::vector<Vec3> gap = cond.gap_at(res.x);
    std::vector<Vec3> trial(n), value(n);
    for (std::size_t p = 0; p < n; ++p) {
      for (int k = 0; k < d; ++k) {
        const auto kk = static_cast<std::size_t>(k);
        const double offset = map.has_prestress() ? map.prestress()[p][kk] : 0.0;
        trial[p][kk] = res.x[p][kk] + offset + r[p] * gap[p][kk];
      }
      const Vec3* gp = g_prev != nullptr ? &(*g_prev)[p] : nullptr;
      Vec3 y = map.law().project(trial[p], state[p], d, &gap[p], gp, dt);
      if (map.has_prestress()) {
        for (int k = 0; k < d; ++k) {
          y[static_cast<std::size_t>(k)] -= map.prestress()[p][static_cast<std::size_t>(k)];
        }
      }
      value[p] = y;
    }

    std::vector<double> residual(size, 0.0);
    change = 0.0;
    for (std::size_t p = 0; p < n; ++p) {
      for (int k = 0; k < d; ++k) {
        const auto kk = static_cast<std::size_t>(k);
        residual[p * static_cast<std::size_t>(d) + kk] = value[p][kk] - res.x[p][kk];
        change = std::max(change, std::abs(value[p][kk] - res.x[p][kk]));
      }
    }
    res.iterations = it;
    res.gap = gap;

    double biggest = 1.0;
    bool finite = std::isfinite(change);
    for (const Vec3& v : res.x) {
      for (int k = 0; k < d; ++k) {
        const double e = v[static_cast<std::size_t>(k)];
        if (!std::isfinite(e)) finite = false;
        biggest = std::max(biggest, std::abs(e));
      }
    }
    if (finite && change <= opt.tolerance * biggest) {
      res.converged = true;
      break;
    }
    if (!finite) break;

    // J = blockdiag(T) (I + r Ghat) - I, with T the law's AD tangent at this trial
    num::Dense jac(size, size);
    for (std::size_t p = 0; p < n; ++p) {
      const Vec3* gp = g_prev != nullptr ? &(*g_prev)[p] : nullptr;
      const Tangent T = map.law().tangent(trial[p], state[p], d, &gap[p], gp, dt);
      for (int a = 0; a < d; ++a) {
        const std::size_t row = p * static_cast<std::size_t>(d) + static_cast<std::size_t>(a);
        for (std::size_t j = 0; j < size; ++j) {
          double acc = 0.0;
          for (int b = 0; b < d; ++b) {
            acc += T(a, b) *
                   trial_jacobian(p * static_cast<std::size_t>(d) + static_cast<std::size_t>(b), j);
          }
          jac(row, j) = acc;
        }
      }
    }
    for (std::size_t i = 0; i < size; ++i) jac(i, i) -= 1.0;

    std::vector<double> rhs(size);
    for (std::size_t i = 0; i < size; ++i) rhs[i] = -residual[i];
    const std::vector<double> step = num::solve_lu(jac, rhs);
    for (std::size_t p = 0; p < n; ++p) {
      for (int k = 0; k < d; ++k) {
        res.x[p][static_cast<std::size_t>(k)] +=
            opt.relaxation * step[p * static_cast<std::size_t>(d) + static_cast<std::size_t>(k)];
      }
    }
  }

  res.change = change;
  res.internal = state;
  return res;
}

// SOLVE x = CD(x) BY RELAXED PICARD ITERATION.
//
// Under-relaxation is not cosmetic. While the fracture STICKS the tangential
// update is a contraction and relaxation = 1 converges; while it SLIDES it is
// not, and the plain iteration settles into a limit cycle of constant amplitude
// rather than converging. Damping restores convergence.
//
// Deliberately separate from ContactMap: the map is the problem, this is one way
// of solving it, and a Newton or Anderson variant would replace only this
// function.
inline FixedPointResult fixed_point(const ContactMap& map, const FixedPointOptions& opt = {},
                                    const std::vector<Vec3>* x0 = nullptr,
                                    const std::vector<State>* internal = nullptr,
                                    const std::vector<Vec3>* g_prev = nullptr, double dt = 0.0) {
  FixedPointResult res;
  res.x = x0 != nullptr ? *x0 : map.initial_guess();
  std::vector<State> state = internal != nullptr ? *internal : map.initial_state();
  const int d = map.dim();

  // CONVERGED MEANS SMALL, which a non-finite iterate never is. Without the
  // finiteness guard a diverging iteration reports success: once x overflows,
  // tolerance * max(|x|, 1) is inf and the test change <= inf passes, so
  // divergence would be indistinguishable from convergence in the returned flag.
  const auto settled = [&](const std::vector<Vec3>& x, double change) {
    if (!std::isfinite(change)) return false;
    double biggest = 1.0;
    for (const Vec3& v : x) {
      for (int k = 0; k < d; ++k) {
        const double e = v[static_cast<std::size_t>(k)];
        if (!std::isfinite(e)) return false;
        biggest = std::max(biggest, std::abs(e));
      }
    }
    return change <= opt.tolerance * biggest;
  };

  double change = std::numeric_limits<double>::infinity();
  for (int it = 1; it <= opt.max_iterations; ++it) {
    const MapEvaluation e = map.evaluate(res.x, &state, g_prev, dt);
    state = e.internal;
    change = 0.0;
    bool finite = true;
    for (std::size_t p = 0; p < map.n_points(); ++p) {
      for (int k = 0; k < d; ++k) {
        const auto kk = static_cast<std::size_t>(k);
        const double step = opt.relaxation * (e.value[p][kk] - res.x[p][kk]);
        res.x[p][kk] += step;
        change = std::max(change, std::abs(step));
        if (!std::isfinite(res.x[p][kk])) finite = false;
      }
    }
    res.iterations = it;
    res.gap = e.gap;
    res.solution = e.solution;
    if (settled(res.x, change)) {
      res.converged = true;
      break;
    }
    if (!finite) break;  // diverged: no point continuing
  }
  res.change = change;
  res.internal = state;
  return res;
}

}  // namespace mimetika::contact
