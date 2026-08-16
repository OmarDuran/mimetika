#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/ad/local.hpp"

// CONTACT LAWS ON A FRACTURE: the constitutive part, free of any degree of
// freedom.
//
// Every law relates the TRACTION on the fracture to the DISPLACEMENT JUMP.
// Both are presented in the facet frame (n, t1, t2) under one fixed
// convention:
//
//     g_n > 0   the fracture is OPEN (a gap)
//     t_n < 0   the fracture is in COMPRESSION
//
// so Signorini reads g_n >= 0, t_n <= 0, g_n t_n = 0. A law never sees a
// degree of freedom, a mesh, or a basis: the driver owns the rotation into
// this frame, the moment/point conversion, assembly and the solve.
//
// THE GENERAL CONTRACT IS IMPLICIT, C(t, g, state) = 0, because that is the
// only form covering unilateral contact -- a compliance g = A t cannot express
// t_n <= 0. Laws differ along axes that each force something on the driver:
//
//   relation form    compliance / implicit    compliance => one linear solve
//   smoothness       smooth / nonsmooth       nonsmooth => outer iteration
//   tangent symmetry symmetric / not          friction is NOT symmetric
//   path dependence  none / incremental       load steps
//   internal state   none / slip              state array, committed per step
//   enforcement      averaged / pointwise     where the projection is applied
//
// SOLUTION STRATEGY. The driver runs an augmented Lagrangian (Uzawa) outer
// iteration in which the multiplier lambda IS the physical contact traction:
// the mechanics is solved with the fracture traction constrained to lambda,
// the gap g is recovered, and then
//
//     lambda <- project(lambda + r g).
//
// So the only thing a nonsmooth law must supply is its projection onto the
// admissible set -- that is what project() is. The augmentation r is not free:
// the iteration contracts only for r < 2 / compliance, so the driver derives it
// from the stiffness the fracture actually sees.
//
// A NOTE ON WHAT IS VERIFIED. Of the laws here, SignoriniCoulomb (which
// subsumes the frictionless case at mu = 0) and FrictionlessBilateral are the
// ones the Novikov et al. (2024) benchmarks exercise, against the mimetic-AFW
// stress product. SlipWeakening derives from SignoriniCoulomb by overriding
// the friction coefficient alone, which is why `friction_at` is virtual: a
// user-supplied law should need to state only what differs.
//
// THE CONSISTENT TANGENT IS NOT WRITTEN DOWN ANYWHERE. dt/dt_trial is what
// turns the fixed-point sweep into a semismooth Newton iteration -- linear
// convergence into quadratic -- and it is the one part of a contact law that
// implementations reliably get wrong: the projection has branches, the
// derivative has the same branches, and the two drift apart the moment either
// is edited. So no law here differentiates itself. Each states its projection
// ONCE, as a template over the scalar type, and `differentiate` re-runs that
// same body on exokal's local AD type. The branch is chosen by the values, the
// derivative is carried by the arithmetic, and the two cannot disagree because
// there is only one body.
//
// That is also why AssociativeMohrCoulomb -- whose closest-point return map has
// four active sets, each with its own closed form -- costs no more to
// differentiate than the identity.

namespace mimetika::contact {

namespace detail {
// the value of a scalar, whichever scalar it is: the only thing the projections
// need that is not arithmetic, and it appears solely where a BRANCH is chosen
inline double val(double x) { return x; }
inline double val(const exokal::ad::Local& x) { return x.value(); }
}  // namespace detail

// A traction or jump at one enforcement point, in the facet frame: normal
// first, then the dim-1 shear components. Held at fixed width 3 so a 2D and a
// 3D law share one type; `dim` says how much of it is live.
//
// Templated on the scalar so ONE projection body serves both the value and its
// derivative: T = double evaluates it, T = exokal::ad::Local differentiates it.
template <class T>
struct VecN {
  T v[3]{};

  T& operator[](std::size_t i) { return v[i]; }
  const T& operator[](std::size_t i) const { return v[i]; }

  const T& normal() const { return v[0]; }

  // |t_t|: the shear magnitude over the live components.
  //
  // AT THE ORIGIN THE NORM IS NOT DIFFERENTIABLE, and sqrt would hand back an
  // infinite derivative rather than say so. Returning the constant 0 selects
  // the subgradient 0 there, which is the choice that makes the projection's
  // tangent the identity on a shear-free trial -- the correct limit from every
  // direction that matters, since the friction radius is nonnegative and a
  // vanishing shear can never exceed it.
  T shear_norm(int dim) const {
    using std::sqrt;
    T acc = T(0.0);
    for (int k = 1; k < dim; ++k) {
      acc = acc + v[static_cast<std::size_t>(k)] * v[static_cast<std::size_t>(k)];
    }
    if (!(detail::val(acc) > 0.0)) return T(0.0);
    return sqrt(acc);
  }
};

using Vec3 = VecN<double>;

// dt_i / dt_trial_j at one enforcement point, in the facet frame. Row-major and
// fixed at 3x3 for the same reason Vec3 is fixed at 3; `dim` says how much is
// live.
struct Tangent {
  double m[9]{};

  double& operator()(int i, int j) { return m[static_cast<std::size_t>(i * 3 + j)]; }
  double operator()(int i, int j) const { return m[static_cast<std::size_t>(i * 3 + j)]; }
};

// The internal variables one enforcement point carries. `n_state` entries are
// live; the rest are unused, so every law shares one container.
struct State {
  static constexpr std::size_t kMax = 2;
  double v[kMax]{0.0, 0.0};

  double& operator[](std::size_t i) { return v[i]; }
  double operator[](std::size_t i) const { return v[i]; }
};

// where a point stands relative to the admissible set; a diagnostic, not a
// control flow input
enum class Status { open = 0, stick = 1, slip = 2 };

// ---------------------------------------------------------------------------

// THE PROJECTION'S DERIVATIVE, BY RUNNING THE PROJECTION.
//
// The trial traction is the independent variable -- dim components, one local
// block -- and the law's own `project_at` template is evaluated on exokal's
// Local. Every branch it takes is decided by the VALUES, which are the same
// values the double instantiation would see, so the derivative returned is the
// derivative OF THE BRANCH ACTUALLY TAKEN. That is what "consistent" means in
// consistent tangent, and here it is structural rather than maintained.
//
// The state and the jump are held fixed: they are the step's data, not the
// unknown, which is why a slip-weakening coefficient enters as a constant. The
// state is copied because a projection may not commit anything -- committing is
// `advance`, and it happens once a step has converged.
template <class Law>
Tangent differentiate(const Law& law, const Vec3& trial, const State& state, int dim,
                      const Vec3* g = nullptr, const Vec3* g_prev = nullptr, double dt = 0.0) {
  namespace ad = exokal::ad;
  const auto n = static_cast<std::size_t>(dim);
  ad::LocalContext ctx(ad::LocalSpace::from_sizes({n}));
  std::vector<double> at(n);
  for (std::size_t k = 0; k < n; ++k) at[k] = trial[k];

  const std::vector<ad::Local> seed = ctx.seed(at);
  VecN<ad::Local> x;
  for (std::size_t k = 0; k < n; ++k) x[k] = seed[k];

  State scratch = state;
  const VecN<ad::Local> t = law.template project_at<ad::Local>(x, scratch, dim, g, g_prev, dt);

  Tangent J;
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j)
      J(i, j) = t[static_cast<std::size_t>(i)].d(static_cast<std::size_t>(j));
  }
  return J;
}

// The two virtual entry points, generated from the one templated body. A law
// writes `project_at` and then this line; there is no second place for the
// projection and its derivative to disagree.
#define MIMETIKA_CONTACT_PROJECTION                                                        \
  Vec3 project(const Vec3& trial, State& state, int dim, const Vec3* g = nullptr,          \
               const Vec3* g_prev = nullptr, double dt = 0.0) const override {             \
    return project_at<double>(trial, state, dim, g, g_prev, dt);                           \
  }                                                                                        \
  Tangent tangent(const Vec3& trial, const State& state, int dim, const Vec3* g = nullptr, \
                  const Vec3* g_prev = nullptr, double dt = 0.0) const override {          \
    return ::mimetika::contact::differentiate(*this, trial, state, dim, g, g_prev, dt);    \
  }

class ContactLaw {
 public:
  virtual ~ContactLaw() = default;
  virtual std::string name() const = 0;

  // internal variables carried per enforcement point
  virtual std::size_t n_state() const { return 0; }

  // THE STATE A POINT STARTS A SIMULATION IN, which is not always zero: a
  // rate-and-state fault begins at theta = theta0, and starting it at zero puts
  // log(0) in the friction coefficient on the first step. The driver asks the
  // law rather than assuming, so a law that needs a nonzero initial state gets
  // one without the driver knowing what it means.
  virtual State initial_state() const { return State{}; }
  // needs the jump of the previous step (slip history)
  virtual bool path_dependent() const { return false; }
  // needs a time increment (slip RATE)
  virtual bool rate_dependent() const { return false; }
  // whether the exact tangent is symmetric -- friction is not
  virtual bool symmetric_tangent() const { return true; }

  // A_f in the facet frame when the law is EXACTLY LINEAR, else absent.
  //
  // The diagonal in the (n, t_1, ..., t_{dim-1}) components. A law that
  // supplies one is solved in a single linear solve, with no outer iteration
  // and no projection -- which is why LinearContact never reaches the fixed
  // point map in practice.
  virtual bool has_linear_compliance() const { return false; }
  virtual Vec3 linear_compliance(int /*dim*/) const { return Vec3{}; }

  // THE PROJECTION onto the admissible set: the whole of what a nonsmooth law
  // must supply. `trial` is lambda + r g in the facet frame.
  virtual Vec3 project(const Vec3& trial, State& state, int dim, const Vec3* g = nullptr,
                       const Vec3* g_prev = nullptr, double dt = 0.0) const = 0;

  // THE CONSISTENT TANGENT dt/dt_trial, which turns the fixed-point sweep into
  // a semismooth Newton iteration.
  //
  // Every law shipped here obtains it from `differentiate` -- its own
  // projection re-run on exokal's AD scalar -- so it is exact and cannot drift
  // out of step with the projection. The default below is a CENTRAL DIFFERENCE,
  // and it exists only so that the contract of this class stays what it says it
  // is: a law must supply its projection, and nothing else. A law that takes
  // the default pays the accuracy of a difference quotient near the
  // nonsmooth branches, which is exactly where a contact law lives.
  virtual Tangent tangent(const Vec3& trial, const State& state, int dim, const Vec3* g = nullptr,
                          const Vec3* g_prev = nullptr, double dt = 0.0) const {
    const double h = 1e-6 * std::max(1.0, std::abs(trial[0]) + trial.shear_norm(dim));
    Tangent J;
    for (int j = 0; j < dim; ++j) {
      Vec3 lo = trial, hi = trial;
      lo[static_cast<std::size_t>(j)] -= h;
      hi[static_cast<std::size_t>(j)] += h;
      State a = state, b = state;
      const Vec3 pl = project(lo, a, dim, g, g_prev, dt);
      const Vec3 ph = project(hi, b, dim, g, g_prev, dt);
      for (int i = 0; i < dim; ++i) {
        J(i, j) = (ph[static_cast<std::size_t>(i)] - pl[static_cast<std::size_t>(i)]) / (2.0 * h);
      }
    }
    return J;
  }

  // Commit internal variables at the end of a converged step.
  virtual void advance(const Vec3& /*traction*/, const Vec3* /*g*/, State& /*state*/, int /*dim*/,
                       const Vec3* /*g_prev*/ = nullptr, double /*dt*/ = 0.0) const {}

  // where the point stands; open by default for laws with no unilateral part
  virtual Status status(const Vec3& /*traction*/, int /*dim*/, double /*tol*/ = 1e-10) const {
    return Status::stick;
  }
};

// ---------------------------------------------------------------------------

// A_f sigma n = [[u]] -- linear springs, always bonded.
//
// Allows tension and interpenetration: there is no unilateral condition. The
// law is exactly representable, so the driver solves it in a single linear
// solve rather than iterating.
class LinearContact final : public ContactLaw {
 public:
  explicit LinearContact(double normal_stiffness = 1.0, double shear_stiffness = 1.0)
      : kn_(normal_stiffness), kt_(shear_stiffness) {
    if (!(kn_ > 0.0) || !(kt_ > 0.0)) {
      throw std::invalid_argument("LinearContact: stiffnesses must be positive");
    }
  }

  std::string name() const override { return "linear"; }
  bool has_linear_compliance() const override { return true; }
  Vec3 linear_compliance(int dim) const override {
    Vec3 c;
    c[0] = 1.0 / kn_;
    for (int k = 1; k < dim; ++k) c[static_cast<std::size_t>(k)] = 1.0 / kt_;
    return c;
  }

  template <class T>
  VecN<T> project_at(const VecN<T>& trial, State& /*state*/, int /*dim*/, const Vec3* = nullptr,
                     const Vec3* = nullptr, double = 0.0) const {
    return trial;  // no constraint at all: the identity
  }
  MIMETIKA_CONTACT_PROJECTION

  double normal_stiffness() const { return kn_; }
  double shear_stiffness() const { return kt_; }

 private:
  double kn_, kt_;
};

// A CLOSED, FRICTIONLESS FAULT: t_t = 0 and g_n = 0.
//
// Bilateral in the normal direction -- the fault is held shut and may carry
// tension -- and free to slide tangentially. The projection keeps the normal
// traction and zeroes the shear, so the converged state has no opening and no
// shear stress: the classical frictionless crack.
//
// WHY NOT SignoriniCoulomb(friction = 0). That law also clips the normal
// traction to compression, which is right for a TOTAL-stress problem and wrong
// for an INCREMENTAL one. A fault sitting under tens of MPa of in-situ
// compression stays firmly closed, so an incremental solve -- where only the
// depletion response is computed -- must not read an incremental normal tension
// as opening. Signorini there would open the fault spuriously wherever the
// increment happens to be tensile. The choice between the two is a modelling
// decision about WHAT THE UNKNOWN IS, not about the physics of the fault.
class FrictionlessBilateral final : public ContactLaw {
 public:
  std::string name() const override { return "frictionless_bilateral"; }

  template <class T>
  VecN<T> project_at(const VecN<T>& trial, State& /*state*/, int dim, const Vec3* = nullptr,
                     const Vec3* = nullptr, double = 0.0) const {
    VecN<T> t;
    t[0] = trial[0];  // whatever holds the fault shut
    for (int k = 1; k < dim; ++k) t[static_cast<std::size_t>(k)] = T(0.0);
    return t;
  }
  MIMETIKA_CONTACT_PROJECTION
};

// UNILATERAL CONTACT WITH COULOMB FRICTION (the Alart-Curnier projection).
//
//   normal      g_n >= 0, t_n <= 0, g_n t_n = 0  -- no interpenetration, no
//               tension; the fracture may open and lose contact
//   tangential  |t_t| <= -mu t_n + c; sticking inside the cone, sliding on it
//
// The projection clips the normal traction to the compressive half-line, then
// projects the tangential traction onto the friction disk whose radius follows
// from the PROJECTED normal traction -- so an open point carries no shear,
// automatically.
//
// State is the accumulated tangential slip, which this law does not itself use
// but which makes the slip path available to callers and to the laws derived
// from it.
//
// `friction_at` IS VIRTUAL, and that is the extension point: a slip-weakening
// or rate-and-state law differs from Coulomb in the coefficient alone, so it
// overrides one short function and inherits the projection, the state handling
// and the status diagnostic unchanged.
class SignoriniCoulomb : public ContactLaw {
 public:
  explicit SignoriniCoulomb(double friction = 0.6, double cohesion = 0.0)
      : friction_(friction), cohesion_(cohesion) {
    if (friction_ < 0.0) throw std::invalid_argument("SignoriniCoulomb: friction must be >= 0");
  }

  std::string name() const override { return "signorini_coulomb"; }
  std::size_t n_state() const override { return 1; }  // accumulated slip magnitude
  bool path_dependent() const override { return true; }
  bool symmetric_tangent() const override { return false; }  // non-associated

  double friction() const { return friction_; }
  double cohesion() const { return cohesion_; }

  // THE COEFFICIENT THIS POINT SEES: constant for Coulomb, and the single thing
  // a weakening or rate-and-state law overrides.
  //
  // It is handed the whole of the step's data -- the state, the current jump,
  // the previous one and the time increment -- because that is what the family
  // spans: slip weakening reads the jump, rate-and-state reads the RATE, and
  // neither should have to restate the projection to get at it.
  virtual double friction_at(const State& /*state*/, const Vec3* /*g*/, const Vec3* /*g_prev*/,
                             double /*dt*/, int /*dim*/) const {
    return friction_;
  }

  template <class T>
  VecN<T> project_at(const VecN<T>& trial, State& state, int dim, const Vec3* g = nullptr,
                     const Vec3* g_prev = nullptr, double dt = 0.0) const {
    using std::max;
    using std::min;
    VecN<T> t;
    // normal: onto the compressive half-line
    t[0] = min(trial[0], T(0.0));
    // tangential: onto the friction disk of radius -mu t_n + c, with the
    // radius taken from the PROJECTED normal traction
    const double mu = friction_at(state, g, g_prev, dt, dim);
    const T radius = max(-mu * t[0] + cohesion_, T(0.0));
    const T mag = trial.shear_norm(dim);
    const T scale = mag > radius ? radius / max(mag, T(1e-300)) : T(1.0);
    for (int k = 1; k < dim; ++k) {
      t[static_cast<std::size_t>(k)] = trial[static_cast<std::size_t>(k)] * scale;
    }
    return t;
  }
  MIMETIKA_CONTACT_PROJECTION

  void advance(const Vec3& /*traction*/, const Vec3* g, State& state, int dim,
               const Vec3* g_prev = nullptr, double = 0.0) const override {
    if (g == nullptr) return;
    double acc = 0.0;
    for (int k = 1; k < dim; ++k) {
      const double prev = g_prev != nullptr ? (*g_prev)[static_cast<std::size_t>(k)] : 0.0;
      const double d = (*g)[static_cast<std::size_t>(k)] - prev;
      acc += d * d;
    }
    state[0] += std::sqrt(acc);
  }

  Status status(const Vec3& traction, int dim, double tol = 1e-10) const override {
    if (traction[0] > -tol) return Status::open;
    const double radius = std::max(-friction_ * traction[0] + cohesion_, 0.0);
    return traction.shear_norm(dim) >= radius - tol ? Status::slip : Status::stick;
  }

 private:
  double friction_, cohesion_;
};

// SLIP-WEAKENING FRICTION (Novikov et al. 2024, Eq. 23): the coefficient falls
// linearly from mu_s to mu_d over a critical slip distance d_c,
//
//     mu(|g_t|) = max(mu_d, mu_s - (mu_s - mu_d) |g_t| / d_c).
//
// Only the coefficient changes, so only friction_at is overridden -- the
// Alart-Curnier projection, the slip accumulation and the status diagnostic all
// come from Coulomb unchanged. This is the law of Benchmark 3, and the reason
// friction_at takes the JUMP as well as the state: the weakening is driven by
// the tangential slip at the current iterate, not only by what has been
// committed.
class SlipWeakening final : public SignoriniCoulomb {
 public:
  SlipWeakening(double mu_static, double mu_dynamic, double critical_slip, double cohesion = 0.0)
      : SignoriniCoulomb(mu_static, cohesion),
        static_(mu_static),
        dynamic_(mu_dynamic),
        d_c_(critical_slip) {
    if (!(d_c_ > 0.0)) {
      throw std::invalid_argument("SlipWeakening: the critical slip distance must be positive");
    }
    if (dynamic_ > static_) {
      throw std::invalid_argument("SlipWeakening: mu_d must not exceed mu_s -- it WEAKENS");
    }
  }

  std::string name() const override { return "slip_weakening"; }

  double friction_at(const State& state, const Vec3* g, const Vec3* /*g_prev*/, double /*dt*/,
                     int dim) const override {
    // the slip that has weakened the fault: what is committed, plus what the
    // current jump adds
    double slip = state[0];
    if (g != nullptr) slip = std::max(slip, g->shear_norm(dim));
    return std::max(dynamic_, static_ - (static_ - dynamic_) * slip / d_c_);
  }

  double mu_static() const { return static_; }
  double mu_dynamic() const { return dynamic_; }
  double critical_slip() const { return d_c_; }

 private:
  double static_, dynamic_, d_c_;
};

// RATE- AND STATE-DEPENDENT FRICTION (regularised, with the aging law).
//
// The coefficient is no longer a property of the fault but of how fast it is
// moving and how long it has been in contact:
//
//     mu(V, theta) = mu0 + a ln(V/V0) + b ln(V0 theta / Dc)
//
// with the slip rate V = |g_t - g_t_prev| / dt and the state variable theta
// evolving by the aging law dtheta/dt = 1 - V theta / Dc. The DIRECT effect
// a ln(V/V0) strengthens the fault as it accelerates; the EVOLUTION effect
// b ln(V0 theta/Dc) strengthens it as contact matures. Whether the fault is
// velocity-weakening -- and so capable of unstable slip -- is the sign of
// a - b, which is why both appear separately rather than as one number.
//
// THE AGING LAW IS INTEGRATED IMPLICITLY,
//
//     theta_new = (theta + dt) / (1 + dt V / Dc),
//
// which is unconditionally stable AND unconditionally positive: theta can never
// be driven negative by too large a step, so log(theta) never fails. The
// explicit form theta + dt(1 - V theta/Dc) does both at dt > Dc/V, which on a
// seismic-cycle problem is every step but the first.
//
// EVERYTHING ELSE IS INHERITED, and that is the point of the taxonomy: the
// unilateral normal condition and the projection onto the friction disk are
// Coulomb's, and only the RADIUS of that disk changes. The law therefore states
// the coefficient, the state evolution, and nothing else -- and its consistent
// tangent comes from the same AD pass as every other law's, with mu frozen at
// the step's rate, which is the linearization the projection actually performs.
class RateAndStateFriction final : public SignoriniCoulomb {
 public:
  RateAndStateFriction(double mu0 = 0.6, double a = 0.010, double b = 0.015, double Dc = 1e-4,
                       double V0 = 1e-6, double theta0 = 1.0, double Vmin = 1e-16)
      : SignoriniCoulomb(mu0),
        mu0_(mu0),
        a_(a),
        b_(b),
        d_c_(Dc),
        v0_(V0),
        theta0_(theta0),
        v_min_(Vmin) {
    if (!(d_c_ > 0.0) || !(v0_ > 0.0) || !(v_min_ > 0.0)) {
      throw std::invalid_argument("RateAndStateFriction: Dc, V0 and Vmin must be positive");
    }
    if (!(theta0_ > 0.0)) {
      throw std::invalid_argument(
          "RateAndStateFriction: theta0 must be positive -- the "
          "coefficient carries log(theta)");
    }
  }

  std::string name() const override { return "rate_and_state"; }
  std::size_t n_state() const override { return 2; }  // accumulated slip, and theta
  bool rate_dependent() const override { return true; }

  State initial_state() const override {
    State s;
    s[1] = theta0_;
    return s;
  }

  double mu0() const { return mu0_; }
  double direct_effect() const { return a_; }
  double evolution_effect() const { return b_; }
  double critical_slip() const { return d_c_; }
  double reference_rate() const { return v0_; }
  double initial_theta() const { return theta0_; }
  double minimum_rate() const { return v_min_; }

  // V = |g_t - g_t_prev| / dt, FLOORED at Vmin. The floor is not cosmetic: a
  // stuck point has V = 0 exactly, and log(0) would take the coefficient to
  // minus infinity on precisely the points that are not moving.
  double slip_rate(const Vec3* g, const Vec3* g_prev, double dt, int dim) const {
    if (g == nullptr || g_prev == nullptr || dt == 0.0) return v_min_;
    double acc = 0.0;
    for (int k = 1; k < dim; ++k) {
      const double d = (*g)[static_cast<std::size_t>(k)] - (*g_prev)[static_cast<std::size_t>(k)];
      acc += d * d;
    }
    return std::max(std::sqrt(acc) / std::abs(dt), v_min_);
  }

  double friction_coefficient(double V, double theta) const {
    const double safe = std::max(theta, 1e-300);
    return std::max(mu0_ + a_ * std::log(V / v0_) + b_ * std::log(v0_ * safe / d_c_), 0.0);
  }

  double friction_at(const State& state, const Vec3* g, const Vec3* g_prev, double dt,
                     int dim) const override {
    return friction_coefficient(slip_rate(g, g_prev, dt, dim), state[1]);
  }

  void advance(const Vec3& /*traction*/, const Vec3* g, State& state, int dim,
               const Vec3* g_prev = nullptr, double dt = 0.0) const override {
    const double V = slip_rate(g, g_prev, dt, dim);
    if (g != nullptr && g_prev != nullptr) {
      double acc = 0.0;
      for (int k = 1; k < dim; ++k) {
        const double d = (*g)[static_cast<std::size_t>(k)] - (*g_prev)[static_cast<std::size_t>(k)];
        acc += d * d;
      }
      state[0] += std::sqrt(acc);
    }
    if (dt != 0.0) state[1] = (state[1] + dt) / (1.0 + dt * V / d_c_);
  }

 private:
  double mu0_, a_, b_, d_c_, v0_, theta0_, v_min_;
};

// MOHR-COULOMB BY CLOSEST-POINT PROJECTION.
//
// The SAME admissible set as SignoriniCoulomb -- the truncated cone
//
//     S* = { t_n <= 0 ,  |t_t| <= c - mu t_n }
//
// -- but a different return mapping, and therefore different physics off the
// stick region. The updated traction is the point of S* nearest the trial in
// the augmentation-weighted metric
//
//     d(t_tr, t)^2 = (t_n,tr - t_n)^2 / eps_n + |t_t,tr - t_t|^2 / eps_t,
//
// which is the return mapping of ASSOCIATIVE elastoplasticity with the
// augmentation parameters in the role of the elastic moduli.
//
// ASSOCIATIVE VERSUS NON-ASSOCIATIVE. SignoriniCoulomb performs the PARTIAL
// return: t_n is clipped first and the shear is then projected RADIALLY at that
// fixed t_n, so sliding never alters the normal traction -- non-associative
// friction, appropriate to a smooth fault. The closest-point projection moves
// along the cone's own normal, so correcting an over-stressed shear state also
// CHANGES the normal traction: shear and normal response are energetically
// coupled, which is the traction-space image of dilatancy on a rough fault. The
// two agree only where the projection happens to be radial; elsewhere they are
// genuinely different constitutive assumptions, not two approximations of one.
//
// WHY THE METRIC MATTERS. eps_n and eps_t are not free numerical knobs here:
// they weight the distance, so they select which point of the cone is nearest.
// At eps_n = eps_t the projection is the plain Euclidean shortest path. They
// must be the values the augmented-Lagrangian update itself uses, or the return
// mapping and the iteration are minimising different things.
//
// HOW IT IS SOLVED. By rotational symmetry about the t_n axis the shear stays
// collinear with the trial, so the problem collapses to two unknowns (t_n, rho)
// with rho = |t_t| >= 0. The convex feasible set has three faces, giving four
// candidate active sets -- the trial itself, the lateral cone, the truncation
// disc t_n = 0, and the axis rho = 0. Each has a closed form, so the projection
// evaluates all four and takes the nearest FEASIBLE one. That is exact and free
// of nested case analysis, which matters: hand-written case logic on a cone is
// where these implementations usually go wrong.
//
// AND ITS TANGENT COSTS NOTHING. Four active sets means a hand-derived tangent
// would carry four separate closed forms of its own, each having to agree with
// the branch the projection took. Here the branch is selected by the values and
// the winning candidate's closed form is then differentiated by re-running it
// -- so the four cases exist once, in the projection, and nowhere else.
class AssociativeMohrCoulomb final : public SignoriniCoulomb {
 public:
  AssociativeMohrCoulomb(double friction = 0.6, double cohesion = 0.0, double eps_n = 1.0,
                         double eps_t = 1.0)
      : SignoriniCoulomb(friction, cohesion), eps_n_(eps_n), eps_t_(eps_t) {
    if (!(eps_n_ > 0.0) || !(eps_t_ > 0.0)) {
      throw std::invalid_argument("AssociativeMohrCoulomb: the metric weights must be positive");
    }
  }

  std::string name() const override { return "associative_mohr_coulomb"; }

  double eps_n() const { return eps_n_; }
  double eps_t() const { return eps_t_; }

  template <class T>
  VecN<T> project_at(const VecN<T>& trial, State& /*state*/, int dim, const Vec3* = nullptr,
                     const Vec3* = nullptr, double = 0.0) const {
    using std::max;
    using std::min;
    const double mu = friction(), c = cohesion();
    const T tn = trial[0];
    const T rho = trial.shear_norm(dim);

    // the four active sets, each in closed form
    T cn[4], cr[4];
    cn[0] = tn;  // the trial itself
    cr[0] = rho;
    const T lam = (rho + mu * tn - c) / (eps_t_ + mu * mu * eps_n_);
    cn[1] = tn - lam * (mu * eps_n_);  // the lateral cone face
    cr[1] = rho - lam * eps_t_;
    cn[2] = T(0.0);  // the truncation disc
    cr[2] = min(rho, T(c));
    cn[3] = min(tn, T(0.0));  // the axis
    cr[3] = T(0.0);

    // THE BRANCH IS CHOSEN BY THE VALUES, and only by the values: nearest
    // among the feasible candidates, ties to the lower index. Under AD this is
    // the same selection the double instantiation makes, which is precisely why
    // the derivative below belongs to the branch actually taken.
    double scale = 1.0;
    for (int q = 0; q < 4; ++q) scale = std::max(scale, std::abs(detail::val(cn[q])));
    const double tol = 1e-12 * scale;
    const double a0 = detail::val(tn), r0 = detail::val(rho);

    int region = 3;  // the axis is always feasible, so this is never the fallback it looks like
    double best = std::numeric_limits<double>::infinity();
    for (int q = 0; q < 4; ++q) {
      const double a = detail::val(cn[q]), b = detail::val(cr[q]);
      if (!(a <= tol && b >= -tol && b <= c - mu * a + tol)) continue;
      const double d = (a - a0) * (a - a0) / eps_n_ + (b - r0) * (b - r0) / eps_t_;
      if (d < best) {
        best = d;
        region = q;
      }
    }

    VecN<T> t;
    t[0] = min(cn[region], T(0.0));
    const T out = max(cr[region], T(0.0));
    // the shear stays collinear with the trial -- rotational symmetry about the
    // normal axis -- so only its magnitude is projected
    const T safe = r0 > 0.0 ? rho : T(1.0);
    for (int k = 1; k < dim; ++k) {
      t[static_cast<std::size_t>(k)] = trial[static_cast<std::size_t>(k)] / safe * out;
    }
    return t;
  }
  MIMETIKA_CONTACT_PROJECTION

 private:
  double eps_n_, eps_t_;
};

}  // namespace mimetika::contact
