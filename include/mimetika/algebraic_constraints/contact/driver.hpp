#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#include "exokal/geometry/reference.hpp"
#include "mimetika/algebraic_constraints/contact/laws.hpp"
#include "mimetika/algebraic_constraints/contact/map.hpp"

// THE CONTACT DRIVER: turns a law into a solve, one step at a time.
//
// The driver owns everything a constitutive law should not know about -- the
// augmentation parameter, the outer iteration, the state carried between steps
// -- and everything the algebraic map should not know about either: the mesh,
// the moduli, the facets. What sits between them is the ContactMechanics
// interface, so the driver names no model and the map names no mesh.
//
// AUGMENTED LAGRANGIAN (UZAWA). The multiplier lambda IS the physical contact
// traction. Each outer iteration
//
//   1. solves the mechanics with the fracture traction CONSTRAINED to lambda
//      (an essential condition, since the traction is a degree of freedom here),
//   2. recovers the gap g from the jump operator, and
//   3. updates lambda <- law.project(lambda + r g).
//
// It matters that the traction is CONSTRAINED rather than tied to lambda + r g
// through a compliance: with the augmented relation inside the operator the
// solved traction is the TRIAL value, so an open fracture would come out
// carrying tension. Constraining it keeps t = lambda exactly, and an open point
// is then genuinely traction free.
//
// AN EXACTLY LINEAR LAW NEEDS NO OUTER ITERATION AT ALL. The driver detects it
// through ContactLaw::has_linear_compliance and does one solve with the
// compliance block instead -- which is why LinearContact never reaches the
// fixed-point map in practice.
//
// STEPPING. solve_step advances ONE step. The caller owns the loop, which is
// what keeps the driver free to be embedded in a staggered poromechanics
// scheme: the same object serves CauchyElasticityModel for pure contact
// mechanics and PoroelasticModel for contact poromechanics, because in both
// cases all it sees is a ContactMechanics.

namespace mimetika::contact {

// Everything carried from one step to the next.
struct ContactState {
  std::vector<Vec3> traction;   // the multiplier, in facet-frame values
  std::vector<State> internal;  // law state per enforcement point
  std::vector<Vec3> jump;       // facet-frame jump at the enforcement points
  std::vector<double> solution; // the mechanics solution of the last evaluation
  int iterations{0};
  bool converged{true};

  bool empty() const { return traction.empty(); }
};

struct DriverOptions {
  // Uzawa under-relaxation; 1.0 is none. Damping is what restores convergence
  // while the fracture SLIDES, where the plain iteration is not a contraction.
  double relaxation{0.5};
  double tolerance{1e-10};
  int max_iterations{200};
  // the augmentation r; non-positive means "derive it from the geometry"
  double augmentation{-1.0};

  // WHICH SOLVER SOLVES x = CD(x).
  //
  //   picard   the relaxed Uzawa sweep. Converges when CD is a contraction,
  //            which needs r to match the fracture compliance AND that
  //            compliance to be close to diagonal.
  //   newton   semismooth Newton on the CONDENSED map, using the law's AD
  //            tangent. Needed whenever the second condition fails -- a fault
  //            that cuts the domain has a dense Ghat, every facet feels every
  //            other, and no scalar r makes I + r Ghat a contraction. It costs
  //            n_points * dim + 1 back-substitutions once, and then touches the
  //            global system not at all.
  enum class Solver { picard, newton };
  Solver solver{Solver::picard};
};

// ---------------------------------------------------------------------------

// A PER-POINT AUGMENTATION PARAMETER r, FROM GEOMETRY AND MODULI.
//
// Uzawa converges only when r is comparable to the STIFFNESS THE FRACTURE SEES:
// the update lambda <- P(lambda + r g) contracts when r < 2 / compliance, and
// oscillates in a two-cycle otherwise. The surrounding rock behaves as a spring
// of compliance L / (2 mu + lam), where L is the distance from the two adjacent
// cell centroids to the facet, so the natural choice is its inverse.
//
// L IS MEASURED DIRECTLY as |(x_f - x_E) . n_f| summed over the two cells. A
// volume/area shortcut would be exact only for boxes -- on a tetrahedron it
// gives h/6 instead of h/4, mis-scaling r badly enough to stall the iteration.
inline std::vector<double> default_augmentation(const exokal::Mesh& mesh, int cell_dim,
                                                const std::vector<graphos::Index>& facets,
                                                double mu, double lam,
                                                std::size_t points_per_facet = 1) {
  const double modulus = 2.0 * mu + lam;
  const graphos::Complex& c = mesh.topology();
  const graphos::CoboundaryOperator cob = graphos::coboundary(c, cell_dim - 1);

  std::vector<double> out;
  out.reserve(facets.size() * points_per_facet);
  for (const graphos::Index f : facets) {
    const auto xf = exokal::centroid(mesh, cell_dim - 1, f);
    const auto b = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f)]);
    const auto e = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f) + 1]);
    // the facet's own normal: both cells measure their standoff along it
    const auto n = exokal::facet_normal_vector(mesh, cell_dim, cob.indices[b], f);
    double area = 0.0;
    for (std::size_t k = 0; k < 3; ++k) area += n[k] * n[k];
    area = std::sqrt(area);
    if (!(area > 0.0)) throw std::runtime_error("default_augmentation: degenerate facet");

    double length = 0.0;
    for (std::size_t m = b; m < e; ++m) {
      const auto xE = exokal::centroid(mesh, cell_dim, cob.indices[m]);
      double d = 0.0;
      for (std::size_t k = 0; k < 3; ++k) d += (xf[k] - xE[k]) * n[k] / area;
      length += std::abs(d);
    }
    if (!(length > 0.0)) {
      throw std::runtime_error("default_augmentation: a facet with no standoff to its cells");
    }
    for (std::size_t p = 0; p < points_per_facet; ++p) out.push_back(modulus / length);
  }
  return out;
}

// ---------------------------------------------------------------------------

class ContactDriver {
 public:
  // `mechanics` is the ONLY route by which boundary conditions, materials or a
  // pore-pressure right-hand side reach the contact problem -- the driver never
  // names one. `law` supplies the projection. Neither is owned.
  // A TEMPORARY LAW IS A COMPILE ERROR, deliberately.
  //
  // The driver holds the law by POINTER -- it does not own it, because a law is
  // configuration that outlives any one step -- so binding a temporary leaves it
  // dangling the moment the constructor returns. That is not a hypothetical: it
  // cost a stack-use-after-scope that stayed latent for two benchmarks, since
  // the freed slot kept plausible bytes until an unrelated change moved the
  // frame. The mistake is easy and silent, so it is refused here instead.
  ContactDriver(const ContactMechanics&, ContactLaw&&, std::vector<double>,
                DriverOptions = {}) = delete;

  ContactDriver(const ContactMechanics& mechanics, const ContactLaw& law,
                std::vector<double> augmentation, DriverOptions options = {})
      : mechanics_(&mechanics), law_(&law), options_(options),
        augmentation_(std::move(augmentation)) {
    if (augmentation_.size() != mechanics.n_points()) {
      throw std::invalid_argument("ContactDriver: one augmentation per enforcement point");
    }
    if (options_.augmentation > 0.0) {
      for (double& r : augmentation_) r = options_.augmentation;
    }
  }

  std::size_t n_points() const { return mechanics_->n_points(); }
  int dim() const { return mechanics_->dim(); }
  const ContactLaw& law() const { return *law_; }
  const std::vector<double>& augmentation() const { return augmentation_; }
  const DriverOptions& options() const { return options_; }

  void set_prestress(std::vector<Vec3> p) { prestress_ = std::move(p); }

  ContactState initial_state() const {
    ContactState s;
    s.traction.assign(n_points(), Vec3{});
    // THE LAW SUPPLIES ITS OWN INITIAL STATE. Zero is right for Coulomb and
    // wrong for rate-and-state, whose theta starts at theta0 and whose
    // coefficient carries log(theta).
    s.internal.assign(n_points(), law_->initial_state());
    s.jump.assign(n_points(), Vec3{});
    return s;
  }

  // ADVANCE ONE LOAD OR TIME STEP by solving x = CD(x). The caller owns the
  // loop -- which is what lets this be embedded in a staggered poromechanics
  // scheme, where the pressure solve sits between two calls to this.
  ContactState solve_step(const ContactState* previous = nullptr, double dt = 0.0) const {
    const ContactState state = previous != nullptr ? *previous : initial_state();

    ContactMap map(*mechanics_, *law_, augmentation_);
    if (!prestress_.empty()) map.set_prestress(prestress_);

    // AN EXACTLY LINEAR LAW IS ONE SOLVE. Its projection is the identity, so
    // the fixed point is reached in a single evaluation and iterating would
    // only re-derive it.
    FixedPointOptions fp;
    fp.relaxation = options_.relaxation;
    fp.tolerance = options_.tolerance;
    fp.max_iterations = law_->has_linear_compliance() ? 1 : options_.max_iterations;

    FixedPointResult res;
    if (options_.solver == DriverOptions::Solver::newton) {
      // one factorization is already done; this is n_points * dim + 1
      // back-substitutions, after which the iteration is dense and small
      const CondensedMap cond = condense(*mechanics_);
      res = newton(map, cond, fp, &state.traction, &state.internal, &state.jump, dt);
      // the condensed iteration never formed the global state, so form it once
      if (!res.x.empty()) {
        std::vector<double> moments;
        mechanics_->to_moments(res.x, moments);
        mechanics_->solution_operator(moments, res.solution);
        mechanics_->gap(res.solution, res.gap);
      }
    } else {
      res = fixed_point(map, fp, &state.traction, &state.internal, &state.jump, dt);
    }

    ContactState out;
    out.traction = res.x;
    out.jump = res.gap;
    out.solution = res.solution;
    out.iterations = res.iterations;
    out.converged = res.converged;

    // COMMIT the internal variables: the slip a weakening law will read next
    // step is what THIS step accumulated.
    out.internal = res.internal;
    for (std::size_t p = 0; p < n_points(); ++p) {
      law_->advance(out.traction[p], &out.jump[p], out.internal[p], dim(), &state.jump[p], dt);
    }
    return out;
  }

  // where each point stands, for reporting and for the slip-patch diagnostics
  // the benchmarks draw
  std::vector<Status> status(const ContactState& s, double tol = 1e-10) const {
    std::vector<Status> out(n_points());
    for (std::size_t p = 0; p < n_points(); ++p) {
      Vec3 total = s.traction[p];
      if (!prestress_.empty()) {
        for (int k = 0; k < dim(); ++k) {
          total[static_cast<std::size_t>(k)] += prestress_[p][static_cast<std::size_t>(k)];
        }
      }
      out[p] = law_->status(total, dim(), tol);
    }
    return out;
  }

 private:
  const ContactMechanics* mechanics_;
  const ContactLaw* law_;
  DriverOptions options_;
  std::vector<double> augmentation_;
  std::vector<Vec3> prestress_;
};

}  // namespace mimetika::contact
