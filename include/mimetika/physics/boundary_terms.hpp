#pragma once

#include <string>
#include <vector>

#include "exokal/ad/axpy.hpp"
#include "exokal/forms/term.hpp"
#include "exokal/hodge/flux_operators.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/model/boundary.hpp"

// The natural boundary terms: the datum for the quantity a mixed form does
// not carry as an unknown.
//
// In the flux-pressure form the interior pressure enters the flux row as
// -div^T p, so on a boundary facet — where there is no second cell — a
// prescribed pressure takes its place. That is the whole of the term: one
// signed contribution, on the flux degree of freedom of the facet being
// integrated over.
//
// A facet the data says nothing about contributes nothing, which is the
// homogeneous natural condition: a drained face at zero pressure needs no
// term, and a model cannot be wrong by failing to mention its free
// boundaries. The conditions that must be stated are the essential ones, and
// those are refused when a degree of freedom is left unconstrained twice.

namespace mimetika::physics::terms {

using exokal::forms::Coupling;
using exokal::forms::Params;
using exokal::forms::Stencil;
using exokal::forms::TermContext;

class PrescribedPressure {
 public:
  PrescribedPressure() = default;
  PrescribedPressure(const Params&, const TermContext& ctx)
      : data_(&ctx.require<BoundaryData>("boundary_pressure")),
        moments_data_(ctx.find<BoundaryMoments>("boundary_pressure_moments")),
        moments_(ctx.require<exokal::hodge::FluxOperators>("flux_operators").moments_per_facet()) {}

  static constexpr std::size_t kQ = 0;

  std::vector<std::string> fields() const { return {"q"}; }

  template <class T>
  void operator()(const Stencil& st, const std::vector<T>& a, std::vector<T>& r) const {
    (void)a;
    if (st.n_cofacets != 1) return;           // not on the boundary after all
    if (!data_->applies(st.support)) return;  // free: the homogeneous case

    // d moments per facet, not one. `q.begin + slot` is the lowest-order
    // indexing -- one flux per facet -- and on the de Rham space it addresses
    // an unrelated unknown.
    //
    // EVERY MOMENT TAKES ITS OWN COEFFICIENT. The row is int_f p_D (tau.n),
    // and with the equilibrated chart the facet Gram is |f| I, so dof b wants
    // (1/|f|) int_f p_D phi_b. A CONSTANT datum puts everything on the
    // constant moment -- the other basis functions are centred, so their
    // means vanish -- which is why one number per facet is the whole datum at
    // lowest order. An AFFINE one does not: dropping its higher coefficients
    // is a consistent O(h) perturbation, and it is what made the BDM products
    // converge at first order on a linear patch they reproduce exactly.
    const auto& q = st.field(kQ);
    const std::size_t slot = st.support_slot[0];
    const std::size_t i = q.begin + slot * static_cast<std::size_t>(moments_);
    if (i >= q.end) return;
    if (moments_data_ != nullptr && moments_data_->applies(st.support)) {
      for (int b = 0; b < moments_; ++b) {
        const std::size_t j = i + static_cast<std::size_t>(b);
        if (j < q.end) {
          r[j] += st.incidence[0] * moments_data_->at(st.support, static_cast<std::size_t>(b));
        }
      }
      return;
    }
    // the same shape the interior -div^T p has, with the datum in place of
    // the missing neighbour, and the facet's own incidence for the sign.
    //
    // The incidence, not `view.signs`. The latter is a local array parallel to
    // view.dofs, indexed 0..n_local; `i` here is a global degree of freedom, so
    // indexing it with `i` reads an unrelated entry or runs off the end. The
    // boundary coefficient of this facet in its one cofacet is what the
    // interior term carries, and st.incidence[0] is exactly that.
    //
    // Terzaghi cannot detect a wrong sign here: its only natural pressure
    // datum is a drained face at p = 0, and zero times a wrong sign is zero.
    r[i] += st.incidence[0] * data_->at(st.support);
  }

 private:
  const BoundaryData* data_{nullptr};
  const BoundaryMoments* moments_data_{nullptr};
  int moments_{1};
};

inline const exokal::forms::RegisterTerm<PrescribedPressure> register_prescribed_pressure{
    "prescribed_pressure", Coupling::boundary, {"q"}};

// The mirror of the prescribed pressure, for mechanics.
//
// In the Hellinger-Reissner form the interior displacement enters the stress
// row as -D^T u, so on a boundary facet a prescribed displacement takes its
// place. Its contribution to the moment of the traction against facet basis
// function b in component k is
//
//     int_e u_k b   =   a_k int_e b  +  sum_c B_kc int_e b (x - x_E)_c
//
// and both integrals are already held by the stress operators. So an affine
// datum is exact here: no quadrature at the boundary, and a linear
// displacement is reproduced rather than approximated.
class PrescribedDisplacement {
 public:
  PrescribedDisplacement() = default;
  PrescribedDisplacement(const Params&, const TermContext& ctx)
      : ops_(&ctx.require<exokal::hodge::StressOperators>("stress_operators")),
        data_(&ctx.require<BoundaryVectorData>("boundary_displacement")) {}

  static constexpr std::size_t kS = 0;

  std::vector<std::string> fields() const { return {"s"}; }

  template <class T>
  void operator()(const Stencil& st, const std::vector<T>& a, std::vector<T>& r) const {
    (void)a;
    if (st.n_cofacets != 1) return;
    if (!data_->applies(st.support)) return;  // free: the homogeneous case

    const auto& S = st.field(kS);
    const auto& c = ops_->compact(st.cells[0]);  // moments and grams: no dense M
    const std::size_t slot = st.support_slot[0];
    if (slot >= c.moment.size()) return;
    const exokal::numerics::Dense& mom = c.moment[slot];
    const exokal::numerics::Dense& gram = c.facet_gram[slot];

    // The datum is a function and must be expanded, not merely integrated.
    //
    // A traction degree of freedom is a moment, m_b = int_f (sigma n) chi_b, so
    // the row it leads pairs against the expansion coefficients of whatever
    // stands opposite. The prescribed displacement is given as a function, so
    // what belongs here is Gram^{-1} int_f u chi_b and not the raw moment: the
    // two differ by |f|, and using the moment makes the boundary datum grow
    // with the facet size -- a patch test then fails by an amount that looks
    // like a discretization error and is not.
    //
    // It is the mirror image of the trace, where no inverse belongs because the
    // residual already emerges in coefficient form. Same Gram, opposite
    // direction, and getting either backwards scales by |f|.
    const std::size_t nb = mom.rows();
    const std::size_t nc = mom.cols() - 1;
    for (std::size_t k = 0; k < nc; ++k) {
      for (std::size_t b = 0; b < nb; ++b) {
        // int_f u_k chi_b, from the moments the operators already hold
        double moment = data_->constant_at(st.support, k) * mom(b, 0);
        for (std::size_t cc = 0; cc < nc; ++cc) {
          moment += data_->gradient_at(st.support, k, cc) * c.scale * mom(b, cc + 1);
        }
        // expand it: the chart is L^2-orthonormal, so the Gram is diagonal and
        // this is one division -- but it is read rather than assumed, so the
        // term stays correct if the chart is ever changed
        const double coeff = moment / gram(b, b);
        // and it replaces -D^T u in the stress row, with the facet's own
        // incidence for the side it is seen from
        const std::size_t i = S.begin + slot * nb * nc + b * nc + k;
        if (i < S.end) r[i] -= st.incidence[0] * coeff;
      }
    }
  }

 private:
  const exokal::hodge::StressOperators* ops_{nullptr};
  const BoundaryVectorData* data_{nullptr};
};

inline const exokal::forms::RegisterTerm<PrescribedDisplacement> register_prescribed_displacement{
    "prescribed_displacement", Coupling::boundary, {"s"}};

// The same datum for the strong family, whose facet carries the six-component
// traction moment vector whole. The model precomputes the expansion
// coefficients of u_D against that basis -- Gram |f| I, so coefficient_b =
// (1/|f|) int_f u_D . basis_b, exact for the affine datum -- and this term
// places them where -Dv^T u stood, with the facet's incidence for the side it
// is seen from.
class StrongPrescribedDisplacement {
 public:
  StrongPrescribedDisplacement() = default;
  StrongPrescribedDisplacement(const Params&, const TermContext& ctx)
      : data_(&ctx.require<StrongDisplacementCoefficients>("strong_boundary_displacement")) {}

  static constexpr std::size_t kS = 0;

  std::vector<std::string> fields() const { return {"s"}; }

  template <class T>
  void operator()(const Stencil& st, const std::vector<T>& a, std::vector<T>& r) const {
    (void)a;
    if (st.n_cofacets != 1) return;
    if (!data_->applies(st.support)) return;  // free: the homogeneous case

    const auto& S = st.field(kS);
    const std::size_t slot = st.support_slot[0];
    const std::size_t q = data_->slots();  // six in space, three in the plane
    for (std::size_t b = 0; b < q; ++b) {
      const std::size_t i = S.begin + slot * q + b;
      if (i < S.end) r[i] -= st.incidence[0] * data_->at(st.support, b);
    }
  }

 private:
  const StrongDisplacementCoefficients* data_{nullptr};
};

inline const exokal::forms::RegisterTerm<StrongPrescribedDisplacement>
    register_strong_prescribed_displacement{"strong_prescribed_displacement", Coupling::boundary,
                                            {"s"}};

// Reservoir pressurization as a load on the mechanics alone.
//
// A depletion or injection benchmark does not solve the flow: the pore pressure
// change is given, cell by cell, and only the mechanical response to it is
// computed. That is how Novikov et al. (2024) pose every one of their induced
// fault-slip cases; it is a different problem from poromechanics, not an
// approximation of it, and p is data.
//
// The term is therefore the Biot coupling with the pressure moved to the
// right-hand side. In the coupled system the stress row carries
//
//     eps = C^{-1} sigma + (alpha / dK) p I ,
//
// contributed as alpha * T^T p with T the discrete trace; with p known, the
// same product is a load and the pressure row does not exist at all. So the
// coefficient, the operator and the sign are the ones BiotCouplingCell already
// uses, so the benchmark cannot disagree with the coupled solver by a factor.
//
// The datum is per cell: a region of cells at a changed pressure, zero
// outside it.
class ReservoirPressurization {
 public:
  ReservoirPressurization() = default;
  ReservoirPressurization(const Params& p, const TermContext& ctx)
      : ops_(&ctx.require<exokal::hodge::StressOperators>("stress_operators")),
        data_(&ctx.require<CellData>("reservoir_pressure")),
        alpha_(p.get("biot", 1.0) * p.get("volumetric_compliance", 1.0)) {}

  static constexpr std::size_t kS = 0;

  std::vector<std::string> fields() const { return {"s"}; }

  template <class T>
  void operator()(const Stencil& st, const std::vector<T>& a, std::vector<T>& r) const {
    (void)a;
    const double p = data_->at(st.support);
    if (p == 0.0) return;  // outside the reservoir: nothing to add
    const auto& S = st.field(kS);
    const auto& c = ops_->compact(st.support);  // the trace row alone: no dense M
    const std::size_t D = S.end - S.begin;
    for (std::size_t i = 0; i < D; ++i) {
      r[S.begin + i] += alpha_ * c.T(0, i) * p;
    }
  }

 private:
  const exokal::hodge::StressOperators* ops_{nullptr};
  const CellData* data_{nullptr};
  double alpha_{1.0};
};

inline const exokal::forms::RegisterTerm<ReservoirPressurization> register_reservoir_pressurization{
    "reservoir_pressurization", exokal::forms::Coupling::closure, {"s"}};

}  // namespace mimetika::physics::terms
