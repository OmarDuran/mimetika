#pragma once

#include <string>
#include <vector>

#include "exokal/ad/axpy.hpp"
#include "exokal/forms/term.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/model/boundary.hpp"

// THE NATURAL BOUNDARY TERMS: the datum for the quantity a mixed form does
// NOT carry as an unknown.
//
// In the flux-pressure form the interior pressure enters the flux row as
// -div^T p, so on a boundary facet — where there is no second cell — a
// prescribed pressure takes its place. That is the whole of the term: one
// signed contribution, on the flux degree of freedom of the facet being
// integrated over.
//
// A FACET THE DATA SAYS NOTHING ABOUT CONTRIBUTES NOTHING, which is exactly
// the homogeneous natural condition. So a drained face at zero pressure needs
// no term, and writing one changes nothing — which means a model cannot be
// wrong by failing to mention its free boundaries. The conditions that must
// be stated are the essential ones, and those are refused loudly when a
// degree of freedom is left unconstrained twice.

namespace mimetika::physics::terms {

using exokal::forms::Coupling;
using exokal::forms::Params;
using exokal::forms::Stencil;
using exokal::forms::TermContext;

class PrescribedPressure {
 public:
  PrescribedPressure() = default;
  PrescribedPressure(const Params&, const TermContext& ctx)
      : data_(&ctx.require<BoundaryData>("boundary_pressure")) {}

  static constexpr std::size_t kQ = 0;

  std::vector<std::string> fields() const { return {"q"}; }

  template <class T>
  void operator()(const Stencil& st, const std::vector<T>& a, std::vector<T>& r) const {
    (void)a;
    if (st.n_cofacets != 1) return;           // not on the boundary after all
    if (!data_->applies(st.support)) return;  // free: the homogeneous case

    const auto& q = st.field(kQ);
    const std::size_t slot = st.support_slot[0];
    const std::size_t i = q.begin + slot;
    if (i >= q.end) return;
    // the same shape the interior -div^T p has, with the datum in place of
    // the missing neighbour, and the facet's own incidence for the sign
    r[i] += st.view.signs[i] * data_->at(st.support);
  }

 private:
  const BoundaryData* data_{nullptr};
};

inline const exokal::forms::RegisterTerm<PrescribedPressure> register_prescribed_pressure{
    "prescribed_pressure", Coupling::boundary, {"q"}};

// THE MIRROR OF THE PRESCRIBED PRESSURE, for mechanics.
//
// In the Hellinger-Reissner form the interior displacement enters the stress
// row as -D^T u, so on a boundary facet a prescribed displacement takes its
// place. Its contribution to the moment of the traction against facet basis
// function b in component k is
//
//     int_e u_k b   =   a_k int_e b  +  sum_c B_kc int_e b (x - x_E)_c
//
// and both integrals are already held by the stress operators. So an AFFINE
// datum is EXACT here: no quadrature at the boundary, and a linear
// displacement is reproduced rather than approximated — which is what makes a
// patch test a test of the method rather than of the boundary integration.
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
    const auto& c = ops_->cell(st.cells[0]);
    const std::size_t slot = st.support_slot[0];
    if (slot >= c.moment.size()) return;
    const exokal::numerics::Dense& mom = c.moment[slot];

    for (std::size_t k = 0; k < 3; ++k) {
      for (std::size_t b = 0; b < 3; ++b) {
        double v = data_->constant_at(st.support, k) * mom(b, 0);
        for (std::size_t cc = 0; cc < 3; ++cc) {
          v += data_->gradient_at(st.support, k, cc) * c.scale * mom(b, cc + 1);
        }
        // the ProductSpace orders component fastest within a facet
        const std::size_t i = S.begin + slot * 9 + b * 3 + k;
        if (i < S.end) r[i] += v;
      }
    }
  }

 private:
  const exokal::hodge::StressOperators* ops_{nullptr};
  const BoundaryVectorData* data_{nullptr};
};

inline const exokal::forms::RegisterTerm<PrescribedDisplacement>
    register_prescribed_displacement{"prescribed_displacement", Coupling::boundary, {"s"}};

}  // namespace mimetika::physics::terms
