#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "exokal/ad/axpy.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/physics/package.hpp"

// The facet-jump stabilization of diagonal_afw's rotation multiplier.
//
// The rotation row becomes
//
//     r_gamma = As sigma - J gamma
//
// with J the two-point Laplacian over interior facets,
//
//     J_ii = sum_f gamma_f ,   J_ij = -gamma_f ,
//     gamma_f = g_{E1,f} + g_{E2,f} ,   g_{E,f} = c mu |f| delta_{E,f} ,
//
// where the half weights g_{E,f} are what StressOperators::build stores per
// cell when given rotation_jump = c, and delta_{E,f} is the facet's half dual
// volume. J annihilates constant rotations, so a linear displacement is still
// reproduced exactly; on a smooth solution the jump is O(h), the scheme's own
// order.
//
// gamma_f pairs the two cells sharing a facet, so this is a cofacet coupling
// while the star it stabilizes is a closure term. The coupling is also why the
// stabilization does not eliminate cell by cell: exokal's hybrid path refuses
// a nonzero rotation jump.
//
// The wrench star is stable at c = 0. The term adds definiteness to the
// multiplier block at the price of a constant the physics does not supply.

namespace mimetika::physics {
namespace terms {

using exokal::forms::Params;
using exokal::forms::Stencil;
using exokal::forms::TermContext;

class RotationJumpFacet {
 public:
  explicit RotationJumpFacet(const exokal::hodge::StressOperators& ops) : ops_(&ops) {}

  RotationJumpFacet(const Params&, const TermContext& ctx)
      : ops_(&ctx.require<exokal::hodge::StressOperators>("stress_operators")) {}

  static constexpr std::size_t kG = 0;  // rotation

  std::vector<std::string> fields() const { return {"g"}; }

  template <class T>
  void operator()(const Stencil& st, const std::vector<T>& a, std::vector<T>& r) const {
    if (st.n_cofacets != 2) return;  // a boundary facet has no jump
    const exokal::hodge::StressOperators::Cell& ca = ops_->compact(st.cells[0]);
    const exokal::hodge::StressOperators::Cell& cb = ops_->compact(st.cells[1]);
    // empty whenever the stabilization is off, or the realization carries none
    if (ca.rotation_jump.empty() || cb.rotation_jump.empty()) return;
    const double gamma =
        ca.rotation_jump[st.support_slot[0]] + cb.rotation_jump[st.support_slot[1]];

    const auto& ga = st.field(kG, 0);
    const auto& gb = st.field(kG, 1);
    const std::size_t ng = ga.end - ga.begin;
    for (std::size_t k = 0; k < ng; ++k) {
      // -(J gamma), componentwise and antisymmetric between the pair: gamma_f
      // on each diagonal entry and -gamma_f on each off-diagonal one. Written
      // as four axpy so it holds for the AD value types as well as for double.
      exokal::axpy(r[ga.begin + k], -gamma, a[ga.begin + k]);
      exokal::axpy(r[ga.begin + k], gamma, a[gb.begin + k]);
      exokal::axpy(r[gb.begin + k], -gamma, a[gb.begin + k]);
      exokal::axpy(r[gb.begin + k], gamma, a[ga.begin + k]);
    }
  }

 private:
  const exokal::hodge::StressOperators* ops_{nullptr};
};

inline const exokal::forms::RegisterTerm<RotationJumpFacet> register_rotation_jump_facet{
    "rotation_jump_facet", exokal::forms::Coupling::cofacet, {"g"}};

}  // namespace terms
}  // namespace mimetika::physics
