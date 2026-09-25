#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "exokal/ad/axpy.hpp"
#include "mimetika/physics/package.hpp"

// t(tau, v_c): the interface trace coupling of the mixed contact formulation.
//
//     t(tau, v_c) = -<tau^+ n^+, v_c^+>_Gamma - <tau^- n^-, v_c^->_Gamma
//
// the term that puts T in the momentum-balance row of the contact system and
// T^T in the constitutive one. It is the only coupling between the bulk stress
// and the fracture, and it is a CARRIER term: evaluated on the facets of the
// ambient stratum, reaching the fracture cell each of them carries.
//
// IT NEEDS NO QUADRATURE, because the two sides of the pairing are already
// written in the same basis. A traction unknown of a facet is a moment,
// m_b = int_f (tau nu) chi_b, against the facet's own chart; the trace u_c is
// expanded in that same chart, u_c = sum_b c_b chi_b. So
//
//     <tau nu, v_c>_f = sum_b c_b m_b
//
// exactly, and the local block is the identity -- the interface coupling is as
// topological as the divergence is, and for the same reason. Choosing a trace
// space that is NOT the facet's traction space would put a Gram here; the
// formulation does not, taking Lambda_c,h(f) = T_h(f) and U_c,h(f) = T_h(f)^2.
//
// THE SIDE IS THE INCIDENCE. n^s is the outward normal of the cell on side s,
// which is the facet's canonical normal times its incidence there, so
// <tau^s n^s, v_c^s> = incidence_s * sum_b c^s_b m^s_b. After the cut each
// facet copy has exactly one cofacet, so its incidence is +1 on one side and
// -1 on the other and no geometric test is needed. u_c's first half is the
// side whose incidence is +1 and its second half the other, which fixes which
// is u_c^+ once the fracture normal is taken to be the facet's canonical one.
//
// THE MIRROR OF A PRESCRIBED DISPLACEMENT. With v_c a datum rather than an
// unknown this is exactly boundary_terms.hpp's PrescribedDisplacement, which
// writes -incidence times the trace's expansion coefficient into the stress
// row; the sign convention is taken from there rather than chosen here. What is
// added is the second direction: u_c is an unknown, so the same coefficient
// appears again in its own row, which is what makes the block pair T with T^T.
//
// THE FREE-STANDING COPY CONTRIBUTES NOTHING. The cut leaves the original facet
// in the ambient complex, detached, carrying the fracture cell but belonging to
// no bulk cell. Stencil::couples() is false there -- there is no traction to
// pair -- and the term returns.

namespace mimetika::physics {
namespace terms {

using exokal::forms::Params;
using exokal::forms::Stencil;
using exokal::forms::TermContext;

class ContactTraceFacet {
 public:
  ContactTraceFacet() = default;
  explicit ContactTraceFacet(const Params&) {}

  static constexpr std::size_t kS = 0;   // the bulk stress, on this stratum
  static constexpr std::size_t kUc = 0;  // the two-sided trace, on the fracture

  std::vector<std::string> fields() const { return {"s"}; }
  std::vector<std::string> carried_fields() const { return {"uc"}; }

  template <class T>
  void operator()(const Stencil& st, const std::vector<T>& a, std::vector<T>& r) const {
    if (!st.couples()) return;

    const auto& S = st.field(kS, 0);
    const auto& U = st.carried_field(kUc);
    const std::size_t both = U.end - U.begin;
    if (both == 0 || both % 2 != 0) return;  // not a two-sided trace
    const std::size_t k = both / 2;          // dim T_h(f): 6 for the wrench, d^2 for BDM

    // the facet's own tractions inside the cell's block: the site's slot times
    // the per-facet count, which the trace space matches by construction
    const std::size_t slot = st.support_slot[0];
    const std::size_t base = S.begin + slot * k;
    if (base + k > S.end) return;  // the trace does not match the facet's tractions

    const double side = st.incidence[0];
    const std::size_t opposite = U.begin + (side > 0.0 ? 0 : k);
    for (std::size_t j = 0; j < k; ++j) {
      // T^T u_c in the stress row, and T sigma in the trace's own: one
      // coefficient, written twice, which is the block and its transpose
      exokal::axpy(r[base + j], -side, a[opposite + j]);
      exokal::axpy(r[opposite + j], -side, a[base + j]);
    }
  }
};

inline const exokal::forms::RegisterTerm<ContactTraceFacet> register_contact_trace_facet{
    "contact_trace_facet", exokal::forms::Coupling::carrier, {"s"}, {"uc"}};

}  // namespace terms
}  // namespace mimetika::physics
