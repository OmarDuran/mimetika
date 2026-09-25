#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "exokal/ad/axpy.hpp"
#include "mimetika/physics/contact_interface.hpp"
#include "mimetika/physics/package.hpp"

// The two interface rows of the linear contact system, both face-local and both
// evaluated on the cells of the fracture stratum.
//
//     (D_c t_c, s_c)_Gamma + j(s_c, u_c) = 0          the compliance law
//     t(sigma, v_c)        + j(t_c, v_c) = 0          traction equilibrium
//
// with j(s_c, v_c) = -<s_c, [[v_c]]> and [[v_c]] = v_c^+ - v_c^-. The first
// term below is A_c, the second is J, and t() is the carrier term in
// terms/contact_trace.hpp.
//
// J APPEARS IN BOTH ROWS, which is why one term writes both: j(s_c, u_c) tests
// the compliance row against the jump of the unknown trace, and j(t_c, v_c)
// tests the equilibrium row against the unknown traction. They are the same
// bilinear form, so the second is the transpose of the first -- J and J^T of
// the block system -- and writing them together is what keeps them so.
//
// THE TWO HALVES OF u_c. The first k components are the side whose facet copy
// has incidence +1 and the second k the other, which is the convention
// terms/contact_trace.hpp fixes. The jump is their difference, so J = -G[I, -I]
// for a chart Gram G, and the sign of the pair is what makes a positive
// compliance oppose an opening jump.

namespace mimetika::physics {
namespace terms {

using exokal::forms::Params;
using exokal::forms::Stencil;
using exokal::forms::TermContext;

// (D_c t_c, s_c)_Gamma: the face's compliance, and the only place c_n and c_T
// enter. A nonlinear contact law replaces this term and nothing else.
class ContactComplianceCell {
 public:
  explicit ContactComplianceCell(const ContactInterface& data) : data_(&data) {}
  ContactComplianceCell(const Params&, const TermContext& ctx)
      : data_(&ctx.require<ContactInterface>("contact_interface")) {}

  static constexpr std::size_t kT = 0;

  std::vector<std::string> fields() const { return {"t"}; }

  template <class T>
  void operator()(const Stencil& st, const std::vector<T>& a, std::vector<T>& r) const {
    const auto& tc = st.field(kT, 0);
    const std::size_t k = data_->k;
    if (tc.end - tc.begin != k) return;
    const double* A = data_->compliance_at(static_cast<std::size_t>(st.support));
    for (std::size_t i = 0; i < k; ++i) {
      for (std::size_t j = 0; j < k; ++j) {
        exokal::axpy(r[tc.begin + i], A[i * k + j], a[tc.begin + j]);
      }
    }
  }

 private:
  const ContactInterface* data_{nullptr};
};

// j(s_c, u_c) and j(t_c, v_c): the jump, in both rows at once.
class ContactJumpCell {
 public:
  explicit ContactJumpCell(const ContactInterface& data) : data_(&data) {}
  ContactJumpCell(const Params&, const TermContext& ctx)
      : data_(&ctx.require<ContactInterface>("contact_interface")) {}

  static constexpr std::size_t kT = 0;
  static constexpr std::size_t kUc = 1;

  std::vector<std::string> fields() const { return {"t", "uc"}; }

  template <class T>
  void operator()(const Stencil& st, const std::vector<T>& a, std::vector<T>& r) const {
    const auto& tc = st.field(kT, 0);
    const auto& uc = st.field(kUc, 0);
    const std::size_t k = data_->k;
    if (tc.end - tc.begin != k || uc.end - uc.begin != 2 * k) return;
    const double* G = data_->gram_at(static_cast<std::size_t>(st.support));
    const std::size_t plus = uc.begin, minus = uc.begin + k;
    for (std::size_t i = 0; i < k; ++i) {
      for (std::size_t j = 0; j < k; ++j) {
        const double g = G[i * k + j];
        // -<s_c, [[u_c]]> in the compliance row
        exokal::axpy(r[tc.begin + i], -g, a[plus + j]);
        exokal::axpy(r[tc.begin + i], g, a[minus + j]);
        // and its transpose in the equilibrium row, the jump being a difference
        exokal::axpy(r[plus + j], -g, a[tc.begin + i]);
        exokal::axpy(r[minus + j], g, a[tc.begin + i]);
      }
    }
  }

 private:
  const ContactInterface* data_{nullptr};
};

inline const exokal::forms::RegisterTerm<ContactComplianceCell> register_contact_compliance_cell{
    "contact_compliance_cell", exokal::forms::Coupling::closure, {"t"}};

inline const exokal::forms::RegisterTerm<ContactJumpCell> register_contact_jump_cell{
    "contact_jump_cell", exokal::forms::Coupling::closure, {"t", "uc"}};

}  // namespace terms
}  // namespace mimetika::physics
