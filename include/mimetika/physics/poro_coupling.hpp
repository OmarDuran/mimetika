#pragma once

#include <string>
#include <vector>

#include "exokal/ad/axpy.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/physics/package.hpp"

// The Biot coupling.
//
// Poromechanics is flow plus mechanics plus the exchange between them, and that
// exchange contributes no field of its own: it reads a pressure that flow
// declared and a stress that mechanics declared, and adds two terms.
//
//     r_sigma += -c T^T p        the pore pressure in the momentum balance
//     r_p     += +c T sigma      the volumetric response in the mass balance
//
// T is the discrete trace, tr_h(tau)_E = (1/|E|) sum_e int_e (tau n_e).(x - x_E),
// which is the volumetric coupling every stress space exposes.
//
// The coefficient is not the Biot number alone. What multiplies the trace is
//
//     c = alpha * (1 - 2nu) / (2 mu (1 - 2nu + d nu))
//
// the Biot coefficient times the skeleton's volumetric compliance — 1/(dK) in
// three dimensions. With alpha by itself the coupling would be independent of
// the skeleton stiffness, so a rigid medium would respond to pressure as a soft
// one does.
//
// Written this way rather than as alpha/(dK) because it stays finite at
// nu = 1/2, where it is zero: the incompressible limit arrives continuously
// instead of through a division by an infinite bulk modulus.
//
// Both blocks are written from one array, so the (sigma, p) and (p, sigma)
// blocks are exact transposes rather than two kernels that can drift.

namespace mimetika::physics {

namespace terms {

class BiotCouplingCell {
 public:
  BiotCouplingCell() = default;
  BiotCouplingCell(const Params& p, const TermContext& ctx)
      : ops_(&ctx.require<exokal::hodge::StressOperators>("stress_operators")),
        alpha_(p.get("biot", 1.0) * p.get("volumetric_compliance", 1.0)) {}

  static constexpr std::size_t kS = 0;  // stress
  static constexpr std::size_t kP = 1;  // pressure

  std::vector<std::string> fields() const { return {"s", "p"}; }

  template <class T>
  void operator()(const Stencil& st, const std::vector<T>& a, std::vector<T>& r) const {
    const auto& S = st.field(kS);
    const auto& P = st.field(kP);
    const auto& c = ops_->compact(st.support);  // the trace row alone: no dense M
    const std::size_t D = S.end - S.begin;

    for (std::size_t i = 0; i < D; ++i) {
      const std::size_t ri = S.begin + i;  // already in ProductSpace order
      // The same sign on both rows. The Biot coupling is a constitutive
      // symmetry, not a differential adjoint: both terms are second derivatives
      // of one free energy, so the block is symmetric and
      //
      //     eps = C^{-1} sigma + (alpha/dK) p I     the stress row
      //     zeta = (alpha/dK) tr sigma + S p        the mass balance
      //
      // carry the same coefficient with the same sign. The antisymmetric
      // convention [M, -B^T; +B, 0] belongs to the div/grad pair of the Darcy
      // system, where the two blocks are adjoint differential operators.
      // Undrained confined compression separates them: it must give
      // p = sigma_0/alpha, and with the sign flipped it gives 5 sigma_0/13 at
      // mu = lam = alpha = 1, on the right clock and with the right profile
      // shape.
      const double t = alpha_ * c.T(0, i);
      exokal::axpy(r[ri], t, a[P.begin]);
      exokal::axpy(r[P.begin], t, a[ri]);
    }
  }

 private:
  const exokal::hodge::StressOperators* ops_{nullptr};
  double alpha_{1.0};
};

inline const exokal::forms::RegisterTerm<BiotCouplingCell> register_biot{
    "biot_coupling_cell", exokal::forms::Coupling::closure, {"s", "p"}};

}  // namespace terms

struct PoroCouplingOptions {
  double biot{1.0};
  double volumetric_compliance{1.0};  // (1-2nu)/(2mu(1-2nu+d nu))
};

class PoroCoupling final : public Package {
 public:
  PoroCoupling() = default;
  explicit PoroCoupling(PoroCouplingOptions o) : opt_(o) {}

  std::string name() const override { return "PoroCoupling"; }

  // No fields: the coupling reads what the two physics it joins already
  // declared, so single-phase and compositional multiphase poromechanics are
  // two catalogue rows over one implementation.
  Requirements requirements(int, int) const override {
    Requirements r;
    r.needs = {"pressure", "displacement"};
    r.slots = {{"biot", Scope::rock}, {"normal_permeability", Scope::interface}};
    return r;
  }

  void attach(exokal::forms::Model& model, const exokal::forms::TermContext&) const override {
    model.add("biot_coupling_cell", exokal::forms::On::all(),
              {{"biot", opt_.biot}, {"volumetric_compliance", opt_.volumetric_compliance}});
  }

 private:
  PoroCouplingOptions opt_;
};

}  // namespace mimetika::physics
