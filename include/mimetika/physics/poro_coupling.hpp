#pragma once

#include <string>
#include <vector>

#include "exokal/ad/axpy.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/physics/package.hpp"

// THE BIOT COUPLING — and this package is the argument of the whole
// catalogue, made concrete.
//
// Poromechanics is not a third physics beside flow and mechanics. It is the
// two of them plus the exchange between them, and that exchange contributes
// NO FIELD OF ITS OWN: it reads a pressure that flow declared and a stress
// that mechanics declared, and adds two terms.
//
//     r_sigma += -c T^T p        the pore pressure in the momentum balance
//     r_p     += +c T sigma      the volumetric response in the mass balance
//
// T is the discrete trace, tr_h(tau)_E = (1/|E|) sum_e int_e (tau n_e).(x - x_E),
// which is the volumetric coupling every stress space exposes.
//
// THE COEFFICIENT IS NOT THE BIOT NUMBER ALONE. What multiplies the trace is
//
//     c = alpha * (1 - 2nu) / (2 mu (1 - 2nu + d nu))
//
// the Biot coefficient times the skeleton's VOLUMETRIC COMPLIANCE — 1/(dK) in
// three dimensions. Using alpha by itself is not a scaling error to be
// absorbed elsewhere: it makes the coupling independent of how stiff the
// skeleton is, so a rigid medium would respond to pressure exactly as a soft
// one does.
//
// Written this way rather than as alpha/(dK) because it stays FINITE at
// nu = 1/2, where it is zero: the incompressible limit arrives continuously
// instead of through a division by an infinite bulk modulus.
//
// BOTH BLOCKS COME FROM ONE ARRAY. Writing them separately would let them
// drift, and a poroelastic system whose coupling is not adjoint loses the
// energy structure that makes it solvable — quietly, since it still runs.

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
    const auto& c = ops_->cell(st.support);
    const std::size_t D = S.end - S.begin;

    for (std::size_t i = 0; i < D; ++i) {
      const std::size_t ri = S.begin + i;  // already in ProductSpace order
      // THE SAME SIGN ON BOTH ROWS. The Biot coupling is a CONSTITUTIVE
      // symmetry, not a differential adjoint: both terms are second
      // derivatives of one free energy, so the block is symmetric and
      //
      //     eps = C^{-1} sigma + (alpha/dK) p I     the stress row
      //     zeta = (alpha/dK) tr sigma + S p        the mass balance
      //
      // carry the SAME coefficient with the SAME sign. The antisymmetric
      // convention [M, -B^T; +B, 0] belongs to the div/grad pair of the Darcy
      // system, where the two blocks really are adjoint differential
      // operators; applying it here instead is a sign error with no visible
      // symptom. Undrained confined compression is where it shows: it must
      // give p = sigma_0/alpha, and with the sign flipped it gives
      // 5 sigma_0/13 at mu = lam = alpha = 1 -- a plausible number, on the
      // right clock, with the right profile shape.
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

  // NO FIELDS. That is the point: the coupling reads what the two physics it
  // joins already declared, so `{single-phase poromechanics, compositional
  // multiphase poromechanics}` is two catalogue rows and zero
  // implementations.
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
