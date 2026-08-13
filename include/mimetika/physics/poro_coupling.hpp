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
//     r_sigma += -alpha T^T p        the pore pressure in the momentum balance
//     r_p     += +alpha T sigma      the volumetric response in the mass balance
//
// T is the discrete trace, tr_h(tau)_E = (1/|E|) sum_e int_e (tau n_e).(x - x_E),
// which is the volumetric coupling every stress space exposes.
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
        alpha_(p.get("biot", 1.0)) {}

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
      const std::size_t fi = i / 9, ki = (i % 9) / 3, bi = i % 3;
      const std::size_t ri = S.begin + fi * 9 + bi * 3 + ki;  // to ProductSpace order
      const double t = alpha_ * c.T(0, i);
      exokal::axpy(r[ri], -t, a[P.begin]);
      exokal::axpy(r[P.begin], t, a[ri]);  // the adjoint, from the same coefficient
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
    model.add("biot_coupling_cell", exokal::forms::On::all(), {{"biot", opt_.biot}});
  }

 private:
  PoroCouplingOptions opt_;
};

}  // namespace mimetika::physics
