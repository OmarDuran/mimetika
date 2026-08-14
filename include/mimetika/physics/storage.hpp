#pragma once

#include <string>
#include <vector>

#include "exokal/ad/axpy.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/physics/package.hpp"

// FLUID STORAGE: the pressure's own contribution to the mass balance, and the
// last piece the consolidation problem needs.
//
// The mass balance of a poroelastic medium is
//
//     d/dt ( alpha tr(eps) + S p ) + div q = 0
//
// The first term is the Biot coupling, already carried by PoroCoupling; the
// second is here, and its coefficient is NOT the fluid compressibility alone:
//
//     S = d alpha^2 (1 - 2nu) / (2 mu (1 - 2nu + d nu))  +  1/M
//
// The first part is the SKELETON's storage, and it survives an incompressible
// fluid. Dropping it — taking S = 1/M and setting 1/M to zero for Terzaghi's
// incompressible constituents — leaves the pressure with no time derivative
// at all, so the column solves at every step and consolidates at none: a
// static answer dressed as a transient one, which is the failure mode this
// benchmark exists to catch. Together with the divergence they make the balance
// TRANSIENT, which is what makes consolidation a process rather than a state.
//
// THE TIME DISCRETIZATION NEEDS NO TERM OF ITS OWN. Backward Euler over a
// step is
//
//     alpha T sigma + (|E|/M) p + dt div q = (the same at the old state)
//
// and dt appears only against the divergence. Rescaling the flux to the
// quantity that actually crosses a facet during the step, q~ = dt q, absorbs
// it: the constitutive row becomes (M_q/dt) q~ = div^T p, which is exactly
// the Darcy term with its mobility set to dt. So the step system is assembled
// from the steady terms plus this one, with no time-stepping machinery at
// all — and the flux that comes out is a volume rather than a rate, which is
// what a balance over a step is written in anyway.

namespace mimetika::physics {

namespace terms {

class StorageCell {
 public:
  StorageCell() = default;
  StorageCell(const Params& p, const TermContext& ctx)
      : ops_(&ctx.require<exokal::hodge::StressOperators>("stress_operators")),
        storage_(p.get("storage", 0.0)) {}

  static constexpr std::size_t kP = 0;

  std::vector<std::string> fields() const { return {"p"}; }

  template <class T>
  void operator()(const Stencil& st, const std::vector<T>& a, std::vector<T>& r) const {
    const auto& P = st.field(kP);
    const double c = storage_ * ops_->cell(st.support).volume;
    exokal::axpy(r[P.begin], c, a[P.begin]);
  }

 private:
  const exokal::hodge::StressOperators* ops_{nullptr};
  double storage_{0.0};
};

inline const exokal::forms::RegisterTerm<StorageCell> register_storage{
    "storage_cell", exokal::forms::Coupling::closure, {"p"}};

}  // namespace terms

struct StorageOptions {
  double storage{0.0};  // S, the whole coefficient
};

class Storage final : public Package {
 public:
  Storage() = default;
  explicit Storage(StorageOptions o) : opt_(o) {}

  std::string name() const override { return "Storage"; }

  Requirements requirements(int, int) const override {
    Requirements r;
    r.needs = {"pressure"};
    r.slots = {{"biot_modulus", Scope::fluid}};
    return r;
  }

  void attach(exokal::forms::Model& model, const exokal::forms::TermContext&) const override {
    model.add("storage_cell", exokal::forms::On::all(), {{"storage", opt_.storage}});
  }

 private:
  StorageOptions opt_;
};

}  // namespace mimetika::physics
