#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/ad/axpy.hpp"
#include "mimetika/physics/constitutive/immiscible.hpp"
#include "exokal/forms/stencil.hpp"
#include "exokal/forms/term.hpp"
#include "exokal/hodge/flux_operators.hpp"

// THE MIXED DARCY CELL TERM: the first form written against a real
// constitutive operator rather than a test kernel.
//
// On one cell, with the flux q on the facets and the pressure p on the
// cell, the two rows are
//
//     r_q = M(lambda) q - div^T p        the constitutive relation
//     r_p = div q                        the mass balance
//
// where M is the discrete Hodge star of the flux space and div is the
// signed sum of facet fluxes — the boundary coefficients graphos already
// carries, which the stencil hands over as view.signs. Nothing here
// computes a divergence: the topology supplies it, and the term only
// pairs it.
//
// THE HODGE IS OPAQUE TO THE TERM. It arrives as a matrix per cell from
// whichever realization the model chose, so the same kernel runs on the
// stabilized product over polytopes and on the consistency-only RT_0
// product over simplices. That interchangeability is not a convenience —
// it is the reason ⋆ and the space were separated in the first place.
//
// THE STATE ENTERS AS ONE DIVISION. Every realization satisfies
// M(lambda) = M(1)/lambda, so the geometry stays frozen in double while
// the mobility carries the derivatives. With a constant mobility the term
// is affine and its Jacobian is assembled once; with a state-dependent
// one it is not, and the AD produces the tangent from the same source.
//
// THE COUPLING BLOCKS ARE ADJOINT BY CONSTRUCTION. div^T appears in the
// flux row and div in the pressure row, written from the same signs, so
// the assembled (q,p) and (p,q) blocks are exact transposes — never two
// hand-written kernels that can drift.

namespace mimetika::physics::terms {

// THE TERM INTERFACE IS EXOKAL'S; THE PHYSICS IS MIMETIKA'S. exokal supplies
// the coupling, the stencil and the registry — machinery with no constitutive
// content — while the mobility models and the Darcy balance below are physics
// and so belong here, per exokal's docs/scope.md. These declarations bring the
// interface into scope so the kernel reads as an ordinary term.
using exokal::forms::Coupling;
using exokal::forms::empty_context;
using exokal::forms::KernelTerm;
using exokal::forms::Params;
using exokal::forms::RegisterTerm;
using exokal::forms::Slots;
using exokal::forms::Stencil;
using exokal::forms::Term;
using exokal::forms::TermContext;
using exokal::forms::TermInfo;
namespace hodge = exokal::hodge;
namespace numerics = exokal::numerics;

// A MOBILITY MODEL DECLARES ITS INPUTS. That declaration is the whole
// mechanism by which exokal learns a function dependency, and it exists
// because two different things are going on:
//
//   NUMERICAL dependency — the values and derivatives — is DISCOVERED. The
//   AD fills whatever blocks the kernel actually reads, and nothing has to
//   be told in advance.
//
//   STRUCTURAL dependency — WHICH fields — must be known BEFORE anything
//   is assembled: the product space has to contain them, and the global
//   sparsity pattern has to include those blocks. A pattern derived from
//   what the AD happened to fill would change shape whenever a branch or a
//   vanishing coefficient made a block go quiet. So it is declared.
//
// inputs() is that declaration. A term's field list is its own structural
// fields followed by the model's, and the two can then be CHECKED against
// each other: the blocks the AD fills must be a subset of the blocks the
// declaration allows. declared_fields() below composes the list, and the
// tests assert the containment.
//
// The field order is positional: the stencil's blocks follow the product
// space's field order, so a term declaring {q, p, h} reads block 0 as the
// flux, 1 as the pressure and 2 as the enthalpy. That is a real constraint
// on how a space is built, and it is checked at assembly rather than
// assumed.
struct ConstantMobility {
  double value{1.0};

  ConstantMobility() = default;
  explicit ConstantMobility(const Params& p, const TermContext& = empty_context())
      : value(p.get("mobility", 1.0)) {}

  // no state dependence at all: the term stays affine
  std::vector<std::string> inputs() const { return {}; }

  template <class T>
  T at(const Stencil&, const std::vector<T>&, const std::size_t*) const {
    return T{value};
  }
};

// lambda(p, h) = lambda_0 (1 + a p)(1 + b h): the smallest model with a
// genuine MULTI-field dependency, which is what makes the declaration
// matter — the flux row acquires a block against the enthalpy that a
// constant mobility never touches.
struct PressureEnthalpyMobility {
  double reference{1.0};
  double pressure_coefficient{0.0};
  double enthalpy_coefficient{0.0};

  PressureEnthalpyMobility() = default;
  explicit PressureEnthalpyMobility(const Params& p, const TermContext& = empty_context())
      : reference(p.get("mobility", 1.0)),
        pressure_coefficient(p.get("mobility_pressure_coefficient", 0.0)),
        enthalpy_coefficient(p.get("mobility_enthalpy_coefficient", 0.0)) {}

  std::vector<std::string> inputs() const { return {"p", "h"}; }

  template <class T>
  T at(const Stencil& st, const std::vector<T>& x, const std::size_t* k) const {
    const T p = x[st.field(k[0]).begin];
    const T h = x[st.field(k[1]).begin];
    return reference * (1.0 + pressure_coefficient * p) * (1.0 + enthalpy_coefficient * h);
  }
};

// THE MOBILITY AS AN OBJECT, fetched from the context rather than told in
// numbers. This is the case a scalar parameter cannot express: lambda is a
// weight of the whole state (p, h, z), computed by a fluid model the
// caller owns and the term merely reads.
//
// It is also what forces inputs() to be a PER-INSTANCE member rather than
// a static: how many composition fields a fluid reads depends on its
// component context, which is a property of the object, not of its type.
class FluidMobility {
 public:
  FluidMobility() = default;
  FluidMobility(const Params&, const TermContext& ctx)
      : fluid_(&ctx.require<constitutive::ImmiscibleFluid>("mobility")) {}

  std::vector<std::string> inputs() const {
    if (fluid_ == nullptr) return {};
    std::vector<std::string> f{"p", "h"};
    for (std::size_t a = 0; a < fluid_->n_phases(); ++a) f.push_back("z" + std::to_string(a));
    return f;
  }

  template <class T>
  T at(const Stencil& st, const std::vector<T>& x, const std::size_t* k) const {
    const std::size_t n = fluid_->n_phases();
    std::vector<T> z;
    z.reserve(n);
    for (std::size_t a = 0; a < n; ++a) z.push_back(x[st.field(k[2 + a]).begin]);
    const constitutive::State<T> state{x[st.field(k[0]).begin], x[st.field(k[1]).begin], z};
    return fluid_->evaluate(state).total_mobility;
  }

 private:
  const constitutive::ImmiscibleFluid* fluid_{nullptr};
};

// The field list a term actually touches: its own, then whatever the
// mobility model declares. This is what a model validates the space
// against, and what the sparsity pattern is built from. Taken from an
// INSTANCE, because a fluid's arity is instance data.
template <class MobilityModel>
std::vector<std::string> declared_fields(const MobilityModel& model) {
  std::vector<std::string> f{"q", "p"};
  for (const std::string& e : model.inputs()) {
    if (std::find(f.begin(), f.end(), e) == f.end()) f.push_back(e);
  }
  return f;
}

template <class MobilityModel>
std::vector<std::string> declared_fields() {
  return declared_fields(MobilityModel{});
}

template <class MobilityModel = ConstantMobility>
class MixedDarcyCell {
 public:
  MixedDarcyCell(const hodge::FluxOperators& hodge, MobilityModel mobility)
      : hodge_(&hodge), mobility_(std::move(mobility)) {
    map_mobility_inputs();
  }

  // THE NAMED PATH. Params carries the mobility because it is numbers;
  // the Hodge comes from the context because it is data. The key is fixed
  // — a model provides its flux Hodge under "flux_operators" — so composing
  // this term needs no C++ at all, which is the point of the registry.
  MixedDarcyCell(const Params& p, const TermContext& ctx)
      : hodge_(&ctx.require<hodge::FluxOperators>("flux_operators")), mobility_(p, ctx) {
    map_mobility_inputs();
  }

  const MobilityModel& mobility() const { return mobility_; }
  std::vector<std::string> fields() const { return declared_fields(mobility_); }

  // FIELDS BY NAME. kQ and kP index this term's own DECLARATION — the order
  // it listed {"q", "p", ...} in — and the Stencil maps that through the
  // resolution made against whichever space this stratum has. A composition
  // that puts a displacement ahead of the flux moves every block and
  // changes nothing here.
  static constexpr std::size_t kQ = 0;
  static constexpr std::size_t kP = 1;

  template <class T>
  void operator()(const Stencil& st, const std::vector<T>& a, std::vector<T>& r) const {
    const auto& q = st.field(kQ);
    const auto& p = st.field(kP);
    const std::size_t F = q.end - q.begin;

    const numerics::Dense& M = hodge_->cell(st.support);
    if (M.rows() != F) {
      throw std::invalid_argument("MixedDarcyCell: the Hodge has " + std::to_string(M.rows()) +
                                  " unknowns and the flux block has " + std::to_string(F) +
                                  "; the layout and the realization disagree");
    }

    // the mobility reads whatever fields it declared, so it carries the
    // state and its derivatives while M stays frozen
    const T inv_lambda = T{1.0} / mobility_.at(st, a, mob_k_.data());

    // THE DIVERGENCE PAIRS THE CONSTANT MOMENT ONLY. With d moments per
    // facet the flux carries a linear profile across each face, but the
    // pressure is cell-wise, so what the balance sees is the net flow — the
    // constant moment. The higher moments are invisible to the divergence,
    // exactly as the curl-enriched modes are, and pairing them would make
    // the discrete equilibrium depend on how the flux is distributed within
    // a facet.
    const auto per_facet = static_cast<std::size_t>(hodge_->moments_per_facet());
    for (std::size_t i = 0; i < F; ++i) {
      const std::size_t qi = q.begin + i;
      for (std::size_t j = 0; j < F; ++j) {
        // M(lambda) q, with the whole state dependence in one division
        r[qi] += M(i, j) * inv_lambda * a[q.begin + j];
      }
      if (i % per_facet != 0) continue;  // not the constant moment
      // -div^T p in the flux row, div q in the pressure row: written from
      // the SAME signs, so the two blocks are exact transposes
      exokal::axpy(r[qi], -st.view.signs[qi], a[p.begin]);
      exokal::axpy(r[p.begin], st.view.signs[qi], a[qi]);
    }
  }

 private:
  // Where each of the mobility's OWN inputs lands in this term's field
  // declaration. The two lists differ because the declaration deduplicates:
  // a mobility reading "p" reuses the pressure the term already declared
  // rather than asking for a second copy of it.
  void map_mobility_inputs() {
    const std::vector<std::string> df = declared_fields(mobility_);
    for (const std::string& n : mobility_.inputs()) {
      mob_k_.push_back(static_cast<std::size_t>(std::find(df.begin(), df.end(), n) - df.begin()));
    }
  }

  const hodge::FluxOperators* hodge_;
  MobilityModel mobility_;
  std::vector<std::size_t> mob_k_;
};

// Registered at static init, so a model composes it by name:
//
//     TermContext ctx;  ctx.provide("flux_operators", hodge);
//     model.use(ctx);
//     model.add("mixed_darcy_cell", On::all(), {{"mobility", 2.0}});
//
// An inline variable, so the registration happens once however many
// translation units include this header.
// Each instantiation is its own catalogue entry, declaring the fields its
// mobility reads. Extending the PHYSICS stays a compile-time act; choosing
// among what was compiled in stays a runtime one.
inline const RegisterTerm<MixedDarcyCell<ConstantMobility>> register_mixed_darcy_cell{
    "mixed_darcy_cell", Coupling::closure, declared_fields<ConstantMobility>()};

inline const RegisterTerm<MixedDarcyCell<PressureEnthalpyMobility>> register_mixed_darcy_cell_ph{
    "mixed_darcy_cell_ph", Coupling::closure, declared_fields<PressureEnthalpyMobility>()};

// The fluid-driven variant. Its registered field list is only the
// structural minimum — the composition fields depend on the fluid the
// context supplies, so the exact list comes from the CONSTRUCTED term.
inline const RegisterTerm<MixedDarcyCell<FluidMobility>> register_mixed_darcy_cell_fluid{
    "mixed_darcy_cell_fluid", Coupling::closure, {"q", "p", "h"}};

// The direct path, for a caller already holding the Hodge in C++.
template <class MobilityModel = ConstantMobility>
std::unique_ptr<Term> mixed_darcy_cell(const hodge::FluxOperators& hodge, const Params& p = {}) {
  TermInfo info{"mixed_darcy_cell", Coupling::closure, declared_fields<MobilityModel>()};
  return std::make_unique<KernelTerm<MixedDarcyCell<MobilityModel>>>(
      std::move(info), MixedDarcyCell<MobilityModel>(hodge, MobilityModel(p)));
}

}  // namespace mimetika::physics::terms
