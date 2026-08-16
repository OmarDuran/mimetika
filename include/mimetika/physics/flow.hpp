#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "exokal/forms/stencil.hpp"
#include "mimetika/physics/terms/darcy.hpp"
#include "mimetika/physics/package.hpp"

// FLOW IN A POROUS MEDIUM, in mixed form.
//
//     r_q = ⋆_{(λK)⁻¹} q − div^T p        the constitutive relation
//     r_p = div q                          the mass balance
//
// The first row is a CLOSURE coupling pairing (n−1)-cochains with each
// other and with the n-cochain of the pressure; the second is the adjoint
// of the same signs. Neither reaches out of a cell, so both are algebraic
// in exokal's sense — the discrete Hodge is the whole of the metric content
// and the divergence is pure topology.
//
// ONE PACKAGE FOR BOTH FLOW ROWS OF THE CATALOGUE. Single-phase flow is
// compositional flow at one component: the equations are the same and only
// the number of composition fields differs, which is data the fluid model
// carries rather than a property of this type. A package that hard-coded a
// phase count would turn one catalogue row into two implementations, which
// is exactly the multiplication this layer exists to prevent.

namespace mimetika::physics {

struct FlowOptions {
  // How many composition fields accompany the pressure. Zero is the
  // single-phase case; the fields are named z0, z1, ... and the closure
  // that reads them decides what they mean.
  std::size_t components{0};
  bool thermal{false};
  // Which registered Darcy term to attach. The mobility model is a
  // compile-time choice of the application, not of the library.
  std::string darcy_term{"mixed_darcy_cell"};
  // The mobility multiplying the Hodge: M(lambda) = M(1)/lambda. A transient
  // step sets it to dt, which rescales the flux to the volume that crosses a
  // facet during the step and is what makes backward Euler need no
  // time-stepping machinery at all.
  double mobility{1.0};
  // MOMENTS PER FACET OF THE FLUX SPACE. Zero means d, the de Rham/BDM_1
  // layout. One is the lowest-order single-flux-per-facet space -- RT_0 in its
  // de Rham realization, or the stabilized polytopal product.
  //
  // The package does not choose it and does not know which inner product will
  // be built: it lays out a space, and the layout has to match whatever star
  // lands on it. The driver derives both from ONE realization so they cannot
  // disagree -- a mismatch here is not a wrong answer, it is a space of the
  // wrong size, and every index downstream is off.
  int flux_moments{0};
};

class Flow final : public Package {
 public:
  Flow() = default;
  explicit Flow(FlowOptions o) : opt_(std::move(o)) {}

  std::string name() const override { return "Flow"; }

  // FIELDS ARE NAMED BY CODIMENSION. The same package placed on the ambient
  // stratum and on a fracture immersed in it declares q_0/p_0 and q_1/p_1 —
  // the same physics, distinct unknowns of one global system. A term
  // declares the quantity and exokal resolves it per stratum, so the
  // hierarchy costs one attachment rather than one package per depth.
  Requirements requirements(int dim, int codim = 0) const override {
    const auto at = [codim](std::string_view base) { return exokal::forms::field_at(base, codim); };
    Requirements r;
    // the flux is a cochain on the facets, the pressure one per cell: the
    // lowest-order mixed pair, and the layouts that make the first row an
    // (n−1, n−1) pairing and the second an (n, n−1) one
    // THE MOMENT COUNT COMES FROM THE REALIZATION. d of them is the BDM_1 /
    // de Rham space, which is what a coupled poroelastic model needs because
    // the stress space it pairs with has d^2; one is RT_0 or the stabilized
    // polytopal product. These are DIFFERENT discretizations, not a finer and
    // a coarser version of one, so the choice belongs to whoever states the
    // model rather than to this package.
    r.fields.push_back(
        {at("q"),
         DofLayout::moments(dim, dim - 1, opt_.flux_moments > 0 ? opt_.flux_moments : dim)});
    r.fields.push_back({at("p"), DofLayout::cell_wise(dim)});
    if (opt_.thermal) r.fields.push_back({at("h"), DofLayout::cell_wise(dim)});
    for (std::size_t a = 0; a < opt_.components; ++a) {
      r.fields.push_back({at("z" + std::to_string(a)), DofLayout::cell_wise(dim)});
    }

    r.provides = {"pressure", "mass_balance"};
    if (opt_.thermal) r.provides.push_back("enthalpy");

    r.slots = {{"density", Scope::fluid},
               {"viscosity", Scope::fluid},
               {"porosity", Scope::rock},
               {"permeability", Scope::rock},
               // exchange across a codimension gap. Declared unconditionally
               // because whether the mesh HAS a substratum is a property of
               // the mesh, and a package may not branch on that.
               {"normal_permeability", Scope::interface}};
    if (opt_.thermal) r.slots.push_back({"enthalpy", Scope::fluid});
    return r;
  }

  void attach(exokal::forms::Model& model, const exokal::forms::TermContext& ctx) const override {
    (void)ctx;  // the term reads its own data from the context by name
    model.add(opt_.darcy_term, exokal::forms::On::all(), {{"mobility", opt_.mobility}});
  }

  const FlowOptions& options() const { return opt_; }

 private:
  FlowOptions opt_;
};

}  // namespace mimetika::physics
