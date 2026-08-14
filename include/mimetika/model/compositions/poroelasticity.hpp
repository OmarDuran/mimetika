#pragma once

#include "mimetika/model/compositions/elasticity.hpp"
#include "mimetika/physics/catalogue.hpp"
#include "mimetika/physics/flow.hpp"
#include "mimetika/physics/mechanics.hpp"
#include "mimetika/physics/poro_coupling.hpp"
#include "mimetika/physics/storage.hpp"

// POROELASTICITY AND CONSOLIDATION: flow, plus mechanics, plus the coupling
// between them. Including this file registers linear_elasticity too, since
// these entries are built from the same Mechanics package and a consumer of
// the coupled models invariably wants the uncoupled one to compare against.

namespace mimetika::compositions {

// THE BUDGET IS TEN LINES, and this is the entry that had to justify it:
// poroelasticity is flow, plus mechanics, plus the coupling between them.
// No new term, no new field, no new package.
inline const physics::RegisterModel poroelasticity{
    "poroelasticity", "Single-phase flow coupled to linear elasticity (Biot)",
    [](const physics::ModelOptions& o) {
      physics::Composition c;
      c.emplace<physics::Flow>(physics::FlowOptions{0, o.thermal, "mixed_darcy_cell", 1.0, o.flux_moments});
      c.emplace<physics::Mechanics>(
          physics::MechanicsOptions{"mixed_elasticity_cell", o.traction_moments});
      c.emplace<physics::PoroCoupling>();
      return c;
    }};

// CONSOLIDATION: poroelasticity made transient by the fluid's own storage.
// Still a composition — the fourth package adds one term and no field.
inline const physics::RegisterModel consolidation{
    "consolidation", "Terzaghi consolidation: poroelasticity with fluid storage",
    [](const physics::ModelOptions& o) {
      physics::Composition c;
      c.emplace<physics::Flow>(
          physics::FlowOptions{0, o.thermal, "mixed_darcy_cell", o.mobility, o.flux_moments});
      c.emplace<physics::Mechanics>(
          physics::MechanicsOptions{"mixed_elasticity_cell", o.traction_moments});
      c.emplace<physics::PoroCoupling>(
          physics::PoroCouplingOptions{o.biot, o.volumetric_compliance});
      c.emplace<physics::Storage>(physics::StorageOptions{o.storage});
      return c;
    }};

}  // namespace mimetika::compositions
