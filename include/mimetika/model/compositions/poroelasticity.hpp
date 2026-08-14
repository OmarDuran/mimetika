#pragma once

#include "mimetika/physics/catalogue.hpp"
#include "mimetika/physics/flow.hpp"
#include "mimetika/physics/mechanics.hpp"
#include "mimetika/physics/poro_coupling.hpp"
#include "mimetika/physics/storage.hpp"

namespace mimetika::compositions {

inline const physics::RegisterModel linear_elasticity{
    "linear_elasticity", "Weakly-symmetric mixed elasticity (Hellinger-Reissner)",
    [](const physics::ModelOptions&) {
      physics::Composition c;
      c.emplace<physics::Mechanics>();
      return c;
    }};

// THE BUDGET IS TEN LINES, and this is the entry that had to justify it:
// poroelasticity is flow, plus mechanics, plus the coupling between them.
// No new term, no new field, no new package.
inline const physics::RegisterModel poroelasticity{
    "poroelasticity", "Single-phase flow coupled to linear elasticity (Biot)",
    [](const physics::ModelOptions& o) {
      physics::Composition c;
      c.emplace<physics::Flow>(physics::FlowOptions{0, o.thermal, "mixed_darcy_cell", 1.0, o.flux_moments});
      c.emplace<physics::Mechanics>();
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
      c.emplace<physics::Mechanics>();
      c.emplace<physics::PoroCoupling>(
          physics::PoroCouplingOptions{o.biot, o.volumetric_compliance});
      c.emplace<physics::Storage>(physics::StorageOptions{o.storage});
      return c;
    }};

}  // namespace mimetika::compositions
