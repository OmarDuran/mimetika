#pragma once

#include "mimetika/physics/catalogue.hpp"
#include "mimetika/physics/flow.hpp"

// CATALOGUE ENTRIES. Each is a composition and nothing else — the budget
// is a few lines, and exceeding it means a package is missing.

namespace mimetika::compositions {

inline const physics::RegisterModel single_phase_flow{
    "single_phase_flow", "Darcy flow of one phase, mixed form",
    [](const physics::ModelOptions& o) {
      physics::Composition c;
      c.emplace<physics::Flow>(physics::FlowOptions{0, o.thermal, "mixed_darcy_cell", 1.0, o.flux_moments});
      return c;
    }};

// The SAME package, told a different component count. Two catalogue rows,
// one implementation — which is the property the catalogue lives or dies on.
inline const physics::RegisterModel compositional_flow{
    "compositional_flow", "Darcy flow of a compositional mixture, mixed form",
    [](const physics::ModelOptions& o) {
      physics::Composition c;
      c.emplace<physics::Flow>(
          physics::FlowOptions{o.components, o.thermal, "mixed_darcy_cell_fluid", 1.0,
                               o.flux_moments});
      return c;
    }};

}  // namespace mimetika::compositions
