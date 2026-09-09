#pragma once

#include "mimetika/physics/catalogue.hpp"
#include "mimetika/physics/flow.hpp"

// Catalogue entries: each names a composition of packages.

namespace mimetika::compositions {

inline const physics::RegisterModel flow{
    "flow", "Darcy flow of one phase, mixed form", [](const physics::ModelOptions& o) {
      physics::Composition c;
      c.emplace<physics::Flow>(
          physics::FlowOptions{0, o.thermal, "mixed_darcy_cell", 1.0, o.flux_moments});
      return c;
    }};

// The same package, told a different component count: two catalogue rows,
// one implementation.
inline const physics::RegisterModel compositional_flow{
    "compositional_flow", "Darcy flow of a compositional mixture, mixed form",
    [](const physics::ModelOptions& o) {
      physics::Composition c;
      c.emplace<physics::Flow>(physics::FlowOptions{o.components, o.thermal,
                                                    "mixed_darcy_cell_fluid", 1.0, o.flux_moments});
      return c;
    }};

}  // namespace mimetika::compositions
