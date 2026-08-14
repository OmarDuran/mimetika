#pragma once

#include "mimetika/physics/catalogue.hpp"
#include "mimetika/physics/mechanics.hpp"

// LINEAR ELASTICITY, as a catalogue entry: one package and nothing else.
//
// It lives apart from the poroelastic entries because a consumer that wants
// only mechanics should not have to include -- and thereby REGISTER -- flow,
// the Biot coupling and fluid storage to get it. Registration happens at
// static initialization, so an include is not a suggestion: whatever a
// translation unit includes is what its catalogue contains.

namespace mimetika::compositions {

inline const physics::RegisterModel linear_elasticity{
    "linear_elasticity", "Weakly-symmetric mixed elasticity (Hellinger-Reissner)",
    [](const physics::ModelOptions& o) {
      physics::Composition c;
      c.emplace<physics::Mechanics>(
          physics::MechanicsOptions{"mixed_elasticity_cell", o.traction_moments});
      return c;
    }};

}  // namespace mimetika::compositions
