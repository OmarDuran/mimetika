#pragma once

#include "mimetika/physics/catalogue.hpp"
#include "mimetika/physics/cauchy_mechanics.hpp"
#include "mimetika/physics/terms/rotation_jump.hpp"

// Linear elasticity as a catalogue entry: one package.
//
// It lives apart from the poroelastic entries because a consumer that wants
// only mechanics should not have to include -- and thereby register -- flow,
// the Biot coupling and fluid storage to get it. Registration happens at
// static initialization: whatever a translation unit includes is what its
// catalogue contains.

namespace mimetika::compositions {

inline const physics::RegisterModel linear_elasticity{
    "linear_elasticity", "Weakly-symmetric mixed elasticity (Hellinger-Reissner)",
    [](const physics::ModelOptions& o) {
      physics::Composition c;
      // The term follows the formulation: four fields is a different pairing,
      // not the same one with a field appended, and the strong family is a
      // different space, with the symmetry in the reconstruction rather than
      // against a rotation multiplier.
      const char* term =
          o.strong_symmetry
              ? (o.total_pressure ? "strong_elasticity_total_cell" : "strong_elasticity_cell")
              : (o.total_pressure ? "mixed_elasticity_total_cell" : "mixed_elasticity_cell");
      c.emplace<physics::Mechanics>(physics::MechanicsOptions{
          term, o.traction_moments, o.traction_components, o.total_pressure, o.strong_symmetry,
          o.rotation_jump, o.shear_modulus});
      return c;
    }};

}  // namespace mimetika::compositions
