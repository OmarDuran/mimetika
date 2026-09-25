#pragma once

#include <stdexcept>
#include <string>

#include "mimetika/physics/package.hpp"
#include "mimetika/physics/terms/contact_interface.hpp"
#include "mimetika/physics/terms/contact_trace.hpp"

// The fracture's half of the mixed contact system.
//
// The block system is
//
//     [ A   B^T  0    T^T ] [sigma]
//     [ B   0    0    0   ] [u    ]
//     [ 0   0    A_c  J   ] [t_c  ]
//     [ T   0    J^T  0   ] [u_c  ]
//
// and the first two rows are ORDINARY MIXED ELASTICITY on the cut complex:
// `Mechanics` writes them unchanged, knowing nothing about the fracture,
// because the cut is a property of the mesh. What is new is the third and
// fourth rows, and the unknowns they are written in: a traction t_c on the
// fracture and its two-sided displacement trace u_c, both broken face by face,
// which is what makes A_c and J block diagonal and a contact law local.
//
// This package declares those two fields and attaches the three terms that
// write the new rows -- A_c, J and the trace T. It declares nothing at
// codimension zero, so the composition is
//
//     c.only_at(0).emplace<LinearContact>(...)
//
// with the bulk confined to the ambient stratum and this one to the fracture.
//
// THE SIZES ARE THE AMBIENT FACET'S. t_c is a traction on a fracture face and
// u_c is a displacement trace tested against the same chart, so both are sized
// by however the ambient stress carries a facet's moments -- six wrench
// coefficients under strong symmetry, d moments in d components under weak.
// The driver reads that off the operators and states it here; it cannot be
// inferred, because the fracture stratum never sees the ambient layout.

namespace mimetika::physics {

struct ContactOptions {
  int traction_moments{0};     // per facet, as the ambient stress carries them
  int traction_components{0};  // 1 for the wrench layout, d for the componentwise one
};

class LinearContact final : public Package {
 public:
  LinearContact() = default;
  explicit LinearContact(ContactOptions o) : opt_(o) {}

  std::string name() const override { return "LinearContact"; }

  Requirements requirements(int dim, int codim = 0) const override {
    Requirements r;
    // the bulk rows are Mechanics's, and there is no contact between cells
    if (codim != 1) return r;
    const int ambient = dim + codim;
    const int nb = opt_.traction_moments > 0 ? opt_.traction_moments : ambient;
    const int nc = opt_.traction_components > 0 ? opt_.traction_components : ambient;
    const int k = nb * nc;
    const auto at = [codim](std::string_view b) { return exokal::forms::field_at(b, codim); };
    r.fields.push_back({at("t"), DofLayout::cell_wise(dim, 1, k)});
    // the two sides in one field: the jump is a difference of its halves, and
    // keeping them together is what lets J be written face by face
    r.fields.push_back({at("uc"), DofLayout::cell_wise(dim, 1, 2 * k)});
    r.provides = {"contact_traction", "displacement_jump"};
    return r;
  }

  void attach(exokal::forms::Model& model, const exokal::forms::TermContext&) const override {
    // On::all() throughout: the trace is a carrier coupling and runs only where
    // a substratum is carried, the other two resolve only where t_c and u_c
    // exist, so each lands on the one stratum that hosts it.
    model.add("contact_trace_facet", exokal::forms::On::all(), exokal::forms::Params{});
    model.add("contact_compliance_cell", exokal::forms::On::all(), exokal::forms::Params{});
    model.add("contact_jump_cell", exokal::forms::On::all(), exokal::forms::Params{});
  }

 private:
  ContactOptions opt_;
};

}  // namespace mimetika::physics
