#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "exokal/forms/epoch.hpp"
#include "exokal/geometry/embedding.hpp"
#include "mimetika/model/constraints.hpp"

// BOUNDARY CONDITIONS IN A MIXED FORM, where which kind is which is the
// opposite of what the primal form trains you to expect.
//
// The rule is simply that the quantity carried as a degree of freedom is
// imposed ESSENTIALLY, and the one that is not enters NATURALLY:
//
//   mixed flow     the FLUX is the unknown, so a prescribed flux is
//                  essential and a prescribed PRESSURE is natural
//   mixed elasticity  the TRACTION is the unknown, so a prescribed traction
//                  is essential and a prescribed DISPLACEMENT is natural
//
// They are mirror images, and Terzaghi's column needs all four: a traction
// on the loaded face and rollers on the sides (essential, on stress),
// sealed faces (essential, on flux), and a drained face (natural, a
// pressure — zero here, which is why it costs nothing).
//
// Getting the classification wrong does not crash. The column simply stops
// being one-dimensional: it bulges, or drains from the wrong face, or never
// reaches the right settlement. That is what makes the closed form worth
// having.

namespace mimetika {

using exokal::forms::Epoch;
using exokal::forms::Index;
using exokal::forms::StratifiedEpoch;

// The facets on the boundary of a stratum: exactly one cofacet.
inline std::vector<Index> boundary_facets(const graphos::Complex& c, int cell_dim) {
  const graphos::CoboundaryOperator cob = graphos::coboundary(c, cell_dim - 1);
  std::vector<Index> out;
  for (Index f = 0; f < c.count(cell_dim - 1); ++f) {
    const auto b = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f)]);
    const auto e = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f) + 1]);
    if (e - b == 1) out.push_back(f);
  }
  return out;
}

// A named selection of boundary facets, chosen by where they are. A predicate
// on the centroid is enough for every face of a box, and it keeps the
// selection in the driver where the geometry of the problem is known — a
// stratum has no idea which of its faces is "the loaded one".
class FacetSelector {
 public:
  using Predicate = std::function<bool(const exokal::Mesh::Point&)>;

  static std::vector<Index> where(const exokal::Mesh& mesh, int cell_dim, const Predicate& p) {
    std::vector<Index> out;
    for (const Index f : boundary_facets(mesh.topology(), cell_dim)) {
      if (p(exokal::centroid(mesh, cell_dim - 1, f))) out.push_back(f);
    }
    return out;
  }

  // the common case: a face of a box, at a given coordinate
  static Predicate at(int axis, double value, double tol = 1e-9) {
    return [axis, value, tol](const exokal::Mesh::Point& x) {
      return std::abs(x[static_cast<std::size_t>(axis)] - value) < tol;
    };
  }
};

// ---------------------------------------------------------------- essential
//
// Pinning the degrees of freedom a facet carries, for the quantity that IS
// the unknown. The value is supplied per (component, facet-basis) moment, so
// a sealed face pins every moment to zero while a traction pins the constant
// moment of each component to the applied load and the higher moments to
// zero — which is what "uniform traction" means discretely.
struct FacetDofs {
  std::vector<Index> dofs;
  std::vector<int> component;  // which vector component each dof carries
  std::vector<int> moment;     // which facet-basis function
};

// The degrees of freedom one field places on a facet, with what each means.
inline FacetDofs facet_dofs(const exokal::spaces::ProductSpace& space, const std::string& field,
                            int cell_dim, Index facet) {
  const std::size_t f = space.index_of(field);
  const exokal::spaces::DofMap& map = space.map(f);
  const exokal::spaces::DofLayout& lay = map.layout();
  FacetDofs out;
  const int k = cell_dim - 1;
  for (int l = 0; l < lay.on(k); ++l) {
    for (int cp = 0; cp < lay.components; ++cp) {
      out.dofs.push_back(space.offset(f) + map.global(k, facet, l, cp));
      out.component.push_back(cp);
      out.moment.push_back(l);
    }
  }
  return out;
}

// Pin every moment of a field on the given facets — a sealed face, or a
// clamped one.
inline void pin_facets(Constraints& c, const exokal::spaces::ProductSpace& space,
                       const std::string& field, int cell_dim,
                       const std::vector<Index>& facets, Index offset = 0) {
  for (const Index f : facets) {
    const FacetDofs d = facet_dofs(space, field, cell_dim, f);
    for (const Index dof : d.dofs) c.pin(dof + offset, 0.0);
  }
}

// Pin only the named COMPONENTS, leaving the others free — which is what a
// roller is: the normal component of the displacement is held and the
// tangential ones are not.
inline void pin_facet_components(Constraints& c, const exokal::spaces::ProductSpace& space,
                                 const std::string& field, int cell_dim,
                                 const std::vector<Index>& facets,
                                 const std::vector<int>& components, Index offset = 0) {
  for (const Index f : facets) {
    const FacetDofs d = facet_dofs(space, field, cell_dim, f);
    for (std::size_t i = 0; i < d.dofs.size(); ++i) {
      for (const int cp : components) {
        if (d.component[i] == cp) c.pin(d.dofs[i] + offset, 0.0);
      }
    }
  }
}

// A uniform applied load: the constant moment of each component carries it,
// the higher moments are zero. `load` is the traction per component, already
// scaled by the facet measure the moment is taken against.
inline void pin_facet_load(Constraints& c, const exokal::spaces::ProductSpace& space,
                           const std::string& field, int cell_dim,
                           const std::vector<Index>& facets, const std::vector<double>& load,
                           Index offset = 0) {
  for (const Index f : facets) {
    const FacetDofs d = facet_dofs(space, field, cell_dim, f);
    for (std::size_t i = 0; i < d.dofs.size(); ++i) {
      const double v = d.moment[i] == 0
                           ? load[static_cast<std::size_t>(d.component[i])]
                           : 0.0;
      c.pin(d.dofs[i] + offset, v);
    }
  }
}

// ------------------------------------------------------------------ natural
//
// The datum for the quantity that is NOT a degree of freedom, per facet. A
// term reads it from the context; facets it says nothing about contribute
// nothing, which is exactly a homogeneous natural condition and is why a
// drained face at zero pressure costs nothing to impose.
class BoundaryData {
 public:
  explicit BoundaryData(std::size_t n_facets) : value_(n_facets, 0.0), set_(n_facets, 0) {}

  void set(const std::vector<Index>& facets, double v) {
    for (const Index f : facets) {
      value_[static_cast<std::size_t>(f)] = v;
      set_[static_cast<std::size_t>(f)] = 1;
    }
  }

  bool applies(Index f) const { return set_[static_cast<std::size_t>(f)] != 0; }
  double at(Index f) const { return value_[static_cast<std::size_t>(f)]; }

 private:
  std::vector<double> value_;
  std::vector<char> set_;
};

// A VECTOR DATUM, affine per facet: u(x) = a + B (x - x_E).
//
// Affine rather than constant because that is what a patch test needs. A
// mixed method of this family reproduces linear displacement fields exactly,
// and the only way to see that is to prescribe one — a constant datum would
// pass on a method that had lost the property. The moments of an affine field
// against the facet basis are already computed by the stress product, so this
// costs no quadrature at the boundary.
class BoundaryVectorData {
 public:
  explicit BoundaryVectorData(std::size_t n_facets)
      : constant_(n_facets * 3, 0.0), gradient_(n_facets * 9, 0.0), set_(n_facets, 0) {}

  void set(const std::vector<Index>& facets, const std::array<double, 3>& a) {
    for (const Index f : facets) {
      const auto i = static_cast<std::size_t>(f);
      for (std::size_t k = 0; k < 3; ++k) constant_[i * 3 + k] = a[k];
      set_[i] = 1;
    }
  }

  // u(x) = a + B (x - x_E), with B row-major
  void set_affine(const std::vector<Index>& facets, const std::array<double, 3>& a,
                  const std::array<double, 9>& B) {
    set(facets, a);
    for (const Index f : facets) {
      const auto i = static_cast<std::size_t>(f);
      for (std::size_t k = 0; k < 9; ++k) gradient_[i * 9 + k] = B[k];
    }
  }

  bool applies(Index f) const { return set_[static_cast<std::size_t>(f)] != 0; }
  double constant_at(Index f, std::size_t k) const {
    return constant_[static_cast<std::size_t>(f) * 3 + k];
  }
  double gradient_at(Index f, std::size_t k, std::size_t c) const {
    return gradient_[static_cast<std::size_t>(f) * 9 + k * 3 + c];
  }

 private:
  std::vector<double> constant_, gradient_;
  std::vector<char> set_;
};

}  // namespace mimetika
