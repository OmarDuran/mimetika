#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/forms/epoch.hpp"
#include "exokal/geometry/embedding.hpp"
#include "exokal/geometry/reference.hpp"
#include "mimetika/model/constraints.hpp"

// BOUNDARY CONDITIONS AS FORMS ON THE DEGREES OF FREEDOM.
//
// In a mixed method a boundary condition is a statement about a QUANTITY, and
// the quantity is generally a linear form on a facet's unknowns rather than one
// of them. Writing conditions as forms is what makes the same statement mean
// the same thing on a box, on a borehole wall, and in either dimension:
//
//     n . (sigma n) = g        prescribed normal traction
//     t_a . (sigma n) = 0      free slip: no shear traction
//     sigma n = g              the full traction vector
//     q . n = g                prescribed normal flux; g = 0 is a sealed facet
//     a (q.n) + b p = c        Robin, coupling a facet flux to its cell pressure
//
// On an axis-aligned facet the first two collapse to pinning single components,
// which is why a component-pinning code appears to work on a box. On any other
// facet they do not, and the difference is not a refinement issue: the wrong
// condition is imposed, exactly, forever.
//
// WHICH KIND IS WHICH is the opposite of what the primal form trains you to
// expect, and it follows from one rule -- the quantity carried as an unknown is
// imposed STRONGLY, and the one that is not enters naturally:
//
//   mixed flow          the FLUX is the unknown, so a prescribed flux is
//                       strong and a prescribed PRESSURE is natural
//   mixed elasticity    the TRACTION is the unknown, so a prescribed traction
//                       is strong and a prescribed DISPLACEMENT is natural
//
// A roller is the pair: the vanishing shear traction is strong (the forms
// below) and the vanishing normal displacement is natural, enforced by leaving
// the normal traction FREE so that its own equation reads u.n = 0 weakly.

namespace mimetika {

using exokal::forms::Epoch;
using exokal::forms::Index;
using exokal::forms::StratifiedEpoch;
using Point = exokal::Mesh::Point;

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

// A FACET'S ORTHONORMAL FRAME: its canonical normal and d-1 tangents.
//
// The normal is the CANONICAL one -- the direction the space numbers the
// facet's unknowns in -- so a form written here needs no knowledge of which
// cell is asking. The tangents come from the same argmin-|n_k| rule the
// discrete facet basis is built with, so the frame a condition is expressed in
// and the frame the degrees of freedom live in are the same frame.
struct FacetFrame {
  Point normal{0.0, 0.0, 0.0};   // CANONICAL: the direction the dofs are numbered in
  Point outward{0.0, 0.0, 0.0};  // OUTWARD: the direction a condition is about
  double incidence{1.0};         // outward = incidence * normal
  std::array<Point, 2> tangent{Point{0.0, 0.0, 0.0}, Point{0.0, 0.0, 0.0}};
  int n_tangents{0};  // d - 1
  double measure{0.0};

  // THE FRAME COMES FROM THE REFERENCE-SPACE NORMAL, so it is right for a facet
  // of a volume cell, of a surface cell tilted anywhere in space, and of a line.
  // Turning an edge tangent a quarter turn in the xy-plane is the special case
  // that happens to work on a planar mesh lying in that plane and silently
  // produces a vector outside the surface on any other.
  //
  // `cell` supplies the plane the facet's normal lives in. For a volume cell it
  // is ignored; for a surface cell it is the whole content of the question.
  static FacetFrame of(const exokal::Mesh& mesh, int cell_dim, Index cell, Index facet) {
    FacetFrame fr;
    const exokal::Point av = exokal::facet_normal_vector(mesh, cell_dim, cell, facet);
    fr.measure = std::sqrt(av[0] * av[0] + av[1] * av[1] + av[2] * av[2]);
    if (!(fr.measure > 0.0)) throw std::runtime_error("FacetFrame: degenerate facet");
    for (int k = 0; k < 3; ++k) fr.normal[k] = av[k] / fr.measure;
    // AND THE OUTWARD ONE, which is what a boundary condition means. The two
    // differ by the stored boundary coefficient on about half the facets of
    // any mesh, and leaving that conversion to the caller is what produces
    // sign errors no assertion catches.
    const exokal::Point ov = exokal::outward_normal_vector(mesh, cell_dim, cell, facet);
    for (int k = 0; k < 3; ++k) fr.outward[k] = ov[k] / fr.measure;
    fr.incidence = (fr.outward[0] * fr.normal[0] + fr.outward[1] * fr.normal[1] +
                    fr.outward[2] * fr.normal[2]) < 0.0
                       ? -1.0
                       : 1.0;
    fr.n_tangents = cell_dim - 1;

    if (cell_dim == 3) {
      // the same argmin-|n_k| rule the discrete facet basis uses, so the frame
      // a condition is written in is the frame its unknowns live in
      std::size_t a = 0;
      for (std::size_t i = 1; i < 3; ++i) {
        if (std::abs(fr.normal[i]) < std::abs(fr.normal[a])) a = i;
      }
      exokal::Point e{0.0, 0.0, 0.0};
      e[a] = 1.0;
      const double dp = e[0] * fr.normal[0] + e[1] * fr.normal[1] + e[2] * fr.normal[2];
      exokal::Point t1{e[0] - dp * fr.normal[0], e[1] - dp * fr.normal[1],
                       e[2] - dp * fr.normal[2]};
      const double len = std::sqrt(t1[0] * t1[0] + t1[1] * t1[1] + t1[2] * t1[2]);
      for (double& c : t1) c /= len;
      fr.tangent[0] = t1;
      fr.tangent[1] = {fr.normal[1] * t1[2] - fr.normal[2] * t1[1],
                       fr.normal[2] * t1[0] - fr.normal[0] * t1[2],
                       fr.normal[0] * t1[1] - fr.normal[1] * t1[0]};
    } else if (cell_dim == 2) {
      // one tangent: the edge's own direction, taken from the FACET rather
      // than from a coordinate rotation of the normal
      const exokal::Frame ef = exokal::tangent_frame(mesh, 1, facet);
      fr.tangent[0] = ef.axis[0];
    } else if (cell_dim != 1) {
      throw std::invalid_argument("FacetFrame: cell dimension " + std::to_string(cell_dim));
    }
    return fr;
  }
};

// The one cell behind a boundary facet. A boundary condition is a statement
// about a facet, but the facet's normal lives in the plane of the cell it
// bounds, so the cell has to be recovered -- and on the boundary there is
// exactly one, which is what makes it a boundary facet.
inline Index cofacet_of(const exokal::Mesh& mesh, int cell_dim, Index facet) {
  const graphos::CoboundaryOperator cob = graphos::coboundary(mesh.topology(), cell_dim - 1);
  const auto b = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(facet)]);
  const auto e = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(facet) + 1]);
  if (e - b != 1) {
    throw std::invalid_argument("cofacet_of: facet " + std::to_string(facet) +
                                " is interior; a boundary condition needs a boundary facet");
  }
  return cob.indices[b];
}

// A named selection of facets, chosen by where they are or by what they look
// like. A predicate keeps the selection in the driver, where the geometry of
// the problem is known -- a stratum has no idea which of its facets is "the
// borehole wall".
class FacetSelector {
 public:
  using Predicate = std::function<bool(const Point&)>;

  static std::vector<Index> where(const exokal::Mesh& mesh, int cell_dim, const Predicate& p) {
    std::vector<Index> out;
    for (const Index f : boundary_facets(mesh.topology(), cell_dim)) {
      if (p(exokal::centroid(mesh, cell_dim - 1, f))) out.push_back(f);
    }
    return out;
  }

  // a face of a box, at a given coordinate
  static Predicate at(int axis, double value, double tol = 1e-9) {
    return [axis, value, tol](const Point& x) {
      return std::abs(x[static_cast<std::size_t>(axis)] - value) < tol;
    };
  }

  // a cylindrical surface about an axis -- the borehole wall, and the outer
  // boundary of the annulus around it
  static Predicate radius(double r, int axis = 2, double tol = 1e-9,
                          const Point& centre = {0.0, 0.0, 0.0}) {
    return [r, axis, tol, centre](const Point& x) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) {
        if (k == axis) continue;
        const double dk = x[static_cast<std::size_t>(k)] - centre[static_cast<std::size_t>(k)];
        s += dk * dk;
      }
      return std::abs(std::sqrt(s) - r) < tol;
    };
  }

  static Predicate all() {
    return [](const Point&) { return true; };
  }
};

// ------------------------------------------------------------------ forms
//
// The degrees of freedom one field places on a facet. A vector field on a facet
// carries `moments` basis functions per component; the ProductSpace orders the
// component fastest, so the (component, moment) pair addresses them.
struct FacetDofs {
  std::vector<Index> dofs;
  int components{0};
  int moments{0};

  Index at(int component, int moment) const {
    return dofs[static_cast<std::size_t>(moment * components + component)];
  }
};

inline FacetDofs facet_dofs(const exokal::spaces::ProductSpace& space, const std::string& field,
                            int cell_dim, Index facet, Index offset = 0) {
  const std::size_t fi = space.index_of(field);
  const exokal::spaces::DofMap& map = space.map(fi);
  const exokal::spaces::DofLayout& lay = map.layout();
  const int k = cell_dim - 1;
  FacetDofs out;
  out.components = lay.components;
  out.moments = lay.on(k);
  out.dofs.reserve(static_cast<std::size_t>(out.components * out.moments));
  for (int l = 0; l < out.moments; ++l) {
    for (int cp = 0; cp < lay.components; ++cp) {
      out.dofs.push_back(space.offset(fi) + map.global(k, facet, l, cp) + offset);
    }
  }
  return out;
}

// PRESCRIBE A DIRECTIONAL COMPONENT OF A VECTOR-VALUED FACET QUANTITY.
//
//     e . (sigma n) = value      on every facet of the set
//
// with `e` any direction, supplied per facet so it can be that facet's own
// normal or tangent. The value is the pointwise one; it lands entirely on the
// CONSTANT moment scaled by the measure it is integrated against, and the
// higher moments are set to zero, which is what "uniform over the facet" means
// discretely.
inline void impose_component(Constraints& c, const exokal::spaces::ProductSpace& space,
                             const std::string& field, int cell_dim, const exokal::Mesh& mesh,
                             const std::vector<Index>& facets,
                             const std::function<Point(const FacetFrame&)>& direction,
                             const std::function<double(const Point&)>& value,
                             Index offset = 0) {
  for (const Index f : facets) {
    const FacetFrame fr = FacetFrame::of(mesh, cell_dim, cofacet_of(mesh, cell_dim, f), f);
    const Point e = direction(fr);
    const FacetDofs d = facet_dofs(space, field, cell_dim, f, offset);
    const double g = value(exokal::centroid(mesh, cell_dim - 1, f));
    for (int b = 0; b < d.moments; ++b) {
      std::vector<Index> dofs;
      std::vector<double> coeff;
      for (int k = 0; k < d.components; ++k) {
        if (e[static_cast<std::size_t>(k)] == 0.0) continue;
        dofs.push_back(d.at(k, b));
        coeff.push_back(e[static_cast<std::size_t>(k)]);
      }
      if (dofs.empty()) continue;
      c.constrain(std::move(dofs), std::move(coeff), b == 0 ? g * fr.measure : 0.0);
    }
  }
}

// n . (sigma n) = value: the normal traction, or the normal flux of a scalar
// field whose single component IS the normal one.
inline void impose_normal(Constraints& c, const exokal::spaces::ProductSpace& space,
                          const std::string& field, int cell_dim, const exokal::Mesh& mesh,
                          const std::vector<Index>& facets, double value = 0.0,
                          Index offset = 0) {
  impose_component(
      c, space, field, cell_dim, mesh, facets, [](const FacetFrame& fr) { return fr.normal; },
      [value](const Point&) { return value; }, offset);
}

// t_a . (sigma n) = 0 for every tangent: FREE SLIP, the strong half of a
// roller. Works on any facet orientation, because the tangents are that
// facet's own.
inline void impose_free_slip(Constraints& c, const exokal::spaces::ProductSpace& space,
                             const std::string& field, int cell_dim, const exokal::Mesh& mesh,
                             const std::vector<Index>& facets, Index offset = 0) {
  const int nt = mesh.dim() - 1;
  for (int a = 0; a < nt; ++a) {
    impose_component(
        c, space, field, cell_dim, mesh, facets,
        [a](const FacetFrame& fr) { return fr.tangent[static_cast<std::size_t>(a)]; },
        [](const Point&) { return 0.0; }, offset);
  }
}

// sigma n = g, the whole traction vector, from the STRESS TENSOR rather than a
// traction vector. Passing a tensor is what makes it safe: the caller never has
// to know which way a facet's canonical normal points, and a vector assembled
// against the wrong one is silently sign-flipped.
inline void impose_traction(Constraints& c, const exokal::spaces::ProductSpace& space,
                            const std::string& field, int cell_dim, const exokal::Mesh& mesh,
                            const std::vector<Index>& facets,
                            const std::function<std::array<double, 9>(const Point&)>& stress,
                            Index offset = 0) {
  const int d = mesh.dim();
  for (const Index f : facets) {
    const FacetFrame fr = FacetFrame::of(mesh, cell_dim, cofacet_of(mesh, cell_dim, f), f);
    const auto s = stress(exokal::centroid(mesh, cell_dim - 1, f));
    const FacetDofs dd = facet_dofs(space, field, cell_dim, f, offset);
    for (int k = 0; k < d; ++k) {
      double t = 0.0;
      for (int j = 0; j < d; ++j) {
        t += s[static_cast<std::size_t>(k * 3 + j)] * fr.normal[static_cast<std::size_t>(j)];
      }
      for (int b = 0; b < dd.moments; ++b) {
        c.constrain({dd.at(k, b)}, {1.0}, b == 0 ? t * fr.measure : 0.0);
      }
    }
  }
}

// The constant-tensor case, which is what a step load is.
inline void impose_traction(Constraints& c, const exokal::spaces::ProductSpace& space,
                            const std::string& field, int cell_dim, const exokal::Mesh& mesh,
                            const std::vector<Index>& facets, const std::array<double, 9>& stress,
                            Index offset = 0) {
  impose_traction(c, space, field, cell_dim, mesh, facets,
                  [stress](const Point&) { return stress; }, offset);
}

// q . n = value on a SCALAR facet field, whose one component is already the
// normal one, so the form is a single term per moment. Zero is a sealed facet.
inline void impose_normal_flux(Constraints& c, const exokal::spaces::ProductSpace& space,
                               const std::string& field, int cell_dim, const exokal::Mesh& mesh,
                               const std::vector<Index>& facets, double value = 0.0,
                               Index offset = 0) {
  for (const Index f : facets) {
    const FacetFrame fr = FacetFrame::of(mesh, cell_dim, cofacet_of(mesh, cell_dim, f), f);
    const FacetDofs d = facet_dofs(space, field, cell_dim, f, offset);
    for (int b = 0; b < d.moments; ++b) {
      for (int k = 0; k < d.components; ++k) {
        c.constrain({d.at(k, b)}, {1.0}, b == 0 ? value * fr.measure : 0.0);
      }
    }
  }
}

// A ROBIN CONDITION: a (q.n) + b p_E = g, coupling a facet's normal flux to the
// pressure of the cell behind it. One form, two fields -- which is exactly why
// the constraint layer takes forms rather than values.
inline void impose_robin(Constraints& c, const exokal::spaces::ProductSpace& space,
                         const std::string& flux_field, const std::string& cell_field,
                         int cell_dim, const exokal::Mesh& mesh, const std::vector<Index>& facets,
                         double a, double b, double g, Index offset = 0) {
  const graphos::CoboundaryOperator cob = graphos::coboundary(mesh.topology(), cell_dim - 1);
  const std::size_t ci = space.index_of(cell_field);
  const exokal::spaces::DofMap& cmap = space.map(ci);
  for (const Index f : facets) {
    const auto k0 = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f)]);
    const auto k1 = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f) + 1]);
    if (k1 - k0 != 1) throw std::invalid_argument("impose_robin: facet is not on the boundary");
    const Index cell = cob.indices[k0];
    const FacetFrame fr = FacetFrame::of(mesh, cell_dim, cofacet_of(mesh, cell_dim, f), f);
    const FacetDofs d = facet_dofs(space, flux_field, cell_dim, f, offset);
    const Index p = space.offset(ci) + cmap.global(cell_dim, cell, 0, 0) + offset;
    for (int m = 0; m < d.moments; ++m) {
      for (int comp = 0; comp < d.components; ++comp) {
        if (m == 0) {
          c.constrain({d.at(comp, m), p}, {a, b}, g * fr.measure);
        } else {
          c.constrain({d.at(comp, m)}, {a}, 0.0);
        }
      }
    }
  }
}

// ------------------------------------------------------------------ natural
//
// The datum for the quantity that is NOT an unknown, per facet. A term reads it
// from the context; facets it says nothing about contribute nothing, which is
// exactly a homogeneous natural condition and is why a drained face at zero
// pressure costs nothing to impose.
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
// Affine rather than constant because that is what a patch test needs: a mixed
// method of this family reproduces linear displacement fields exactly, and the
// only way to see that is to prescribe one.
// A SCALAR PER CELL: a reservoir is a region at a changed pressure, and zero
// outside it. The facet-indexed holders above are about boundaries; this one is
// about a body load.
class CellData {
 public:
  CellData() = default;
  explicit CellData(std::size_t n_cells) : value_(n_cells, 0.0) {}

  void set(const std::vector<Index>& cells, double v) {
    for (const Index e : cells) value_[static_cast<std::size_t>(e)] = v;
  }
  void set_all(double v) { std::fill(value_.begin(), value_.end(), v); }
  double at(Index e) const { return value_[static_cast<std::size_t>(e)]; }
  std::size_t size() const { return value_.size(); }

 private:
  std::vector<double> value_;
};

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
