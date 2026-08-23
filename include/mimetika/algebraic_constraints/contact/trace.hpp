#pragma once

#include <stdexcept>
#include <vector>

#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/algebraic_constraints/contact/laws.hpp"
#include "mimetika/model/boundary.hpp"

// The trace operator on a fracture: the displacement jump, as the adjoint of
// the divergence and the asymmetry.
//
// A contact law needs two things from the mechanics -- the traction on the
// fracture, which in the mixed form is already a degree of freedom, and the
// displacement jump, which is not. The jump is what the trace operator
// supplies, and it is not a separate construction:
//
//     [| D^T u + A^T gamma |]_f = (D^T u + A^T gamma)_f^+ - (...)_f^-
//
// The right-hand side is not something to be assembled twice and subtracted.
// D is the discrete divergence, a map from facet tractions to cell vectors, and
// its adjoint D^T maps cell displacements back onto facets; each cell's
// contribution carries that cell's outward incidence on the facet, so the two
// cofaces of an interior facet enter with opposite signs and the assembled row
// is the difference. The jump comes out of the adjoint for free. The same holds
// for A and the rotation gamma, which supplies the rigid-rotation part of the
// displacement field the facet sees.
//
// So the operator is the constitutive row of the unfractured system,
//
//     g_f = -( M sigma - D^T u - A^T gamma )_f ,
//
// evaluated on the solution. Three consequences:
//
//   * It is linear, so it can be applied even though that row was replaced by
//     the contact constraint. The row the fractured system solves is gone; the
//     functional it used to express is not.
//   * It must be the unfractured row. At the solution the fractured row is
//     satisfied exactly and its residual is zero, whereas the unfractured
//     residual is precisely the jump this exists to extract.
//   * No Gram^{-1}. A traction degree of freedom is a moment
//     m = int_e (sigma n) b, so recovering its pointwise values needs
//     Gram^{-1} m. The jump term int_e [[u]].(tau n) is instead paired against
//     that moment: writing (tau n) = sum_b phi_b b_b gives m = Gram phi, so the
//     pairing already carries a Gram^{-1} and the residual emerges as the
//     expansion coefficients of the jump, ready to evaluate against the basis.
//     A second inversion divides the jump by |e|, and the slip then grows like
//     1/h under refinement -- a mesh-dependent answer that converges to nothing.
//
// The Python reference computes the same functional through the folded
// four-field product, which carries an extra hydrostatic degree of freedom.
// That makes no difference here: three- and four-field evaluate one functional,
// and the contact driver wants only the jump.

namespace mimetika::contact {

using graphos::Index;

// The fracture: a set of facets on which a contact law is active, with the
// per-facet frames and the dof addressing the trace needs.
//
// An instance of a driver owns one of these, so several fractures -- several
// laws, several friction coefficients -- coexist in one problem as several
// drivers over disjoint facet sets.
class Fracture {
 public:
  Fracture(const exokal::Mesh& mesh, int cell_dim, std::vector<Index> facets, int moments_per_facet)
      : mesh_(&mesh), dim_(cell_dim), facets_(std::move(facets)), nb_(moments_per_facet) {
    if (facets_.empty()) throw std::invalid_argument("Fracture: no facets");
    if (nb_ < 1 || nb_ > dim_) {
      throw std::invalid_argument("Fracture: moments per facet must lie in [1, d]");
    }
    const graphos::Complex& c = mesh.topology();
    const graphos::CoboundaryOperator cob = graphos::coboundary(c, cell_dim - 1);
    frames_.reserve(facets_.size());
    for (const Index f : facets_) {
      const auto b = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f)]);
      const auto e = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f) + 1]);
      if (e - b != 2) {
        throw std::invalid_argument(
            "Fracture: facet " + std::to_string(f) + " has " + std::to_string(e - b) +
            " cofaces; a fracture is INTERIOR -- a jump needs two sides, and a "
            "boundary facet has one");
      }
      // the frame is the facet's, so both cofaces read one convention: the
      // canonical normal the traction dofs are numbered in, and the tangents
      // derived from it alone
      frames_.push_back(FacetFrame::of(mesh, cell_dim, cob.indices[b], f));
    }
  }

  const exokal::Mesh& mesh() const { return *mesh_; }
  int dim() const { return dim_; }
  int moments_per_facet() const { return nb_; }
  const std::vector<Index>& facets() const { return facets_; }
  std::size_t size() const { return facets_.size(); }
  const FacetFrame& frame(std::size_t i) const { return frames_[i]; }

  // One enforcement point per facet: the law is applied to the facet-mean
  // traction, which is what most discrete-fracture codes do. Applying it at the
  // facet quadrature points instead resolves partial contact within a facet at
  // the cost of state per point; the choice belongs to the caller and the law is
  // written the same way either way.
  std::size_t n_points() const { return facets_.size(); }

  // rotate a facet-frame vector into the ambient components the dofs carry
  std::array<double, 3> to_ambient(const Vec3& v, std::size_t i) const {
    const FacetFrame& fr = frames_[i];
    std::array<double, 3> out{0.0, 0.0, 0.0};
    for (std::size_t k = 0; k < 3; ++k) {
      out[k] = v[0] * fr.normal[k];
      for (int a = 0; a < fr.n_tangents; ++a) {
        out[k] += v[static_cast<std::size_t>(a + 1)] * fr.tangent[static_cast<std::size_t>(a)][k];
      }
    }
    return out;
  }

  // and back: ambient components into the facet frame
  Vec3 to_frame(const std::array<double, 3>& a, std::size_t i) const {
    const FacetFrame& fr = frames_[i];
    Vec3 out;
    for (std::size_t k = 0; k < 3; ++k) out[0] += a[k] * fr.normal[k];
    for (int t = 0; t < fr.n_tangents; ++t) {
      double acc = 0.0;
      for (std::size_t k = 0; k < 3; ++k) {
        acc += a[k] * fr.tangent[static_cast<std::size_t>(t)][k];
      }
      out[static_cast<std::size_t>(t + 1)] = acc;
    }
    return out;
  }

 private:
  const exokal::Mesh* mesh_;
  int dim_;
  std::vector<Index> facets_;
  int nb_;
  std::vector<FacetFrame> frames_;
};

}  // namespace mimetika::contact
