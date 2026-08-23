#pragma once

#include <stdexcept>
#include <vector>

#include "mimetika/algebraic_constraints/contact/map.hpp"
#include "mimetika/algebraic_constraints/contact/trace.hpp"
#include "mimetika/model/cauchy_mechanics_model.hpp"

// Contact over Cauchy elasticity: the adapter, and nothing more.
//
// ContactMechanics asks for three operations and this supplies them by
// delegation -- it computes nothing itself:
//
//   to_moments          the Fracture's facet frames, and the fact that the
//                       constant chart function is 1 so a uniform traction
//                       lands entirely on the leading moment
//   solution_operator   the model's affine solution operator S, which
//                       factorizes once and back-substitutes thereafter
//   gap                 the model's trace, rotated into the facet frame
//
// The contact code above it names no model; the model below it names no
// contact. Replacing CauchyMechanicsModel with PoroelasticModel is a second
// file of this shape -- the fracture pressure enters the model's own trace
// through the Biot coupling and its right-hand side is nonzero on the
// prescribed rows, which is why `gap` must be the row residual rather than
// J z -- and laws.hpp, map.hpp and driver.hpp do not change.

namespace mimetika::contact {

class CauchyContactMechanics final : public ContactMechanics {
 public:
  // The model must already have been told which facets carry a prescribed
  // traction, and built: the structure of the constrained problem is fixed at
  // build time and only the values move afterwards.
  CauchyContactMechanics(const CauchyMechanicsModel& model, Fracture fracture)
      : model_(&model), fracture_(std::move(fracture)) {
    if (fracture_.dim() != model.dim()) {
      throw std::invalid_argument(
          "CauchyContactMechanics: the fracture and the model disagree "
          "on the dimension");
    }
    if (model.prescribed_traction().size() != fracture_.size()) {
      throw std::invalid_argument("CauchyContactMechanics: the model was built prescribing " +
                                  std::to_string(model.prescribed_traction().size()) +
                                  " facets and the fracture has " +
                                  std::to_string(fracture_.size()) +
                                  "; call prescribe_traction(fracture.facets()) BEFORE build()");
    }
    for (std::size_t i = 0; i < fracture_.size(); ++i) {
      if (model.prescribed_traction()[i] != fracture_.facets()[i]) {
        throw std::invalid_argument(
            "CauchyContactMechanics: the prescribed facets and the "
            "fracture's facets are not in the same order");
      }
    }
    ndf_ = static_cast<std::size_t>(fracture_.dim()) *
           static_cast<std::size_t>(fracture_.moments_per_facet());
  }

  std::size_t n_points() const override { return fracture_.n_points(); }
  int dim() const override { return fracture_.dim(); }
  std::size_t n_dofs() const override { return model_->simulation().n_dofs(); }

  const Fracture& fracture() const { return fracture_; }

  // Facet-frame values -> traction moments.
  //
  // A traction dof is the moment m_b = int_f (sigma n) chi_b. The facet chart is
  // orthonormalized in L^2(f) with chi_0 = 1 and int_f chi_b = 0 for b >= 1, so
  // a traction taken constant over the facet -- which is what one enforcement
  // point per facet means -- lands entirely on the leading moment, t_k |f|, and
  // contributes nothing to the higher ones. The same statement TractionBC makes
  // for a prescribed boundary traction, and for the same reason.
  void to_moments(const std::vector<Vec3>& x, std::vector<double>& moments) const override {
    if (x.size() != n_points()) throw std::invalid_argument("to_moments: one value per point");
    moments.assign(fracture_.size() * ndf_, 0.0);
    const int d = dim();
    for (std::size_t i = 0; i < fracture_.size(); ++i) {
      const std::array<double, 3> t = fracture_.to_ambient(x[i], i);
      const double area = fracture_.frame(i).measure;
      for (int k = 0; k < d; ++k) {
        // moment 0, component k: the ProductSpace orders the component fastest
        moments[i * ndf_ + static_cast<std::size_t>(k)] = t[static_cast<std::size_t>(k)] * area;
      }
    }
  }

  // S : m -> z(m). One factorization, done at build time; this is a
  // back-substitution against a moved right-hand side.
  void solution_operator(const std::vector<double>& moments,
                         std::vector<double>& z) const override {
    z = model_->solution_operator(moments);
  }

  // The gap, from the trace: the residual of the unfractured constitutive row,
  // rotated into the facet frame. It is a displacement in metres, and its
  // leading moment is the facet mean -- the quantity one enforcement point per
  // facet is about.
  void gap(const std::vector<double>& z, std::vector<Vec3>& g) const override {
    g.assign(n_points(), Vec3{});
    const int d = dim();
    for (std::size_t i = 0; i < fracture_.size(); ++i) {
      const std::vector<double> row = model_->trace(fracture_.facets()[i], z);
      std::array<double, 3> ambient{0.0, 0.0, 0.0};
      for (int k = 0; k < d; ++k) {
        ambient[static_cast<std::size_t>(k)] = row[static_cast<std::size_t>(k)];
      }
      g[i] = fracture_.to_frame(ambient, i);
    }
  }

 private:
  const CauchyMechanicsModel* model_;
  Fracture fracture_;
  std::size_t ndf_{0};
};

}  // namespace mimetika::contact
