#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/model/boundary_conditions.hpp"
#include "mimetika/model/compositions/elasticity.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/solver/linear.hpp"

// CAUCHY ELASTICITY, STATED AS DATA -- the poroelastic model with the flow
// taken out, and the smallest problem that exercises the stress space alone.
//
// Weakly-symmetric mixed form (Hellinger-Reissner), THREE fields:
//
//     r_sigma = M sigma - D^T u - A^T gamma      the constitutive relation
//     r_u     = + D sigma                        momentum balance
//     r_gamma = + A sigma                        symmetry, imposed weakly
//
// gamma is not decoration. The space carries no symmetry constraint -- that is
// what makes it usable -- so gamma is the multiplier enforcing sigma = sigma^T
// against the rigid rotations. Dropping it does not simplify the method, it
// changes it.
//
// It earns its keep the way SinglePhaseModel does for the flux: when a
// poroelastic answer is wrong, this is what says whether the mechanics half is.
// Both of its closed forms hold in any dimension:
//
//     a column    uniaxial extension: constant stress, LINEAR displacement,
//                 which every one of these spaces contains exactly
//     an annulus  Lame's thick-walled tube: sigma_rr, sigma_tt carry a 1/r^2
//                 and u_r a 1/r, which none of them contains
//
// so the first is a statement about the space and the second about resolution.

namespace mimetika {

// Isotropic linear elasticity, in the two parameters the operators want. The
// engineering constants are derived rather than stored, so a caller can state
// whichever pair it has.
struct ElasticMaterial {
  double shear{1.0};  // mu
  double lame{1.0};   // lambda

  static ElasticMaterial from_young_poisson(double E, double nu) {
    if (!(nu > -1.0 && nu < 0.5)) {
      throw std::invalid_argument(
          "ElasticMaterial: Poisson's ratio must lie in (-1, 1/2); at 1/2 exactly the material "
          "is incompressible and lambda is not finite -- state mu and lambda directly");
    }
    return {E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))};
  }

  double poisson() const { return lame / (2.0 * (lame + shear)); }
  double young() const {
    return shear * (3.0 * lame + 2.0 * shear) / (lame + shear);
  }
  // the oedometer modulus: lam + 2mu in any dimension, since confinement
  // removes every lateral strain whatever d is
  double oedometer() const { return lame + 2.0 * shear; }
};

class CauchyElasticityModel {
 public:
  // WHICH DISCRETE STRESS HODGE, and only the two that are elements.
  //
  //   derham          d copies of the mimetic-BDM plus a rank-one volumetric
  //                   fold-back. Consistency-only: the scalar layer is
  //                   unisolvent, so N is square and nothing is stabilized.
  //                   Any cell type, either dimension.
  //   stabilized_afw  the same d^2 degrees of freedom reconstructed on the FULL
  //                   linear tensor space [P_1]^{dxd}, m = d^2(d+1) modes. On a
  //                   simplex D = m and the stabilization VANISHES -- there it
  //                   is the conforming AFW/BDM_1 element, checked by congruence
  //                   in exokal's hodge.test_afw_equivalence. On a polytope a
  //                   stabilization remains: 4 on a quadrilateral, 18 on a hex.
  //
  // The third realization exokal offers, derham_rt, is deliberately NOT here. It
  // is a sound inner product -- unisolvent, positive definite, exact on the
  // compliance energy -- and it is not an element: one constant traction vector
  // per facet cannot control the rigid rotations across a mesh, the inf-sup for
  // gamma degenerates, and the saddle point comes out singular. Offering it
  // would be offering a model that does not solve. See
  // tests/model/test_dimensions.cpp, which pins exactly that.
  using Realization = exokal::hodge::StressOperators::Realization;

  CauchyElasticityModel(const exokal::Mesh& mesh, int cell_dim, ElasticMaterial material,
                        Realization how = Realization::derham_afw)
      : mesh_(&mesh), dim_(cell_dim), material_(material), how_(how) {
    if (how != Realization::derham_afw && how != Realization::stabilized_afw) {
      throw std::invalid_argument(
          "CauchyElasticityModel: only the derham and stabilized_afw stress products are "
          "elements; derham_rt is unisolvent but its weak-symmetry inf-sup degenerates");
    }
  }

  MechanicsBoundary& mechanics() { return mechanics_; }
  const MechanicsBoundary& mechanics() const { return mechanics_; }

  int dim() const { return dim_; }
  Realization realization() const { return how_; }
  const char* realization_name() const { return exokal::hodge::StressOperators::name(how_); }
  const ElasticMaterial& material() const { return material_; }
  const Simulation& simulation() const { return *sim_; }
  const solver::SparseSystem& system() const { return system_; }
  const std::vector<double>& rhs() const { return rhs_; }
  std::size_t n_cells() const { return n_cells_; }
  // how many cells needed a stabilization: zero on a simplex mesh for either
  // realization, and that is the construction rather than a coincidence
  std::size_t n_stabilized() const { return stress_.n_stabilized(); }

  void build() {
    const graphos::Complex& c = mesh_->topology();
    // the K-independent mode selection, which only the de Rham product has
    const bool derham = how_ == Realization::derham_afw;
    if (derham) geometry_ = exokal::hodge::DeRhamGeometryCache::build(*mesh_, dim_);
    stress_ = exokal::hodge::StressOperators::build(*mesh_, dim_, material_.shear,
                                                    material_.lame, how_,
                                                    derham ? &geometry_ : nullptr);
    ctx_.provide("stress_operators", stress_);

    // THE SPACE FOLLOWS THE STAR: d^2 traction moments per facet for both of
    // these, read off the operators rather than restated, so the layout and the
    // product cannot drift apart.
    physics::ModelOptions o;
    o.traction_moments = stress_.moments_per_facet();
    sim_ = std::make_unique<Simulation>(
        physics::Catalogue::instance().build("linear_elasticity", o),
        std::vector<StratumSpec>{StratumSpec{"ambient", &c, dim_, 0}}, ctx_);

    // the conditions resolve against the space, which is what gives each of
    // them its dofs, and the strong ones then hand their forms to the
    // constraint set
    const auto& sp = sim_->epoch().stratum(0).space();
    mechanics_.resolve(*mesh_, dim_, sp);
    mechanics_.impose(sim_->constraints());
    sim_->freeze_constraints();

    exokal::forms::TripletSink jac(sim_->n_dofs());
    sim_->jacobian(jac);
    system_ = solver::SparseSystem::from(jac);

    // THE LOAD. Everything the terms contribute that does not depend on the
    // unknowns is a load, and the residual at the zero state is exactly minus
    // that -- so it is read off rather than re-derived. The strongly imposed
    // rows carry their own datum, scaled as their equation is.
    std::vector<double> r;
    sim_->state().assign(sim_->n_dofs(), 0.0);
    sim_->residual(r);
    rhs_.assign(sim_->n_dofs(), 0.0);
    for (std::size_t i = 0; i < sim_->n_dofs(); ++i) {
      rhs_[i] = sim_->constraints().pinned(i)
                    ? sim_->constraints().scale_at(i) * sim_->constraints().rhs_at(i)
                    : -r[i];
    }

    s_offset_ = static_cast<std::size_t>(sp.offset(sp.index_of("s_0")));
    u_offset_ = static_cast<std::size_t>(sp.offset(sp.index_of("u_0")));
    n_cells_ = static_cast<std::size_t>(c.count(dim_));
  }

  void accept(std::vector<double> x) { state_ = std::move(x); }
  const std::vector<double>& state() const { return state_; }

  // THE CELL DISPLACEMENT, in the sign and the scale a caller means by it.
  //
  // Two conversions, and both belong here rather than at every read site. The
  // unknown is the MOMENT of u over the cell, not a nodal value, so the mean is
  // that divided by the measure. And it carries the opposite SIGN to the
  // physical displacement: the mixed form is written [M, -B^T; +B, 0], so the
  // multiplier standing in the constitutive row is -u. That convention is not a
  // free choice once several physics land in one system -- two of them meeting
  // there would give a matrix neither symmetric nor antisymmetric -- so the
  // place to undo it is the accessor, once, and not in whatever measures a
  // result. Reading the raw dof gives an answer that is exactly twice the
  // solution away from it, which looks like a discretization error and is not.
  double displacement(Index cell, int axis) const {
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& mu = sp.map(sp.index_of("u_0"));
    return -state_[u_offset_ + static_cast<std::size_t>(mu.global(dim_, cell, 0, axis))] /
           exokal::measure(*mesh_, dim_, cell);
  }

  // THE NORMAL TRACTION on a facet, read through the same form a condition
  // would impose there: n . (sigma n), against the facet's CANONICAL normal and
  // divided by the measure it was integrated against.
  double normal_traction(Index facet) const {
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& ms = sp.map(sp.index_of("s_0"));
    const FacetFrame fr = FacetFrame::of(*mesh_, dim_, cofacet_of(*mesh_, dim_, facet), facet);
    double t = 0.0;
    for (int k = 0; k < dim_; ++k) {
      t += fr.normal[static_cast<std::size_t>(k)] *
           state_[s_offset_ + static_cast<std::size_t>(ms.global(dim_ - 1, facet, 0, k))];
    }
    return t / fr.measure;
  }

  // and the traction along an arbitrary direction, which a curved boundary
  // needs: e . (sigma n) with n canonical
  double traction_along(Index facet, const std::array<double, 3>& e) const {
    const auto& sp = sim_->epoch().stratum(0).space();
    const auto& ms = sp.map(sp.index_of("s_0"));
    const FacetFrame fr = FacetFrame::of(*mesh_, dim_, cofacet_of(*mesh_, dim_, facet), facet);
    double t = 0.0;
    for (int k = 0; k < dim_; ++k) {
      t += e[static_cast<std::size_t>(k)] *
           state_[s_offset_ + static_cast<std::size_t>(ms.global(dim_ - 1, facet, 0, k))];
    }
    return t / fr.measure;
  }

 private:
  const exokal::Mesh* mesh_;
  int dim_;
  ElasticMaterial material_;
  Realization how_;
  MechanicsBoundary mechanics_;

  exokal::hodge::DeRhamGeometryCache geometry_;
  exokal::hodge::StressOperators stress_;
  exokal::forms::TermContext ctx_;
  std::unique_ptr<Simulation> sim_;
  solver::SparseSystem system_;
  std::vector<double> rhs_, state_;
  std::size_t s_offset_{0}, u_offset_{0}, n_cells_{0};
};

}  // namespace mimetika
