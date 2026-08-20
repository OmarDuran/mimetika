#pragma once

#include <string>
#include <vector>

#include "exokal/ad/axpy.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/physics/package.hpp"

// LINEAR ELASTICITY in weakly-symmetric mixed form (Hellinger-Reissner).
//
//     r_sigma = M sigma - D^T u - A^T gamma
//     r_u     = + D sigma
//     r_gamma = + A sigma
//
// Three fields, because the symmetry of the stress is imposed WEAKLY: the
// space carries no symmetry constraint — that is what makes it usable — and
// the rotation gamma is the multiplier enforcing it against the rigid
// rotations. Dropping gamma does not simplify the method, it changes it.
//
// THE SIGN CONVENTION IS THE ONE THE FLUX TERM ALREADY USES: [M, -B^T; +B, 0],
// so an off-diagonal pair is the NEGATIVE transpose of its partner rather
// than its transpose. It is not a free choice once a model composes several
// physics — flow, mechanics and their coupling land in one system, and two
// conventions meeting there would produce a matrix that is neither symmetric
// nor antisymmetric and whose structure no solver could exploit.
//
// Every off-diagonal pair is written from the SAME operator, so the relation
// holds by construction rather than by two hand-written kernels agreeing.
//
// M, D and A are geometry and material alone. They are built once into
// StressOperators and read here, so the term is opaque to which stress
// product produced them.

namespace mimetika::physics {

namespace terms {

using exokal::forms::Params;
using exokal::forms::Stencil;
using exokal::forms::TermContext;

class MixedElasticityCell {
 public:
  MixedElasticityCell() = default;
  MixedElasticityCell(const Params&, const TermContext& ctx)
      : ops_(&ctx.require<exokal::hodge::StressOperators>("stress_operators")) {}

  // the term's own declaration order
  static constexpr std::size_t kS = 0;  // stress
  static constexpr std::size_t kU = 1;  // displacement
  static constexpr std::size_t kG = 2;  // rotation

  std::vector<std::string> fields() const { return {"s", "u", "g"}; }

  template <class T>
  void operator()(const Stencil& st, const std::vector<T>& a, std::vector<T>& r) const {
    const auto& S = st.field(kS);
    const auto& U = st.field(kU);
    const auto& G = st.field(kG);
    const auto& c = ops_->compact(st.support);
    const std::size_t D = S.end - S.begin;
    // A DIAGONAL STAR STAYS DIAGONAL: the compact cell carries only diag, and
    // reading M(i, j) here would materialize the dense zeros for every cell
    // -- exactly the storage the compact form exists to avoid. One multiply
    // per unknown, and the jacobian keeps the compact sparsity.
    const bool diagonal = !c.diag.empty();
    if (D != (diagonal ? c.diag.size() : c.M.rows())) {
      throw std::invalid_argument(
          "MixedElasticityCell: the stress block and the operators "
          "disagree on the degree-of-freedom count");
    }

    // The operators arrive in the ProductSpace's own degree-of-freedom order:
    // StressOperators permutes them once when it builds them, so the index
    // here is the index, and the inner loop is a straight contiguous walk of
    // one row of M rather than D gathers through a divide-and-modulus.
    // the DIMENSION comes from the operators, not from a constant. A
    // displacement has d components and a rotation d(d-1)/2 -- three in three
    // dimensions and ONE in two, where skew(2) is a line.
    const std::size_t nu = c.Dv.rows();
    const std::size_t ng = c.As.rows();
    for (std::size_t i = 0; i < D; ++i) {
      const std::size_t ri = S.begin + i;
      // M sigma
      if (diagonal) {
        exokal::axpy(r[ri], c.diag[i], a[S.begin + i]);
      } else {
        for (std::size_t j = 0; j < D; ++j) {
          exokal::axpy(r[ri], c.M(i, j), a[S.begin + j]);
        }
      }
      // -D^T u, with its adjoint in the displacement row
      for (std::size_t k = 0; k < nu; ++k) {
        exokal::axpy(r[ri], -c.Dv(k, i), a[U.begin + k]);
        exokal::axpy(r[U.begin + k], c.Dv(k, i), a[ri]);
      }
      // -A^T gamma, with its adjoint in the rotation row
      for (std::size_t k = 0; k < ng; ++k) {
        exokal::axpy(r[ri], -c.As(k, i), a[G.begin + k]);
        exokal::axpy(r[G.begin + k], c.As(k, i), a[ri]);
      }
    }
  }

 private:
  const exokal::hodge::StressOperators* ops_{nullptr};
};

inline const exokal::forms::RegisterTerm<MixedElasticityCell> register_mixed_elasticity{
    "mixed_elasticity_cell", exokal::forms::Coupling::closure, {"s", "u", "g"}};

// THE SAME PAIRING WITH THE TOTAL PRESSURE INDEPENDENT.
//
// p = lambda div u is carried as a field of its own, one scalar per cell, and
// the compliance is then lambda-free: a tr(sigma) is exactly p, so C^-1
// reduces to (2 mu)^-1 -- the same construction at a = 0. The two extra rows
// are exokal's:
//
//     sigma-row:  ... - (2 mu)^-1 T^T p
//     p-row:      (2 mu)^-1 T sigma - c_p |E| p = 0 ,  c_p = d/(2 mu) + 1/lambda
//
// WHY IT IS NOT AN OPTION BUT A DIFFERENT DISCRETIZATION: condensing p does not
// return the three-field operator. p is one scalar per cell, so the volumetric
// response is resolved to P0 and the Schur complement is a rank-one fold-back,
// where the three-field pairing has rank d+1. They agree only where tr sigma is
// constant on a cell. A cell-centred p is also what a two-point realization
// needs, which is why diagonal_afw exists only here.
class MixedElasticityTotalCell {
 public:
  MixedElasticityTotalCell() = default;
  // The compliance the four-field form keeps is (2 mu)^-1, and mu is the only
  // material number this term needs: lambda is gone from M and survives only
  // in c_p, which the operators already carry.
  MixedElasticityTotalCell(const Params& p, const TermContext& ctx)
      : ops_(&ctx.require<exokal::hodge::StressOperators>("stress_operators")),
        half_(1.0 / (2.0 * p.get("shear_modulus", 1.0))) {}

  static constexpr std::size_t kS = 0;  // stress
  static constexpr std::size_t kU = 1;  // displacement
  static constexpr std::size_t kG = 2;  // rotation
  static constexpr std::size_t kP = 3;  // total pressure

  std::vector<std::string> fields() const { return {"s", "u", "g", "p"}; }

  template <class T>
  void operator()(const Stencil& st, const std::vector<T>& a, std::vector<T>& r) const {
    const auto& S = st.field(kS);
    const auto& U = st.field(kU);
    const auto& G = st.field(kG);
    const auto& P = st.field(kP);
    const auto& c = ops_->compact(st.support);
    const std::size_t D = S.end - S.begin;
    // A DIAGONAL STAR STAYS DIAGONAL: the compact cell carries only diag, and
    // reading M(i, j) here would materialize the dense zeros for every cell
    // -- exactly the storage the compact form exists to avoid. One multiply
    // per unknown, and the jacobian keeps the compact sparsity.
    const bool diagonal = !c.diag.empty();
    if (D != (diagonal ? c.diag.size() : c.M.rows())) {
      throw std::invalid_argument(
          "MixedElasticityTotalCell: the stress block and the operators "
          "disagree on the degree-of-freedom count");
    }
    const std::size_t nu = c.Dv.rows();
    const std::size_t ng = c.As.rows();
    // the cell's own measure is on the operators, so no geometry is consulted
    const double half = half_;
    const double mass = ops_->hydrostatic_mass() * c.volume;

    for (std::size_t i = 0; i < D; ++i) {
      const std::size_t ri = S.begin + i;
      if (diagonal) {
        exokal::axpy(r[ri], c.diag[i], a[S.begin + i]);
      } else {
        for (std::size_t j = 0; j < D; ++j) {
          exokal::axpy(r[ri], c.M(i, j), a[S.begin + j]);
        }
      }
      for (std::size_t k = 0; k < nu; ++k) {
        exokal::axpy(r[ri], -c.Dv(k, i), a[U.begin + k]);
        exokal::axpy(r[U.begin + k], c.Dv(k, i), a[ri]);
      }
      for (std::size_t k = 0; k < ng; ++k) {
        exokal::axpy(r[ri], -c.As(k, i), a[G.begin + k]);
        exokal::axpy(r[G.begin + k], c.As(k, i), a[ri]);
      }
      // -(2 mu)^-1 T^T p, and its adjoint (2 mu)^-1 T sigma in the p row
      exokal::axpy(r[ri], -half * c.T(0, i), a[P.begin]);
      exokal::axpy(r[P.begin], half * c.T(0, i), a[ri]);
    }
    // and the cell's own mass, which closes the system: c_p |E| p
    exokal::axpy(r[P.begin], -mass, a[P.begin]);
  }

 private:
  const exokal::hodge::StressOperators* ops_{nullptr};
  double half_{0.5};
};

inline const exokal::forms::RegisterTerm<MixedElasticityTotalCell> register_mixed_elasticity_total{
    "mixed_elasticity_total_cell", exokal::forms::Coupling::closure, {"s", "u", "g", "p"}};

// STRONG SYMMETRY: the rigid-motion ansatz (exokal's vem_operators), TWO
// fields.
//
//     r_sigma = M sigma - Dv^T u
//     r_u     = + Dv sigma
//
// Symmetry lives in the reconstruction space, so there is no rotation
// multiplier: the displacement carries the whole of RM(E) -- six coefficients
// per cell in space, translations and rotations at once -- and Dv is the full
// rigid-motion pairing. The stress carries the six-component traction moment
// vector per facet whole rather than d copies of a scalar layout.
//
// The sign convention is the shared one, [M, -B^T; +B, 0], as for every other
// pairing in the catalogue.
class StrongElasticityCell {
 public:
  StrongElasticityCell() = default;
  StrongElasticityCell(const Params&, const TermContext& ctx)
      : ops_(&ctx.require<exokal::hodge::StressOperators>("stress_operators")) {}

  static constexpr std::size_t kS = 0;  // stress
  static constexpr std::size_t kU = 1;  // displacement, as RM coefficients

  std::vector<std::string> fields() const { return {"s", "u"}; }

  template <class T>
  void operator()(const Stencil& st, const std::vector<T>& a, std::vector<T>& r) const {
    const auto& S = st.field(kS);
    const auto& U = st.field(kU);
    const auto& c = ops_->compact(st.support);
    const std::size_t D = S.end - S.begin;
    // A DIAGONAL STAR STAYS DIAGONAL: the compact cell carries only diag, and
    // reading M(i, j) here would materialize the dense zeros for every cell
    // -- exactly the storage the compact form exists to avoid. One multiply
    // per unknown, and the jacobian keeps the compact sparsity.
    const bool diagonal = !c.diag.empty();
    if (D != (diagonal ? c.diag.size() : c.M.rows())) {
      throw std::invalid_argument(
          "StrongElasticityCell: the stress block and the operators "
          "disagree on the degree-of-freedom count");
    }
    const std::size_t nu = c.Dv.rows();
    for (std::size_t i = 0; i < D; ++i) {
      const std::size_t ri = S.begin + i;
      if (diagonal) {
        exokal::axpy(r[ri], c.diag[i], a[S.begin + i]);
      } else {
        for (std::size_t j = 0; j < D; ++j) {
          exokal::axpy(r[ri], c.M(i, j), a[S.begin + j]);
        }
      }
      for (std::size_t k = 0; k < nu; ++k) {
        exokal::axpy(r[ri], -c.Dv(k, i), a[U.begin + k]);
        exokal::axpy(r[U.begin + k], c.Dv(k, i), a[ri]);
      }
    }
  }

 private:
  const exokal::hodge::StressOperators* ops_{nullptr};
};

inline const exokal::forms::RegisterTerm<StrongElasticityCell> register_strong_elasticity{
    "strong_elasticity_cell", exokal::forms::Coupling::closure, {"s", "u"}};

// THE SAME ANSATZ WITH THE TOTAL PRESSURE INDEPENDENT: exokal's
// strong_symmetry_total, and the only formulation the diagonal member of the
// VEM family admits. The p rows are the ones the weak total form carries --
// -(2 mu)^-1 T^T p in the stress row, (2 mu)^-1 T sigma - c_p |E| p = 0 to
// close -- read off the same operators.
class StrongElasticityTotalCell {
 public:
  StrongElasticityTotalCell() = default;
  StrongElasticityTotalCell(const Params& p, const TermContext& ctx)
      : ops_(&ctx.require<exokal::hodge::StressOperators>("stress_operators")),
        half_(1.0 / (2.0 * p.get("shear_modulus", 1.0))) {}

  static constexpr std::size_t kS = 0;  // stress
  static constexpr std::size_t kU = 1;  // displacement, as RM coefficients
  static constexpr std::size_t kP = 2;  // total pressure

  std::vector<std::string> fields() const { return {"s", "u", "p"}; }

  template <class T>
  void operator()(const Stencil& st, const std::vector<T>& a, std::vector<T>& r) const {
    const auto& S = st.field(kS);
    const auto& U = st.field(kU);
    const auto& P = st.field(kP);
    const auto& c = ops_->compact(st.support);
    const std::size_t D = S.end - S.begin;
    // A DIAGONAL STAR STAYS DIAGONAL: the compact cell carries only diag, and
    // reading M(i, j) here would materialize the dense zeros for every cell
    // -- exactly the storage the compact form exists to avoid. One multiply
    // per unknown, and the jacobian keeps the compact sparsity.
    const bool diagonal = !c.diag.empty();
    if (D != (diagonal ? c.diag.size() : c.M.rows())) {
      throw std::invalid_argument(
          "StrongElasticityTotalCell: the stress block and the operators "
          "disagree on the degree-of-freedom count");
    }
    const std::size_t nu = c.Dv.rows();
    const double half = half_;
    const double mass = ops_->hydrostatic_mass() * c.volume;
    for (std::size_t i = 0; i < D; ++i) {
      const std::size_t ri = S.begin + i;
      if (diagonal) {
        exokal::axpy(r[ri], c.diag[i], a[S.begin + i]);
      } else {
        for (std::size_t j = 0; j < D; ++j) {
          exokal::axpy(r[ri], c.M(i, j), a[S.begin + j]);
        }
      }
      for (std::size_t k = 0; k < nu; ++k) {
        exokal::axpy(r[ri], -c.Dv(k, i), a[U.begin + k]);
        exokal::axpy(r[U.begin + k], c.Dv(k, i), a[ri]);
      }
      exokal::axpy(r[ri], -half * c.T(0, i), a[P.begin]);
      exokal::axpy(r[P.begin], half * c.T(0, i), a[ri]);
    }
    exokal::axpy(r[P.begin], -mass, a[P.begin]);
  }

 private:
  const exokal::hodge::StressOperators* ops_{nullptr};
  double half_{0.5};
};

inline const exokal::forms::RegisterTerm<StrongElasticityTotalCell>
    register_strong_elasticity_total{"strong_elasticity_total_cell",
                                     exokal::forms::Coupling::closure,
                                     {"s", "u", "p"}};

}  // namespace terms

struct MechanicsOptions {
  std::string term{"mixed_elasticity_cell"};
  // TRACTION MOMENTS PER FACET PER COMPONENT. Zero means d, which is the
  // mimetic-AFW space: d copies of the mimetic-BDM, so d^2 per facet -- the
  // traction vector AND its d-1 in-facet linear variations. One is d copies of
  // the mimetic-RT instead: a single CONSTANT traction vector per facet, d per
  // facet and nothing more.
  //
  // The package does not choose it. It lays out a space, and the layout has to
  // match whatever star lands on it; the driver derives both from one
  // realization so they cannot disagree.
  int traction_moments{0};
  // 1 for the wrench layout, d for the componentwise one; 0 means d
  int traction_components{0};
  // THE TOTAL PRESSURE AS A FIELD OF ITS OWN, which is exokal's
  // weak_symmetry_total. It adds one scalar per cell and changes the term, so
  // the package cannot infer it from the layout: the driver derives this and
  // the realization from one formulation, as it does for the moments.
  bool total_pressure{false};
  // STRONG SYMMETRY: six traction moments per facet carried whole, six
  // rigid-motion coefficients per cell for the displacement, no rotation
  // field. A different space, so a different layout, not a re-parametrized
  // one.
  bool strong_symmetry{false};
  double shear_modulus{1.0};
};

class Mechanics final : public Package {
 public:
  Mechanics() = default;
  explicit Mechanics(MechanicsOptions o) : opt_(std::move(o)) {}

  std::string name() const override { return "Mechanics"; }

  Requirements requirements(int dim, int codim = 0) const override {
    const auto at = [codim](std::string_view b) { return exokal::forms::field_at(b, codim); };
    Requirements r;
    if (opt_.strong_symmetry) {
      // the rigid-motion ansatz: the facet carries the six-component traction
      // moment vector WHOLE -- six moments of one scalar layout, not d copies
      // -- and the displacement is the six rigid-motion coefficients of the
      // cell, translations and rotations at once. No rotation field: the
      // symmetry lives in the reconstruction space.
      r.fields.push_back({at("s"), DofLayout::moments(dim, dim - 1, 6, 1)});
      r.fields.push_back({at("u"), DofLayout::cell_wise(dim, 1, 6)});
      if (opt_.total_pressure) {
        r.fields.push_back({at("p"), DofLayout::cell_wise(dim, 1, 1)});
      }
      r.provides = {"displacement", "momentum_balance", "stress"};
      if (opt_.total_pressure) r.provides.push_back("total_pressure");
      r.slots = {{"shear_modulus", Scope::rock}, {"poisson", Scope::rock}};
      return r;
    }
    // the stress carries part of the facet P_1 basis in each of d components:
    // all d of it for the BDM layer, so d^2 per facet, or the constant alone
    // for the RT layer, so d. The displacement and the rotation are cell-wise,
    // the rotation with d(d-1)/2 components -- and those do not move, because
    // the rigid rotations of a cell are the same however the traction on its
    // facets is measured.
    r.fields.push_back(
        {at("s"), DofLayout::moments(
                      dim, dim - 1, opt_.traction_moments > 0 ? opt_.traction_moments : dim,
                      opt_.traction_components > 0 ? opt_.traction_components : dim)});
    r.fields.push_back({at("u"), DofLayout::cell_wise(dim, 1, dim)});
    r.fields.push_back({at("g"), DofLayout::cell_wise(dim, 1, dim * (dim - 1) / 2)});
    // p = lambda div u, one scalar per cell: the volumetric response resolved
    // to P0 rather than to the trace of the stress space
    if (opt_.total_pressure) {
      r.fields.push_back({at("p"), DofLayout::cell_wise(dim, 1, 1)});
    }
    r.provides = {"displacement", "momentum_balance", "stress"};
    if (opt_.total_pressure) r.provides.push_back("total_pressure");
    r.slots = {{"shear_modulus", Scope::rock}, {"poisson", Scope::rock}};
    return r;
  }

  void attach(exokal::forms::Model& model, const exokal::forms::TermContext&) const override {
    exokal::forms::Params p;
    p.set("shear_modulus", opt_.shear_modulus);
    model.add(opt_.term, exokal::forms::On::all(), p);
  }

 private:
  MechanicsOptions opt_;
};

}  // namespace mimetika::physics
