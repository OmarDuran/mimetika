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
    const auto& c = ops_->cell(st.support);
    const std::size_t D = S.end - S.begin;
    if (D != c.M.rows()) {
      throw std::invalid_argument("MixedElasticityCell: the stress block and the operators "
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
      for (std::size_t j = 0; j < D; ++j) {
        exokal::axpy(r[ri], c.M(i, j), a[S.begin + j]);
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

}  // namespace terms

struct MechanicsOptions {
  std::string term{"mixed_elasticity_cell"};
};

class Mechanics final : public Package {
 public:
  Mechanics() = default;
  explicit Mechanics(MechanicsOptions o) : opt_(std::move(o)) {}

  std::string name() const override { return "Mechanics"; }

  Requirements requirements(int dim, int codim = 0) const override {
    const auto at = [codim](std::string_view b) { return exokal::forms::field_at(b, codim); };
    Requirements r;
    // the stress carries the facet P_1 basis in each of d components, so
    // d^2 per facet; the displacement and the rotation are cell-wise, the
    // rotation with d(d-1)/2 components
    r.fields.push_back({at("s"), DofLayout::moments(dim, dim - 1, dim, dim)});
    r.fields.push_back({at("u"), DofLayout::cell_wise(dim, 1, dim)});
    r.fields.push_back({at("g"), DofLayout::cell_wise(dim, 1, dim * (dim - 1) / 2)});
    r.provides = {"displacement", "momentum_balance", "stress"};
    r.slots = {{"shear_modulus", Scope::rock}, {"poisson", Scope::rock}};
    return r;
  }

  void attach(exokal::forms::Model& model, const exokal::forms::TermContext&) const override {
    model.add(opt_.term, exokal::forms::On::all());
  }

 private:
  MechanicsOptions opt_;
};

}  // namespace mimetika::physics
