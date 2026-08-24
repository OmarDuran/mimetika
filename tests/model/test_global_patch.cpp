#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "../mimetika_test.hpp"
#include "mimetika/linear_solver/petsc.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/cauchy_mechanics_model.hpp"
#include "mimetika/model/flow_model.hpp"

// The linear global patch test, pure Dirichlet, on fewer than ten cells.
//
// The one configuration both physics share: the exact field is prescribed as
// natural data on every boundary facet -- the pressure for the flow, the affine
// displacement for the mechanics -- nothing is strongly constrained, and the
// discrete answer must be the field itself. It is the smallest test that
// exercises the whole boundary machinery: the datum terms, the facet frames,
// the incidence signs, and the pairing they feed. A boundary condition that
// stops being applied fails here, on a mesh small enough to debug by hand.
//
// The field is a full affine map -- dilation, shear and rotation at once --
// because a pure dilation cannot see a rotation readback that broke, and an
// axis-aligned gradient cannot see a tangent frame that rotated. Every
// realization is asserted on its own claim: the consistent products
// everywhere, the two-point stars where the mesh is face-orthogonal with
// isotropic second moment, which the cartesian patch is.

using graphos::Index;
using mimetika::CauchyMechanicsModel;
using mimetika::ElasticMaterial;
using mimetika::FlowModel;
using mimetika::mesh::Family;

namespace {

constexpr double kMu = 1.3;
constexpr double kLam = 2.7;

// the affine gradient, deliberately with no zero and no symmetry
constexpr std::array<double, 9> kGrad = {0.4, 0.1,  -0.2,   //
                                         0.3, -0.5, 0.25,   //
                                         -0.15, 0.05, 0.6};

std::array<double, 9> gradient(int dim) {
  std::array<double, 9> g{};
  for (int i = 0; i < dim; ++i) {
    for (int j = 0; j < dim; ++j) {
      g[static_cast<std::size_t>(i * 3 + j)] = kGrad[static_cast<std::size_t>(i * 3 + j)];
    }
  }
  return g;
}

// a patch of fewer than ten cells, per family: 6 tetrahedra, 8 hexahedra,
// 8 prisms -- small enough to debug by hand, closed enough to have interior
// facets on every axis
exokal::Mesh patch_mesh(int dim, Family family) {
  switch (family) {
    case Family::simplex: return mimetika::mesh::box({1, 1, 1}, dim, family);
    case Family::prism: return mimetika::mesh::box({2, 2, 1}, dim, family);
    default: return mimetika::mesh::box({2, 2, 2}, dim, family);
  }
}

// ---- the flow half ----------------------------------------------------------

double flow_patch(int dim, Family family, FlowModel::Realization how,
                  bool affine_datum = true) {
  const exokal::Mesh m = patch_mesh(dim, family);
  CHECK(m.topology().count(dim) < 10);

  FlowModel prob(m, dim, 1.0, how);
  const std::array<double, 3> a{0.7, -0.4, dim == 3 ? 0.5 : 0.0};
  const auto p = [&](const exokal::Mesh::Point& x) {
    return a[0] * x[0] + a[1] * x[1] + a[2] * x[2];
  };
  // the exact pressure on every boundary facet, at its own centroid: pure
  // Dirichlet, naturally imposed, no facet strongly constrained
  // THE DATUM IS THE FIELD, not a number: value at the facet centroid AND the
  // gradient, which is what a facet carrying d flux moments tests against its
  // centred basis functions. `affine_datum = false` is the old one-number form,
  // kept because the difference between them is a test of its own.
  const std::array<double, 3> slope = affine_datum ? a : std::array<double, 3>{0.0, 0.0, 0.0};
  for (const Index f : mimetika::boundary_facets(m.topology(), dim)) {
    prob.flow().emplace<mimetika::PressureBC>(
        std::vector<Index>{f}, p(exokal::centroid(m, dim - 1, f)), slope);
  }
  prob.build();

  mimetika::solver::PetscSolver petsc;
  std::vector<double> x;
  const auto rep = petsc.solve(prob.system(), prob.rhs(), x);
  if (!rep.converged) throw std::runtime_error("flow patch: " + rep.reason);
  prob.accept(x);

  double worst = 0.0;
  for (Index e = 0; e < static_cast<Index>(prob.n_cells()); ++e) {
    worst = std::max(worst, std::abs(prob.cell_pressure(e) - p(exokal::centroid(m, dim, e))));
    // and the flux the pressure drives: q = -grad p, constant, in the space
    const auto q = prob.cell_flux(e);
    for (int k = 0; k < dim; ++k) {
      worst = std::max(worst, std::abs(q[static_cast<std::size_t>(k)] +
                                       a[static_cast<std::size_t>(k)]));
    }
  }
  return worst;
}

MIMETIKA_TEST(the_flow_patch_is_exact_where_the_datum_is_complete) {
  // one moment per facet: the constant the datum supplies IS the whole facet
  // trace, so nothing of a linear pressure is dropped and the answer is the
  // field. stabilized_rt and adaptive_rt everywhere; derham_rt where its
  // enrichment is consistent (see the pinned deficit below for the prism).
  using R = FlowModel::Realization;
  for (const int dim : {2, 3}) {
    for (const Family family : {Family::cartesian, Family::simplex, Family::prism}) {
      for (const R how : {R::stabilized_rt, R::adaptive_rt}) {
        CHECK(flow_patch(dim, family, how) < 1e-10);
      }
    }
    CHECK(flow_patch(dim, Family::cartesian, R::derham_rt) < 1e-10);
    CHECK(flow_patch(dim, Family::simplex, R::derham_rt) < 1e-10);
  }
  CHECK(flow_patch(2, Family::prism, R::derham_rt) < 1e-10);
}

// THE CHECK HAS FLIPPED, and the two halves are why the gradient exists.
//
// A BDM facet carries d flux moments and the row is int_f p_D (tau.n), so the
// datum is tested against d basis functions. The chart is equilibrated -- the
// facet Gram is |f| I -- so moment b takes (1/|f|) int_f p_D phi_b. The higher
// basis functions are CENTRED: blind to a constant, and seeing exactly the
// variation of p across the facet. So one number per facet is the whole datum
// at lowest order and is missing a term at BDM order, and the missing term is
// not small -- the patch it should reproduce exactly comes out at 1e-2 on a
// Kuhn box and 1e0 on a simplicial annulus, with the flux worse than the
// pressure. Given the gradient, every product is exact on every family.
//
// Writing the same NUMBER into every moment instead is not the fix and was
// measured: it destroys the Cartesian patch too (2e-15 -> 1e-1), the centred
// basis functions having no business seeing a constant.
MIMETIKA_TEST(the_bdm_flow_datum_needs_the_gradient_and_is_exact_with_it) {
  using R = FlowModel::Realization;
  for (const int dim : {2, 3}) {
    for (const R how : {R::derham_bdm, R::stabilized_bdm}) {
      const double affine = flow_patch(dim, Family::cartesian, how, true);
      const double number = flow_patch(dim, Family::cartesian, how, false);
      std::printf("  %dD %-16s affine datum %.2e   value alone %.2e\n", dim,
                  exokal::hodge::FluxOperators::name(how), affine, number);
      CHECK(affine < 1e-12);
      CHECK(number > 1e-3);
    }
  }
}

MIMETIKA_TEST(derham_rt_on_prisms_is_not_yet_consistent_in_the_plane) {
  // A pinned defect, exokal's: the curl-enriched derham_rt on a 3D prism
  // reproduces an axial linear pressure to round-off and loses an in-plane one
  // at O(1e-2), while stabilized_rt is exact in every direction on the same
  // mesh with the same datum -- so the datum path is exonerated and the
  // enriched product's consistency is what fails. This check flips the day the
  // enrichment is fixed, and the exactness claim above then absorbs the prism
  // row.
  using R = FlowModel::Realization;
  CHECK(flow_patch(3, Family::prism, R::derham_rt) > 1e-4);
  CHECK(flow_patch(3, Family::prism, R::stabilized_rt) < 1e-10);
}

MIMETIKA_TEST(the_flow_patch_is_exact_for_the_two_point_star_where_it_claims) {
  // the cartesian patch is K-orthogonal, which is the whole of TPFA's claim
  using R = FlowModel::Realization;
  for (const int dim : {2, 3}) {
    CHECK(flow_patch(dim, Family::cartesian, R::diagonal_tpfa) < 1e-10);
  }
}

// ---- the mechanics half -----------------------------------------------------

struct ElasticPatch {
  double u{0.0}, rot{0.0}, stress{0.0}, pressure{0.0};
};

ElasticPatch elastic_patch(int dim, Family family, CauchyMechanicsModel::Realization how,
                           CauchyMechanicsModel::Formulation form) {
  const exokal::Mesh m = patch_mesh(dim, family);
  CHECK(m.topology().count(dim) < 10);

  CauchyMechanicsModel model(m, dim, ElasticMaterial{kMu, kLam}, how, form);
  const std::array<double, 9> g = gradient(dim);
  // u = G (x - x_E) + G x_E on every boundary facet: the affine datum stated
  // about the facet's one cofacet, exactly as the external drivers state it
  for (const Index f : mimetika::boundary_facets(m.topology(), dim)) {
    const auto xE = exokal::centroid(m, dim, mimetika::cofacet_of(m, dim, f));
    std::array<double, 3> constant{};
    for (int i = 0; i < dim; ++i) {
      for (int j = 0; j < dim; ++j) {
        constant[static_cast<std::size_t>(i)] +=
            g[static_cast<std::size_t>(i * 3 + j)] * xE[static_cast<std::size_t>(j)];
      }
    }
    model.prescribe_displacement({f}, constant, g);
  }
  model.build();
  mimetika::solver::PetscSolver petsc;
  std::vector<double> x;
  const auto rep = petsc.solve(model.system(), model.rhs(), x);
  if (!rep.converged) throw std::runtime_error("elastic patch: " + rep.reason);
  model.accept(std::move(x));

  const double tr = g[0] + g[4] + g[8];
  ElasticPatch out;
  for (Index e = 0; e < static_cast<Index>(model.n_cells()); ++e) {
    const auto x_c = exokal::centroid(m, dim, e);
    for (int i = 0; i < dim; ++i) {
      double want = 0.0;
      for (int j = 0; j < dim; ++j) {
        want += g[static_cast<std::size_t>(i * 3 + j)] * x_c[static_cast<std::size_t>(j)];
      }
      out.u = std::max(out.u, std::abs(model.displacement(e, i) - want));
    }
    // rotation: skw(grad u), in the (i < j) generator order
    int p = 0;
    for (int i = 0; i < dim; ++i) {
      for (int j = i + 1; j < dim; ++j, ++p) {
        const double want = 0.5 * (g[static_cast<std::size_t>(i * 3 + j)] -
                                   g[static_cast<std::size_t>(j * 3 + i)]);
        out.rot = std::max(out.rot, std::abs(model.rotation(e, p) - want));
      }
    }
    // the full stress: sigma = 2 mu sym(G) + lam tr(G) I
    const auto s = model.cell_stress(e);
    for (int i = 0; i < dim; ++i) {
      for (int j = 0; j < dim; ++j) {
        const double sym = 0.5 * (g[static_cast<std::size_t>(i * 3 + j)] +
                                  g[static_cast<std::size_t>(j * 3 + i)]);
        const double want = 2.0 * kMu * sym + (i == j ? kLam * tr : 0.0);
        out.stress =
            std::max(out.stress, std::abs(s[static_cast<std::size_t>(i * 3 + j)] - want));
      }
    }
    if (form == CauchyMechanicsModel::Formulation::weak_symmetry_total ||
        form == CauchyMechanicsModel::Formulation::strong_symmetry_total) {
      out.pressure = std::max(out.pressure, std::abs(model.total_pressure(e) - kLam * tr));
    }
  }
  return out;
}

MIMETIKA_TEST(the_elastic_patch_is_exact_for_the_weak_family) {
  using R = CauchyMechanicsModel::Realization;
  using F = CauchyMechanicsModel::Formulation;
  for (const int dim : {2, 3}) {
    for (const Family family : {Family::cartesian, Family::simplex, Family::prism}) {
      for (const R how : {R::derham_bdm, R::stabilized_bdm}) {
        const ElasticPatch o = elastic_patch(dim, family, how, F::weak_symmetry);
        CHECK(o.u < 1e-10);
        CHECK(o.rot < 1e-10);
        CHECK(o.stress < 1e-9);
      }
      const ElasticPatch o = elastic_patch(dim, family, R::stabilized_bdm,
                                           F::weak_symmetry_total);
      CHECK(o.u < 1e-10);
      CHECK(o.pressure < 1e-9);
    }
  }
}

MIMETIKA_TEST(the_elastic_patch_is_exact_for_the_strong_family) {
  using R = CauchyMechanicsModel::Realization;
  using F = CauchyMechanicsModel::Formulation;
  for (const Family family : {Family::cartesian, Family::simplex, Family::prism}) {
    for (const F form : {F::strong_symmetry, F::strong_symmetry_total}) {
      const ElasticPatch o = elastic_patch(3, family, R::stabilized_vem, form);
      CHECK(o.u < 1e-10);
      CHECK(o.rot < 1e-10);
      CHECK(o.stress < 1e-9);
      if (form == F::strong_symmetry_total) CHECK(o.pressure < 1e-9);
    }
    // the derived selection at its default is the stabilized product
    const ElasticPatch a = elastic_patch(3, family, R::adaptive_vem, F::strong_symmetry_total);
    CHECK(a.u < 1e-10);
    CHECK(a.rot < 1e-10);
    CHECK(a.stress < 1e-9);
  }
}

// The boundary machinery, separated from the star's consistency: on a simplex
// the two-point members are inconsistent by their own claim, and a failed patch
// there says nothing about whether the data arrived. Two tests that a broken
// datum fails and an inconsistent-but-correctly-driven star cannot:
//
//   * a rigid motion has sigma = 0, so M sigma vanishes for any M: every
//     realization must reproduce it exactly on every family, and the only way
//     to miss it is to mis-state the boundary pairing.
//   * the load identity: the datum pairs the space, not the operator, so the
//     assembled right-hand side of the diagonal star must equal the
//     stabilized product's entry for entry on the same mesh with the same
//     data.
MIMETIKA_TEST(a_rigid_motion_is_exact_for_the_diagonal_star_on_every_family) {
  using R = CauchyMechanicsModel::Realization;
  using F = CauchyMechanicsModel::Formulation;
  const std::array<double, 9> w = {0.0, 0.3, -0.2, -0.3, 0.0, 0.5, 0.2, -0.5, 0.0};
  const std::array<double, 3> a = {0.1, -0.2, 0.05};
  for (const Family family : {Family::cartesian, Family::simplex, Family::prism}) {
    for (const R how : {R::diagonal_vem, R::adaptive_vem, R::stabilized_vem}) {
      const exokal::Mesh m = patch_mesh(3, family);
      CauchyMechanicsModel model(m, 3, ElasticMaterial{kMu, kLam}, how,
                                  F::strong_symmetry_total);
      for (const Index f : mimetika::boundary_facets(m.topology(), 3)) {
        const auto xE = exokal::centroid(m, 3, mimetika::cofacet_of(m, 3, f));
        std::array<double, 3> constant = a;
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            constant[static_cast<std::size_t>(i)] +=
                w[static_cast<std::size_t>(i * 3 + j)] * xE[static_cast<std::size_t>(j)];
          }
        }
        model.prescribe_displacement({f}, constant, w);
      }
      model.build();
      mimetika::solver::PetscSolver petsc;
      std::vector<double> x;
      const auto rep = petsc.solve(model.system(), model.rhs(), x);
      CHECK(rep.converged);
      model.accept(std::move(x));
      for (Index e = 0; e < static_cast<Index>(model.n_cells()); ++e) {
        const auto xc = exokal::centroid(m, 3, e);
        for (int i = 0; i < 3; ++i) {
          double want = a[static_cast<std::size_t>(i)];
          for (int j = 0; j < 3; ++j) {
            want += w[static_cast<std::size_t>(i * 3 + j)] * xc[static_cast<std::size_t>(j)];
          }
          CHECK(std::abs(model.displacement(e, i) - want) < 1e-12);
        }
        const auto s = model.cell_stress(e);
        for (int k = 0; k < 9; ++k) CHECK(std::abs(s[static_cast<std::size_t>(k)]) < 1e-12);
      }
    }
  }
}

MIMETIKA_TEST(the_diagonal_stars_load_is_the_stabilized_products_on_simplexes) {
  using R = CauchyMechanicsModel::Realization;
  using F = CauchyMechanicsModel::Formulation;
  const exokal::Mesh m = patch_mesh(3, Family::simplex);
  const std::array<double, 9> g = gradient(3);
  std::array<std::vector<double>, 2> rhs;
  int slot = 0;
  for (const R how : {R::diagonal_vem, R::stabilized_vem}) {
    CauchyMechanicsModel model(m, 3, ElasticMaterial{kMu, kLam}, how,
                                F::strong_symmetry_total);
    for (const Index f : mimetika::boundary_facets(m.topology(), 3)) {
      const auto xE = exokal::centroid(m, 3, mimetika::cofacet_of(m, 3, f));
      std::array<double, 3> constant{};
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          constant[static_cast<std::size_t>(i)] +=
              g[static_cast<std::size_t>(i * 3 + j)] * xE[static_cast<std::size_t>(j)];
        }
      }
      model.prescribe_displacement({f}, constant, g);
    }
    model.build();
    rhs[static_cast<std::size_t>(slot++)] = model.rhs();
  }
  CHECK(rhs[0].size() == rhs[1].size());
  double worst = 0.0;
  for (std::size_t i = 0; i < rhs[0].size(); ++i) {
    worst = std::max(worst, std::abs(rhs[0][i] - rhs[1][i]));
  }
  // entry for entry: the datum never consulted M, so a difference here is a
  // boundary-condition defect and nothing else
  CHECK(worst < 1e-14);
}

MIMETIKA_TEST(the_elastic_patch_is_exact_for_the_two_point_stars_where_they_claim) {
  // the cartesian patch is face-orthogonal with isotropic second moment,
  // which is the whole of the diagonal members' claim
  using R = CauchyMechanicsModel::Realization;
  using F = CauchyMechanicsModel::Formulation;
  for (const int dim : {2, 3}) {
    const ElasticPatch t = elastic_patch(dim, Family::cartesian, R::diagonal_afw,
                                         F::weak_symmetry_total);
    CHECK(t.u < 1e-10);
    CHECK(t.pressure < 1e-9);
  }
  const ElasticPatch v = elastic_patch(3, Family::cartesian, R::diagonal_vem,
                                       F::strong_symmetry_total);
  CHECK(v.u < 1e-10);
  CHECK(v.rot < 1e-10);
  CHECK(v.stress < 1e-9);
  CHECK(v.pressure < 1e-9);
}

}  // namespace

MIMETIKA_TEST_MAIN()
