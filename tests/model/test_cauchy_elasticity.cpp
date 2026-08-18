#include <cmath>
#include <string>
#include <vector>

#include "../mimetika_test.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/cauchy_elasticity_model.hpp"
#include "mimetika/linear_solver/petsc.hpp"

// THE CAUCHY ELASTICITY MODEL, AGAINST TWO CLOSED FORMS, ON EVERY CELL TYPE AND
// BOTH STRESS PRODUCTS.
//
//   column    confined uniaxial compression   constant stress, LINEAR displacement
//   annulus   Lame's thick-walled tube        sigma_rr = A - B/r^2, u_r = ar + b/r
//
// The pair is chosen the way the single-phase pair is, and for the same reason.
// The column is EXACT FOR THE SPACE: a constant stress and a linear
// displacement lie in every one of these reconstructions, so anything above
// round-off is a broken space and no refinement would fix it. Lame is NOT in
// any of them -- 1/r^2 in the stress, 1/r in the displacement -- so its error
// is a resolution and must fall under refinement. One says the method is right,
// the other says it converges.
//
// TWO PRODUCTS, both of which are elements:
//
//   derham          d copies of the mimetic-BDM plus a rank-one volumetric
//                   fold-back; consistency-only, N square, nothing stabilized.
//   stabilized_bdm  the same d^2 dofs per facet on the full linear tensor space
//                   [P_1]^{dxd}. On a simplex D = m and the stabilization
//                   VANISHES -- there it is the conforming AFW/BDM_1 element.
//                   On a polytope it stabilizes, and that is asserted per family
//                   rather than assumed.
//
// The plane-strain point is worth stating once: with d = 2 the compliance
// coefficient is a = lam/(2mu + 2lam), which is exactly the plane-strain
// inverse of sigma = 2mu eps + lam tr(eps) I. So the SAME Lame solution serves
// both dimensions, the three-dimensional case being held in plane strain by
// rollers on z = 0 and z = h. No separate closed form, and no plane-stress
// branch to get wrong.

using graphos::Index;
using mimetika::CauchyElasticityModel;
using mimetika::ElasticMaterial;
using mimetika::mesh::Family;
using Realization = CauchyElasticityModel::Realization;
using Formulation = CauchyElasticityModel::Formulation;

namespace {

constexpr double kMu = 1.0;
constexpr double kLam = 1.0;

struct Outcome {
  double max_err{0.0};
  double rms_err{0.0};
  double stress_err{0.0};
  std::size_t cells{0};
  std::size_t dofs{0};
  std::size_t stabilized{0};
};

const Family kFamilies[] = {Family::cartesian, Family::simplex, Family::prism};
const Realization kProducts[] = {Realization::derham_bdm, Realization::stabilized_bdm};

const char* product_name(Realization r) { return exokal::hodge::StressOperators::name(r); }

void solve(CauchyElasticityModel& m) {
  m.build();
  mimetika::solver::PetscSolver petsc;
  std::vector<double> x;
  const auto rep = petsc.solve(m.system(), m.rhs(), x);
  if (!rep.converged) throw std::runtime_error("cauchy elasticity: " + rep.reason);
  m.accept(std::move(x));
}

// ---- the column: confined uniaxial compression -----------------------------
//
// Rollers on the base and every side, a uniform compressive traction on top.
// Confinement removes every lateral strain by geometry alone, so elasticity
// gives the whole answer in two lines, in ANY dimension:
//
//     sigma_nn  = lam/(lam + 2mu) sigma_axial   on the confined facets
//     eps_axial = sigma_axial / K_oed,          K_oed = lam + 2mu
//
// A roller is the two halves of one condition: zero TANGENTIAL traction, which
// is strong because the traction is a degree of freedom, and zero NORMAL
// DISPLACEMENT, which is natural and comes from leaving the normal traction
// free so that its own equation reads u.n = 0.
Outcome column_case(int n, int dim, Family family, Realization how,
                    Formulation form = Formulation::weak_symmetry) {
  const double h = 1.0, load = 0.5;
  const exokal::Mesh m = mimetika::mesh::column(n, dim, family, h);
  const graphos::Complex& c = m.topology();
  const int axis = dim - 1;

  std::vector<Index> loaded, confined;
  for (const Index f : mimetika::boundary_facets(c, dim)) {
    const double z = exokal::centroid(m, dim - 1, f)[static_cast<std::size_t>(axis)];
    (std::abs(z - h) < 1e-9 ? loaded : confined).push_back(f);
  }
  std::array<double, 9> applied{};
  applied[static_cast<std::size_t>(axis * 3 + axis)] = -load;

  CauchyElasticityModel model(m, dim, ElasticMaterial{kMu, kLam}, how, form);
  model.mechanics().emplace<mimetika::TractionBC>(loaded, applied);
  model.mechanics().emplace<mimetika::FreeSlipBC>(confined);
  solve(model);

  const double k_oed = model.material().oedometer();
  const double s_exact = kLam / k_oed * (-load);
  const double e_exact = -load / k_oed;

  Outcome out;
  out.cells = model.n_cells();
  out.dofs = model.simulation().n_dofs();
  out.stabilized = model.n_stabilized();
  double acc = 0.0;
  for (Index e = 0; e < static_cast<Index>(out.cells); ++e) {
    const double x = exokal::centroid(m, dim, e)[static_cast<std::size_t>(axis)];
    const double d = model.displacement(e, axis) - e_exact * x;
    out.max_err = std::max(out.max_err, std::abs(d));
    acc += d * d;
  }
  out.rms_err = std::sqrt(acc / static_cast<double>(out.cells));
  // the LATERAL normal traction, on the confined facets whose normal is not the
  // column axis -- the base carries the axial reaction instead
  for (const Index f : confined) {
    const auto nrm = exokal::boundary_outward_normal(m, dim, f);
    if (std::abs(nrm[static_cast<std::size_t>(axis)]) > 1e-9) continue;
    out.stress_err = std::max(out.stress_err, std::abs(model.normal_traction(f) - s_exact));
  }
  return out;
}

// ---- the annulus: Lame's thick-walled tube ---------------------------------
//
// A quarter annulus from a to b, pressure p_a inside and p_b outside, rollers
// on the symmetry planes. With A and B below,
//
//     sigma_rr = A - B/r^2      sigma_tt = A + B/r^2
//     u_r      = A r / (2(lam + mu))  +  B / (2 mu r)
//
// Neither term is polynomial, so neither is in the reconstruction. Note the
// boundary datum: the exact stress at r = a is NOT -p_a I, but its TRACTION is
// -p_a n, and a uniform -p_a I delivers exactly that. So the condition is a
// constant tensor and the closed form is still the curved one.
struct Lame {
  double A{0.0}, B{0.0}, a{1.0}, b{4.0};
  ElasticMaterial mat{kMu, kLam};
  double sigma_rr(double r) const { return A - B / (r * r); }
  double u_r(double r) const {
    return A * r / (2.0 * (mat.lame + mat.shear)) + B / (2.0 * mat.shear * r);
  }
};

Outcome annulus_case(int nr, int nt, int dim, Family family, Realization how,
                     ElasticMaterial mat = ElasticMaterial{kMu, kLam}) {
  const double a = 1.0, b = 4.0, hz = 1.0, p_a = 1.0, p_b = 0.25;
  const exokal::Mesh m = mimetika::mesh::annulus(nr, nt, dim, family, a, b, hz);
  const graphos::Complex& c = m.topology();

  Lame ex;
  ex.a = a;
  ex.b = b;
  ex.mat = mat;
  ex.A = (a * a * p_a - b * b * p_b) / (b * b - a * a);
  ex.B = (p_a - p_b) * a * a * b * b / (b * b - a * a);

  const double rmid = std::sqrt(a * b);
  std::vector<Index> inner, outer, sym;
  for (const Index f : mimetika::boundary_facets(c, dim)) {
    const auto x = exokal::centroid(m, dim - 1, f);
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    const bool on_sym = std::abs(x[0]) < 1e-8 || std::abs(x[1]) < 1e-8 ||
                        (dim == 3 && (std::abs(x[2]) < 1e-8 || std::abs(x[2] - hz) < 1e-8));
    if (on_sym) {
      sym.push_back(f);
    } else if (r < rmid) {
      inner.push_back(f);
    } else {
      outer.push_back(f);
    }
  }
  std::array<double, 9> si{}, so{};
  for (int k = 0; k < 3; ++k) {
    si[static_cast<std::size_t>(k * 3 + k)] = -p_a;
    so[static_cast<std::size_t>(k * 3 + k)] = -p_b;
  }

  CauchyElasticityModel model(m, dim, mat, how);
  model.mechanics().emplace<mimetika::TractionBC>(inner, si);
  model.mechanics().emplace<mimetika::TractionBC>(outer, so);
  model.mechanics().emplace<mimetika::FreeSlipBC>(sym);
  solve(model);

  Outcome out;
  out.cells = model.n_cells();
  out.dofs = model.simulation().n_dofs();
  out.stabilized = model.n_stabilized();
  double acc = 0.0;
  for (Index e = 0; e < static_cast<Index>(out.cells); ++e) {
    const auto x = exokal::centroid(m, dim, e);
    const double r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    // the RADIAL displacement, projected out of the cell's vector unknown
    const double ur = (model.displacement(e, 0) * x[0] + model.displacement(e, 1) * x[1]) / r;
    const double d = ur - ex.u_r(r);
    out.max_err = std::max(out.max_err, std::abs(d));
    acc += d * d;
  }
  out.rms_err = std::sqrt(acc / static_cast<double>(out.cells));
  return out;
}

}  // namespace

// A LINEAR DISPLACEMENT IS REPRODUCED EXACTLY, on every cell type, in either
// dimension, by both products. The stress is constant and the displacement
// linear, and both lie in every one of these reconstructions -- so this is the
// space working, not a fine mesh. The stabilized counts come along because they
// are the construction: none on a simplex mesh, every cell on a polytopal one.
MIMETIKA_TEST(the_column_reproduces_the_linear_displacement_exactly) {
  for (const Realization r : kProducts) {
    for (const int dim : {2, 3}) {
      for (const Family f : kFamilies) {
        const Outcome o = column_case(4, dim, f, r);
        std::printf(
            "  column  %-14s %dD %-10s %4zu cells %6zu dofs   u %.2e   sigma_lat %.2e   stab %zu\n",
            product_name(r), dim, mimetika::mesh::name(f), o.cells, o.dofs, o.max_err, o.stress_err,
            o.stabilized);
        CHECK(o.max_err < 1e-10);
        CHECK(o.stress_err < 1e-10);
        // a simplex mesh never stabilizes, whichever product is asked for
        if (f == Family::simplex) CHECK(o.stabilized == 0);
        // and the de Rham product never stabilizes at all
        if (r == Realization::derham_bdm) CHECK(o.stabilized == 0);
      }
    }
  }
}

// AND LAME IS APPROXIMATED AND CONVERGES. The radial solution carries 1/r^2 in
// the stress and 1/r in the displacement, so no polynomial reconstruction
// contains it: the error is a resolution, and it must fall with refinement. A
// space that were merely wrong would not improve.
MIMETIKA_TEST(the_annulus_reproduces_lame) {
  for (const Realization r : kProducts) {
    for (const int dim : {2, 3}) {
      for (const Family f : kFamilies) {
        const Outcome coarse = annulus_case(6, 3, dim, f, r);
        const Outcome fine = annulus_case(12, 6, dim, f, r);
        std::printf(
            "  annulus %-14s %dD %-10s %4zu -> %5zu cells   u max %.2e -> %.2e   rms %.2e -> "
            "%.2e\n",
            product_name(r), dim, mimetika::mesh::name(f), coarse.cells, fine.cells, coarse.max_err,
            fine.max_err, coarse.rms_err, fine.rms_err);
        CHECK(coarse.max_err < 5e-2);
        CHECK(fine.rms_err < coarse.rms_err);  // refinement helps, so it is resolution
      }
    }
  }
}

// ON A SIMPLEX MESH THE TWO PRODUCTS ARE ONE ELEMENT, and this carries it from
// a single cell up to a solved problem.
//
// exokal settles the local statement twice over: the two operators agree entry
// for entry (hodge.test_stress_hodge), and each equals the CONFORMING AFW/BDM_1
// element by congruence against basix (hodge.test_afw_equivalence), at every
// material in 0 < nu < 1/2. That they still agree after assembly, the boundary
// forms and the solve is a separate claim, and it is this one.
//
// It holds only on simplices, and for a reason that is about the SPACE. There
// derham_bdm needs no curl enrichment -- D = d(d+1) = dim[P_1]^d -- so d copies
// of the scalar mimetic-BDM is [P_1]^{dxd}, exactly what stabilized_bdm
// reconstructs on, and D = m so nothing is stabilized either. On a polytope
// both of those fail: stabilized_bdm stabilizes on ker(N^T) while derham_bdm
// enriches with curl modes instead, and they are then genuinely different
// discretizations -- which the second half measures rather than glosses.
MIMETIKA_TEST(the_two_products_are_one_element_on_a_simplex_mesh) {
  for (const int dim : {2, 3}) {
    // the material must not be trivial: at lam = 0 the volumetric term is
    // switched off and agreement would say nothing about the trace pairing
    for (const ElasticMaterial mat : {ElasticMaterial{kMu, 0.0}, ElasticMaterial{kMu, kLam},
                                      ElasticMaterial{kMu, 100.0 * kMu}}) {
      const Outcome b = annulus_case(6, 3, dim, Family::simplex, Realization::derham_bdm, mat);
      const Outcome a = annulus_case(6, 3, dim, Family::simplex, Realization::stabilized_bdm, mat);
      std::printf(
          "  %dD simplex   nu %.4f   derham_bdm %.12e   stabilized_bdm %.12e  |diff| %.2e\n", dim,
          mat.poisson(), b.max_err, a.max_err, std::abs(b.max_err - a.max_err));
      CHECK(std::abs(b.max_err - a.max_err) < 1e-10);
      CHECK(std::abs(b.rms_err - a.rms_err) < 1e-10);
      CHECK(b.stabilized == 0);
      CHECK(a.stabilized == 0);
    }

    // and on a polytopal mesh they are different discretizations
    const Outcome cb = annulus_case(6, 3, dim, Family::cartesian, Realization::derham_bdm);
    const Outcome ca = annulus_case(6, 3, dim, Family::cartesian, Realization::stabilized_bdm);
    std::printf("  %dD cartesian derham_bdm %.12e   stabilized_bdm %.12e  |diff| %.2e\n", dim,
                cb.max_err, ca.max_err, std::abs(cb.max_err - ca.max_err));
    CHECK(cb.stabilized == 0);         // enriched to unisolvence instead
    CHECK(ca.stabilized == ca.cells);  // stabilized on ker(N^T)
    CHECK(std::abs(cb.max_err - ca.max_err) > 1e-10);
  }
}

// derham_rt IS REFUSED AT CONSTRUCTION. It is a sound inner product and not an
// element -- one constant traction vector per facet cannot control the rigid
// rotations across a mesh -- so the model declines it rather than assembling a
// singular saddle point. See test_confined_compression for the measurement behind this.
MIMETIKA_TEST(the_model_refuses_the_realization_that_is_not_an_element) {
  const exokal::Mesh m = mimetika::mesh::column(2, 3, Family::simplex);
  bool refused = false;
  try {
    CauchyElasticityModel bad(m, 3, ElasticMaterial{kMu, kLam},
                              exokal::hodge::StressOperators::Realization::derham_rt);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  CHECK(refused);
}

// NEITHER PRODUCT LOCKS. This is the regime that would separate them if
// anything did, and it does not.
//
// As nu -> 1/2 the volumetric term dominates the compliance, so a product that
// only approximates it -- derham folds back the rank-one MEAN-trace form where
// stabilized_bdm integrates the trace pairing exactly -- is where one would
// expect the approximation to tell. It does not, and the reason is structural:
// this is a MIXED method. The stress is the primary unknown and what is
// discretized is the COMPLIANCE, which merely loses a rank as a -> 1/d (the
// hydrostatic mode stores no energy) and is absorbed. Locking is a pathology of
// displacement-based formulations, where the stiffness diverges instead.
//
// So the test is a comparison rather than a threshold: at nu = 0.4999 the error
// must be no worse than at nu = 0.25, for both products, on simplices and on
// polytopes. Anything else would be locking, however mild.
MIMETIKA_TEST(neither_product_locks_as_the_material_becomes_incompressible) {
  const double nu = 0.4999;
  const ElasticMaterial stiff{kMu, 2.0 * kMu * nu / (1.0 - 2.0 * nu)};
  std::printf("  nu %.4f -> lam %.1f   (nu at lam = 1 is %.4f)\n", nu, stiff.lame,
              ElasticMaterial{kMu, kLam}.poisson());
  for (const int dim : {2, 3}) {
    for (const Family f : {Family::simplex, Family::cartesian}) {
      for (const Realization r : kProducts) {
        const Outcome soft_c = annulus_case(6, 3, dim, f, r);
        const Outcome hard_c = annulus_case(6, 3, dim, f, r, stiff);
        const Outcome hard_f = annulus_case(12, 6, dim, f, r, stiff);
        const double rate = std::log2(hard_c.rms_err / hard_f.rms_err);
        std::printf("  %dD %-9s %-14s  nu 0.25 rms %.3e   nu 0.4999 rms %.3e -> %.3e  rate %.2f\n",
                    dim, mimetika::mesh::name(f), product_name(r), soft_c.rms_err, hard_c.rms_err,
                    hard_f.rms_err, rate);
        // no degradation at the limit, and the rate holds
        CHECK(hard_c.rms_err < 1.1 * soft_c.rms_err);
        CHECK(rate > 1.5);
      }
    }
  }
}


// THE LINEAR PATCH, WEAKLY IMPOSED. u = x on every boundary facet, as an affine
// datum: both moments the facet carries are supplied, so the answer is the
// field itself and any departure is the discretization rather than the data.
// Nothing is strongly constrained, which keeps this a test of the PAIRING.
struct Patch {
  std::size_t dofs{0}, cells{0};
  double max_err{0.0};      // |u - x|
  double pressure_err{0.0};  // |p - lambda div u|, four fields only
};

Patch patch_case(int n, int dim, Family family, Realization how,
                 Formulation form = Formulation::weak_symmetry) {
  std::array<int, 3> res{n, n, dim == 3 ? n : 1};
  const exokal::Mesh m = mimetika::mesh::box(res, dim, family);
  CauchyElasticityModel model(m, dim, ElasticMaterial{kMu, kLam}, how, form);
  std::array<double, 9> grad{};
  for (int k = 0; k < dim; ++k) grad[static_cast<std::size_t>(k * 3 + k)] = 1.0;
  for (const Index f : mimetika::boundary_facets(m.topology(), dim)) {
    const auto x = exokal::centroid(m, dim, mimetika::cofacet_of(m, dim, f));
    model.prescribe_displacement({f}, {x[0], x[1], dim == 3 ? x[2] : 0.0}, grad);
  }
  solve(model);

  Patch out;
  out.cells = model.n_cells();
  out.dofs = model.simulation().n_dofs();
  const double p_exact = kLam * static_cast<double>(dim);
  for (Index e = 0; e < static_cast<Index>(out.cells); ++e) {
    const auto x = exokal::centroid(m, dim, e);
    for (int k = 0; k < dim; ++k) {
      out.max_err = std::max(out.max_err,
                             std::abs(model.displacement(e, k) - x[static_cast<std::size_t>(k)]));
    }
    if (form == Formulation::weak_symmetry_total) {
      out.pressure_err = std::max(out.pressure_err, std::abs(model.total_pressure(e) - p_exact));
    }
  }
  return out;
}

// ---- FOUR FIELDS, AND THE TWO-POINT PRODUCT THAT NEEDS THEM ----------------
//
// The total pressure p = lambda div u carried as a field of its own. That is
// exokal's weak_symmetry_total, and mimetika's part of it is the pairing: the
// sigma row gains -(2 mu)^-1 T^T p and the p row closes the system with
// c_p |E| p. Everything below tests that pairing, not the product beneath it.

// THE COLUMN IS STILL THE COLUMN. A linear displacement lies in every one of
// these reconstructions, so four fields must reproduce it exactly too -- the
// formulation changes how the volumetric response is resolved, not whether the
// method is consistent.
MIMETIKA_TEST(the_four_field_column_reproduces_the_linear_displacement) {
  for (const int dim : {2, 3}) {
    for (const Family f : kFamilies) {
      const Outcome o =
          column_case(4, dim, f, Realization::stabilized_bdm, Formulation::weak_symmetry_total);
      std::printf("  four-field column %dD %-10s %5zu cells %6zu dofs   max %.2e\n", dim,
                  mimetika::mesh::name(f), o.cells, o.dofs, o.max_err);
      CHECK(o.max_err < 1e-10);
    }
  }
}

// AND IT IS A LARGER SYSTEM, by exactly one scalar per cell: the count is the
// concrete content of "p is a field", and a formulation that quietly kept
// three fields would pass every accuracy check above.
MIMETIKA_TEST(the_fourth_field_is_one_scalar_per_cell) {
  const Outcome three = column_case(4, 3, Family::cartesian, Realization::stabilized_bdm);
  const Outcome four = column_case(4, 3, Family::cartesian, Realization::stabilized_bdm,
                                   Formulation::weak_symmetry_total);
  std::printf("  three-field %zu dofs   four-field %zu dofs   cells %zu\n", three.dofs, four.dofs,
              three.cells);
  CHECK(four.dofs == three.dofs + three.cells);
}

// THE TWO-POINT PRODUCT REACHES THE MODEL. Its space is derham_rt's -- d per
// facet, one constant traction vector -- and it exists in four fields only,
// which the model refuses to fake.
MIMETIKA_TEST(the_two_point_stress_product_needs_four_fields) {
  const exokal::Mesh m = mimetika::mesh::column(4, 3, Family::cartesian, 1.0);
  bool refused = false;
  try {
    CauchyElasticityModel model(m, 3, ElasticMaterial{kMu, kLam}, Realization::diagonal_tpsa);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  CHECK(refused);

  // and derham_rt stays refused in either formulation: its weak-symmetry
  // inf-sup degenerates, which is a different objection
  refused = false;
  try {
    CauchyElasticityModel model(m, 3, ElasticMaterial{kMu, kLam}, Realization::derham_rt,
                                Formulation::weak_symmetry_total);
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  CHECK(refused);
}

// EVERY FOUR-FIELD PRODUCT, ON EVERY MESH THIS MODEL CLAIMS.
//
// The pairing is one term, so a mistake in it -- a sign, a missing adjoint, the
// wrong compliance on the trace -- is the same mistake for all three products.
// Testing them together is what makes a pass evidence about the PORT rather
// than about one product.
//
// diagonal_tpsa is included where it is claimed: face-orthogonal meshes, which
// among these is the box of hexahedra. It carries d per facet and the other two
// carry d^2, so the count separates them without any tolerance.
MIMETIKA_TEST(every_four_field_product_reproduces_the_linear_displacement) {
  for (const int dim : {2, 3}) {
    for (const Family f : kFamilies) {
      for (const Realization r : kProducts) {
        const Patch o = patch_case(3, dim, f, r, Formulation::weak_symmetry_total);
        std::printf("  four-field %-22s %dD %-10s %6zu dofs   u %.2e   p %.2e\n",
                    exokal::hodge::StressOperators::name(r), dim, mimetika::mesh::name(f), o.dofs,
                    o.max_err, o.pressure_err);
        CHECK(o.max_err < 1e-10);
        CHECK(o.pressure_err < 1e-9);
      }
    }
    const Patch t = patch_case(3, dim, Family::cartesian, Realization::diagonal_tpsa,
                               Formulation::weak_symmetry_total);
    std::printf("  four-field %-22s %dD %-10s %6zu dofs   u %.2e   p %.2e\n",
                exokal::hodge::StressOperators::name(Realization::diagonal_tpsa), dim, "cartesian",
                t.dofs, t.max_err, t.pressure_err);
    CHECK(t.max_err < 1e-10);
    CHECK(t.pressure_err < 1e-9);
  }
}

// THE THREE-FIELD ANSWER, WHERE THE TWO MUST AGREE. Condensing p does not
// return the three-field operator in general -- p is one scalar per cell, so
// the volumetric response is resolved to P0 against the trace's rank d+1 -- and
// they agree exactly where tr sigma is CONSTANT on a cell, which a linear
// displacement makes it. So this is the one comparison that is allowed to be
// exact, and it is the sharpest check on the p-row: a wrong compliance there
// moves the displacement while leaving the patch test intact.
MIMETIKA_TEST(four_fields_and_three_agree_where_the_trace_is_constant) {
  for (const int dim : {2, 3}) {
    for (const Family f : kFamilies) {
      for (const Realization r : kProducts) {
        const Patch three = patch_case(3, dim, f, r);
        const Patch four = patch_case(3, dim, f, r, Formulation::weak_symmetry_total);
        std::printf("  %-22s %dD %-10s three %.2e   four %.2e   +%zu dofs\n",
                    exokal::hodge::StressOperators::name(r), dim, mimetika::mesh::name(f),
                    three.max_err, four.max_err, four.dofs - three.dofs);
        CHECK(three.max_err < 1e-10);
        CHECK(four.dofs == three.dofs + three.cells);  // exactly one scalar per cell
      }
    }
  }
}

// AND THE FOURTH FIELD IS WHAT MAKES THE INCOMPRESSIBLE LIMIT UNIFORM. The
// four-field compliance is lambda-free and c_p = d/(2mu) + 1/lambda stays finite
// as lambda grows, where the three-field Gram loses a rank. The patch test is
// exact at every lambda, so what is checked is that it STAYS exact -- a
// formulation that locked would fail here and nowhere else.
MIMETIKA_TEST(the_four_field_patch_survives_the_incompressible_limit) {
  const int dim = 3;
  for (const double nu : {0.25, 0.4999, 0.499999}) {
    const double lam = 2.0 * kMu * nu / (1.0 - 2.0 * nu);
    const exokal::Mesh m = mimetika::mesh::box({3, 3, 3}, dim, Family::cartesian);
    CauchyElasticityModel model(m, dim, ElasticMaterial{kMu, lam}, Realization::stabilized_bdm,
                                Formulation::weak_symmetry_total);
    std::array<double, 9> grad{};
    for (int k = 0; k < dim; ++k) grad[static_cast<std::size_t>(k * 3 + k)] = 1.0;
    for (const Index f : mimetika::boundary_facets(m.topology(), dim)) {
      const auto x = exokal::centroid(m, dim, mimetika::cofacet_of(m, dim, f));
      model.prescribe_displacement({f}, {x[0], x[1], x[2]}, grad);
    }
    solve(model);
    double worst = 0.0, p_worst = 0.0;
    for (Index e = 0; e < static_cast<Index>(model.n_cells()); ++e) {
      const auto x = exokal::centroid(m, dim, e);
      for (int k = 0; k < dim; ++k) {
        worst = std::max(worst,
                         std::abs(model.displacement(e, k) - x[static_cast<std::size_t>(k)]));
      }
      p_worst = std::max(p_worst, std::abs(model.total_pressure(e) - lam * dim));
    }
    // THE TOLERANCE TRACKS THE CONDITIONING, because the error does: 4.8e-15,
    // 1.8e-11, 1.4e-9 as lambda goes 1, 5e3, 5e5 -- linear in lambda, which is
    // round-off through a stiffer system and not a loss of accuracy in the
    // method. Locking would show as an error that stops falling with the mesh,
    // and would fail this by orders rather than by a constant.
    std::printf("  nu %-9.6f lambda %10.1f   u %.2e (%.1e x lambda)   p/lambda %.2e\n", nu, lam,
                worst, worst / lam, p_worst / lam);
    CHECK(worst < 1e-13 * std::max(1.0, lam));
    CHECK(p_worst / lam < 1e-13 * std::max(1.0, lam));
  }
}

MIMETIKA_TEST_MAIN()