#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "../mimetika_test.hpp"
#include "exokal/hodge/stress_operators.hpp"
#include "mimetika/linear_solver/petsc.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/boundary.hpp"
#include "mimetika/model/boundary_conditions.hpp"
#include "mimetika/model/compositions/cauchy_mechanics.hpp"
#include "mimetika/model/mixed_dimensional.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/physics/contact_interface.hpp"
#include "mimetika/physics/contact_mechanics.hpp"

// THE PATCH TEST OF THE ASSEMBLED 4 x 4, and the first check that the system is
// a system rather than five blocks that each look right.
//
//     [ A   B^T  0    T^T ] [sigma]   [ ]
//     [ B   0    0    0   ] [u    ] = [f]
//     [ 0   0    A_c  J   ] [t_c  ]   [ ]
//     [ T   0    J^T  0   ] [u_c  ]   [ ]
//
// A BONDED FRACTURE IS INVISIBLE. Set the compliance to zero and the third row
// reads [[u_c]] = 0: the two sides move together, the fourth row makes the
// multiplier t_c the traction both sides transmit, and the cut problem is the
// uncut problem written in more unknowns. So the answer must be the uncut one,
// to round-off, and not merely close to it.
//
// WHY THIS AND NOT A BLOCK CHECK. Every earlier test on these terms asserts a
// pattern -- where entries land, that T^T is T transposed, that A_c and J are
// the matrices the chart supplies. None of them can see a GLOBAL SIGN: flip T
// and J together and every one still passes, because the flip is absorbed by
// u_c -> -u_c. What no relabelling absorbs is the relative sign of the two
// sides in T against the difference in J. Get it wrong and a bonded interface
// either transmits the wrong side's traction or goes singular, and only solving
// it shows that.
//
// The uncut reference is not taken on faith: confined compression is exact in
// tests/model/test_confined_compression.cpp and the affine patch in
// tests/model/test_global_patch.cpp, so agreeing with it is an absolute claim.
// The closed form is checked here too, directly, for the family whose dofs can
// be read without a projection.

using exokal::hodge::DeRhamGeometryCache;
using exokal::hodge::StressOperators;
using graphos::Index;
using mimetika::Simulation;
using mimetika::StratumSpec;
using mimetika::mesh::Family;
using mimetika::physics::Catalogue;
using mimetika::physics::Composition;

namespace {

constexpr double kMu = 1.0, kLam = 1.0, kLoad = 1.0;
constexpr int kAxis = 2;  // confined uniaxial compression along z

// the bonded interface is the limit c_n -> infinity, and closes by nothing
double c_n_or_inf(double c_n) { return c_n > 0.0 ? c_n : std::numeric_limits<double>::infinity(); }

struct Cut {
  exokal::Mesh mesh;
  exokal::MixedDimensionalMesh md;
};

// The unit cube, n cells per side, cut by the plane z = 1/2 -- across the load,
// so the fracture carries the whole axial traction and a lost sign cannot hide
// in a direction nothing pushes on.
Cut cut_cube(int n, Family family) {
  exokal::Mesh mesh = mimetika::mesh::box({n, n, n}, 3, family);
  std::vector<Index> nodes;
  for (Index v = 0; v < mesh.topology().count(0); ++v) {
    if (std::fabs(mesh.point(v)[kAxis] - 0.5) < 1e-12) nodes.push_back(v);
  }
  exokal::MixedDimensionalMesh md = exokal::mixed_dimensional_mesh(mesh, {nodes});
  return Cut{std::move(mesh), std::move(md)};
}

struct Family_ {
  StressOperators::Realization how;
  StressOperators::Formulation form;
  bool strong;
  const char* name;
};

const Family_ kWeak{StressOperators::Realization::derham_bdm,
                    StressOperators::Formulation::weak_symmetry, false, "weak"};
const Family_ kStrong{StressOperators::Realization::stabilized_vem,
                      StressOperators::Formulation::strong_symmetry, true, "strong"};

// Confined uniaxial compression: a uniform axial traction on the top facets,
// rollers everywhere else. Stated the same way on the cut mesh and the uncut
// one, so the two answers are comparable.
//
// Through the BOUNDARY CONDITIONS and not the impose_* helpers: the wrench
// carries a facet's tangential traction in three of its six slots rather than
// in components, so "the tangential part" is a different set of dofs in the
// two families and only the conditions know which. The helpers resolve a
// direction against components, which on one component is three copies of the
// same row -- the constraint set rejects it, as it should.
void load(Simulation& sim, mimetika::MechanicsBoundary& bcs,
          const exokal::spaces::ProductSpace& sp, const exokal::Mesh& m,
          const std::vector<Index>& boundary, Index offset) {
  std::vector<Index> loaded, confined;
  for (const Index f : boundary) {
    const auto x = exokal::centroid(m, 2, f);
    (std::fabs(x[kAxis] - 1.0) < 1e-9 ? loaded : confined).push_back(f);
  }
  std::array<double, 9> applied{};
  applied[kAxis * 3 + kAxis] = -kLoad;
  bcs.emplace<mimetika::TractionBC>(std::move(loaded), applied);
  bcs.emplace<mimetika::FreeSlipBC>(std::move(confined));
  bcs.resolve(m, 3, sp, offset, false);
  bcs.impose(sim.constraints());
}

std::vector<double> solve(Simulation& sim, const char* what) {
  exokal::forms::TripletSink jac(sim.n_dofs());
  sim.jacobian(jac);
  const auto A = mimetika::solver::SparseSystem::from(jac);
  std::vector<double> b(sim.n_dofs(), 0.0), x;
  for (std::size_t i = 0; i < sim.n_dofs(); ++i) {
    if (sim.constraints().pinned(i)) {
      b[i] = sim.constraints().scale_at(i) * sim.constraints().rhs_at(i);
    }
  }
  mimetika::solver::PetscSolver petsc;
  const auto rep = petsc.solve(A, b, x);
  if (!rep.converged) throw std::runtime_error(std::string(what) + ": " + rep.reason);
  return x;
}

mimetika::physics::ModelOptions bulk_options(const StressOperators& ops, const Family_& f) {
  mimetika::physics::ModelOptions o;
  o.strong_symmetry = f.strong;
  o.traction_moments = ops.moments_per_facet();
  o.shear_modulus = kMu;
  return o;
}

// ---- the uncut reference ---------------------------------------------------
struct Reference {
  std::vector<double> x;
  const exokal::spaces::ProductSpace* space{nullptr};
};

std::vector<double> uncut(const exokal::Mesh& m, const Family_& f,
                          std::vector<double>& u_out, std::vector<double>& s_out) {
  const graphos::Complex& c = m.topology();
  DeRhamGeometryCache geo;
  if (!f.strong) geo = DeRhamGeometryCache::build(m, 3);
  const StressOperators ops =
      StressOperators::build(m, 3, kMu, kLam, f.how, f.form, f.strong ? nullptr : &geo);
  exokal::forms::TermContext ctx;
  ctx.provide("stress_operators", ops);

  const Composition comp = Catalogue::instance().build("linear_elasticity", bulk_options(ops, f));
  Simulation sim(comp, {StratumSpec{"ambient", &c, 3, 0}}, ctx);
  const auto& sp = sim.epoch().stratum(0).space();
  mimetika::MechanicsBoundary bcs;
  load(sim, bcs, sp, m, mimetika::boundary_facets(c, 3), 0);
  sim.freeze_constraints();
  std::vector<double> x = solve(sim, "uncut");

  // copied out by (cell, dof) and (facet, dof), which is the numbering the cut
  // problem shares wherever it did not cut
  const auto& mu = sp.map(sp.index_of("u_0"));
  const auto u0 = static_cast<std::size_t>(sp.offset(sp.index_of("u_0")));
  const int nu = mu.layout().components;
  u_out.assign(static_cast<std::size_t>(c.count(3) * nu), 0.0);
  for (Index e = 0; e < c.count(3); ++e) {
    for (int k = 0; k < nu; ++k) {
      u_out[static_cast<std::size_t>(e * nu + k)] =
          x[u0 + static_cast<std::size_t>(mu.global(3, e, 0, k))];
    }
  }
  const auto& ms = sp.map(sp.index_of("s_0"));
  const auto s0 = static_cast<std::size_t>(sp.offset(sp.index_of("s_0")));
  const int nb = ms.layout().on(2), nc = ms.layout().components;
  s_out.assign(static_cast<std::size_t>(c.count(2) * nb * nc), 0.0);
  for (Index f2 = 0; f2 < c.count(2); ++f2) {
    for (int b = 0; b < nb; ++b) {
      for (int k = 0; k < nc; ++k) {
        s_out[static_cast<std::size_t>((f2 * nb + b) * nc + k)] =
            x[s0 + static_cast<std::size_t>(ms.global(2, f2, b, k))];
      }
    }
  }
  return x;
}

// ---- the cut problem, with a bonded fracture -------------------------------
struct Bonded {
  double u{0.0};           // worst |u_cut - u_uncut|, cell by cell
  double s{0.0};           // worst |s_cut - s_uncut| away from the fracture
  double sigma_lateral{0.0};
  double displacement{0.0};
  double jump{0.0};  // the shortening the fracture itself contributes
  std::size_t n_dofs{0}, n_extra{0};
};

// `c_n` zero is the bonded interface; positive is the linear law at that normal
// stiffness. The tangential one is deliberately different, and nothing here
// shears, so a c_T that leaked into the normal direction would show.
Bonded bonded(int n, Family family, const Family_& f, bool closed_form, double c_n = 0.0) {
  const Cut c = cut_cube(n, family);
  std::vector<double> u_ref, s_ref;
  uncut(c.mesh, f, u_ref, s_ref);

  const exokal::Mesh& m = c.md.top.mesh;
  const graphos::Complex& tc = m.topology();
  DeRhamGeometryCache geo;
  if (!f.strong) geo = DeRhamGeometryCache::build(m, 3);
  const StressOperators ops =
      StressOperators::build(m, 3, kMu, kLam, f.how, f.form, f.strong ? nullptr : &geo);

  mimetika::physics::ContactOptions co;
  co.traction_moments = f.strong ? 6 : ops.moments_per_facet();
  co.traction_components = f.strong ? 1 : 3;
  Composition comp = Catalogue::instance().build("linear_elasticity", bulk_options(ops, f));
  comp.only_at(0).emplace<mimetika::physics::LinearContact>(co);

  // the chart of the ambient facet, and the rigid law on it
  const auto ambient_space = comp.space(tc, 3, 0);
  const auto chart = mimetika::physics::interface_chart(c.md, ambient_space, "s_0", ops, 3);
  const auto ci = c_n > 0.0 ? mimetika::physics::linear_contact(chart, c_n, 7.0)
                            : mimetika::physics::rigid_contact(chart);

  exokal::forms::TermContext ctx;
  ctx.provide("stress_operators", ops);
  ctx.provide("contact_interface", ci);
  Simulation sim(comp, mimetika::strata_of(c.md, c.mesh), ctx);

  const std::size_t ia = sim.epoch().index_of("ambient");
  const auto off = static_cast<Index>(sim.epoch().offset(ia));
  const auto& sp = sim.epoch().stratum(ia).space();

  // the boundary of the cut complex is not the boundary of the domain: each
  // carrying copy has one cofacet and looks exactly like one
  const mimetika::FractureFacets fr = mimetika::fracture_facets(c.md);
  const std::set<Index> on_fracture(fr.carrying.begin(), fr.carrying.end());
  std::vector<Index> outer;
  for (const Index fa : mimetika::boundary_facets(tc, 3)) {
    if (on_fracture.count(fa) == 0) outer.push_back(fa);
  }
  mimetika::MechanicsBoundary bcs;
  load(sim, bcs, sp, m, outer, off);
  // and the detached original carries no bulk equation at all
  for (const Index fa : fr.detached) {
    sim.constraints().pin_all(mimetika::facet_dofs(sp, "s_0", 3, fa, off).dofs, 0.0);
  }
  sim.freeze_constraints();
  const std::vector<double> x = solve(sim, "bonded");

  Bonded out;
  out.n_dofs = sim.n_dofs();
  out.n_extra = sim.n_dofs() - u_ref.size() - s_ref.size();

  const auto& cells = c.md.top.to_ambient.index[3];
  const auto& facets = c.md.top.to_ambient.index[2];
  const auto& mu = sp.map(sp.index_of("u_0"));
  const auto u0 = static_cast<std::size_t>(sp.offset(sp.index_of("u_0"))) + off;
  const int nu = mu.layout().components;
  for (Index e = 0; e < tc.count(3); ++e) {
    const auto a = cells[static_cast<std::size_t>(e)];
    for (int k = 0; k < nu; ++k) {
      const double got = x[u0 + static_cast<std::size_t>(mu.global(3, e, 0, k))];
      out.u = std::max(out.u, std::fabs(got - u_ref[static_cast<std::size_t>(a * nu + k)]));
    }
  }
  const auto& ms = sp.map(sp.index_of("s_0"));
  const auto s0 = static_cast<std::size_t>(sp.offset(sp.index_of("s_0"))) + off;
  const int nb = ms.layout().on(2), nc = ms.layout().components;
  std::set<Index> cut_facets(fr.carrying.begin(), fr.carrying.end());
  cut_facets.insert(fr.detached.begin(), fr.detached.end());
  for (Index f2 = 0; f2 < tc.count(2); ++f2) {
    if (cut_facets.count(f2)) continue;  // no counterpart: the cut made these
    const auto a = facets[static_cast<std::size_t>(f2)];
    for (int b = 0; b < nb; ++b) {
      for (int k = 0; k < nc; ++k) {
        const double got = x[s0 + static_cast<std::size_t>(ms.global(2, f2, b, k))];
        out.s = std::max(out.s, std::fabs(got - s_ref[static_cast<std::size_t>((a * nb + b) * nc +
                                                                              k)]));
      }
    }
  }

  // THE STRESS IS STATICALLY DETERMINATE. Uniaxial compression transmits
  // sigma_zz = -kLoad through any interface that stays closed, however
  // compliant, so the stress comparison above holds at every c_n and only the
  // displacement moves. That is what makes the compliant case a closed form
  // too, and it is asserted rather than assumed.
  if (!closed_form) return out;
  // and the closed form itself, for the componentwise family whose leading
  // moments ARE the traction components
  const double k_oed = kLam + 2.0 * kMu;
  const double s_exact = kLam / k_oed * (-kLoad);
  const double e_exact = -kLoad / k_oed;
  for (const Index fa : outer) {
    const auto frame = mimetika::FacetFrame::of(m, 3, mimetika::cofacet_of(m, 3, fa), fa);
    if (std::fabs(frame.normal[kAxis]) > 1e-9) continue;  // the lateral facets
    double t = 0.0;
    for (int k = 0; k < 3; ++k) {
      t += frame.normal[static_cast<std::size_t>(k)] *
           x[s0 + static_cast<std::size_t>(ms.global(2, fa, 0, k))];
    }
    out.sigma_lateral = std::max(out.sigma_lateral, std::fabs(t / frame.measure - s_exact));
  }
  const double u_jump = -kLoad / c_n_or_inf(c_n);
  for (Index e = 0; e < tc.count(3); ++e) {
    const double vol = exokal::measure(m, 3, e);
    const double z = exokal::centroid(m, 3, e)[kAxis];
    const double uz = -x[u0 + static_cast<std::size_t>(mu.global(3, e, 0, kAxis))] / vol;
    // below the fracture the roller at z = 0 fixes the field; above it, every
    // cell has also taken the whole of the interface's own closure
    const double want = e_exact * z + (z > 0.5 ? u_jump : 0.0);
    out.displacement = std::max(out.displacement, std::fabs(uz - want));
    if (z > 0.5) out.jump = std::max(out.jump, std::fabs((uz - e_exact * z) - u_jump));
  }
  return out;
}

}  // namespace

MIMETIKA_TEST(a_bonded_fracture_is_invisible_to_the_bulk) {
  for (const Family_& f : {kWeak, kStrong}) {
    for (const Family family : {Family::cartesian, Family::simplex}) {
      const Bonded r = bonded(2, family, f, false);
      std::printf("  %-6s %-9s %5zu dofs (%zu of them the fracture's)   u %.2e   sigma %.2e\n",
                  f.name, mimetika::mesh::name(family), r.n_dofs, r.n_extra, r.u, r.s);
      CHECK(r.n_extra > 0);  // the cut did add unknowns, and they changed nothing
      CHECK(r.u < 1e-10);
      CHECK(r.s < 1e-10);
    }
  }
}

MIMETIKA_TEST(the_bonded_patch_reproduces_the_closed_form) {
  for (const Family family : {Family::cartesian, Family::simplex}) {
    const Bonded r = bonded(2, family, kWeak, true);
    std::printf("  %-9s sigma_lateral %.2e   u %.2e\n", mimetika::mesh::name(family),
                r.sigma_lateral, r.displacement);
    CHECK(r.sigma_lateral < 1e-11);
    CHECK(r.displacement < 1e-11);
  }
}

// The compliance the rigid case cannot see.
//
// With A_c = 0 the fourth row is T s + J^T t = 0 and the third is J u_c = 0, so
// flipping the sign of T in the fourth row alone is absorbed by t -> -t: the
// multiplier is determined only up to that sign and nothing measures it. Set
// A_c nonzero and it is measured, because the compliance ties t to a length.
//
// The closed form is still exact, and for the same reason the rigid one was:
// uniaxial compression is statically determinate, so sigma_zz = -kLoad crosses
// any closed interface whatever its stiffness. The stress is therefore the
// uncut stress at every c_n -- asserted below -- and the whole effect of the
// fracture is one extra shortening,
//
//     [[u_n]] = sigma_nn / c_n = -kLoad / c_n ,
//
// which every cell above the plane carries. NEGATIVE: compression closes a
// fracture. Read through the bulk displacement rather than through u_c, so the
// claim is about the domain and not about which side the cut labelled `+`.
MIMETIKA_TEST(a_compliant_fracture_closes_by_the_compliance_and_moves_no_stress) {
  for (const double c_n : {2.0, 20.0}) {
    for (const Family family : {Family::cartesian, Family::simplex}) {
      const Bonded r = bonded(2, family, kWeak, true, c_n);
      std::printf("  c_n %4.0f %-9s closure %.2e of %.4f   u %.2e   sigma vs uncut %.2e\n", c_n,
                  mimetika::mesh::name(family), r.jump, -kLoad / c_n, r.displacement, r.s);
      CHECK(r.jump < 1e-11);          // the closure is the compliance, to round-off
      CHECK(r.displacement < 1e-11);  // and the field around it is still exact
      CHECK(r.s < 1e-10);             // while the stress has not moved at all
      CHECK(r.sigma_lateral < 1e-11);
    }
  }
}

MIMETIKA_TEST_MAIN()
