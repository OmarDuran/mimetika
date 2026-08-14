#include <cmath>
#include <string>
#include <vector>

#include "../../mimetika_test.hpp"
#include "mimetika/mesh/structured.hpp"
#include "mimetika/model/cauchy_elasticity_model.hpp"
#include "mimetika/solver/petsc.hpp"

// THE TRACE OPERATOR, AGAINST A SOLVED SYSTEM.
//
//     g_f = -( M sigma - D^T u - A^T gamma )_f
//         = - sum_{E in {E+, E-}} ( M_E sigma - D_E^T u_E - A_E^T gamma_E )|_f
//
// Two claims, and the first is what fixes the formula:
//
//   THE JUMP VANISHES ON A BONDED INTERIOR FACET. Per cell the constitutive row
//   is M_E sigma - D_E^T u - A_E^T gamma = -int_{dE} u.(tau n); the two cofaces
//   of an interior facet carry opposite outward normals, so when the
//   displacement is continuous their boundary terms cancel and the assembled
//   residual is zero. That IS the discrete statement [[u]] = 0.
//
//   EVERY TERM IS NEEDED. Dropping M sigma leaves 4e-2 on a unit column instead
//   of round-off. The adjoint pair D^T u + A^T gamma names the operators whose
//   adjoints see the facet from both sides -- and the outward incidence is
//   already inside them, which is why the assembled SUM is a difference and no
//   explicit sign belongs in the loop -- but it is not by itself the gap: on the
//   annulus it sums to +3.26e-04 and is cancelled exactly by M sigma's -3.26e-04.
//
//   AND g IS IN METRES. [M sigma] = [D^T u] = [A^T gamma] = m, independently of
//   the dimension: a displacement, not a moment. That is the dimensional content
//   of carrying no Gram^{-1} -- a second inversion would give m^{2-d}, i.e. 1/m
//   in space, which is the 1/h growth under refinement seen as a units error.

using graphos::Index;
using mimetika::CauchyElasticityModel;
using mimetika::ElasticMaterial;
using mimetika::mesh::Family;

namespace {

constexpr double kMu = 1.0, kLam = 1.0;

// the worst |g_f| over every INTERIOR facet of a solved problem
struct Worst {
  double value{0.0};
  std::size_t facets{0};
};

Worst worst_interior_jump(const CauchyElasticityModel& model, const exokal::Mesh& m, int d) {
  const graphos::Complex& c = m.topology();
  const graphos::CoboundaryOperator cob = graphos::coboundary(c, d - 1);
  Worst out;
  for (Index f = 0; f < c.count(d - 1); ++f) {
    const auto b = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f)]);
    const auto e = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f) + 1]);
    if (e - b != 2) continue;  // interior only: a jump needs two sides
    ++out.facets;
    for (const double v : model.trace(f, model.state())) {
      out.value = std::max(out.value, std::abs(v));
    }
  }
  return out;
}

void solve(CauchyElasticityModel& model) {
  model.build();
  mimetika::solver::PetscSolver petsc;
  std::vector<double> x;
  const auto rep = petsc.solve(model.system(), model.rhs(), x);
  if (!rep.converged) throw std::runtime_error("trace: " + rep.reason);
  model.accept(std::move(x));
}

// confined uniaxial compression: a UNIFORM stress state
Worst column_case(int n, int dim, Family family) {
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

  CauchyElasticityModel model(m, dim, ElasticMaterial{kMu, kLam});
  model.mechanics().emplace<mimetika::TractionBC>(loaded, applied);
  model.mechanics().emplace<mimetika::FreeSlipBC>(confined);
  solve(model);
  return worst_interior_jump(model, m, dim);
}

// Lame's tube: sigma_rr = A - B/r^2, a genuinely NON-UNIFORM stress state, which
// is where a formula that happens to work by symmetry stops working
Worst annulus_case(int nr, int nt, int dim, Family family) {
  const double a = 1.0, b = 4.0, hz = 1.0, p_a = 1.0, p_b = 0.25;
  const exokal::Mesh m = mimetika::mesh::annulus(nr, nt, dim, family, a, b, hz);
  const graphos::Complex& c = m.topology();
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
  CauchyElasticityModel model(m, dim, ElasticMaterial{kMu, kLam});
  model.mechanics().emplace<mimetika::TractionBC>(inner, si);
  model.mechanics().emplace<mimetika::TractionBC>(outer, so);
  model.mechanics().emplace<mimetika::FreeSlipBC>(sym);
  solve(model);
  return worst_interior_jump(model, m, dim);
}

}  // namespace

// THE JUMP VANISHES WHEREVER THE MATERIAL IS CONTINUOUS -- on every interior
// facet, every cell type, both dimensions. This is the identity the trace rests
// on, and the one that decided its formula: it is the discrete [[u]] = 0.
MIMETIKA_TEST(the_jump_vanishes_on_every_bonded_interior_facet) {
  for (const int dim : {2, 3}) {
    for (const Family f : {Family::cartesian, Family::simplex, Family::prism}) {
      const Worst w = column_case(4, dim, f);
      std::printf("  column  %dD %-10s %4zu interior facets   worst |g_f| %.2e\n", dim,
                  mimetika::mesh::name(f), w.facets, w.value);
      CHECK(w.facets > 0);
      CHECK(w.value < 1e-12);
    }
  }
}

// AND UNDER A NON-UNIFORM STRESS STATE, which is the case that distinguishes
// the formula from one that merely works by symmetry: on the column the two
// cofaces contribute exactly equal halves of M sigma, on Lame's tube they do
// not, and the identity holds either way.
MIMETIKA_TEST(the_jump_vanishes_under_a_non_uniform_stress_state) {
  for (const int dim : {2, 3}) {
    for (const Family f : {Family::cartesian, Family::simplex}) {
      const Worst w = annulus_case(6, 3, dim, f);
      std::printf("  annulus %dD %-10s %4zu interior facets   worst |g_f| %.2e\n", dim,
                  mimetika::mesh::name(f), w.facets, w.value);
      CHECK(w.facets > 0);
      CHECK(w.value < 1e-12);
    }
  }
}

// EVERY TERM IS LOAD BEARING. Removing M sigma leaves a residual of the order of
// the displacement across a cell -- 4e-2 on this column against a 1e-16 total --
// so the adjoint pair alone is not the gap, and the test says so with the number
// rather than by assertion.
MIMETIKA_TEST(the_compliance_term_is_not_optional) {
  const double h = 1.0, load = 0.5;
  const int dim = 3;
  const exokal::Mesh m = mimetika::mesh::column(4, dim, Family::cartesian, h);
  const graphos::Complex& c = m.topology();
  std::vector<Index> loaded, confined;
  for (const Index f : mimetika::boundary_facets(c, dim)) {
    const double z = exokal::centroid(m, dim - 1, f)[2];
    (std::abs(z - h) < 1e-9 ? loaded : confined).push_back(f);
  }
  std::array<double, 9> applied{};
  applied[8] = -load;
  CauchyElasticityModel model(m, dim, ElasticMaterial{kMu, kLam});
  model.mechanics().emplace<mimetika::TractionBC>(loaded, applied);
  model.mechanics().emplace<mimetika::FreeSlipBC>(confined);
  solve(model);

  // the adjoint pair alone, assembled over both cofaces
  const auto& sp = model.simulation().epoch().stratum(0).space();
  const auto& mu = sp.map(sp.index_of("u_0"));
  const auto& mg = sp.map(sp.index_of("g_0"));
  const auto u_off = static_cast<std::size_t>(sp.offset(sp.index_of("u_0")));
  const auto g_off = static_cast<std::size_t>(sp.offset(sp.index_of("g_0")));
  const auto& z = model.state();
  const graphos::CoboundaryOperator cob = graphos::coboundary(c, dim - 1);

  Index face = -1;
  for (Index f = 0; f < c.count(dim - 1); ++f) {
    if (std::abs(exokal::centroid(m, dim - 1, f)[2] - 0.5) < 1e-9) { face = f; break; }
  }
  CHECK(face >= 0);

  const std::size_t ndf = 9;
  double adjoint_only = 0.0;
  const auto b = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(face)]);
  for (std::size_t mm = b; mm < b + 2; ++mm) {
    const Index cell = cob.indices[mm];
    const auto& op = model.stress_operators().cell(cell);
    std::size_t slot = 0;
    for (std::size_t i = 0; i < op.faces.size(); ++i) if (op.faces[i] == face) slot = i;
    const std::size_t local = slot * ndf + 2;  // moment 0, normal component
    for (std::size_t j = 0; j < op.Dv.rows(); ++j) {
      adjoint_only -= op.Dv(j, local) *
                      z[u_off + static_cast<std::size_t>(mu.global(dim, cell, 0, static_cast<int>(j)))];
    }
    for (std::size_t j = 0; j < op.As.rows(); ++j) {
      adjoint_only -= op.As(j, local) *
                      z[g_off + static_cast<std::size_t>(mg.global(dim, cell, 0, static_cast<int>(j)))];
    }
  }
  const double full = model.trace(face, z)[2];
  std::printf("  adjoint pair alone %.6e   full residual %.2e\n", adjoint_only, full);

  // THE FULL RESIDUAL VANISHES and the adjoint pair does not, by four orders
  CHECK(std::abs(full) < 1e-12);
  CHECK(std::abs(adjoint_only) > 1e-3);

  // AND IT IS THE DISPLACEMENT OVER THE TWO HALF-CELLS, in metres: with
  // K_oed = lam + 2mu = 3, eps = -load/K_oed, over 2 x h/(2n) = 0.25 m
  const double eps = -load / (kLam + 2.0 * kMu);
  CHECK(std::abs(std::abs(adjoint_only) - std::abs(eps * 0.25)) < 1e-12);
}

MIMETIKA_TEST_MAIN()
