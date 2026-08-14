// STEADY DARCY IN THE ANNULUS: no mechanics, one known answer.
//
//   div q = 0,  q = -k grad p,   p = p1 at r = a,  p = p0 at r = R
//   =>  p(r) = p0 + (p1 - p0) ln(R/r)/ln(R/a)
//
// The symmetry planes are sealed (strong, q.n = 0) and both radial boundaries
// carry a prescribed PRESSURE, which in the mixed form is natural -- exactly
// the path Terzaghi never exercised, because its only natural datum is zero.
#include <cmath>
#include <cstdio>
#include <vector>
#include "exokal/constitutive/coefficient.hpp"
#include "exokal/hodge/flux_hodge.hpp"
#include "mimetika/model/boundary.hpp"
#include "mimetika/model/simulation.hpp"
#include "mimetika/models/single_phase_flow.hpp"
#include "mimetika/physics/boundary_terms.hpp"
#include "mimetika/solver/petsc.hpp"
using namespace mimetika;
using graphos::Index;

int main(int argc, char** argv) {
  const int nr = argc > 1 ? std::atoi(argv[1]) : 40;
  const int nt = argc > 2 ? std::atoi(argv[2]) : 12;
  // THE DATUM ENTERS THE FLUX ROW against -B^T p, whose entries are the
  // incidence and NOT the measure: the flux unknown is already the
  // measure-weighted moment. Scaling by the measure imposes the condition at a
  // strength that varies facet by facet on a graded mesh -- which this problem
  // exposes and a uniform column never can.
  const bool scale_by_measure = argc > 3 && std::string(argv[3]) == "scaled";
  const double a = 1.0, R = 40.0, p0 = 1.0, p1 = 2.0;
  const double pi = 3.14159265358979323846;

  std::vector<exokal::Mesh::Point> pts;
  const auto vid = [nt](int i, int j) { return static_cast<Index>(i * (nt + 1) + j); };
  const double growth = std::pow(R / a, 1.0 / nr);
  for (int i = 0; i <= nr; ++i) {
    const double r = a * std::pow(growth, i);
    for (int j = 0; j <= nt; ++j) {
      const double th = 0.5 * pi * j / nt;
      pts.push_back({r * std::cos(th), r * std::sin(th), 0.0});
    }
  }
  std::vector<std::vector<Index>> cl;
  for (int i = 0; i < nr; ++i)
    for (int j = 0; j < nt; ++j)
      cl.push_back({vid(i,j), vid(i+1,j), vid(i+1,j+1), vid(i,j+1)});
  const exokal::Mesh m = exokal::Mesh::from_polygons(std::move(pts), cl);
  const graphos::Complex& c = m.topology();
  const int d = 2;

  const auto geo = exokal::hodge::DeRhamGeometryCache::build(m, d);
  const auto hodge = exokal::hodge::FluxHodge::build(
      m, d, exokal::constitutive::Coefficient::uniform(1.0),
      exokal::hodge::FluxHodge::Realization::derham, &geo);

  BoundaryData pdata(static_cast<std::size_t>(c.count(d - 1)));
  std::vector<Index> sym;
  const double rmid = 0.5 * (a + R);
  for (const Index f : boundary_facets(c, d)) {
    const auto x = exokal::centroid(m, d - 1, f);
    const double r = std::sqrt(x[0]*x[0] + x[1]*x[1]);
    if (std::abs(x[0]) < 1e-8 || std::abs(x[1]) < 1e-8) { sym.push_back(f); continue; }
    const double v = (r < rmid) ? p1 : p0;
    pdata.set({f}, scale_by_measure ? v * exokal::measure(m, d - 1, f) : v);
  }
  exokal::forms::TermContext ctx;
  ctx.provide("flux_hodge", hodge);
  ctx.provide("boundary_pressure", pdata);

  Simulation sim(physics::Catalogue::instance().build("single_phase_flow", {}),
                 {StratumSpec{"ambient", &c, d, 0}}, ctx);
  sim.model().add("prescribed_pressure", exokal::forms::On::all(), {});
  const auto& sp = sim.epoch().stratum(0).space();
  impose_normal_flux(sim.constraints(), sp, "q_0", d, m, sym);
  sim.freeze_constraints();

  exokal::forms::TripletSink jac(sim.n_dofs());
  sim.jacobian(jac);
  const auto A = solver::SparseSystem::from(jac);
  std::vector<double> r0;
  sim.state().assign(sim.n_dofs(), 0.0);
  sim.residual(r0);
  std::vector<double> b(sim.n_dofs(), 0.0), x;
  for (std::size_t i = 0; i < sim.n_dofs(); ++i) {
    b[i] = sim.constraints().pinned(i)
               ? sim.constraints().scale_at(i) * sim.constraints().rhs_at(i)
               : -r0[i];
  }
  // does the natural datum reach the residual at all?
  double rn = 0.0; std::size_t nz = 0;
  for (std::size_t i = 0; i < sim.n_dofs(); ++i) {
    if (sim.constraints().pinned(i)) continue;
    rn = std::max(rn, std::abs(r0[i]));
    nz += std::abs(r0[i]) > 1e-30 ? 1 : 0;
  }
  std::printf("  residual at x=0 on free rows: max |r| = %.3e over %zu nonzeros of %zu\n",
              rn, nz, sim.n_dofs());
  std::size_t applied = 0;
  for (Index f = 0; f < c.count(d - 1); ++f) applied += pdata.applies(f) ? 1 : 0;
  std::printf("  boundary data set on %zu facets\n", applied);

  solver::PetscSolver petsc;
  const auto rep = petsc.solve(A, b, x);
  if (!rep.converged) { std::printf("no: %s\n", rep.reason.c_str()); return 1; }

  const auto p_off = (std::size_t)sp.offset(sp.index_of("p_0"));
  double worst = 0.0;
  std::printf("steady Darcy, %lld cells, datum %s\n", (long long)c.count(d),
              scale_by_measure ? "x measure" : "raw");
  for (Index e = 0; e < c.count(d); ++e) {
    const auto xc = exokal::centroid(m, d, e);
    const double r = std::sqrt(xc[0]*xc[0] + xc[1]*xc[1]);
    const double got = (x[p_off + (std::size_t)e] - p0) / (p1 - p0);
    const double ex = std::log(R / r) / std::log(R / a);
    worst = std::max(worst, std::abs(got - ex));
    if (e % (c.count(d) / 6) == 0)
      std::printf("  r=%-9.4f  pbar=%+.6f  exact=%+.6f\n", r, got, ex);
  }
  std::printf("  worst |pbar - exact| = %.3e\n", worst);
}
