#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "exokal/io/gmsh.hpp"
#include "mimetika/algebraic_constraints/contact/cauchy_mechanics.hpp"
#include "mimetika/algebraic_constraints/contact/driver.hpp"
#include "mimetika/benchmarks/novikov_2024.hpp"
#include "mimetika/model/cauchy_elasticity_model.hpp"
#include "mimetika/linear_solver/petsc.hpp"

// THE INCLINED DISPLACED FAULT of Novikov et al. (2024), shared by benchmarks 2
// and 3. They are the SAME problem under two friction laws -- constant Coulomb,
// and slip weakening -- so the geometry, the depletion field, the roller frame
// and the in-situ prestress are stated once here and the benchmarks differ only
// in the law they hand the driver.

using exokal::Geometry;
using exokal::ShapeId;
using graphos::Index;
using mimetika::CauchyElasticityModel;
using mimetika::ElasticMaterial;
using mimetika::benchmarks::Parameters;
using mimetika::contact::CauchyContactMechanics;
using mimetika::contact::ContactDriver;
using mimetika::contact::ContactState;
using mimetika::contact::DriverOptions;
using mimetika::contact::Fracture;
using mimetika::contact::SignoriniCoulomb;
using mimetika::contact::Vec3;

namespace novikov {

// THE PAPER'S OWN DATASET (4TU, doi 10.4121/d77f1a2c-29ea-4572-ad72-e33ed8dc8d22),
// exported column-wise from the published spreadsheet.
//
// The semi-analytical post-slip solution is not a closed form -- it solves
// Cauchy singular integral equations -- so the only way to compare pointwise is
// against the authors' own numbers. Reading .xlsx from C++ would be a parser
// nobody wants to own, so the Python reference (which already knows which rows
// are genuine and which are corrupt) exports them once to CSV and the test
// reads that. It is the DATA that crosses, not the code.
struct Reference {
  std::map<std::string, std::vector<double>> column;

  bool has(const std::string& name) const { return column.count(name) != 0; }
  const std::vector<double>& operator[](const std::string& name) const { return column.at(name); }

  // linear interpolation of (x, v) at xq, with x sorted ascending
  static double at(const std::vector<double>& x, const std::vector<double>& v, double xq) {
    if (xq <= x.front()) return v.front();
    if (xq >= x.back()) return v.back();
    const auto it = std::lower_bound(x.begin(), x.end(), xq);
    const std::size_t j = static_cast<std::size_t>(it - x.begin());
    const double t = (xq - x[j - 1]) / (x[j] - x[j - 1]);
    return v[j - 1] + t * (v[j] - v[j - 1]);
  }
};

inline Reference read_reference(const std::string& name) {
  Reference r;
  const std::string path = std::string(MIMETIKA_TEST_DATA_DIR) + "/" + name;
  std::ifstream in(path);
  if (!in) throw std::runtime_error("reference not found: " + path);
  std::string line;
  std::getline(in, line);
  std::vector<std::string> keys;
  {
    std::stringstream ss(line);
    std::string k;
    while (std::getline(ss, k, ',')) keys.push_back(k);
  }
  for (const std::string& k : keys) r.column[k];
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string cell;
    for (const std::string& k : keys) {
      if (!std::getline(ss, cell, ',')) break;
      if (!cell.empty()) r.column[k].push_back(std::stod(cell));
    }
  }
  return r;
}

constexpr double kDip = 70.0;

bool close(double got, double want, double rel) {
  return std::abs(got - want) <= rel * std::abs(want);
}

// Table 2 on the wide domain: the analytic solution is posed on an UNBOUNDED
// medium and the paper's own results need W = 18 km to match it.
Parameters wide() {
  Parameters p;
  p.width = 18000.0;
  p.dip = kDip;
  return p;
}

// -- the geometry, as a shape complex -------------------------------------------

struct Setup {
  exokal::Mesh mesh;
  std::vector<Index> fault;     // the fault facets, ordered by y
  std::vector<double> y;        // their centroids, in METRES
  std::vector<Index> depleted;  // the offset reservoir's cells
  double unit{1.0}, length{1.0};
};

// THE DOMAIN, DIVIDED BY THE FAULT AND BY THE RESERVOIR INTERFACES.
//
// Five horizontal levels -- the domain floor and roof and the four reservoir
// edges y = +-a, +-b -- cut both the fault and each side wall, giving five bands
// per side and five fault pieces. The reservoir is bands [-b, a] on one side of
// the fault and [-a, b] on the other: the THROW, which is what loads the fault
// in shear. Without it the two sides would face reservoir everywhere and there
// would be nothing to slip.
Setup build(const Parameters& p, double fault_size, double far_size, double band,
            double interface_size, double corner_size) {
  const double L = p.height, W = p.width / L, H = 1.0;
  const double a = p.fault_a / L, b = p.fault_b / L;
  const double cot = 1.0 / std::tan(kDip * M_PI / 180.0);
  const std::vector<double> levels = {-H / 2, -b, -a, a, b, H / 2};

  Geometry g(2);
  std::vector<ShapeId> on_fault, on_left, on_right;
  for (const double y : levels) {
    on_fault.push_back(g.vertex(y * cot, y));
    on_left.push_back(g.vertex(-W / 2, y));
    on_right.push_back(g.vertex(W / 2, y));
  }
  std::vector<ShapeId> fault_pieces, left_rail, right_rail;
  for (std::size_t i = 0; i + 1 < levels.size(); ++i) {
    fault_pieces.push_back(g.curve(on_fault[i], on_fault[i + 1]));
    left_rail.push_back(g.curve(on_left[i], on_left[i + 1]));
    right_rail.push_back(g.curve(on_right[i], on_right[i + 1]));
  }
  std::vector<ShapeId> left_rung, right_rung;
  for (std::size_t i = 0; i < levels.size(); ++i) {
    left_rung.push_back(g.curve(on_left[i], on_fault[i]));
    right_rung.push_back(g.curve(on_fault[i], on_right[i]));
  }

  // the five bands on each side, and which of them the reservoir occupies:
  // [-b, a] on the left of the fault, [-a, b] on the right
  std::vector<ShapeId> reservoir, seal;
  for (std::size_t i = 0; i + 1 < levels.size(); ++i) {
    const ShapeId l = g.surface({left_rung[i], fault_pieces[i], left_rung[i + 1], left_rail[i]});
    const ShapeId r = g.surface({right_rung[i], right_rail[i], right_rung[i + 1], fault_pieces[i]});
    (i == 1 || i == 2 ? reservoir : seal).push_back(l);  // left: bands (-b,-a) and (-a,a)
    (i == 2 || i == 3 ? reservoir : seal).push_back(r);  // right: bands (-a,a) and (a,b)
  }
  g.name(reservoir, "reservoir");
  g.name(seal, "seal");
  g.name(fault_pieces, "fault");

  // GRADE ABOUT THE RESERVOIR INTERFACES AS WELL AS THE FAULT. Refining only
  // about the fault leaves the 225 m reservoir band represented by far-field
  // cells away from it, and the depletion load is then smeared over cells taller
  // than the band itself -- which shows up not as a blurred answer but as the
  // WRONG one: the shear increment peaks at the fault centre instead of at the
  // reservoir edges, and the two slip patches never separate.
  std::vector<ShapeId> interfaces;
  for (std::size_t i = 1; i + 1 < levels.size(); ++i) {
    interfaces.push_back(left_rung[i]);
    interfaces.push_back(right_rung[i]);
  }
  // AND HARDEST AT THE FOUR RESERVOIR CORNERS ON THE FAULT.
  //
  // Sigma_parallel is near-singular exactly where the reservoir edge meets the
  // fault -- the loading jumps there -- and the paper's Fig. 8 shows it spiking
  // to 27 MPa at y = +-a against 18 MPa at the centre. That spike IS the
  // benchmark: it is what carries the fault past the friction threshold, so a
  // mesh that smears it over 25 m facets reports the fault slipping at its
  // centre, where the paper has it firmly stuck, and stuck on the flanks, where
  // the paper has it slipping. The centre was right all along; only the corners
  // were unresolved.
  //
  // They are POINTS, so they cost almost nothing to resolve -- a point field
  // over a few tens of metres, not a finer fault over five kilometres.
  const std::vector<ShapeId> corners = {on_fault[1], on_fault[2], on_fault[3], on_fault[4]};

  exokal::io::MeshOptions opt;
  opt.size = far_size / L;
  opt.refinements.push_back({fault_pieces, fault_size / L, band / L});
  opt.refinements.push_back({interfaces, interface_size / L, band / L});
  opt.refinements.push_back({corners, corner_size / L, 40.0 / L});
  // AND THE ACTIVE STRETCH OF THE FAULT ITSELF, which is |y| <= b and nothing
  // else. The slip patches live there and nowhere else, and at 12 m their
  // interiors FRAGMENT: facets alternate stick / slip because the cone is met
  // to within a facet's resolution, so one patch is reported as four. Refining
  // the whole 4.8 km fault to fix a 300 m stretch would be the wrong trade;
  // refining the three middle pieces is cheap and is where the answer is.
  opt.refinements.push_back(
      {{fault_pieces[1], fault_pieces[2], fault_pieces[3]}, 4.0 / L, 25.0 / L});
  const auto meshed = exokal::io::mesh(g, opt);

  Setup s{meshed.mesh.mesh, {}, {}, {}, p.shear_modulus, L};
  std::vector<std::pair<double, Index>> ordered;
  for (const Index f : meshed.mesh.facets("fault")) {
    ordered.emplace_back(exokal::centroid(s.mesh, 1, f)[1], f);
  }
  std::sort(ordered.begin(), ordered.end());
  for (const auto& [yi, f] : ordered) {
    s.fault.push_back(f);
    s.y.push_back(yi * L);
  }
  s.depleted = meshed.mesh.cells("reservoir");
  return s;
}

// -- the mechanics ---------------------------------------------------------------

std::unique_ptr<CauchyElasticityModel> make_model(const Setup& s, const Parameters& p,
                                                  bool prescribe_fault = true) {
  const int dim = 2;
  Parameters q = p;
  q.shear_modulus = p.shear_modulus / s.unit;
  q.depletion = p.depletion / s.unit;

  const graphos::Complex& c = s.mesh.topology();
  std::vector<Index> top, rollers;
  for (const Index f : mimetika::boundary_facets(c, dim)) {
    (std::abs(exokal::centroid(s.mesh, 1, f)[1] - 0.5) < 1e-9 ? top : rollers).push_back(f);
  }
  std::fprintf(stderr, "  [mk] ctor (mu=%g lam=%g, %zu top, %zu rollers)\n", q.shear_modulus,
               q.lame(), top.size(), rollers.size());
  auto model = std::make_unique<CauchyElasticityModel>(s.mesh, dim,
                                                       ElasticMaterial{q.shear_modulus, q.lame()});
  model->mechanics().emplace<mimetika::TractionBC>(top, std::array<double, 9>{});
  model->mechanics().emplace<mimetika::FreeSlipBC>(rollers);
  std::fprintf(stderr, "  [mk] bcs done; pressurize %zu cells dp=%g\n", s.depleted.size(),
               q.depletion);
  model->pressurize(s.depleted, q.depletion, q.biot, q.volumetric_compliance(dim));
  // BEFORE build(): prescribing changes which equations the system has. The
  // LOCKED fault does not prescribe at all -- it is the plain continuous medium
  // under the depletion load, which is what Fig. 8 plots.
  std::fprintf(stderr, "  [mk] prescribe %zu (%d)\n", s.fault.size(), (int)prescribe_fault);
  if (prescribe_fault) model->prescribe_traction(s.fault);
  std::fprintf(stderr, "  [mk] build()\n");
  model->build();
  std::fprintf(stderr, "  [mk] built, %zu dofs\n", model->simulation().n_dofs());
  return model;
}

// THE IN-SITU TRACTION, RESOLVED IN EACH FACET'S OWN FRAME. On an inclined fault
// BOTH components are nonzero, which is the whole difference from benchmark 1:
// the fault already carries shear before any depletion, and the depletion adds
// to it.
//
// THE NORMAL COMPONENT IS EFFECTIVE, not total (paper Eq. 29): the Coulomb
// threshold acts on sigma'_n = sigma_n + alpha p. And where only one side of the
// fault is depleted -- over the throw, a < |y| < b -- the fault takes the
// DEPLETED side's pressure rather than the mean of the two. That is not an
// approximation chosen for convenience: the Python records that the 4TU
// Sigma_slip curve of Fig. 8 matches the one-sided value and is off by
// mu alpha |dp| / 2 under the mean.
std::vector<Vec3> insitu_prestress(const Setup& s, const Parameters& p, const Fracture& fr,
                                   double depletion) {
  const graphos::CoboundaryOperator cob = graphos::coboundary(s.mesh.topology(), 1);
  std::vector<bool> is_depleted(static_cast<std::size_t>(s.mesh.topology().count(2)), false);
  for (const Index e : s.depleted) is_depleted[static_cast<std::size_t>(e)] = true;

  std::vector<Vec3> out(s.fault.size());
  for (std::size_t i = 0; i < s.fault.size(); ++i) {
    const double y = s.y[i];
    const std::array<double, 4> sigma = p.stress_tensor(y);
    const auto& frame = fr.frame(i);
    const std::array<double, 2> n = {frame.normal[0], frame.normal[1]};
    const std::array<double, 2> t = {frame.tangent[0][0], frame.tangent[0][1]};

    // the depleted side's pressure: min over the two cofaces
    double dp = 0.0;
    const auto lo = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(s.fault[i])]);
    const auto hi = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(s.fault[i]) + 1]);
    for (std::size_t m = lo; m < hi; ++m) {
      if (is_depleted[static_cast<std::size_t>(cob.indices[m])]) dp = std::min(dp, depletion);
    }
    const double p_fault = p.pressure(y) + dp;

    out[i][0] = (Parameters::contract(n, sigma, n) + p.biot * p_fault) / s.unit;
    out[i][1] = Parameters::contract(t, sigma, n) / s.unit;
  }
  return out;
}

// THE PRE-SLIP STATE ON THE LOCKED FAULT -- Fig. 8. No contact anywhere: the
// fault is not allowed to slip, so what it carries is the DRIVING stress, and
// Sigma_C = |Sigma_parallel| - Sigma_sl is the excess over the friction
// threshold. Where it is positive the fault must slip; where negative it holds.
struct PreSlip {
  std::vector<double> parallel, threshold, coulomb;
  // the locked INCREMENTAL traction in each facet frame, in units of G: the
  // warm start the contact Newton needs (see `simulate`)
  std::vector<Vec3> traction;
};

PreSlip pre_slip(const Setup& s, const Parameters& p, double depletion) {
  const int dim = 2;
  Parameters q = p;
  q.depletion = depletion;
  const std::unique_ptr<CauchyElasticityModel> model = make_model(s, q, false);
  mimetika::solver::PetscSolver petsc;
  std::vector<double> x;
  const auto rep = petsc.solve(model->system(), model->rhs(), x);
  if (!rep.converged) throw std::runtime_error("locked fault: " + rep.reason);
  model->accept(std::move(x));

  const Fracture fr(s.mesh, dim, s.fault, model->stress_operators().moments_per_facet());
  const std::vector<Vec3> pre = insitu_prestress(s, p, fr, depletion);

  PreSlip out;
  for (std::size_t i = 0; i < s.fault.size(); ++i) {
    // the INCREMENT the locked solve carries, in the facet frame, plus the
    // in-situ state the fault already sits on
    const std::array<double, 3> t = model->facet_traction(s.fault[i]);
    const auto& frame = fr.frame(i);
    double dn = 0.0, dt = 0.0;
    for (int k = 0; k < dim; ++k) {
      dn += frame.normal[static_cast<std::size_t>(k)] * t[static_cast<std::size_t>(k)];
      dt += frame.tangent[0][static_cast<std::size_t>(k)] * t[static_cast<std::size_t>(k)];
    }
    const double sn = (dn + pre[i][0]) * s.unit;  // effective normal
    const double st = (dt + pre[i][1]) * s.unit;  // shear
    Vec3 incremental;
    incremental[0] = dn;
    incremental[1] = dt;
    out.traction.push_back(incremental);
    out.parallel.push_back(std::abs(st));
    out.threshold.push_back(-p.friction * sn);
    out.coulomb.push_back(std::abs(st) + p.friction * sn);
  }
  return out;
}

struct Slipped {
  ContactState state;

  // the signed tangential jump in metres, which is what pair_average filters
  std::vector<double> state_jump_tangential(double length) const {
    std::vector<double> out;
    for (const Vec3& g : state.jump) out.push_back(g[1] * length);
    return out;
  }
  std::vector<double> slip, normal, shear, excess;
  int iterations{0};
  bool converged{false};
  double peak{0.0};
};

// WARM-STARTING FROM THE LOCKED SOLUTION.
//
// A COLD START DIVERGES, and not marginally. With the fault carrying nothing on
// the first trial, the projection clips metre-scale excursions and the
// iteration leaves the physical basin for a spurious WHOLE-FAULT runaway --
// thousands of millimetres of slip across the entire plane, at levels where the
// fault is in fact fully stuck. It is not a resolution artefact and no mesh
// fixes it: the fixed point simply has another, unphysical attractor and the
// origin is in its basin on a confined domain.
//
// The locked tractions are ONE DIRECT SOLVE away and sit next to the contact
// solution -- their map residual is only the augmentation times the enforcement
// truncation -- so Newton converges from there in a handful of steps.
// PRECONDITIONING, not a guess. Zero tractions with a zero jump is not a state
// the mechanics could ever be in: it says the fault carries nothing while the
// rock around it is fully loaded, so the first trial asks the projection to
// absorb the entire in-situ imbalance and it clips metre-scale excursions. The
// locked solve gives an EQUILIBRATED alternative -- a traction field the
// continuum actually produces under this load -- one direct solve away.
//
// THE INTERNAL STATE IS CARRIED, NOT RESET, when the caller has one. A
// slip-weakening law reads accumulated slip, so replacing the history with
// `initial_state()` at each level would undo the weakening the continuation
// exists to accumulate: the fault would arrive at every level as though it had
// never slipped.
inline ContactState locked_start(const Setup& s, const Parameters& p, double depletion,
                                 const mimetika::contact::ContactLaw& law,
                                 const ContactState* carry = nullptr) {
  const PreSlip locked = pre_slip(s, p, depletion);
  ContactState warm;
  warm.traction = locked.traction;
  if (carry != nullptr && carry->internal.size() == s.fault.size()) {
    warm.internal = carry->internal;
    warm.jump = carry->jump;
  } else {
    warm.internal.assign(s.fault.size(), law.initial_state());
    warm.jump.assign(s.fault.size(), Vec3{});
  }
  return warm;
}

// RAMPING THE DEPLETION WITH WARM RESTARTS.
//
// A friction cone is path dependent in the solver's sense even when the law is
// not: the fixed point has more than one attractor, and jumping straight to the
// full load can land in the wrong one or converge onto a stick/slip partition
// that a continuation would have walked past. The Python ramps in `substeps`
// with each stage warm-started from the last, and that is what makes the patch
// boundaries come out where the reference has them rather than merely somewhere
// admissible.
Slipped simulate_ramped(const Setup& s, const Parameters& p, double depletion,
                        const mimetika::contact::ContactLaw& law, int substeps,
                        const ContactState* warm = nullptr);

Slipped simulate(const Setup& s, const Parameters& p, double depletion,
                 const mimetika::contact::ContactLaw& law, const ContactState* warm = nullptr) {
  const int dim = 2;
  Parameters q = p;
  q.depletion = depletion;
  std::fprintf(stderr, "[sim] make_model\n");
  const std::unique_ptr<CauchyElasticityModel> model = make_model(s, q);
  std::fprintf(stderr, "[sim] fracture\n");
  const Fracture fr(s.mesh, dim, s.fault, model->stress_operators().moments_per_facet());
  std::fprintf(stderr, "[sim] mech\n");
  const CauchyContactMechanics mech(*model, fr);

  DriverOptions opt;
  opt.relaxation = 1.0;
  opt.tolerance = 1e-10;
  opt.max_iterations = 400;
  opt.solver = DriverOptions::Solver::newton;
  const double mu = p.shear_modulus / s.unit;
  const double lam = 2.0 * mu * p.poisson / (1.0 - 2.0 * p.poisson);
  ContactDriver driver(mech, law,
                       mimetika::contact::default_augmentation(s.mesh, dim, s.fault, mu, lam), opt);
  const std::vector<Vec3> pre = insitu_prestress(s, p, fr, depletion);
  driver.set_prestress(pre);
  std::fprintf(stderr, "[sim] solve_step\n");
  const ContactState state = driver.solve_step(warm);
  std::fprintf(stderr, "[sim] solved, conv=%d sol=%zu\n", (int)state.converged,
               state.solution.size());
  Slipped out;
  out.state = state;
  out.iterations = state.iterations;
  out.converged = state.converged;

  // THE TRACTION IS READ OFF THE FACET DEGREES OF FREEDOM of the solved state,
  // not taken from the multiplier.
  //
  // The two agree only where the constraint is met exactly. The multiplier is
  // the ITERATE -- what the projection last proposed -- while the facet dofs
  // are what the mechanics actually solved with, and it is the latter the
  // reference reports. Fig. 8 reaches 0.4 MPa against the dataset on precisely
  // this path, through the same accessor, on the locked problem.
  // A DIVERGED STEP HAS NO SOLUTION TO READ. The driver leaves it empty rather
  // than pushing a non-finite iterate through the factorization, so there is
  // nothing on the facets -- and for benchmark 3 that is not an error but the
  // measurement: the level past the fold is the one that has no equilibrium.
  if (state.solution.empty()) {
    out.slip.assign(s.fault.size(), 0.0);
    out.normal.assign(s.fault.size(), 0.0);
    out.shear.assign(s.fault.size(), 0.0);
    out.excess.assign(s.fault.size(), 0.0);
    out.converged = false;
    out.peak = std::numeric_limits<double>::infinity();  // past the fold
    return out;
  }

  model->accept(state.solution);
  for (std::size_t i = 0; i < s.fault.size(); ++i) {
    const std::array<double, 3> t = model->facet_traction(s.fault[i]);
    const auto& frame = fr.frame(i);
    double dn = 0.0, dt = 0.0;
    for (int k = 0; k < dim; ++k) {
      dn += frame.normal[static_cast<std::size_t>(k)] * t[static_cast<std::size_t>(k)];
      dt += frame.tangent[0][static_cast<std::size_t>(k)] * t[static_cast<std::size_t>(k)];
    }
    const double tn = (dn + pre[i][0]) * s.unit;
    const double tt = (dt + pre[i][1]) * s.unit;
    out.slip.push_back(std::abs(state.jump[i][1]) * s.length);
    out.normal.push_back(tn);
    out.shear.push_back(tt);
    // the Coulomb function: |t_t| - (-mu t_n). Zero on a slipping patch,
    // negative where the fault is stuck.
    out.excess.push_back(std::abs(tt) + p.friction * tn);
    out.peak = std::max(out.peak, out.slip.back());
  }
  return out;
}

// A TWO-FACET MOVING AVERAGE, on facet-midpoint positions.
//
// THE EXACT SET-VALUED LAW RESOLVES A NEAR-THRESHOLD PLATEAU WITH AN
// ALTERNATING ACTIVE SET. Where Sigma_C grazes zero over a stretch, adjacent
// facets settle into stick / at-threshold / stick, so the reaction traction --
// and with it the slip -- carries a facet-scale, MEAN-ZERO oscillation. It is
// not under-resolution and it does not refine away: it is what an exact
// complementarity condition does on a plateau, and the Python reference records
// the same behaviour.
//
// Averaging adjacent facets removes exactly that mode and nothing else, which
// is why it is the right filter rather than a smoothing: a mean-zero
// oscillation at the sampling frequency is annihilated by a two-point mean,
// while a resolved feature is untouched to second order.
inline std::pair<std::vector<double>, std::vector<double>> pair_average(
    const std::vector<double>& y, const std::vector<double>& v) {
  std::vector<double> ym, vm;
  for (std::size_t i = 0; i + 1 < y.size(); ++i) {
    ym.push_back(0.5 * (y[i] + y[i + 1]));
    vm.push_back(0.5 * (v[i] + v[i + 1]));
  }
  return {ym, vm};
}

// THE CONNECTED RUNS OF SLIPPING FACETS, as (y_lo, y_hi), on the pair-averaged
// slip. The threshold is relative to the peak: a converged solve leaves points
// sitting exactly on the cone with slip at the fixed point's own tolerance, and
// counting those as patches says nothing about the fault.
inline std::vector<std::pair<double, double>> patches(const Setup& s, const Slipped& r,
                                                      double fraction_of_peak = 0.01) {
  const auto [y, slip] = pair_average(s.y, r.slip);
  const double tol = fraction_of_peak * r.peak;
  std::vector<std::pair<double, double>> out;
  bool inside = false;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const bool slipping = slip[i] > tol;
    if (slipping && !inside) {
      out.emplace_back(y[i], y[i]);
      inside = true;
    } else if (slipping) {
      out.back().second = y[i];
    } else {
      inside = false;
    }
  }
  return out;
}

inline Slipped simulate_ramped(const Setup& s, const Parameters& p, double depletion,
                               const mimetika::contact::ContactLaw& law, int substeps,
                               const ContactState* warm) {
  ContactState carried;
  bool have = warm != nullptr;
  if (have) carried = *warm;
  Slipped r;
  for (int k = 1; k <= substeps; ++k) {
    const double level = depletion * static_cast<double>(k) / substeps;
    r = simulate(s, p, level, law, have ? &carried : nullptr);
    if (!r.converged) return r;
    carried = r.state;
    have = true;
  }
  return r;
}

// -- PREPARE ONCE, SOLVE MANY ---------------------------------------------------
//
// The coupling the benchmarks need, and the one the Python has: the mechanics is
// built ONCE for a mesh and a material, and a sweep then moves only the load and
// the contact iterate over it.
//
//   the MATRIX depends on the mesh, the moduli, and which facets are prescribed
//   the LOAD depends on the depletion
//   the CONTACT ITERATE depends on neither
//
// So one construction, one assembly, one factorization, one condensation --
// and after that a depletion sweep is a right-hand side per level and a dense
// projection per outer iteration. Rebuilding the model per solve, which is what
// this file did before, re-ran all four on every step of an outer loop that a
// slip-weakening branch tracker runs hundreds of times.
struct Prepared {
  std::unique_ptr<CauchyElasticityModel> model;
  std::unique_ptr<Fracture> fracture;
  std::unique_ptr<CauchyContactMechanics> mechanics;
  double mu{1.0}, lam{1.0};

  // GHAT AND g_0 DEPEND ON THE MECHANICS, NOT ON THE LAW OR THE ITERATE, so an
  // outer loop that only changes the friction coefficient reuses them. Building
  // them costs n_points * dim + 1 global back-substitutions -- 751 here -- and
  // doing that once per outer iteration rather than once per LEVEL is the
  // difference between a sweep that finishes and one that does not.
  std::unique_ptr<mimetika::contact::CondensedMap> condensed;
  double condensed_at{1.0};  // the depletion Ghat/g_0 were built at
};

inline Prepared prepare(const Setup& s, const Parameters& p) {
  Prepared out;
  out.model = make_model(s, p);  // built at p.depletion; set_depletion moves it
  out.fracture = std::make_unique<Fracture>(s.mesh, 2, s.fault,
                                            out.model->stress_operators().moments_per_facet());
  out.mechanics = std::make_unique<CauchyContactMechanics>(*out.model, *out.fracture);
  out.mu = p.shear_modulus / s.unit;
  out.lam = 2.0 * out.mu * p.poisson / (1.0 - 2.0 * p.poisson);
  return out;
}

// One level of the sweep on an already-prepared mechanics: the load moves, the
// factorization does not.
inline Slipped solve_on(Prepared& prep, const Setup& s, const Parameters& p, double depletion,
                        const mimetika::contact::ContactLaw& law,
                        const ContactState* warm = nullptr) {
  const int dim = 2;
  prep.model->set_depletion(depletion / s.unit);

  DriverOptions opt;
  opt.relaxation = 1.0;
  opt.tolerance = 1e-10;
  opt.max_iterations = 400;
  opt.solver = DriverOptions::Solver::newton;
  ContactDriver driver(
      *prep.mechanics, law,
      mimetika::contact::default_augmentation(s.mesh, dim, s.fault, prep.mu, prep.lam), opt);
  const std::vector<Vec3> pre = insitu_prestress(s, p, *prep.fracture, depletion);
  driver.set_prestress(pre);

  // the condensation is per LEVEL: rebuild only when the load moved
  if (prep.condensed == nullptr || prep.condensed_at != depletion) {
    prep.condensed = std::make_unique<mimetika::contact::CondensedMap>(driver.condensed());
    prep.condensed_at = depletion;
  }
  const ContactState state = driver.solve_step_condensed(prep.condensed.get(), warm);
  Slipped out;
  out.state = state;
  out.iterations = state.iterations;
  out.converged = state.converged;
  if (state.solution.empty()) {
    out.slip.assign(s.fault.size(), 0.0);
    out.normal.assign(s.fault.size(), 0.0);
    out.shear.assign(s.fault.size(), 0.0);
    out.excess.assign(s.fault.size(), 0.0);
    out.converged = false;
    out.peak = std::numeric_limits<double>::infinity();
    return out;
  }
  prep.model->accept(state.solution);
  for (std::size_t i = 0; i < s.fault.size(); ++i) {
    const std::array<double, 3> t = prep.model->facet_traction(s.fault[i]);
    const auto& frame = prep.fracture->frame(i);
    double dn = 0.0, dt = 0.0;
    for (int k = 0; k < dim; ++k) {
      dn += frame.normal[static_cast<std::size_t>(k)] * t[static_cast<std::size_t>(k)];
      dt += frame.tangent[0][static_cast<std::size_t>(k)] * t[static_cast<std::size_t>(k)];
    }
    const double tn = (dn + pre[i][0]) * s.unit;
    const double tt = (dt + pre[i][1]) * s.unit;
    out.slip.push_back(std::abs(state.jump[i][1]) * s.length);
    out.normal.push_back(tn);
    out.shear.push_back(tt);
    out.excess.push_back(std::abs(tt) + p.friction * tn);
    out.peak = std::max(out.peak, out.slip.back());
  }
  return out;
}

}  // namespace novikov
