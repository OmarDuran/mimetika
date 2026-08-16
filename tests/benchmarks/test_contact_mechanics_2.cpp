#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "../mimetika_test.hpp"
#include "exokal/io/gmsh.hpp"
#include "mimetika/algebraic_constraints/contact/cauchy_mechanics.hpp"
#include "mimetika/algebraic_constraints/contact/driver.hpp"
#include "mimetika/benchmarks/novikov_2024.hpp"
#include "mimetika/model/cauchy_elasticity_model.hpp"
#include "mimetika/solver/petsc.hpp"

// BENCHMARK 2 of Novikov et al. (2024): the INCLINED displaced fault, with
// constant Coulomb friction (paper 4.1).
//
// The reservoir of benchmark 1, but the fault dips 70 degrees from horizontal
// and carries friction mu = 0.52. Depletion loads it in shear; where the shear
// traction reaches the slip threshold -mu times the EFFECTIVE normal traction,
// the fault slips. At -25 MPa the slip occurs in two separate patches around
// the reservoir edges at y = +-75 m; the patches grow with depletion and merge
// near -27 MPa (the paper's Fig. 12).
//
// THIS IS THE FIRST BENCHMARK WITH A REAL FRICTION CONE. Benchmark 1 was
// frictionless, so its projection was affine and the tangent was trivially
// block-diagonal. Here the cone couples the shear to the normal at every
// sliding point, the tangent stops being trivial, and the semismooth Newton is
// solving a genuinely nonlinear system -- which is what the AD tangent was
// built for.
//
// THE GEOMETRY IS STATED AS A SHAPE COMPLEX. The fault spans the domain, so it
// is a shared BOUNDARY of the rock on either side rather than an immersed
// curve, and the reservoir interfaces divide each side into seal / reservoir /
// seal. Every region is named and the mesh is graded about the fault. Nothing
// here is a mesher script: it is the geometry, and gmsh is asked to realize it.
//
// WHAT IS COMPARED. The paper's post-slip solution is semi-analytical (Cauchy
// singular integral equations) and is published as a dataset rather than a
// formula; its slip rows are corrupt, which the Python reference records, so
// the paper's own REPORTED OBSERVABLES are what a solver can be held to: the
// two-patch structure, where the patches sit, and that they grow with
// depletion. Those are checked here, together with the conditions the law
// itself encodes -- which hold pointwise and are not a matter of reference data.

#include <string>

#include "novikov_fault.hpp"

using novikov::build;
using novikov::close;
using novikov::kDip;
using novikov::patches;
using novikov::pre_slip;
using novikov::PreSlip;
using novikov::Setup;
using novikov::simulate;
using novikov::Slipped;
using novikov::wide;

// THE FACET FRAMES ALONG THE FAULT MUST NOT FLIP.
//
// The in-situ shear prestress is t . sigma . n -- BILINEAR in the frame, so it
// changes sign if one facet's frame is oriented opposite to its neighbour's.
// The normal component n . sigma . n is quadratic and cannot. So a frame flip
// is invisible in the normal traction and inverts the shear, which is exactly
// the asymmetry a fault slipping on one side only would show.
//
// Benchmark 1 could never have caught this: a vertical fault carries no in-situ
// shear, so the sign of a quantity that is identically zero does not matter.
MIMETIKA_TEST(the_fault_facet_frames_do_not_flip) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const Parameters p = wide();
  const Setup s = build(p, 12.0, 1500.0, 400.0, 40.0, 2.0);
  const auto model = novikov::make_model(s, p);
  const Fracture fr(s.mesh, 2, s.fault, model->stress_operators().moments_per_facet());
  const auto pre = novikov::insitu_prestress(s, p, fr, p.depletion);

  std::printf("      y [m]     n=(nx,ny)         t=(tx,ty)       t_n' [MPa]  t_t [MPa]\n");
  int flips = 0;
  double last = 0.0;
  for (std::size_t i = 0; i < s.fault.size(); ++i) {
    const auto& F = fr.frame(i);
    if (i > 0 && last * F.normal[0] < 0.0) ++flips;
    last = F.normal[0];
    if (std::abs(s.y[i]) < 160.0) {
      std::printf("   %9.1f   %+.4f %+.4f    %+.4f %+.4f    %+9.3f  %+9.3f\n", s.y[i], F.normal[0],
                  F.normal[1], F.tangent[0][0], F.tangent[0][1], pre[i][0] * s.unit / 1e6,
                  pre[i][1] * s.unit / 1e6);
    }
  }
  std::printf("  NORMAL-DIRECTION FLIPS ALONG THE FAULT: %d of %zu facets\n", flips,
              s.fault.size());
  CHECK(flips == 0);
}

// IS THE SOLUTION SYMMETRIC AS THE GEOMETRY IS?
//
// The offset reservoir and the dipping fault are invariant under the POINT
// REFLECTION (x, y) -> (-x, -y): the left band [-b, a] maps onto the right band
// [-a, b], and the fault line y = x tan(theta) maps onto itself. The in-situ
// state is not symmetric -- it has a depth gradient -- but that gradient is
// small over +-100 m compared with the depletion response, so the SLIP should
// be very nearly mirror-symmetric in y.
//
// The reference is: patches [-77.8, -36.6] and [+30.9, +77.8], peaks 8.15 and
// 9.23 mm -- mirror images to within the depth gradient. Mine freeze one side
// at [-82, -55] through an entire depletion sweep while the other grows, which
// is not a small asymmetry but a qualitative one.
//
// The gap is the first quantity whose sign is NOT invariant under a facet frame
// flip: [[u]] depends on which side is "+", so it must flip with the canonical
// normal and `Fracture::to_frame` must undo that. Where the two conventions
// disagree, a facet's slip is signed backwards -- and a backwards facet can
// never join a growing patch.
MIMETIKA_TEST(the_slip_is_symmetric_as_the_geometry_is) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const Parameters p = wide();
  const Setup s = build(p, 12.0, 1500.0, 400.0, 40.0, 2.0);
  const SignoriniCoulomb law(p.friction);
  novikov::Prepared prep = novikov::prepare(s, p);
  const Slipped r = novikov::solve_on(prep, s, p, p.depletion, law);
  CHECK(r.converged);

  // PAIR-AVERAGED, because that is the quantity the reference reports.
  //
  // The exact set-valued law leaves an alternating active set on a near-threshold
  // plateau: every point there sits ON the cone (Sigma_C = 0) so the TRACTION is
  // determined, while the slip MAGNITUDE is not, facet to facet. That mode is
  // mean-zero at the sampling frequency, so a two-facet average annihilates it
  // and leaves everything else to second order. Comparing raw per-facet slip
  // against a reference measures that oscillation, not the solution.
  const auto [av_y, av_slip_raw] = novikov::pair_average(s.y, r.state_jump_tangential(s.length));
  const std::vector<double>& av_slip = av_slip_raw;
  (void)av_y;
  std::printf("      y [m]     signed slip [mm]   |mirror(-y)| [mm]   Sigma_C [MPa]\n");
  double worst = 0.0, peak = 0.0;
  int pairs = 0, sign_flips = 0;
  for (std::size_t i = 0; i < s.fault.size(); ++i) {
    if (s.y[i] <= 0.0 || s.y[i] > 100.0) continue;
    // the facet nearest -y
    std::size_t j = 0;
    double best = 1e300;
    for (std::size_t k = 0; k < s.fault.size(); ++k) {
      if (std::abs(s.y[k] + s.y[i]) < best) {
        best = std::abs(s.y[k] + s.y[i]);
        j = k;
      }
    }
    if (best > 4.0) continue;
    ++pairs;
    const double a = av_slip[i], b = av_slip[j];
    if (a * b < 0.0 && std::abs(a) > 1e-6 && std::abs(b) > 1e-6) ++sign_flips;
    peak = std::max(peak, std::max(std::abs(a), std::abs(b)));
    worst = std::max(worst, std::abs(std::abs(a) - std::abs(b)));
    if (pairs % 3 == 1) {
      std::printf("   %9.1f   %+12.5f      %12.5f      %+8.3f\n", s.y[i], 1e3 * a,
                  1e3 * std::abs(b), r.excess[i] / 1e6);
    }
  }
  std::printf(
      "  %d mirror pairs   worst |mismatch| %.4f mm on a %.4f mm peak (%.1f%%)"
      "   SIGNED-SLIP DISAGREEMENTS: %d\n",
      pairs, 1e3 * worst, 1e3 * peak, 100.0 * worst / std::max(peak, 1e-12), sign_flips);
  CHECK(pairs > 5);
  CHECK(worst < 0.15 * peak);
}

// -- the geometry ----------------------------------------------------------------

// THE SHAPE COMPLEX IS THE DOMAIN. The fault spans it, so it is a shared
// boundary; the reservoir is named surfaces, so its area is exact rather than
// the result of a centroid test; and the mesh conforms to both by construction.
MIMETIKA_TEST(the_geometry_is_a_shape_complex_with_the_fault_as_a_shared_boundary) {
  // UNBUFFERED, so that a crash inside a solve does not take the record of
  // which test was running down with it
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const Parameters p = wide();
  const Setup s = build(p, 12.0, 1500.0, 400.0, 40.0, 2.0);
  const graphos::Complex& c = s.mesh.topology();
  c.validate();

  // the fault facets tile the chord, and every one is interior
  double length = 0.0;
  const graphos::CoboundaryOperator cob = graphos::coboundary(c, 1);
  for (const Index f : s.fault) {
    length += exokal::measure(s.mesh, 1, f) * s.length;
    const auto lo = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f)]);
    const auto hi = static_cast<std::size_t>(cob.offsets[static_cast<std::size_t>(f) + 1]);
    CHECK(hi - lo == 2);
  }
  const double chord = p.height / std::sin(kDip * M_PI / 180.0);

  double reservoir = 0.0;
  for (const Index e : s.depleted) {
    reservoir += exokal::measure(s.mesh, 2, e) * s.length * s.length;
  }
  // TWO BANDS OF THICKNESS a + b, one on each side of the fault -- and their
  // areas are NOT mirror images, because the fault leans. Integrating the two
  // strips against the fault line x = y cot(theta),
  //
  //     A = W (a + b) + cot(theta) (a^2 - b^2) ,
  //
  // which is smaller than the vertical-fault value by 6142 m^2 here. Asserting
  // the naive W (a + b) would be asserting benchmark 1's geometry.
  const double cot = 1.0 / std::tan(kDip * M_PI / 180.0);
  const double want =
      p.width * p.reservoir_height() + cot * (p.fault_a * p.fault_a - p.fault_b * p.fault_b);
  std::printf(
      "  %lld cells, %zu fault facets   fault %.2f m (chord %.2f)   reservoir %.4e m^2 (%.4e)\n",
      (long long)c.count(2), s.fault.size(), length, chord, reservoir, want);
  CHECK(close(length, chord, 1e-9));
  CHECK(close(reservoir, want, 1e-9));
}

// THE THROW IS WHAT LOADS THE FAULT. The reservoir is offset across it, so each
// side faces seal over part of the throw; with no offset there would be no shear
// and nothing to slip.
MIMETIKA_TEST(the_reservoir_is_offset_across_the_inclined_fault) {
  const Parameters p = wide();
  const Setup s = build(p, 12.0, 1500.0, 400.0, 40.0, 2.0);
  const double cot = 1.0 / std::tan(kDip * M_PI / 180.0);
  double left_lo = 1e300, left_hi = -1e300, right_lo = 1e300, right_hi = -1e300;
  for (const Index e : s.depleted) {
    const exokal::Mesh::Point x = exokal::centroid(s.mesh, 2, e);
    const double y = x[1] * s.length;
    // left of the fault: x - y cot < 0, in the nondimensional frame
    if (x[0] - x[1] * cot < 0.0) {
      left_lo = std::min(left_lo, y);
      left_hi = std::max(left_hi, y);
    } else {
      right_lo = std::min(right_lo, y);
      right_hi = std::max(right_hi, y);
    }
  }
  std::printf("  left  y in [%+7.1f, %+7.1f] m   right y in [%+7.1f, %+7.1f] m   (a %.0f b %.0f)\n",
              left_lo, left_hi, right_lo, right_hi, p.fault_a, p.fault_b);
  CHECK(left_lo < right_lo);  // offset in y: the throw
  CHECK(left_hi < right_hi);
  // CENTROIDS, so they lie strictly INSIDE their band and never reach its edge
  CHECK(left_lo > -p.fault_b && left_hi < p.fault_a);
  CHECK(right_lo > -p.fault_a && right_hi < p.fault_b);
  // and the offset is the throw, to within a cell
  CHECK(std::abs((right_hi - left_hi) - p.throw_()) < 60.0);
  CHECK(std::abs((right_lo - left_lo) - p.throw_()) < 60.0);
}

// -- the friction cone ------------------------------------------------------------

// THE COULOMB CONDITION HOLDS AT EVERY POINT, which is the constitutive content
// of the law and does not depend on any reference data:
//
//     t_n <= 0        no tension on a fault under kilometres of overburden
//     |t_t| <= -mu t_n    inside the cone
//     slip > 0 only where |t_t| = -mu t_n
//
// The last is the complementarity: a point that slips must sit ON the cone, and
// a point inside it must be stuck. This is where a real friction cone differs
// from benchmark 1's frictionless law, and where the AD tangent is doing work.
MIMETIKA_TEST(the_coulomb_condition_holds_at_every_point_of_the_fault) {
  const Parameters p = wide();
  const Setup s = build(p, 12.0, 1500.0, 400.0, 40.0, 2.0);
  const Slipped r = simulate(s, p, p.depletion, SignoriniCoulomb(p.friction));
  std::printf("  %d Newton iterations, converged=%d\n", r.iterations,
              static_cast<int>(r.converged));
  CHECK(r.converged);

  const double scale = std::abs(p.slip_stress_scale());
  double worst_tension = -1e300, worst_outside = -1e300, worst_complementarity = 0.0;
  std::size_t slipping = 0;
  for (std::size_t i = 0; i < s.fault.size(); ++i) {
    worst_tension = std::max(worst_tension, r.normal[i]);
    worst_outside = std::max(worst_outside, r.excess[i]);
    if (r.slip[i] > 1e-6) {
      ++slipping;
      worst_complementarity = std::max(worst_complementarity, std::abs(r.excess[i]));
    }
  }
  std::printf("  max t_n %+.3e Pa   max (|t_t| + mu t_n) %+.3e Pa (%.1e of C)   %zu slipping\n",
              worst_tension, worst_outside, worst_outside / scale, slipping);
  std::fprintf(stderr, "      y [m]    t_n [MPa]   t_t [MPa]   |t_t|+mu t_n [MPa]   slip [m]\n");
  for (std::size_t i = 0; i < s.fault.size(); ++i) {
    if (std::abs(s.y[i]) > 400.0) continue;
    std::fprintf(stderr, "   %9.1f  %+10.3f  %+10.3f  %+14.4f  %12.6f\n", s.y[i], r.normal[i] / 1e6,
                 r.shear[i] / 1e6, r.excess[i] / 1e6, r.slip[i]);
  }
  CHECK(worst_tension < 0.0);           // compressive everywhere
  CHECK(worst_outside < 1e-6 * scale);  // and inside the cone everywhere
  CHECK(slipping > 0);
  // where it slips, it sits ON the cone
  CHECK(worst_complementarity < 1e-6 * scale);
}

// -- the two slip patches -----------------------------------------------------------

// SLIP OCCURS IN TWO SEPARATE PATCHES, one about each reservoir edge y = +-a.
// That is the paper's central observation for this benchmark, and it is a
// consequence of the throw: the driving Coulomb stress peaks where reservoir
// faces seal, and the fault is still stuck in between.
MIMETIKA_TEST(slip_occurs_in_two_patches_about_the_reservoir_edges) {
  const Parameters p = wide();
  const Setup s = build(p, 12.0, 1500.0, 400.0, 40.0, 2.0);
  const Slipped r = simulate(s, p, p.depletion, SignoriniCoulomb(p.friction));
  CHECK(r.converged);

  const auto found = patches(s, r);
  std::printf("  peak slip %.2f mm   %zu patch(es):", 1e3 * r.peak, found.size());
  for (const auto& [lo, hi] : found) std::printf("  [%+.0f, %+.0f] m", lo, hi);
  std::printf("   (paper: two, about y = +-%.0f)\n", p.fault_a);
  CHECK(found.size() == 2);
  // one on each side of the origin, about the reservoir edges
  CHECK(found[0].second < 0.0 && found[1].first > 0.0);
  CHECK(std::abs(found[0].first) < 3.0 * p.fault_b);
  CHECK(std::abs(found[1].second) < 3.0 * p.fault_b);
}

// AND THEY GROW WITH DEPLETION, which is what Fig. 12 tracks. The patches
// lengthen as the pressure falls and eventually merge; here it is enough that
// the trend is monotone, which no amount of reference data is needed to state.
MIMETIKA_TEST(the_patches_grow_with_depletion) {
  const Parameters p = wide();
  const Setup s = build(p, 12.0, 1500.0, 400.0, 40.0, 2.0);
  double previous_extent = 0.0, previous_peak = 0.0;
  for (const double level : {-15e6, -20e6, -25e6}) {
    const Slipped r = simulate(s, p, level, SignoriniCoulomb(p.friction));
    CHECK(r.converged);
    const auto found = patches(s, r);
    double extent = 0.0;
    for (const auto& [lo, hi] : found) extent += hi - lo;
    std::printf("  dp %6.1f MPa   %zu patch(es), total extent %7.1f m   peak slip %.5f m\n",
                level / 1e6, found.size(), extent, r.peak);
    CHECK(extent >= previous_extent);
    CHECK(r.peak >= previous_peak);
    previous_extent = extent;
    previous_peak = r.peak;
  }
}

// -- Fig. 3: the initial stresses on the 70-degree line -------------------------

// THE IN-SITU STATE RESOLVED ON THE FAULT PLANE, before anything is depleted.
// Both components are linear in depth and both follow from Table 2, so this is
// a check on the SETUP -- and on the rotation, which benchmark 1 never
// exercised because a vertical fault carries no in-situ shear.
MIMETIKA_TEST(the_initial_stresses_on_the_seventy_degree_line) {
  const Parameters p = wide();
  std::vector<double> y, perp, para;
  for (int i = 0; i <= 20; ++i) {
    const double yi = -2250.0 + 4500.0 * i / 20.0;
    const auto r = p.resolved(yi, kDip);
    y.push_back(yi);
    perp.push_back(r[0]);
    para.push_back(r[1]);
  }
  const auto fit_n = mimetika::benchmarks::linear_fit(y, perp);
  const auto fit_t = mimetika::benchmarks::linear_fit(y, para);
  std::printf("  sigma_perp   %+8.2f MPa %+8.2f kPa/m  (paper -60.04, +17.15)\n", fit_n[0] / 1e6,
              fit_n[1] / 1e3);
  std::printf("  sigma_para   %+8.2f MPa %+8.2f kPa/m  (paper   8.21,  -2.35)\n", fit_t[0] / 1e6,
              fit_t[1] / 1e3);
  CHECK(close(fit_n[0], -60.04e6, 5e-3));
  CHECK(close(fit_n[1], 17.15e3, 5e-3));
  CHECK(close(std::abs(fit_t[0]), 8.21e6, 5e-3));
  CHECK(close(std::abs(fit_t[1]), 2.35e3, 5e-3));
  // both are EXACTLY linear -- the paper states them as such
  for (std::size_t i = 0; i < y.size(); ++i) {
    CHECK(std::abs(perp[i] - (fit_n[0] + fit_n[1] * y[i])) < 1e-6);
    CHECK(std::abs(para[i] - (fit_t[0] + fit_t[1] * y[i])) < 1e-6);
  }
}

// -- Fig. 8: the pre-slip stresses on the locked fault ---------------------------

// THE DRIVING STRESS, and the figure that says whether the fault slips at all.
//
// Sigma_parallel SPIKES at the four reservoir corners -- the loading jumps
// there -- reaching about 27 MPa against 18 MPa at the fault centre, and it is
// that spike which carries the fault past the threshold. Sigma_C is therefore
// POSITIVE in two bands about y = +-a and NEGATIVE between them, which is the
// two-patch structure before any contact solve is done.
//
// Getting this wrong is what a coarse mesh does: smear the corners and the
// excess appears at the centre instead, so the fault slips in the one place the
// paper says it does not.
MIMETIKA_TEST(the_pre_slip_coulomb_stress_peaks_at_the_reservoir_corners) {
  const Parameters p = wide();
  const Setup s = build(p, 12.0, 1500.0, 400.0, 40.0, 2.0);
  const PreSlip r = pre_slip(s, p, p.depletion);

  double centre_para = 0.0, centre_coulomb = 0.0, corner_para = 0.0, corner_coulomb = -1e300;
  double corner_y = 0.0;
  for (std::size_t i = 0; i < s.fault.size(); ++i) {
    const double y = s.y[i];
    if (std::abs(y) < 8.0) {
      centre_para = r.parallel[i];
      centre_coulomb = r.coulomb[i];
    }
    if (std::abs(std::abs(y) - p.fault_a) < 12.0 && r.parallel[i] > corner_para) {
      corner_para = r.parallel[i];
      corner_coulomb = r.coulomb[i];
      corner_y = y;
    }
  }
  std::printf("  centre  Sigma_par %6.2f MPa  Sigma_C %+6.2f MPa   (paper 18.5, -0.5)\n",
              centre_para / 1e6, centre_coulomb / 1e6);
  std::printf("  y=%+5.0f  Sigma_par %6.2f MPa  Sigma_C %+6.2f MPa   (paper 27.0, +7.0)\n",
              corner_y, corner_para / 1e6, corner_coulomb / 1e6);

  // THE SPIKE IS THERE: the corner carries markedly more shear than the centre
  CHECK(corner_para > centre_para);
  // the centre is BELOW the threshold and the corner ABOVE it -- the two-patch
  // structure, read off the locked solve before any contact iteration
  CHECK(centre_coulomb < 0.0);
  CHECK(corner_coulomb > 0.0);
}

// -- Fig. 8, POINTWISE against the paper's own dataset ---------------------------
//
// The semi-analytical solution is not a closed form, so this is the only
// comparison that can be made pointwise: the authors' published Sigma_shear and
// Sigma_slip on the locked fault at p = -25 MPa, interpolated onto the fault
// facets and differenced.
//
// AT THE PAPER'S FAULT RESOLUTION, 2 m. Sigma_parallel is near-singular at the
// four reservoir corners, and a coarser fault mesh does not merely blur it -- it
// loses the peak that decides whether the fault slips at all.
MIMETIKA_TEST(the_pre_slip_stresses_match_the_published_dataset) {
  const Parameters p = wide();
  const Setup s = build(p, 2.0, 1500.0, 300.0, 20.0, 1.0);
  const PreSlip r = pre_slip(s, p, p.depletion);
  const auto ref = novikov::read_reference("8_left.csv");

  const std::vector<double>& ry = ref["y"];
  const std::vector<double>& rshear = ref["Sigma_shear"];
  const std::vector<double>& rslip = ref["Sigma_slip"];
  std::printf("  %zu fault facets, reference has %zu points\n", s.fault.size(), ry.size());

  double worst_par = 0.0, worst_thr = 0.0, worst_c = 0.0;
  int compared = 0;
  for (std::size_t i = 0; i < s.fault.size(); ++i) {
    const double y = s.y[i];
    if (y < ry.front() || y > ry.back()) continue;
    // skip a cell either side of each corner: the reference is singular there
    // and no cell-centred value can follow it, which is the paper's own caveat
    bool near_corner = false;
    for (const double e : {-p.fault_b, -p.fault_a, p.fault_a, p.fault_b}) {
      near_corner = near_corner || std::abs(y - e) < 4.0;
    }
    if (near_corner) continue;
    ++compared;
    const double want_par = std::abs(novikov::Reference::at(ry, rshear, y));
    const double want_thr = novikov::Reference::at(ry, rslip, y);
    worst_par = std::max(worst_par, std::abs(r.parallel[i] - want_par));
    worst_thr = std::max(worst_thr, std::abs(r.threshold[i] - want_thr));
    worst_c = std::max(worst_c, std::abs(r.coulomb[i] - (want_par - want_thr)));
  }
  std::printf(
      "  %d points   worst |dSigma_par| %.3f MPa   |dSigma_sl| %.3f MPa"
      "   |dSigma_C| %.3f MPa\n",
      compared, worst_par / 1e6, worst_thr / 1e6, worst_c / 1e6);
  CHECK(compared > 100);
  CHECK(worst_par < 2.0e6);
  CHECK(worst_thr < 2.0e6);
  CHECK(worst_c < 2.5e6);
}

// -- Figs. 9 and 10: the POST-SLIP Coulomb function, pointwise -------------------
//
// Sigma_C on the slipped state: identically zero on the slipping patches -- a
// point that slides sits ON the cone -- and negative where the fault still
// holds. So this one curve carries both the stress answer and the patch
// structure, and its zeros ARE the patch boundaries that Fig. 12 tracks.
//
// Fig. 9 is the Table 2 domain and Fig. 10 the wide one. The paper plots them
// separately because the fault answer is insensitive to the truncation while
// benchmark 1's slip was not; both are run here for that reason.
MIMETIKA_TEST(the_post_slip_coulomb_function_matches_the_published_dataset) {
  const auto ref = novikov::read_reference("9_10_left.csv");
  for (const double width : {4500.0, 18000.0}) {
    Parameters p = wide();
    p.width = width;
    const Setup s = build(p, 2.0, 1500.0, 300.0, 20.0, 1.0);
    const SignoriniCoulomb law(p.friction);

    for (const double level : {-25e6, -27e6}) {
      const std::string tag = level == -25e6 ? "25" : "27";
      const std::vector<double>& ry = ref["y_" + tag];
      const std::vector<double>& rc = ref["Sigma_C_post_" + tag];
      // warm-started from the locked solution: a cold start diverges here
      // THE LOCKED WARM START IS FOR THE CONFINED DOMAIN ONLY, which is what
      // the Python does: `warm_from_locked = width <= 4500`. On W = 4500 the
      // cold start diverges into a spurious whole-fault runaway; on the wide
      // domain it does not, and forcing the locked start there lands the
      // iteration on a DIFFERENT admissible stick/slip partition -- converged,
      // Coulomb-satisfied, and not the reference's.
      const bool confined = width <= 4500.0;
      const ContactState warm = confined ? novikov::locked_start(s, p, level, law) : ContactState{};
      const Slipped r = simulate(s, p, level, law, confined ? &warm : nullptr);
      CHECK(r.converged);

      // THE REFERENCE COMPARISON IS AN RMS OVER |y| <= 100 m, which is what the
      // Python reports and the only statistic that means anything here.
      //
      // Sigma_C is near-singular at the four reservoir corners and the profile
      // runs to +-250 m, so a worst-case over the full window is dominated by
      // two cells next to a logarithmic singularity -- it measures the mesh at
      // the corner, not the agreement on the fault. The Python quotes
      // rms = 0.100 MPa against a scale of 8.9 MPa over the slipping region.
      // PAIR-AVERAGED, for the same reason the slip is: the plateau's active set
      // alternates, so Sigma_C carries a facet-scale mean-zero mode that a
      // pointwise difference against a smooth reference measures instead of the
      // agreement.
      const auto [cy, cc] = novikov::pair_average(s.y, r.excess);
      double sum = 0.0;
      int compared = 0;
      for (std::size_t i = 0; i < cy.size(); ++i) {
        const double y = cy[i];
        if (std::abs(y) > 100.0) continue;
        if (y < ry.front() || y > ry.back()) continue;
        ++compared;
        const double e = cc[i] - novikov::Reference::at(ry, rc, y);
        sum += e * e;
      }
      const double rms = std::sqrt(sum / std::max(1, compared));
      const auto found = patches(s, r);
      std::printf(
          "  W %5.0f m  dp %5.1f MPa   %d pts, Sigma_C rms %.3f MPa"
          "   peak slip %6.2f mm   %zu patch(es)",
          width, level / 1e6, compared, rms / 1e6, 1e3 * r.peak, found.size());
      for (const auto& [lo, hi] : found) std::printf("  [%+.0f,%+.0f]", lo, hi);
      std::printf("\n");
      CHECK(compared > 40);
      CHECK(rms < 0.5e6);  // the Python reports 0.100 MPa on a 8.9 MPa scale
    }
  }
}

// -- Fig. 12: the slip patches merge at a definite pressure ----------------------
//
// The two patches grow inward as the reservoir depletes and eventually meet.
// That MERGING PRESSURE is the sharpest scalar this benchmark produces -- the
// paper's dataset puts it at -26.87 MPa and the Python port at -26.9 -- because
// it is a topological change in the solution, not a value read off a curve.
//
// This is what `prepare` is for. The matrix, its factorization and Ghat do not
// depend on the depletion, so a sweep is a right-hand side per level and a dense
// projection per iterate. Rebuilding per level would make the bisection below
// unaffordable.
MIMETIKA_TEST(the_slip_patches_merge_at_the_published_pressure) {
  const Parameters p = wide();
  const Setup s = build(p, 2.0, 1500.0, 300.0, 20.0, 1.0);
  const SignoriniCoulomb law(p.friction);
  novikov::Prepared prep = novikov::prepare(s, p);

  // separate (2 patches) or merged (1)?
  const auto count_at = [&](double level) {
    const Slipped r = novikov::solve_on(prep, s, p, level, law);
    const auto found = patches(s, r);
    std::printf("   %6.2f MPa   %zu patch(es)   peak %6.2f mm", level / 1e6, found.size(),
                1e3 * r.peak);
    for (const auto& [lo, hi] : found) std::printf("  [%+.0f,%+.0f]", lo, hi);
    std::printf("\n");
    return found.size();
  };

  // bracket: separated well above the merge, single below it
  double separate = -25e6, merged = -29e6;
  CHECK(count_at(separate) == 2);
  CHECK(count_at(merged) == 1);

  // BISECT ON THE TOPOLOGY. The merge is a change in the number of connected
  // slipping runs, so it is bracketed exactly rather than interpolated.
  for (int k = 0; k < 5; ++k) {
    const double mid = 0.5 * (separate + merged);
    (count_at(mid) == 1 ? merged : separate) = mid;
  }
  const double p_merge = 0.5 * (separate + merged);
  std::printf("  merging pressure %.2f MPa   (4TU -26.87, Python port -26.9)\n", p_merge / 1e6);
  CHECK(std::abs(p_merge / 1e6 + 26.87) < 1.0);
}

MIMETIKA_TEST_MAIN()
