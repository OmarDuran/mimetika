#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

// THE FAULT-REACTIVATION BENCHMARKS of Novikov, Voskov et al. (2024),
// "Benchmark study of fault reactivation induced by pressure depletion".
//
// A depleting reservoir at depth loads a fault; the question is when, where and
// how much it slips. Four cases share one setup, which is what lives here:
// Table 2's parameters and the in-situ state they imply. Nothing in this file
// solves anything -- it is the DATA and the CLOSED FORMS a solve is judged
// against, so an error in the setup surfaces before any solver runs.
//
// GEOMETRY AND SIGN CONVENTIONS. y is measured UPWARDS from the reservoir
// reference level, so depth is D0 - y. Stresses are TENSION POSITIVE throughout
// -- the paper's figures are too -- which makes every in-situ stress negative.
// Effective stress uses the Biot convention sigma' = sigma + alpha p.
//
// THE IN-SITU STATE IS DERIVED, NOT TABULATED. Everything in the paper's
// eqs. (17)-(19) follows from the Table 2 parameters:
//
//     rho_b        = phi rho_fl + (1 - phi) rho_s          bulk density
//     sigma_yy(y)  = -rho_b g (D0 - y)                     lithostatic
//     p(y)         = p0 - rho_fl g y                       hydrostatic
//     sigma'_xx    = K0 sigma'_yy                          lateral earth pressure
//     sigma_xx     = sigma'_xx - alpha p
//
// Reproducing the paper's printed coefficients FROM the parameters, rather than
// pasting them in, is what makes the setup checkable -- and the tests do exactly
// that before they run a solver.
//
// ONE DEVIATION, and it is deliberate. The tabulated fluid density gives a
// pressure gradient of 1020 * 9.81 = 10.01 kPa/m, while the paper quotes
// 10.06 kPa/m. The latter corresponds to rho_fl = 1025, which is the value used
// here so that the published in-situ profiles are reproduced exactly.

namespace mimetika::benchmarks {

// Table 2, in SI units.
struct Parameters {
  double width{4500.0};   // W
  double height{4500.0};  // H
  double depth{3500.0};   // D0, to the reservoir reference level
  double fault_a{75.0};   // a: reservoir edge on the shallow side of the throw
  double fault_b{150.0};  // b: reservoir edge on the deep side

  double shear_modulus{6500e6};  // G
  double poisson{0.15};          // nu
  double biot{0.9};              // alpha
  double earth_pressure{0.5};    // K0, effective horizontal / vertical

  double depletion{-25e6};          // Delta p
  double reference_pressure{35e6};  // p0 at y = 0
  double fluid_density{1025.0};     // rho_fl (see the note above)
  double solid_density{2650.0};     // rho_s
  double porosity{0.15};            // phi
  double gravity{9.81};             // g

  double dip{90.0};       // theta, degrees from horizontal
  double friction{0.52};  // mu

  // -- derived geometry ------------------------------------------------------

  // h = a + b = 225 m. Table 2 gives the domain but not the reservoir's own
  // thickness; it is fixed by the displaced-fault geometry of benchmark 1,
  // where the reservoir spans [-b, a] on one side of the fault and [-a, b] on
  // the other. That reproduces the reported compaction of -0.32 m, which is the
  // independent confirmation that the reading is right.
  double reservoir_height() const { return fault_a + fault_b; }
  // vertical offset of the reservoir across the fault, b - a = 75 m
  double throw_() const { return fault_b - fault_a; }

  // -- derived material constants --------------------------------------------

  double bulk_density() const {
    return porosity * fluid_density + (1.0 - porosity) * solid_density;
  }
  // Kv = 2G(1-nu)/(1-2nu): the oedometer (confined) modulus
  double uniaxial_modulus() const {
    return 2.0 * shear_modulus * (1.0 - poisson) / (1.0 - 2.0 * poisson);
  }
  double lame() const { return 2.0 * shear_modulus * poisson / (1.0 - 2.0 * poisson); }

  // -- the in-situ state -----------------------------------------------------

  // d sigma_yy / dy -- positive: the rock gets lighter upwards
  double vertical_gradient() const { return bulk_density() * gravity; }
  // dp/dy -- negative: hydrostatic pressure falls upwards
  double pressure_gradient() const { return -fluid_density * gravity; }

  double pressure(double y) const { return reference_pressure + pressure_gradient() * y; }
  double vertical_stress(double y) const { return -bulk_density() * gravity * (depth - y); }
  double horizontal_stress(double y) const {
    const double effective = earth_pressure * (vertical_stress(y) + biot * pressure(y));
    return effective - biot * pressure(y);
  }

  // the in-situ total stress: diagonal in (x, y), row-major 2x2
  std::array<double, 4> stress_tensor(double y) const {
    return {horizontal_stress(y), 0.0, 0.0, vertical_stress(y)};
  }

  // -- resolved on the fault --------------------------------------------------

  // (normal, tangent) of a fault dipping `degrees` from horizontal
  static std::array<double, 2> fault_normal(double degrees) {
    const double t = degrees * M_PI / 180.0;
    return {-std::sin(t), std::cos(t)};
  }
  static std::array<double, 2> fault_tangent(double degrees) {
    const double t = degrees * M_PI / 180.0;
    return {std::cos(t), std::sin(t)};
  }

  // a . S b for a row-major 2x2 S
  static double contract(const std::array<double, 2>& a, const std::array<double, 4>& s,
                         const std::array<double, 2>& b) {
    return a[0] * (s[0] * b[0] + s[1] * b[1]) + a[1] * (s[2] * b[0] + s[3] * b[1]);
  }

  // (sigma_normal, sigma_shear) of the in-situ state on a `degrees` fault.
  // sigma_normal < 0 in compression; the sign of the shear follows the chosen
  // tangent, so only its magnitude is convention free.
  std::array<double, 2> resolved(double y, double degrees) const {
    const auto n = fault_normal(degrees), t = fault_tangent(degrees);
    const auto s = stress_tensor(y);
    return {contract(n, s, n), contract(t, s, n)};
  }

  // -- the depletion response (paper section 2.4) -----------------------------

  // eps_yy = alpha Delta p / Kv under uniaxial (confined) strain
  double vertical_strain() const { return biot * depletion / uniaxial_modulus(); }
  // Delta h = h eps_yy: with h = 225 m this is -0.3207 m, the paper's -0.32
  double compaction() const { return reservoir_height() * vertical_strain(); }
  // Delta sigma'_xx = nu/(1-nu) alpha Delta p
  double horizontal_effective_increment() const {
    return poisson / (1.0 - poisson) * biot * depletion;
  }
  // Delta sigma_xx = Delta sigma'_xx - alpha Delta p
  double horizontal_total_increment() const {
    return horizontal_effective_increment() - biot * depletion;
  }

  // -- the frictionless displaced fault (paper section 3, eqs. 18-21) ---------
  //
  // Jansen & Meulenbroek (2022), quoted by the paper as eqs. (18)-(22). A
  // reservoir offset across a VERTICAL fault by the throw b - a puts reservoir
  // against seal on both sides, which loads the fault in shear; with no friction
  // it slips until it carries no shear stress at all.
  //
  // BOTH ARE DERIVED FOR AN UNBOUNDED MEDIUM, which is why a simulation on a
  // finite box is compared on the profile shape and the peak rather than
  // pointwise -- and why the box has to be made wide before even those agree.

  // C = (1-2nu) alpha Dp / (2 pi (1-nu)) -- eq. (19), -2.95e6 Pa
  double slip_stress_scale() const {
    return (1.0 - 2.0 * poisson) * biot * depletion / (2.0 * M_PI * (1.0 - poisson));
  }
  // A = G / (2 pi (1 - nu)) -- eq. (21), 1.2171e9 Pa
  double slip_stiffness() const { return shear_modulus / (2.0 * M_PI * (1.0 - poisson)); }

  // Sigma_C(y) -- eq. (18). LOGARITHMICALLY SINGULAR at y = +-a and y = +-b,
  // which is not a defect of the formula: those are the four reservoir edges,
  // where the loading jumps, and no cell-centred value can follow it there.
  double analytic_coulomb_stress(double y) const {
    const double a = fault_a, b = fault_b;
    const double num = (y - a) * (y - a) * (y + a) * (y + a);
    const double den = (y - b) * (y - b) * (y + b) * (y + b);
    return 0.5 * slip_stress_scale() * std::log(num / den);
  }

  // delta(y) -- eq. (20), the five-interval piecewise profile. Continuous,
  // piecewise linear, and FLAT at (C/A)(a-b) over the overlap |y| < a where
  // reservoir faces reservoir across the fault.
  double analytic_slip(double y) const {
    const double a = fault_a, b = fault_b;
    const double scale = slip_stress_scale() / slip_stiffness();
    double shape = 0.0;
    if (y <= -b) {
      shape = 0.0;
    } else if (y <= -a) {
      shape = -(y + b);
    } else if (y < a) {
      shape = a - b;
    } else if (y < b) {
      shape = y - b;
    }
    return scale * shape;
  }

  // |delta| over the overlap: 0.18173 m for the published parameters
  double peak_slip() const {
    return std::abs(slip_stress_scale() / slip_stiffness() * (fault_a - fault_b));
  }

  // tr(C^{-1} T)/tr(T), the skeleton's volumetric compliance -- the factor the
  // Biot coupling carries alongside alpha, and the same one the coupled solver
  // computes for itself
  double volumetric_compliance(int d) const {
    return (1.0 - 2.0 * poisson) /
           (2.0 * shear_modulus * (1.0 - 2.0 * poisson + d * poisson));
  }

  // -- Fig. 4: the combined state across the depleted reservoir ---------------

  // The analytic combined profile: EXACTLY piecewise, because the reservoir is
  // infinitely wide. There is no arching, so outside the depleted band the
  // increment is identically zero and the combined stress is the in-situ state,
  // while inside it is the in-situ state plus the uniaxial increment
  // Delta sigma_xx (with Delta sigma_yy = 0). Both branches keep the in-situ
  // depth gradient, so neither is flat.
  std::array<double, 2> combined_analytic(double y, double degrees) const {
    auto s = stress_tensor(y);
    if (std::abs(y) < 0.5 * reservoir_height()) s[0] += horizontal_total_increment();
    const auto n = fault_normal(degrees), t = fault_tangent(degrees);
    return {contract(n, s, n), contract(t, s, n)};
  }

  // Fig. 4's two plateaus, at the reservoir's own level. Outside is simply the
  // in-situ state; inside, the uniaxial increment adds Delta sigma_xx with
  // Delta sigma_yy = 0, resolved on the same plane.
  std::array<double, 2> plateau_outside(double degrees) const {
    return resolved(0.0, degrees);
  }
  std::array<double, 2> plateau_inside(double degrees) const {
    return combined_analytic(0.0, degrees);
  }
};

// (intercept, gradient) of a field the benchmark states as linear
inline std::array<double, 2> linear_fit(const std::vector<double>& y,
                                        const std::vector<double>& v) {
  const auto n = static_cast<double>(y.size());
  double sy = 0.0, sv = 0.0, syy = 0.0, syv = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    sy += y[i];
    sv += v[i];
    syy += y[i] * y[i];
    syv += y[i] * v[i];
  }
  const double gradient = (n * syv - sy * sv) / (n * syy - sy * sy);
  return {(sv - gradient * sy) / n, gradient};
}

}  // namespace mimetika::benchmarks
