#include <cmath>

#include "exokal/ad/dual.hpp"
#include "mimetika/physics/constitutive/immiscible.hpp"
#include "mimetika/physics/constitutive/upwind.hpp"
#include "../mimetika_test.hpp"

using mimetika::physics::constitutive::ImmiscibleFluid;
using mimetika::physics::constitutive::PhaseModel;
using mimetika::physics::constitutive::State;
using mimetika::physics::constitutive::Weights;

namespace {

bool near(double a, double b, double tol = 1e-12) { return std::abs(a - b) <= tol; }

// the state width here is p, h and the compositions: a handful of scalars,
// so the dual is sized to that and not to the default capacity
using D = exokal::ad::Dual<double, 8>;

ImmiscibleFluid two_phase() {
  PhaseModel liquid;
  liquid.reference_density = 1000.0;
  liquid.viscosity = 1.0e-3;
  liquid.heat_capacity = 4200.0;
  liquid.corey_exponent = 2.0;
  PhaseModel vapour;
  vapour.reference_density = 20.0;
  vapour.viscosity = 2.0e-5;
  vapour.heat_capacity = 2000.0;
  vapour.corey_exponent = 2.0;
  return ImmiscibleFluid({liquid, vapour});
}

}  // namespace

// THE FLASH IS EXPLICIT under the bijection: the saturations invert from
// the compositions in closed form, and they invert back
MIMETIKA_TEST(the_composition_inverts_to_saturations_exactly) {
  const ImmiscibleFluid f = two_phase();
  const std::vector<double> z = {0.6, 0.4};
  const State<double> x{5.0e6, 1.0e5, z};
  const Weights<double> w = f.evaluate(x);

  CHECK(near(w.saturation[0] + w.saturation[1], 1.0));
  // s = (z/rho) / sum, so the round trip z = rho_a s_a / rho_mix holds
  for (std::size_t a = 0; a < 2; ++a) {
    CHECK(near(w.density[a] * w.saturation[a] / w.mixture_density, z[a], 1e-14));
  }
  // and the mixture density is the saturation-weighted sum
  CHECK(near(w.mixture_density, w.density[0] * w.saturation[0] + w.density[1] * w.saturation[1],
             1e-9));

  // the light phase occupies far more volume than its mass share
  CHECK(w.saturation[1] > 0.9 && z[1] < 0.5);
}

// the weights are on the simplex, and the derived ones are consistent
// with the fractional flows
MIMETIKA_TEST(the_weights_live_on_the_simplex) {
  const ImmiscibleFluid f = two_phase();
  const std::vector<double> z = {0.5, 0.5};
  const Weights<double> w = f.evaluate(State<double>{1.0e6, 8.0e4, z});

  CHECK(near(w.fraction[0] + w.fraction[1], 1.0, 1e-14));
  CHECK(near(w.mobility[0] + w.mobility[1], w.total_mobility, 1e-12));
  for (std::size_t a = 0; a < 2; ++a) {
    CHECK(w.fraction[a] >= 0.0 && w.fraction[a] <= 1.0);
    CHECK(near(w.fraction[a], w.mobility[a] / w.total_mobility, 1e-14));
  }
  CHECK(near(w.flow_density, w.fraction[0] * w.density[0] + w.fraction[1] * w.density[1], 1e-9));
  CHECK(near(w.mixture_enthalpy,
             w.fraction[0] * w.phase_enthalpy[0] + w.fraction[1] * w.phase_enthalpy[1], 1e-9));
}

// the temperature comes from the enthalpy through the caloric closure,
// the phase mass fractions BEING the compositions here
MIMETIKA_TEST(temperature_follows_from_the_enthalpy) {
  const ImmiscibleFluid f = two_phase();
  const std::vector<double> z = {0.75, 0.25};
  const double h = 1.2e5;
  const Weights<double> w = f.evaluate(State<double>{1.0e6, h, z});

  const double cp = 0.75 * 4200.0 + 0.25 * 2000.0;
  CHECK(near(w.temperature, h / cp, 1e-9));
  CHECK(near(w.phase_enthalpy[0], 4200.0 * w.temperature, 1e-9));

  // doubling the enthalpy doubles the temperature at fixed composition
  const Weights<double> w2 = f.evaluate(State<double>{1.0e6, 2.0 * h, z});
  CHECK(near(w2.temperature, 2.0 * w.temperature, 1e-9));
}

// An immobile phase contributes nothing. And the wholly immobile state is
// UNREACHABLE for a valid composition: the saturations sum to one while
// the residuals sum to less than one, so s_a < s_ra for every phase would
// give 1 < 1. At least one phase is always strictly mobile — the guard in
// evaluate() is defence, not a case the physics reaches.
MIMETIKA_TEST(residual_saturation_and_the_unreachable_immobile_state) {
  PhaseModel a;
  a.residual_saturation = 0.2;
  a.reference_density = 1000.0;
  PhaseModel b;
  b.residual_saturation = 0.1;
  b.reference_density = 1000.0;
  const ImmiscibleFluid f({a, b});

  // equal densities make z and s coincide; below its residual, phase 0 is
  // immobile and its fractional flow vanishes
  const std::vector<double> z = {0.15, 0.85};
  const Weights<double> w = f.evaluate(State<double>{0.0, 1.0, z});
  CHECK(near(w.saturation[0], 0.15, 1e-12));
  CHECK(near(w.mobility[0], 0.0));
  CHECK(near(w.fraction[0], 0.0));
  CHECK(near(w.fraction[1], 1.0, 1e-12));

  // sweep the whole composition edge: total mobility never vanishes
  for (int i = 0; i <= 100; ++i) {
    const double t = double(i) / 100.0;
    const std::vector<double> zz = {t, 1.0 - t};
    const Weights<double> ws = f.evaluate(State<double>{0.0, 1.0, zz});
    CHECK(near(ws.saturation[0] + ws.saturation[1], 1.0, 1e-12));
    CHECK(ws.total_mobility > 0.0);
    CHECK(near(ws.fraction[0] + ws.fraction[1], 1.0, 1e-12));
    CHECK(std::isfinite(ws.mixture_density));
  }
}

// compressibility is what makes the ACCUMULATION nonlinear: the mixture
// density depends on the pressure, so d/dt(phi rho) is not a mass matrix
MIMETIKA_TEST(compressibility_makes_the_accumulation_state_dependent) {
  PhaseModel m;
  m.reference_density = 1000.0;
  m.compressibility = 1.0e-8;
  const ImmiscibleFluid f({m});
  const std::vector<double> z = {1.0};

  const double lo = f.evaluate(State<double>{0.0, 1.0, z}).mixture_density;
  const double hi = f.evaluate(State<double>{1.0e7, 1.0, z}).mixture_density;
  CHECK(hi > lo);
  CHECK(near(hi, 1000.0 * (1.0 + 1.0e-8 * 1.0e7), 1e-9));
}

// THE WEIGHTS DIFFERENTIATE: evaluated at dual scalars they return exact
// derivatives with respect to the state, which is what a term needs
MIMETIKA_TEST(the_weights_carry_dual_scalars) {
  const ImmiscibleFluid f = two_phase();
  const double p = 5.0e6, h = 1.0e5;
  const std::vector<double> z = {0.6, 0.4};

  // seed p, h, z0, z1 as the four independent directions
  std::vector<D> zd = {D::variable(z[0], 2, 4), D::variable(z[1], 3, 4)};
  const State<D> x{D::variable(p, 0, 4), D::variable(h, 1, 4), zd};
  const Weights<D> w = f.evaluate(x);

  const Weights<double> ref = f.evaluate(State<double>{p, h, z});
  CHECK(near(w.total_mobility.value(), ref.total_mobility, 1e-9));

  // d(total mobility)/dz0 against a finite difference
  const double eps = 1e-8;
  const std::vector<double> zp = {z[0] + eps, z[1]};
  const double fd = (f.evaluate(State<double>{p, h, zp}).total_mobility - ref.total_mobility) / eps;
  CHECK(std::abs(w.total_mobility.d(2) - fd) < 1e-4 * std::abs(fd) + 1e-6);

  // and dT/dh is exactly 1/cp, the closure being linear in the enthalpy
  const double cp = 0.6 * 4200.0 + 0.4 * 2000.0;
  CHECK(near(w.temperature.d(1), 1.0 / cp, 1e-12));
}

// the weights feed the upwind operator directly: mobilities carry the
// dual, the densities stay frozen
MIMETIKA_TEST(the_weights_drive_the_upwind_operator) {
  const ImmiscibleFluid f = two_phase();
  const std::vector<double> zl = {0.6, 0.4}, zr = {0.3, 0.7};
  const double p = 5.0e6, h = 1.0e5;

  const Weights<double> wl = f.evaluate(State<double>{p, h, zl});
  const Weights<double> wr = f.evaluate(State<double>{p, h, zr});
  const std::vector<double> mobL = {wl.mobility[0], wl.mobility[1]};
  const std::vector<double> mobR = {wr.mobility[0], wr.mobility[1]};
  const std::vector<double> rho = f.densities_at(p);  // frozen

  const double u = mimetika::physics::constitutive::hybrid_pair(std::span<const double>(mobL),
                                                     std::span<const double>(mobR),
                                                     std::span<const double>(rho), 0, 1, 1.0);
  CHECK(u > 0.0);

  // a shared state reduces it to the continuous pair mobility
  const double same = mimetika::physics::constitutive::hybrid_pair(std::span<const double>(mobL),
                                                        std::span<const double>(mobL),
                                                        std::span<const double>(rho), 0, 1, 1.0);
  CHECK(
      near(same, mimetika::physics::constitutive::pair_mobility(std::span<const double>(mobL), 0, 1), 1e-12));
}

MIMETIKA_TEST_MAIN()
