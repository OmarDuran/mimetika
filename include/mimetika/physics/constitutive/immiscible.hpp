#pragma once

#include <array>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

// The nonlinear weights, evaluated pointwise from the state.
//
// Every constitutive nonlinearity enters through a handful of weights — the
// total mass mobility, the mixture density, the fractional flows, the
// flow-weighted density, the mixture enthalpy — each a pointwise function of
// x = (p, h, z). The differential operators act linearly on the fluxes; only
// these move.
//
// Evaluated here for the immiscible mixture, where the phase and component
// contexts are in bijection, x_ka = δ_ka, so component k lies entirely in phase
// k. The phase equilibrium then inverts in closed form and no flash is solved:
//
//     s_k = (z_k/ρ_k) / Σ_b (z_b/ρ_b),     ρ_mix = 1 / Σ_b (z_b/ρ_b),
//
// and with constant heat capacities T = h / Σ_k z_k c_k, the phase mass
// fractions being the overall compositions under the bijection.
//
// The interface is what matters here, not the correlations behind it. A
// weight is a pointwise map evaluated with whatever scalar the caller chose
// — double for a residual, ad::Dual for a residual with its derivatives. An
// operator-based linearization replaces the closed forms with a
// multilinear interpolant over a tabulated state box; differentiating that
// interpolant with the same dual numbers yields exactly what OBL requires,
// the gradient of the interpolant rather than an interpolant of the
// gradient. So the tables drop into this slot later without anything above
// changing.
//
// The dual width: the state here is p, h and the compositions — a handful
// of scalars — so these evaluations want ad::Dual sized to that, not the
// default capacity. StateDual below is the recommended alias; the
// cell-system width belongs to ad::Local, never here.

namespace mimetika::physics::constitutive {

// raise if a mixture needs more; every array below is sized by it
inline constexpr std::size_t max_phases = 4;

// The primary state at one point. The composition carries all phases and
// sums to one; close() applies the closure to the independent entries.
template <class T>
struct State {
  T pressure{};
  T enthalpy{};
  std::span<const T> composition;
};

template <class T>
std::vector<T> close_composition(std::span<const T> independent) {
  std::vector<T> z(independent.begin(), independent.end());
  T last{1.0};
  for (const T& e : independent) last = last - e;
  z.push_back(last);
  return z;
}

// Everything the balance laws weight their fluxes by, at one point.
template <class T>
struct Weights {
  std::size_t n{0};
  std::array<T, max_phases> saturation{};
  std::array<T, max_phases> density{};
  std::array<T, max_phases> mobility{};
  std::array<T, max_phases> fraction{};
  std::array<T, max_phases> phase_enthalpy{};
  T total_mobility{};
  T mixture_density{};
  T mixture_enthalpy{};
  T flow_density{};  // the rho-bar carried by the gravitational flux
  T temperature{};
};

// One phase's material behaviour: a slightly compressible density, a Corey
// relative permeability, and constant caloric data.
struct PhaseModel {
  double reference_density{1000.0};
  double compressibility{0.0};  // rho = rho0 (1 + c (p - p_ref))
  double viscosity{1.0e-3};
  double heat_capacity{4200.0};
  double residual_saturation{0.0};
  double corey_exponent{2.0};
  double endpoint_permeability{1.0};
};

class ImmiscibleFluid {
 public:
  explicit ImmiscibleFluid(std::vector<PhaseModel> phases, double reference_pressure = 0.0)
      : phases_(std::move(phases)), p_ref_(reference_pressure) {
    if (phases_.empty() || phases_.size() > max_phases) {
      throw std::invalid_argument("ImmiscibleFluid: between 1 and max_phases phases");
    }
    for (const PhaseModel& m : phases_) {
      if (!(m.reference_density > 0.0) || !(m.viscosity > 0.0)) {
        throw std::invalid_argument("ImmiscibleFluid: density and viscosity must be positive");
      }
      residual_total_ += m.residual_saturation;
    }
    if (!(residual_total_ < 1.0)) {
      throw std::invalid_argument("ImmiscibleFluid: residual saturations leave no mobile room");
    }
  }

  std::size_t n_phases() const { return phases_.size(); }
  const PhaseModel& phase(std::size_t a) const { return phases_[a]; }

  // The densities the upwind operator needs: frozen, evaluated at the
  // linearization pressure and not differentiated.
  std::vector<double> densities_at(double p) const {
    std::vector<double> r(phases_.size());
    for (std::size_t a = 0; a < phases_.size(); ++a) r[a] = density(a, p);
    return r;
  }

  template <class T>
  Weights<T> evaluate(const State<T>& x) const {
    using std::pow;
    const std::size_t n = phases_.size();
    if (x.composition.size() != n) {
      throw std::invalid_argument("ImmiscibleFluid::evaluate: composition/phase count mismatch");
    }
    Weights<T> w;
    w.n = n;

    // densities, then the explicit inversion of the composition
    T inv_sum{};
    for (std::size_t a = 0; a < n; ++a) {
      w.density[a] = density_of(a, x.pressure);
      inv_sum = inv_sum + x.composition[a] / w.density[a];
    }
    w.mixture_density = T{1.0} / inv_sum;
    for (std::size_t a = 0; a < n; ++a) {
      w.saturation[a] = (x.composition[a] / w.density[a]) / inv_sum;
    }

    // the phase mass fractions ARE the compositions here, so the caloric
    // closure gives the temperature directly
    T cp{};
    for (std::size_t a = 0; a < n; ++a) {
      cp = cp + x.composition[a] * phases_[a].heat_capacity;
    }
    w.temperature = x.enthalpy / cp;
    for (std::size_t a = 0; a < n; ++a) {
      w.phase_enthalpy[a] = phases_[a].heat_capacity * w.temperature;
    }

    // mobilities and the fractional flows on the simplex
    w.total_mobility = T{};
    for (std::size_t a = 0; a < n; ++a) {
      w.mobility[a] =
          w.density[a] * relative_permeability(a, w.saturation[a]) / phases_[a].viscosity;
      w.total_mobility = w.total_mobility + w.mobility[a];
    }
    // A defence, not a case the physics reaches: the saturations sum to one
    // while the residuals sum to less than one, so s_a < s_ra for every
    // phase would give 1 < 1. At least one phase is always strictly
    // mobile, and the guard exists only so a malformed composition cannot
    // divide by zero.
    const bool mobile = w.total_mobility > T{};
    w.mixture_enthalpy = T{};
    w.flow_density = T{};
    for (std::size_t a = 0; a < n; ++a) {
      // every phase immobile is physically a dead cell, not a division
      w.fraction[a] = mobile ? w.mobility[a] / w.total_mobility : T{};
      w.mixture_enthalpy = w.mixture_enthalpy + w.fraction[a] * w.phase_enthalpy[a];
      w.flow_density = w.flow_density + w.fraction[a] * w.density[a];
    }
    return w;
  }

  // Corey, clamped. The clamp is a branch on a value, so the mobility is
  // Lipschitz at the residual saturation rather than smooth.
  template <class T>
  T relative_permeability(std::size_t a, const T& s) const {
    using std::pow;
    const PhaseModel& m = phases_[a];
    const T se = (s - m.residual_saturation) / (1.0 - residual_total_);
    if (!(se > T{})) return T{};
    if (se >= T{1.0}) return T{m.endpoint_permeability};
    return m.endpoint_permeability * pow(se, m.corey_exponent);
  }

 private:
  double density(std::size_t a, double p) const {
    const PhaseModel& m = phases_[a];
    return m.reference_density * (1.0 + m.compressibility * (p - p_ref_));
  }
  template <class T>
  T density_of(std::size_t a, const T& p) const {
    const PhaseModel& m = phases_[a];
    return m.reference_density * (1.0 + m.compressibility * (p - p_ref_));
  }

  std::vector<PhaseModel> phases_;
  double p_ref_{0.0};
  double residual_total_{0.0};
};

}  // namespace mimetika::physics::constitutive
