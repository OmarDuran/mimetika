# Fault-reactivation contact-mechanics benchmarks

Novikov, Voskov et al. (2024), *Benchmark study of fault reactivation induced by
pressure depletion*. A depleting reservoir loads a fault; the question is when,
where, and how much it slips.

Run any benchmark as a module from the repository root:

```bash
python -m benchmarks.contact_mechanics.benchmark_0
```

Every benchmark is covered by pytest under `tests/benchmarks/`, so the published
numbers are regression-checked rather than merely printed.

## Conventions

| | |
|---|---|
| `y` | measured **upwards** from the reservoir reference level; depth is `D0 - y` |
| stress | **tension positive**, so every in-situ stress is negative |
| effective stress | `sigma' = sigma + alpha p` (Biot) |
| gap / traction | `g_n > 0` open, `t_n < 0` in compression |

## Status

| # | Section | What it covers | State |
|---|---------|----------------|-------|
| 0 | 2.4 | In-situ state and depletion response, no fault | **done** |
| 1 | 3.1 | Vertical displaced fault, frictionless | to do |
| 2 | 3.2 | Inclined 70° fault, constant friction `mu = 0.52` | to do |
| 3 | 3.3 | Slip weakening and nucleation | to do |

## Benchmark 0

Two independent claims, tested separately.

**The in-situ state is derived, not tabulated.** Everything the paper prints in
eqs. (17)–(19) follows from the Table 2 parameters:

```
rho_b = phi rho_fl + (1 - phi) rho_s
sigma_yy(y) = -rho_b g (D0 - y)          p(y) = p0 - rho_fl g y
sigma'_xx   = K0 sigma'_yy               sigma_xx = sigma'_xx - alpha p
```

Reproducing the published coefficients from the parameters — rather than pasting
them in — is what makes the setup checkable.

| field | computed | paper |
|---|---|---|
| `sigma_yy` | `-82.62e6 + 23.61e3 y` | `-82.60e6 + 23.60e3 y` |
| `sigma_xx` | `-57.06e6 + 16.33e3 y` | `-57.05e6 + 16.30e3 y` |
| `p` | `35.00e6 - 10.06e3 y` | `35.00e6 - 10.06e3 y` |
| `sigma_n` (70°) | `-60.05e6 + 17.18e3 y` | `-60.04e6 + 17.15e3 y` |
| `sigma_t` (70°) | `-8.21e6 + 2.34e3 y` | `8.21e6 - 2.35e3 y` |

The shear differs by sign only: that is which way the fault tangent is taken.

**The depletion response is simulated** and must match the closed forms exactly.
A uniformly depleted, laterally confined block compacts uniaxially:

| quantity | simulated | closed form / paper |
|---|---|---|
| `Kv = 2G(1-nu)/(1-2nu)` | — | `1.5786e10` (paper `15.79e9`) |
| `eps_yy = alpha dp / Kv` | `-1.425339e-03` | `-1.425339e-03` |
| `dh = h eps_yy` | `-0.3200 m` | `-0.32 m` |
| `d sigma'_xx` | `-3.9706e6` | `-3.97e6` |
| `d sigma_xx` | `+1.85294e7` | `18.53e6` |
| free-surface `sigma_yy` | `1.1e-07` | `0` |

Agreement is to round-off, on every mesh, because the discrete solution of a
uniform state is exact.

### Two notes on the setup

- **Fluid density.** Table 2's `rho_fl = 1020` gives a pressure gradient of
  `10.01` kPa/m, but the paper quotes `10.06` kPa/m. The latter corresponds to
  `rho_fl = 1025`, which is what `common.py` uses so the published profiles are
  reproduced exactly.
- **Reservoir thickness.** Table 2 gives the domain size but not the reservoir's
  own thickness `h`; the reported `dh = -0.32` m fixes it at `h = 224.5` m.

### Rollers

Lateral confinement is imposed with rollers — prescribed normal displacement,
free tangential slip. In Hellinger–Reissner the two halves land on opposite
sides: the displacement is *natural* (it rides along in `dirichlet`) while the
shear traction is *essential* and must be pinned, which is what
`MixedElasticity.roller_dofs` returns. Restricted to axis-aligned facets; a
general normal would need a rotated facet basis, and the code raises rather than
silently applying the wrong constraint.
