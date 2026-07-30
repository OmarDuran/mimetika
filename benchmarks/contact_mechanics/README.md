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
| 1 | 3 | Vertical displaced fault, frictionless | **done** |
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


## Benchmark 1

A reservoir offset across a vertical fault (`a = 75` m, `b = 150` m, throw
`b - a = 75` m) is depleted by `-25` MPa. The throw puts reservoir against seal,
loading the fault in shear; frictionless, it slips until it carries none.

| constant | computed | paper |
|---|---|---|
| `C = (1-2nu) alpha p / (2 pi (1-nu))` | `-2.9490e6` | `-2.95e6` |
| `A = G / (2 pi (1-nu))` | `1.2171e9` | `1.2171e9` |
| `C/A` | `-0.002423` | `-0.0024` |
| peak slip `(C/A)(a-b)` | `0.18173 m` | Fig. 6 plateau |

Refinement of the peak slip (`W = H = 4500` m):

| `ny` | `dy` [m] | peak slip [m] | rel. err | L2 (profile) |
|---|---|---|---|---|
| 30 | 150.0 | 0.2006 | 10.4% | 0.255 |
| 60 | 75.0 | 0.1909 | 5.1% | 0.141 |
| 120 | 37.5 | 0.1885 | 3.7% | 0.120 |
| 180 | 25.0 | 0.1862 | 2.5% | 0.102 |

The residual gap is expected: the analytic solution assumes an **unbounded**
medium and the simulation uses the paper's finite domain. The shear traction on
the fault, by contrast, is zero to round-off on every mesh — that is the
frictionless condition itself, and it is required exactly.

### Two things this benchmark forced

**The contact law is `FrictionlessBilateral`, not `SignoriniCoulomb(friction=0)`.**
This is an *incremental* problem, and the incremental normal traction reaches
`+8.4` MPa in tension over part of the fault. The in-situ normal stress there is
about `-57` MPa, so the fault is still shut by a wide margin — but a unilateral
law applied to the increment reads that as opening and clips it. Measured: the
associative law at `mu = 0` drives the peak normal traction from `+8.44e6` to
`~0`, i.e. it opens the fault. (At `mu = 0` the Mohr–Coulomb cone degenerates to
a ray, so the associative and partial return mappings are bit-identical there;
the distinction needs a cone.)

**Picard cannot solve it; Newton solves it in 3 iterations.** A fault cutting the
whole domain is compliant over 4500 m of rock, not over its two adjacent cells,
so the geometric augmentation estimate is ~8x too stiff and the iteration
diverges. Rescaling `r` from the condensed compliance is not enough either:
`Ghat` is *dense* — every fault facet feels every other — so no scalar `r` makes
`I + r Ghat` a contraction. Semismooth Newton on `F(x) = CD(x) - x`, using the
condensed `Ghat` and the law's projection tangent, converges in 3 iterations
independently of the mesh.
