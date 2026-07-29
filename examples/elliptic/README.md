# Elliptic examples

Convergence studies for the two elliptic problems solved in mixed
(saddle-point) form on the unit cube.

```bash
python examples/elliptic/diffusion_convergence.py
python examples/elliptic/elasticity_convergence.py
```

Both scripts run from a checkout without installing the package
(`common.py` puts `src/` on the path).

## `diffusion_convergence.py` — mixed Poisson

```
div F = f ,   F = -K grad p ,   p = p_D on the boundary
```

Unknowns: one **normal flux per facet**, one **pressure per cell**.

```
[  M   -Bᵀ ] [ F ]   [ -g_D ]
[  B    0  ] [ p ] = [   b  ]
```

`B` is the *purely topological* discrete divergence — the signed incidence
matrix scaled by facet measures, so `(B F)_E = ∫_E div F` exactly.

Four cases are reported. The isotropic grid-aligned case is included next to
the anisotropic one deliberately: its symmetry makes the leading error terms
cancel, and the flux superconverges far past the guaranteed rate. That is a
property of the configuration, not of the method — the anisotropic tensor
breaks the symmetry and shows the genuine behaviour.

## `elasticity_convergence.py` — mixed elasticity (Mimetic-AFW)

Hellinger–Reissner with **weakly imposed symmetry**, so there are *three*
unknown fields:

```
[  M    Dᵀ   Aᵀ ] [ σ ]   [ g_D ]      σ : traction moments per facet
[  D     0    0 ] [ u ] = [  f  ]      u : displacement per cell
[  A     0    0 ] [ s ]   [  0  ]      s : rotation multiplier per cell
```

`D` is the discrete divergence and `A` the discrete anti-symmetry (Beirão eqs.
(2.12), (2.19)). The third block equation is `as_h(σ) = 0`: symmetry of the
stress is enforced weakly, and `s = skw(∇u)` is its Lagrange multiplier. All
three fields are solved for and all three errors are reported.

The near-incompressible table (`λ = 1e4`) shows the rates are uniform in `λ`.
The stress errors there are large in absolute terms only because `σ` itself
scales with `λ`; the rate is unaffected, which is the point of the mixed
formulation.

## What to expect

| quantity | guaranteed | observed on these structured meshes |
|---|---|---|
| pressure `p` | O(h) | ~O(h²) (superconvergent) |
| flux `F` | O(h) | ~O(h²) anisotropic; ~O(h⁴) isotropic grid-aligned |
| displacement `u` | O(h) | ~O(h²) |
| stress `σ` | O(h) | ~O(h²) |
| rotation `s` | O(h) | ~O(h¹·⁵–h²) |

The coarsest one or two rows are preasymptotic. Rates measured against an
error already at round-off (the exact solution happened to be reproduced) are
printed as `--` rather than as a spurious number.

`stab dim` in the tables is the dimension of the local stabilization space:
**0 on tetrahedra**, where the stabilization vanishes and the schemes coincide
with RT₀ (diffusion) and AFW/BDM₁ (elasticity); positive on hexahedra, which
are genuine polytopes.
