# Fault-mesh examples

Two elliptic problems solved on `examples/meshes/fault_mesh.vtu` — a genuinely
**polytopal** grid (22 056 cells, 6–24 planar faces each) of the kind produced
by cutting a corner-point grid with a fault. No reference element, no shape
functions: just facet unknowns and cell unknowns.

```bash
python examples/fault/darcy_fault.py
python examples/fault/elasticity_fault.py
```

Both run from a checkout without installing (`common.py` puts `src/` on the
path) and use PETSc automatically when `petsc4py` is importable.

## `darcy_fault.py` — Darcy flow

```
div u = f ,   u = -(K/mu) grad p ,   p = p_D on the boundary
```

Unknowns: one normal flux per facet, one pressure per cell (98 777 total).
Anisotropic mobility (bedding-parallel flow 10× the across-bedding value).

**Dirichlet data** (non-trivial, on the whole boundary):

```
p_D = 1 - x + 0.25 sin(2 pi y) + 0.4 (1 - z/0.4)
```

## `elasticity_fault.py` — linear elasticity (Mimetic-AFW)

Hellinger–Reissner with weakly imposed symmetry, solving **three** fields:
stress traction moments per facet, displacement per cell, and the rotation
multiplier `s = skw(∇u)` per cell.

**Dirichlet data** (non-trivial): depth-dependent shear in two directions plus
compaction toward the top surface.

```
u_D = ( A_s (z/Z) sin(pi y),  A_s (z/Z) sin(pi x)/2,  -A_c (1 - z/Z) )
```

The full mesh gives ~690 000 stress unknowns and ~89 M nonzeros, past what a
direct factorisation can hold, so the example runs on a **subregion by
default**. `--box XLO YLO ZLO XHI YHI ZHI` picks another region;
`--full --method minres` attempts the whole mesh.

## Options

| flag | meaning |
|---|---|
| `--backend auto\|petsc\|scipy` | `auto` uses PETSc when `petsc4py` imports |
| `--method direct\|minres` | MUMPS LU, or MINRES + Schur `fieldsplit` |
| `--rtol` | iterative tolerance |
| `--vtk out.vtk` | write cell results |

## Measured on the full mesh (PETSc 3.25.3, MUMPS)

| stage | time |
|---|---|
| read + validate mesh | ~44 s |
| assemble flux inner product | ~50 s |
| direct solve (98 777 unknowns) | **2.2 s** |
| MINRES + fieldsplit | 6 iterations, ~7 s |

## What the scripts verify

Each run ends with checks that are meaningful rather than decorative:

- **Local mass conservation** — `div_h u = 0` on *every* cell to ~1e-12,
  regardless of cell shape. This is structural: `B` is the signed incidence
  matrix times facet measures, so conservation is exact by construction.
- **Discrete maximum principle** — the computed pressure stays inside the range
  of the boundary data.
- **Equilibrium and weak symmetry** — `div_h σ = f` and `as_h σ = 0` to ~1e-14
  relative; the second is a constraint row of the system.
- **Patch test on this very mesh** — a linear pressure / linear displacement is
  reproduced to round-off (~1e-15 for Darcy, ~1e-17 for elasticity with a direct
  solve). This is the payoff from the strong-consistency property `M N = R`:
  the scheme is exact for linear fields on arbitrary polyhedra.

## Note on the mesh file

VTK does **not** require a polyhedron's face loops to be oriented outward, and
this file's are not — they arrive in mixed directions. The reader therefore
orients each loop individually (`mimetika.mesh.readers`). Without that, interior
facets fail to cancel between their two cells and `dd = 0` breaks. After the
fix, the reader's cell volumes and facet areas match the arrays stored in the
file itself to ~1e-14, and `V − E + F − C = 1` as expected for a contractible
domain.
