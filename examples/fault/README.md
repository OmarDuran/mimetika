# Fault-mesh examples

Two elliptic problems solved on `examples/meshes/fault_mesh.vtu` — a genuinely
**polytopal** grid (22 056 cells, 6–24 planar faces each) of the kind produced
by cutting a corner-point grid with a fault. No reference element, no shape
functions: just facet unknowns and cell unknowns.

```bash
python examples/fault/darcy_fault.py --vtu           # -> darcy_fault.vtu
python examples/fault/elasticity_fault.py --vtu      # -> elasticity_fault.vtu
python examples/fault/darcy_fault.py --vtu out.vtu   # or name it yourself
```

Without `--vtu` the scripts solve and report but write nothing.

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
| `--vtu [PATH]` | write the post-processed solution (polyhedra preserved). Bare flag uses `darcy_fault.vtu` / `elasticity_fault.vtu` |

## Plotting: why the raw unknowns are not plottable

This is the part that trips people up. Mimetic unknowns live **on facets**, not
at cells or nodes:

| problem | facet unknown | what it is |
|---|---|---|
| Darcy | `u_e` (1 per facet) | the *normal component* of velocity, averaged over the facet |
| elasticity | 9 per facet | *moments of the traction* `σ·n_e` against the facet `P₁` basis |

Neither is a vector or tensor field — a single scalar per face cannot be
glyphed, and a traction moment is not a stress tensor. To visualise them you
must first **reconstruct a cell-centred object**. Both reconstructions come
from the same identity that underpins the method,

```
Σ_e |e| n_e (x_e − x_E)ᵀ = |E| I        (divergence theorem on x − x_E)
```

giving

```
velocity   u_E = (1/|E|) Σ_e (|e| u_e)(x_e − x_E)
stress     σ_E = (1/|E|) Σ_e t_e ⊗ (x_e − x_E) ,   t_e = ∫_e σ n_e
```

Both are **exact for constant fields** (verified to ~1e-15 in
`tests/postprocess/test_reconstruct.py`), so they are the natural partners of a
scheme that reproduces constant fluxes and stresses exactly — not a smoothing
heuristic. They live in `mimetika.postprocess`:

```python
from mimetika.postprocess import reconstruct_flux, reconstruct_stress, von_mises

velocity = reconstruct_flux(mesh, sol["flux"])         # (ncells, 3)
sigma    = reconstruct_stress(mesh, sol["stress"])     # (ncells, 3, 3)
vm       = von_mises(sigma)                            # (ncells,)
```

### What each script writes, and how to view it

`--vtu out.vtu` writes cell data with the right VTK component counts, keeping
the cells as `VTK_POLYHEDRON` so ParaView shows the true polytopal geometry.

**Darcy** — `pressure`, `velocity` (3-vector), `speed`, `cell_volume`.
In ParaView: colour by `pressure`, then **Filters → Glyph**, orientation array
`velocity`, scale array `velocity`. Use **Stream Tracer** only after a
`Cell Data to Point Data` filter, since the velocity is piecewise constant.

**Elasticity** — `displacement` (3-vector), `rotation` (3-vector), `stress`
(9-component tensor), `von_mises`, `mean_stress`, `sigma_min`, `sigma_max`.
In ParaView: colour by `von_mises`; **Glyph** by `displacement`; or
`Cell Data to Point Data` → **Warp By Vector** with `displacement` to see the
deformed shape. `stress` is a full tensor for **Tensor Glyph** or component
plots.

One honest caveat worth knowing: symmetry of the stress is imposed *weakly*,
so `as_h(σ) = 0` holds exactly on the DOFs (the scripts report ~1e-16) but the
*reconstructed* cell tensor is symmetric only to discretisation error. The
scripts print that asymmetry — it is a useful error indicator. `von_mises` and
`principal_stresses` use the symmetric part.

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
