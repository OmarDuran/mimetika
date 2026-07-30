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
| `--method minres\|direct` | **default** CPR-preconditioned MINRES, or MUMPS LU |
| `--pc cpr\|schur` | `cpr` (default): AMG on the Schur block + cheap leading block, each applied once. `schur`: PETSc's defaults (inner GMRES per block) |
| `--petsc-opts="..."` | options straight to PETSc, e.g. `--petsc-opts="-ksp_view"`. Use the `=` form — argparse rejects a bare value starting with `-` |
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

## Preconditioning: a CPR analogue

CPR (Constrained Pressure Residual) solves the elliptic pressure subsystem with
AMG and follows it with a cheap global smoother. In a **mixed** formulation the
elliptic operator does not need an IMPES-style decoupling to extract — it *is*
the Schur complement `S = B diag(M)⁻¹ Bᵀ`, the cell-centred pressure Laplacian,
which PETSc's `selfp` already assembles. So the CPR recipe maps directly onto
`fieldsplit`/Schur: **AMG (hypre) on the Schur block, a cheap incomplete
factorisation on the leading block, each applied once** (`preonly`).

PETSc's *default* sub-block solvers are an inner GMRES to `rtol 1e-5` with ILU
on **each** block, re-run every outer iteration — which is why the earlier runs
took minutes despite converging in 6 outer iterations. Measured:

| problem | PETSc default | CPR (default here) |
|---|---|---|
| Darcy, 98 777 unknowns | 6 its, 5.7 s | 57 its, **0.5 s** |
| Elasticity subregion, 36 735 | 6 its, 2.4 s | 66 its, **0.28 s** |

CPR is also the *more correct* choice: MINRES requires a **fixed, SPD**
preconditioner, and a nested GMRES solve makes it variable, formally
invalidating the recurrence. `preonly` + ICC/AMG keeps it fixed and symmetric.

One negative result worth recording: `gamg` on the elasticity Schur block
produced an **indefinite** preconditioner (PETSc reason −8, answer wrong by
1.6e-1). `hypre` is used when available, ICC otherwise.

## Verifying which solver actually ran

Do not take the script's word for it — ask PETSc:

```bash
python examples/fault/darcy_fault.py --petsc-opts="-ksp_view -ksp_monitor"
```

`-ksp_view` prints PETSc's own dump of the whole solver stack (KSP type, PC
type, both sub-block solvers, the Schur approximation). The one-line summary the
scripts print is queried from the live `KSP`/`PC` objects, not hard-coded.

## Measured on the full mesh (PETSc 3.25.3)

| stage | Darcy (98 777 dof) | Elasticity `--full` (822 825 dof) |
|---|---|---|
| read + geometry + validate | ~41 s | ~41 s |
| assemble | 5 s | 20 s |
| solve (CPR-MINRES) | **0.5 s**, 72 its | **22 s**, 102 its |

Assembly was 212 s for the full elasticity problem before the local matrices
were rebuilt around the tensor-product structure of the reconstruction space
(and the duplicated pass over cells removed) — a 13.7x reduction.

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
