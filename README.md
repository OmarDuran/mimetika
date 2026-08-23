# mimetika

Efficient construction of **mimetic operators** for PDE, in the
differential-forms / discrete-exterior-calculus tradition, on **3D polytopal
meshes**.

The design principle: the *mimetic* structure (the discrete exterior derivative,
and the identities `curl·grad = 0`, `div·curl = 0`, discrete Stokes) is **purely
topological** and metric-free — it lives in the signed incidence matrices. All
geometric information enters *only* through the Hodge / mass (inner-product)
operators. Keeping these strictly separate is what makes the construction both
correct-by-construction and efficient.

## Architecture

Each layer has a single responsibility and depends only on the layers above it.

| Layer | Package | Responsibility |
|-------|---------|----------------|
| **Topology** | `mimetika.topology` | Combinatorial cell complex (vertex/edge/facet/cell) and signed incidence `∂ₖ`. Metric-free. Gives exact `∂ₖ∂ₖ₊₁ = 0`. Complexes of dimension 0–3. |
| **Geometry** | `mimetika.geometry` | Coordinates and metric: k-cell measures, **true** centroids, facet normals, quadrature, and the per-cell affine frame (`LocalCell`). |
| **Mesh** | `mimetika.mesh` | Container tying topology + geometry; generators, readers, and a [reference cell catalogue](src/mimetika/mesh/reference.py) with analytic measures. Cells are described by their boundary, so arbitrary polytopes flow through unchanged. |
| **DOF** | `mimetika.dof` | Numbering of differential-form degrees of freedom; single-space and mixed/blocked systems. |
| **Operators** | `mimetika.operators` | Exterior derivative `grad/curl/div` (from incidence) and the Hodge/mass interface (from geometry). |
| **Assembly** | `mimetika.assembly` | Global operators (e.g. Hodge Laplacian `Dᵀ M D`), boundary conditions, and scipy⇄PETSc backend conversion. |
| **Solver** | `mimetika.solver` | Linear solve via PETSc KSP; scipy fallback when petsc4py is absent. |
| **Postprocess** | `mimetika.postprocess` | VTK (polyhedral) export and error norms. |

## Layout

```
src/mimetika/
  topology/    geometry/   mesh/       dof/
  operators/   assembly/   solver/     postprocess/
tests/         # mirrors the source package structure
```

## The mimetic core

`CellComplex.from_polyhedra` builds the whole graded complex from a list of
cells, each given as face-loops of vertex ids. Edges and facets are deduced with
consistent signed orientation, so

```python
mesh.complex.verify_complex()   # ∂ₖ ∂ₖ₊₁ == 0, to machine zero
```

The exterior derivatives are just transposes of the incidence matrices:

```python
from mimetika.operators import grad, curl, div
curl(mesh) @ grad(mesh)   # == 0  (topological, exact)
div(mesh)  @ curl(mesh)   # == 0
```

## Mimetic inner products (Laplace & elasticity)

All geometric/material information enters through the inner-product (mass)
matrices, built with the **consistency + stability** construction shared by the
mimetic finite-difference family (Brezzi–Lipnikov–Simoncini) and its elasticity
extension (Beirão da Veiga, ESAIM M2AN 2010). Per element:

```
M_E = M1 + M2
M1 = (1/|E|) R K̄⁻¹ Rᵀ                              # consistency
M2 = s · C Cᵀ,  C = orthonormal basis of ker(Nᵀ)   # stability
```

* `N` (D×m) — DOFs of each reconstruction mode; `R` (D×m) — the moment matrix;
  `K̄` — Gram matrix of the modes. They satisfy `NᵀR = |E|K̄`.
* **Strong consistency**: `M_E N = R` exactly. This is what makes a *local mixed
  solve* reproduce exact polynomial fields — strictly stronger than the energy
  identity `NᵀMN = |E|K̄`, which alone does **not** give exact local solves on
  general polytopes.
* **Stabilization vanishes on simplices**: `M2 = 0 ⟺ dim ker(Nᵀ) = 0 ⟺ D = m`.

Columns of `R` are *canonical* when the mode admits a potential (`K⁻¹w = ∇ψ`),
giving `R_e = |e|(ψ̄_e − ψ̄_E)` by integration by parts; modes without a potential
are completed by the minimum-norm solution of `NᵀR = |E|K̄`. (Completing *every*
column that way reproduces the projection form `M1 = |E|N(NᵀN)⁻¹K̄(NᵀN)⁻¹Nᵀ`,
which is why that form fails strong consistency.)

Everything is **dimension-generic**: a `d`-cell's DOFs live on its `(d−1)`-facets
and all work happens in the cell's own affine frame ([`LocalCell`](src/mimetika/geometry/local_cell.py)).

| Problem | DOFs (`D`) | Reconstruction (`m`) | Simplex | Cube |
|---|---|---|---|---|
| **Laplace** `∫K⁻¹F·G` ([`DiffusionInnerProduct`](src/mimetika/operators/diffusion.py)) | 1/facet | constants (`m=d`) | stab 1 | stab 3 |
| **Laplace**, RT₀ | 1/facet | `{a+bξ}` (`m=d+1`) | **stab 0** | stab 2 |
| **Elasticity / AFW** `∫C⁻¹σ:τ` ([`ElasticityInnerProduct`](src/mimetika/operators/elasticity.py)) | `d²`/facet | `[P₁(E)]^{d×d}` (`m=d²(d+1)`) | **stab 0** | stab 18 |

On a simplex (`d+1` facets) the elasticity DOF count is `D = d²(d+1) = m`: the
DOFs are unisolvent for linear tensors, the stabilization vanishes, and the
scheme reduces to the AFW (BDM₁-based) mixed element. This holds in **every**
dimension — segment `2=2`, triangle `12=12`, tetrahedron `36=36`.

```python
from mimetika.mesh.reference import reference_cells
from mimetika.operators import ElasticityInnerProduct

cells = {c.name: c for c in reference_cells()}
ElasticityInnerProduct(cells["tet-reference"].mesh).stabilization_dim(0)  # -> 0
ElasticityInnerProduct(cells["cube-unit"].mesh).stabilization_dim(0)      # -> 18
```

## Patch tests (local saddle-point exactness)

Form the **local mixed problem** on one cell, impose an exact linear field as
data, **solve**, and require the discrete solution to be exact. See
[`assembly/local.py`](src/mimetika/assembly/local.py).

* **Scalar** — linear potential `p = a + b·x` ⟹ constant flux `F = −K∇p`.
  Recovers the flux DOFs and the element pressure to ~1e-15.
* **Vector** — linear displacement `u = a + Bx` ⟹ constant stress `σ = Cε(u)`.
  Recovers the stress DOFs, the mean displacement `u_E` and the weak-symmetry
  multiplier `s = skw(∇u)` to ~1e-14. Rigid-body motions produce zero stress.

Both run over the whole [reference cell collection](src/mimetika/mesh/reference.py)
— segments, polygons (incl. non-convex L-shapes and tilted planes), and
polyhedra (tet, cube, sheared hex, prism, pyramid, dented non-convex cube) —
and stay exact in the incompressible limit `λ → 10⁷`.

## Global mixed problems

[`assembly/mixed.py`](src/mimetika/assembly/mixed.py) assembles the element
operators into global saddle-point systems and solves them.

**Mixed Poisson** — normal flux per facet, pressure per cell:

```
[  M   -Bᵀ ] [ F ]   [ -g_D ]
[  B    0  ] [ p ] = [   b  ]
```

`B` is the *purely topological* discrete divergence — signed incidence scaled by
facet measures, so `(B F)_E = ∫_E div F` exactly.

**Mixed elasticity (Mimetic-AFW)** — Hellinger–Reissner with weakly imposed
symmetry, solving for **three** fields:

```
[  M    Dᵀ   Aᵀ ] [ σ ]   [ g_D ]     σ : traction moments per facet
[  D     0    0 ] [ u ] = [  f  ]     u : displacement per cell
[  A     0    0 ] [ s ]   [  0  ]     s : rotation multiplier per cell
```

The third block equation is `as_h(σ) = 0`: stress symmetry is enforced weakly,
with `s = skw(∇u)` its Lagrange multiplier.

```python
from mimetika.assembly.mixed import MixedElasticity
from mimetika.mesh import structured_tets

problem = MixedElasticity(structured_tets(4, 4, 4), mu=1.0, lam=1.0)
sol = problem.solve(body_force=f, dirichlet=u_exact)
sol["stress"], sol["displacement"], sol["rotation"]
```

Because the local inner products satisfy `M N = R`, these **global** solves are
exact for linear potentials / displacements on any mesh — verified in
[`tests/assembly/test_mixed.py`](tests/assembly/test_mixed.py).

Convergence studies live in [`examples/elliptic/`](examples/elliptic/):

```bash
python examples/elliptic/diffusion_convergence.py
python examples/elliptic/elasticity_convergence.py
```

## Quick start

```python
import numpy as np
from mimetika.mesh import structured_box
from mimetika.operators import DiagonalHodge
from mimetika.assembly import hodge_laplacian, apply_dirichlet, LinearSystem
from mimetika.solver import solve

mesh  = structured_box(8, 8, 8)                 # unit cube, 8^3 hexes
hodge = DiagonalHodge(mesh.geometry)            # metric enters here
L     = hodge_laplacian(mesh, hodge, k=0)       # weak Poisson operator

# Dirichlet Poisson (source-free): u = x on the boundary.
p = mesh.geometry.points
b = np.zeros(mesh.num_cells(0))
on = np.any((p == 0) | (p == 1), axis=1)
A, rhs = apply_dirichlet(L, b, np.where(on)[0], p[on, 0])

u = solve(LinearSystem(A, rhs))                 # PETSc KSP if available, else scipy
```

## Install & test

```bash
pip install -e ".[dev]"        # add ".[petsc]" for petsc4py + mpi4py
pytest
```

The suite (725 tests) runs without PETSc — the solver layer falls back to scipy.

## Deliberate extension points

The scaffolding is complete and tested; these are the numerics upgrades that slot
in without touching callers:

- **`CircumcentricHodge`** — the geometrically-consistent diagonal DEC star
  `*ₖ = diag(|dualₖ| / |primalₖ|)`. The current `DiagonalHodge` is a lumped
  (measure-diagonal) inner product; the polytopal consistency+stability inner
  products now live in [`operators/inner_product.py`](src/mimetika/operators/inner_product.py).
- **Higher-order form spaces** — `DofHandler.n_dofs_per_cell` and the
  local-to-global map are the only hooks that change.
- **Mesh readers** — `Mesh.from_cells` already accepts arbitrary polyhedra;
  add readers (Gmsh, VTK) that emit the cells-as-face-loops format.
```
