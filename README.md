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
| **Topology** | `mimetika.topology` | Combinatorial cell complex (vertex/edge/facet/cell) and signed incidence `∂ₖ`. Metric-free. Gives exact `∂ₖ∂ₖ₊₁ = 0`. |
| **Geometry** | `mimetika.geometry` | Coordinates and metric: k-cell measures (length/area/volume), centroids, facet normals. |
| **Mesh** | `mimetika.mesh` | Container tying topology + geometry; generators (structured hex) and readers. Cells are described as face-loops, so arbitrary polyhedra flow through unchanged. |
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
M1 = N (NᵀN)⁻¹ (|E| K̄) (NᵀN)⁻¹ Nᵀ     # consistency: exact on the reconstruction space
M2 = s · C Cᵀ,  C = orthonormal basis of ker(Nᵀ)   # stability
```

* `N` (D×m) — DOFs of each reconstruction mode; `K̄` — Gram matrix of the modes
  in the continuous inner product (both by quadrature).
* **Consistency**: for any reconstructed field `g = N c`, `gᵀM1g = |E| cᵀK̄c`
  (verified to machine precision in the tests).
* **Stabilization vanishes on simplices**: `M2 = 0 ⟺ dim ker(Nᵀ) = 0 ⟺ D = m`.

| Problem | DOFs (`D`) | Reconstruction space (`m`) | Tetrahedron | Hexahedron |
|---|---|---|---|---|
| **Laplace** `∫K⁻¹F·G` ([`DiffusionInnerProduct`](src/mimetika/operators/diffusion.py)) | 1/facet | constants (`m=3`) | stab dim 1 | stab dim 3 |
| **Laplace**, RT₀ enriched | 1/facet | `{a+bx}` (`m=4`) | **stab dim 0** | stab dim 2 |
| **Elasticity / AFW** `∫C⁻¹σ:τ` ([`ElasticityInnerProduct`](src/mimetika/operators/elasticity.py)) | 9/facet | `[P₁(E)]³ˣ³` (`m=36`) | **stab dim 0** | stab dim 18 |

The elasticity space is designed (Beirão's Remark 4.1 direction) so that on a
tetrahedron `D = 9·4 = 36 = m`: the DOFs are unisolvent for linear tensors, the
stabilization vanishes, and the scheme reduces to the AFW (BDM₁-based) mixed
element. On genuine polytopes the stabilization is active.

```python
from mimetika.mesh import single_tetrahedron, structured_box
from mimetika.operators import ElasticityInnerProduct

ElasticityInnerProduct(single_tetrahedron()).stabilization_dim(0)   # -> 0
ElasticityInnerProduct(structured_box(1,1,1)).stabilization_dim(0)  # -> 18
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

The suite (34 tests) runs without PETSc — the solver layer falls back to scipy.

## Deliberate extension points

The scaffolding is complete and tested; these are the numerics upgrades that slot
in without touching callers:

- **Saddle-point assembly** — wire the stress inner product into the full
  weakly-symmetric HR system (`div_h`, `as_h`, `tr_h` operators, eqs (2.27)) to
  solve the elasticity problem, and the flux inner product into the mixed
  Laplace system.
- **`CircumcentricHodge`** — the geometrically-consistent diagonal DEC star
  `*ₖ = diag(|dualₖ| / |primalₖ|)`. The current `DiagonalHodge` is a lumped
  (measure-diagonal) inner product; the polytopal consistency+stability inner
  products now live in [`operators/inner_product.py`](src/mimetika/operators/inner_product.py).
- **Higher-order form spaces** — `DofHandler.n_dofs_per_cell` and the
  local-to-global map are the only hooks that change.
- **Mesh readers** — `Mesh.from_cells` already accepts arbitrary polyhedra;
  add readers (Gmsh, VTK) that emit the cells-as-face-loops format.
```
