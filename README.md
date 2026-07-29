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

- **`CircumcentricHodge`** — the geometrically-consistent diagonal DEC star
  `*ₖ = diag(|dualₖ| / |primalₖ|)`. The current `DiagonalHodge` is a lumped
  (measure-diagonal) inner product: SPD and enough to drive the full pipeline.
- **`PolytopalHodge`** — a per-cell consistency + stability inner product
  (`M = M_c + M_s`), the genuinely *mimetic-on-polytopes* mass matrix.
- **Higher-order form spaces** — `DofHandler.n_dofs_per_cell` and the
  local-to-global map are the only hooks that change.
- **Mesh readers** — `Mesh.from_cells` already accepts arbitrary polyhedra;
  add readers (Gmsh, VTK) that emit the cells-as-face-loops format.
```
