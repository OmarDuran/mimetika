# Porting the poroelasticity benchmarks to C++

The destination: reproduce mimetika's Terzaghi consolidation column — the
Coussy §5.2.2 closed form, cartesian and simplex families, 2D and 3D — from
the C++ model, then delete the Python it replaces.

That benchmark is the right target because it exercises every boundary
condition the harder ones rely on, exactly once, against an answer that is
known: an applied traction, rollers imposing uniaxial strain, a drained face,
and sealed faces. Get one wrong and the column stops being one-dimensional,
which the analytic profile shows immediately.

## What exists

| piece | state |
| --- | --- |
| topology, stratification | exokal, on four industrial meshes |
| spaces, global numbering, colouring | exokal |
| local AD: tangent, residual, matrix-free action | exokal, tested and profiled |
| assembly, three paths from one form | exokal |
| flux Hodge (scalar) | exokal — stabilized, de Rham/RT, conforming |
| physics packages, catalogue, closure slots | mimetika |
| **computational model** (`Simulation`) | **mimetika — built** |
| **essential constraints** | **mimetika — built** |
| **stress inner product** (Mimetic-AFW) | **exokal — built, cross-checked** |
| **`Mechanics`, `PoroCoupling` packages** | **mimetika — built** |
| **natural + essential boundary conditions** | **built — `Coupling::boundary`, `FacetSelector`, both terms** |

`Simulation` is the wiring a consumer used to redo by hand: epoch, model,
context, workspace, state and constraints behind one object, exposing
`residual`, `jacobian` and `apply`. All three respect the constraints, which
is the part a hand-wired consumer forgets on one path and not the others.

## What is missing, in dependency order

1. **A sparse linear solver.** None in either repository. The Python uses
   scipy. This is the one genuine dependency decision — see below.

2. **Column meshes** (cartesian and simplex, 2D and 3D) and **post-processing**
   (pressure at elevations, the settlement integral). Both exist in Python and
   are mechanical to port.

## The sign convention

`[M, -B^T; +B, 0]`, as the flux term uses. An off-diagonal pair is the
NEGATIVE transpose of its partner; a constitutive block is symmetric. The
Python negates its second block row instead, and is to be adapted rather than
followed: once a model composes several physics into one system, two
conventions meeting there give a matrix that is neither symmetric nor
antisymmetric, whose structure no solver can exploit — and nothing visibly
breaks.

## The order of the port

Each step ends with the C++ and the Python agreeing on something checkable, so
the Python is deleted only against evidence and never on faith.

1. ~~Natural BCs~~ and the elasticity product — both done, the second checked
   cell by cell against the Python operator.
2. Mechanics alone: a linear elasticity patch test. The pieces are in place —
   an affine displacement datum is exact at the boundary, so what the test
   measures is the method rather than the boundary integration.
4. Poroelastic coupling: the Biot blocks, adjointness, and one time step.
5. Terzaghi, against the closed form and against the Python's own numbers.
6. Delete: `simulation/poromechanics.py`, `operators/elasticity.py`,
   `operators/derham.py`, `solver/`, and the assembly stack under them.

## What stays in Python

Orchestration only: choose a model, supply closures and paths, run, plot,
compare. Nothing that evaluates physics per cell.
