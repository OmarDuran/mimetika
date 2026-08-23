r"""Time series output for **mixed-dimensional** problems: a ``.pvd`` over ``.vtu``.

A mixed-dimensional solution does not fit in one file.  The bulk lives on
``d``-cells and the fracture on a tagged set of ``(d-1)``-facets, with their own
unknowns -- fracture pressure and flux, or contact traction and displacement
jump -- that have no bulk counterpart.  Flattening them into one grid would
either drop the fracture fields or smear them over the cells beside it.

VTK already has the right container.  A ``.pvd`` collection lists
``(timestep, part, file)`` triples, so bulk and fracture are written as separate
**parts** at the same timestep and ParaView loads them as two blocks of one
dataset: colour the fracture by its own fields, glyph its own vectors, and step
both through time together.

    <Collection>
      <DataSet timestep="0" part="0" file="run_bulk_0000.vtu"/>
      <DataSet timestep="0" part="1" file="run_fracture_0000.vtu"/>
      ...

What to plot
------------
The two physics ask for different things:

* **Darcy** (scalar).  Pressure and flux, in the bulk *and along the fracture* --
  the fracture carries its own tangential flow, which is the reason for the
  lower-dimensional unknowns.  :func:`darcy_fields`.
* **Elasticity** (vector).  On the fracture, the **traction** and the
  **displacement jump** and nothing else -- there is no fracture stress field to
  plot, because a fault is a contact interface rather than a thin material, so
  those two *are* its whole state (:func:`contact_fields`).  In the rock, the
  displacement and stress that drive it (:func:`mechanics_fields`); a fault
  plotted without its surroundings cannot be read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from mimetika.mesh.mesh import Mesh
from mimetika.postprocess.vtu import export_facets, export_vtu


@dataclass
class MixedDimensionalSeries:
    """Writes one ``.vtu`` per part per step, tracked by a single ``.pvd``.

    ``path`` is a stem: ``out/run`` gives ``out/run.pvd`` alongside
    ``out/run_bulk_0000.vtu`` and ``out/run_fracture_0000.vtu``.  The collection
    is rewritten after every step, so a run that is interrupted still leaves a
    loadable file covering the steps it completed.
    """

    path: str | Path
    mesh: Mesh
    fracture: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    _entries: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fracture = np.asarray(self.fracture, dtype=np.int64)

    @property
    def collection(self) -> Path:
        return self.path.with_suffix(".pvd")

    def write(self, time: float, bulk=None, fracture=None) -> None:
        """Append one timestep.  Either part may be omitted.

        Every cell is tagged with a ``dim`` field -- ``mesh.dim`` in the bulk and
        ``mesh.dim - 1`` on the fracture.  Once the parts are merged in ParaView
        the block structure is gone, so a ``Threshold`` on ``dim`` is the only
        way left to isolate one dimension; carrying it in the data means that
        works without knowing which block was which.
        """
        step = len(self._entries)
        written = []
        if bulk is not None:
            name = f"{self.path.name}_bulk_{step:04d}.vtu"
            export_vtu(
                self.path.parent / name, self.mesh,
                cell_data=self._tagged(bulk, self.mesh.num_cells(self.mesh.dim),
                                       self.mesh.dim),
            )
            written.append(name)
        if fracture is not None and len(self.fracture):
            name = f"{self.path.name}_fracture_{step:04d}.vtu"
            export_facets(
                self.path.parent / name, self.mesh, self.fracture,
                cell_data=self._tagged(fracture, len(self.fracture),
                                       self.mesh.dim - 1),
            )
            written.append(name)
        self._entries.append((float(time), written))
        self._write_collection()

    @staticmethod
    def _tagged(fields, n_cells: int, dim: int) -> dict:
        """Add the ``dim`` tag, unless the caller supplied one of their own."""
        out = dict(fields)
        out.setdefault("dim", np.full(n_cells, float(dim)))
        return out

    def _write_collection(self) -> None:
        rows = []
        for time, files in self._entries:
            for part, name in enumerate(files):
                rows.append(
                    f'    <DataSet timestep="{time:.17g}" group="" '
                    f'part="{part}" file="{name}"/>'
                )
        self.collection.write_text(
            "\n".join([
                '<?xml version="1.0"?>',
                '<VTKFile type="Collection" version="0.1" '
                'byte_order="LittleEndian">',
                "  <Collection>", *rows, "  </Collection>",
                "</VTKFile>", "",
            ])
        )


# -- turning a solution into plottable fields -----------------------------------------


def facet_vectors(mesh: Mesh, facets, values) -> np.ndarray:
    """Facet-frame components ``(n, dim)`` -> ambient ``(n, 3)`` vectors.

    Contact quantities are computed in each facet's own frame ``(n, t_1, ...)``,
    which is the right basis for the physics and the wrong one for a plot: a
    glyph needs ambient components.  ``facet_frame`` returns those basis vectors
    as ambient rows, so the conversion is one contraction and works in 2D and 3D
    alike.
    """
    facets = np.asarray(facets, dtype=np.int64)
    values = np.atleast_2d(np.asarray(values, dtype=float))
    out = np.zeros((len(facets), 3))
    for i, f in enumerate(facets):
        frame = mesh.geometry.facet_frame(int(f))  # (dim, 3), rows = n, t_1, ...
        out[i] = values[i, : len(frame)] @ frame
    return out


def contact_fields(driver, state) -> dict[str, np.ndarray]:
    """Fracture fields for elasticity: **traction and displacement jump only**.

    Both are given as ambient vectors (glyphable) and split into normal
    and tangential parts, which is how they are actually read: ``t_n < 0`` is
    compression, ``g_n > 0`` is opening, and the tangential magnitude is the slip.
    """
    facets = np.asarray(driver.facets, dtype=np.int64)
    traction = driver.per_facet(driver.tractions(state.solution["stress"]))
    jump = driver.per_facet(state.jump)
    mesh = driver.mesh
    return {
        "traction": facet_vectors(mesh, facets, traction),
        "jump": facet_vectors(mesh, facets, jump),
        "normal_traction": traction[:, 0],
        "shear_traction": np.linalg.norm(traction[:, 1:], axis=1),
        "opening": jump[:, 0],
        "slip": np.linalg.norm(jump[:, 1:], axis=1),
    }


def mechanics_fields(problem, solution, pressure=None) -> dict[str, np.ndarray]:
    """Bulk fields for elasticity: displacement and stress in the rock.

    The companion to :func:`contact_fields`, which covers the fracture.

    ``displacement`` is returned as an ambient 3-vector so ParaView can glyph or
    warp by it -- the DOFs live in the mesh frame, which for a 2D mesh is not the
    ambient basis.  ``pressure`` is optional and is the *load*, not a solution
    field; it is carried so the depleted region is visible.
    """
    d = problem.d
    frame = problem.inner.frame  # (3, d)
    cells = problem.n_cells

    u = np.asarray(solution["displacement"], dtype=float).reshape(cells, d)
    stress = problem.cell_stress(solution["stress"])

    fields = {
        "displacement": u @ frame.T,
        "sigma_xx": stress[:, 0, 0],
        "sigma_yy": stress[:, 1, 1],
        "sigma_xy": stress[:, 0, 1],
        "mean_stress": np.einsum("cii->c", stress) / d,
    }
    if d == 3:
        fields["sigma_zz"] = stress[:, 2, 2]
    if pressure is not None:
        fields["pressure"] = np.broadcast_to(
            np.asarray(pressure, dtype=float), (cells,)
        ).copy()
    return fields


def darcy_fields(problem, solution) -> tuple[dict, dict]:
    """``(bulk, fracture)`` fields for mixed-dimensional Darcy.

    Pressure and a reconstructed Darcy velocity on both sides.  The fracture's
    velocity is *tangential* by construction -- it is reconstructed on the
    fracture's own mesh from its own flux unknowns, so it shows the flow running
    **along** the fracture.
    """
    from mimetika.postprocess.reconstruct import reconstruct_flux

    bulk = {
        "pressure": np.asarray(solution["pressure"], dtype=float),
        "velocity": _bulk_velocity(problem, solution),
    }
    fracture = {
        "pressure": np.asarray(solution["fracture_pressure"], dtype=float),
        "velocity": reconstruct_flux(
            problem.frac_mesh, solution["fracture_flux"]
        ),
    }
    return bulk, fracture


def _bulk_velocity(problem, solution) -> np.ndarray:
    """Cell velocities from duplicated facet DOFs.

    With a duplicating dofmap a fracture facet carries one flux per side, so the
    plain reconstruction -- which indexes the flux by facet id -- would read the
    wrong unknown.  Each cell is given the DOF on *its own* side.
    """
    mesh, dofmap = problem.mesh, problem.dofmap
    d = mesh.dim
    flux = np.asarray(solution["flux"], dtype=float)
    fcent = mesh.geometry.centroids(d - 1)
    ccent = mesh.geometry.centroids(d)
    volume = mesh.geometry.measure(d)

    out = np.zeros((mesh.num_cells(d), 3))
    for c in range(mesh.num_cells(d)):
        acc = np.zeros(3)
        for fid, sign in mesh.complex.facets_of(d, c):
            dof = int(dofmap.dofs(c, fid)[0])
            acc += (sign * flux[dof]) * (fcent[fid] - ccent[c])
        out[c] = acc / volume[c]
    return out
