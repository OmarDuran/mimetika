"""A single cell viewed in its own affine frame.

Mimetic inner products are local and intrinsically ``d``-dimensional: what a
cell needs is its measure, its facets' measures/centroids/outward normals, and
quadrature -- all expressed in the ``d``-dimensional affine hull of the cell,
not in the ambient ``R^3``.  :class:`LocalCell` extracts exactly that, so the
operator code is written once and works for segments, polygons and polyhedra
alike.

Conventions
-----------
* Local coordinates are ``xi = Q^T (x - x_E)`` with ``Q`` an orthonormal basis
  of the affine hull and ``x_E`` the true centroid -- so **the cell centroid is
  the origin** of the local frame.  This is what makes the mimetic consistency
  identities take their simplest form (the element mean of a linear function
  vanishes).
* Facet normals are **outward**.  They are oriented by the star-shapedness
  assumption (``(x_e - x_E) . n_e > 0``), which is uniform across dimensions.
* ``signs[i]`` is the incidence sign of facet ``i`` in the cell, i.e. the factor
  converting a DOF expressed against the *canonical* (global) facet orientation
  into the *outward* (local) one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mimetika.geometry.metric import Geometry


@dataclass
class LocalCell:
    """A top-dimensional cell expressed in its own ``d``-dimensional frame."""

    dim: int
    volume: float
    frame: np.ndarray  # (3, d) orthonormal basis of the affine hull
    origin: np.ndarray  # (3,) the cell centroid in ambient coordinates

    facet_ids: list[int]
    facet_measures: np.ndarray  # (nf,)
    facet_centroids: np.ndarray  # (nf, d) local
    facet_normals: np.ndarray  # (nf, d) local, outward, unit
    facet_tangents: list[np.ndarray]  # each (d-1, d) local orthonormal
    facet_quadrature: list[tuple[np.ndarray, np.ndarray]]  # local pts, weights
    signs: np.ndarray  # (nf,) canonical -> outward conversion

    quad_points: np.ndarray  # (nq, d) local
    quad_weights: np.ndarray  # (nq,)

    # -- construction --------------------------------------------------------

    @classmethod
    def build(
        cls, geometry: Geometry, cell_id: int, frame: np.ndarray | None = None
    ) -> "LocalCell":
        """Build the local view of a cell.

        ``frame`` may be supplied to force a common basis across cells, which
        global assembly needs so that neighbouring cells agree on the meaning of
        the vector components of a shared facet's DOFs.
        """
        cx = geometry.complex
        d = cx.dim
        if d == 0:
            raise ValueError("0-dimensional cells carry no differential structure")

        canonical = getattr(geometry, "_mesh_frame", None)
        cache = getattr(geometry, "_localcell_cache", None)
        cacheable = frame is not None and frame is canonical
        if cacheable and cache is not None and cell_id in cache:
            return cache[cell_id]

        origin = geometry.centroids(d)[cell_id]
        frame = _affine_frame(geometry, cell_id, d) if frame is None else frame

        def to_local(x: np.ndarray) -> np.ndarray:
            return (np.atleast_2d(x) - origin) @ frame

        facets = cx.facets_of(d, cell_id)
        facet_ids = [f for f, _ in facets]
        signs = np.array([s for _, s in facets], dtype=float)

        fmeas = geometry.measure(d - 1)[facet_ids]
        fcent = to_local(geometry.centroids(d - 1)[facet_ids])

        normals, tangents, fquad = [], [], []
        for fid in facet_ids:
            normals.append(_facet_normal(geometry, fid, d, frame))
            tangents.append(_facet_tangents(geometry, fid, d, frame))
            qp, qw = geometry.quadrature(d - 1, fid)
            fquad.append((to_local(qp), qw))
        normals = np.array(normals).reshape(len(facet_ids), d)

        # Orient outward using star-shapedness (centroid is the local origin).
        flip = np.sign(np.einsum("ij,ij->i", fcent, normals))
        flip[flip == 0.0] = 1.0
        normals *= flip[:, None]

        qp, qw = geometry.quadrature(d, cell_id)
        built = cls(
            dim=d,
            volume=float(geometry.measure(d)[cell_id]),
            frame=frame,
            origin=origin,
            facet_ids=facet_ids,
            facet_measures=fmeas,
            facet_centroids=fcent,
            facet_normals=normals,
            facet_tangents=tangents,
            facet_quadrature=fquad,
            signs=signs,
            quad_points=to_local(qp),
            quad_weights=qw,
        )
        if cacheable:
            if cache is None:
                cache = {}
                try:
                    geometry._localcell_cache = cache
                except AttributeError:
                    return built
            cache[cell_id] = built
        return built

    # -- helpers -------------------------------------------------------------

    @property
    def n_facets(self) -> int:
        return len(self.facet_ids)

    def project_tensor(self, K: np.ndarray) -> np.ndarray:
        """Express an ambient ``3x3`` tensor in the local frame (``d x d``)."""
        return self.frame.T @ np.asarray(K, dtype=float) @ self.frame

    def to_local(self, x: np.ndarray) -> np.ndarray:
        """Ambient points -> local coordinates."""
        return (np.atleast_2d(x) - self.origin) @ self.frame

    def to_ambient(self, xi: np.ndarray) -> np.ndarray:
        """Local coordinates -> ambient points."""
        return np.atleast_2d(xi) @ self.frame.T + self.origin

    def facet_scalar_basis(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Linear scalar basis on facet ``i`` evaluated at its quadrature points.

        Returns ``(values (nq, d), weights (nq,))`` -- the constant ``1`` plus
        the ``d-1`` in-plane coordinates scaled by the facet diameter, i.e. the
        ``P_1`` basis on the facet.
        """
        qp, qw = self.facet_quadrature[i]
        rel = qp - self.facet_centroids[i]
        scale = np.sqrt(self.facet_measures[i]) if self.dim > 1 else 1.0
        cols = [np.ones(len(qp))]
        for t in self.facet_tangents[i]:
            cols.append(rel @ t / scale)
        return np.column_stack(cols), qw

    def expand_on_facet(self, i: int, values: np.ndarray) -> np.ndarray:
        """``L^2(e_i)`` expansion coefficients in the facet ``P_1`` basis.

        ``values`` has shape ``(nq, ...)`` (sampled at the facet quadrature
        points); the result has shape ``(d, ...)`` -- one coefficient per basis
        function.  The expansion is *exact* for functions that are linear on the
        facet, which is what the mimetic consistency identities require.
        """
        B, qw = self.facet_scalar_basis(i)
        values = np.asarray(values, dtype=float)
        gram = np.einsum("q,qa,qb->ab", qw, B, B)
        rhs = np.einsum("q,qa,q...->a...", qw, B, values)
        flat = np.linalg.solve(gram, rhs.reshape(len(gram), -1))
        return flat.reshape((len(gram),) + values.shape[1:])


def mesh_frame(geometry: Geometry) -> np.ndarray:
    """A single orthonormal ``(3, d)`` frame for the whole mesh.

    Global assembly must express every cell's DOF components in one common
    basis; for a full-dimensional (``d == 3``) mesh this is just the identity.

    The SVD fixes the *span* but not the basis within it: for a square in the
    ``xy`` plane the two in-plane singular values are equal, so the returned axes
    are an arbitrary rotation.  They are then aligned with the ambient axes,
    which costs nothing (everything downstream is frame-covariant) and buys two
    things: DOF components of a planar mesh read directly in global coordinates,
    and axis-aligned facets stay axis-aligned in the frame -- which is what
    component-wise conditions such as rollers need.
    """

    cached = getattr(geometry, "_mesh_frame", None)
    if cached is not None:
        return cached
    d = geometry.complex.dim
    if d == 3:
        return np.eye(3)
    p = geometry.points
    # reduced SVD: only the (3, 3) row space is wanted, and the default
    # full_matrices=True materialises an (N, N) left factor -- gigabytes and
    # seconds for nothing on a large point cloud
    _, _, vt = np.linalg.svd(p - p.mean(0), full_matrices=False)
    frame = _align_to_axes(vt[:d].T)
    try:
        geometry._mesh_frame = frame
    except AttributeError:  # frozen geometry: correctness does not depend on it
        pass
    return frame


def _align_to_axes(frame: np.ndarray) -> np.ndarray:
    """Rotate an orthonormal ``(3, d)`` frame to sit as close to ``e_i`` as it can.

    Greedy: repeatedly take the ambient axis with the largest remaining in-span
    projection, orthonormalise it against the axes already chosen, and keep it.
    The span is preserved exactly, so this is a change of basis and nothing more.
    """
    projector = frame @ frame.T  # onto the span, in ambient coordinates
    chosen: list[np.ndarray] = []
    remaining = list(range(3))
    for _ in range(frame.shape[1]):
        best, best_vector, best_norm = None, None, -1.0
        for axis in remaining:
            v = projector[:, axis].copy()
            for q in chosen:
                v -= (v @ q) * q
            norm = float(np.linalg.norm(v))
            if norm > best_norm:
                best, best_vector, best_norm = axis, v, norm
        if best_norm < 1e-12:  # degenerate: fall back to the SVD basis
            return frame
        chosen.append(best_vector / best_norm)
        remaining.remove(best)
    aligned = np.column_stack(chosen)
    # Preserve handedness.  The greedy pick can land on a *reflection* of the
    # original basis, and in 2D the rotation multiplier is a pseudo-scalar that
    # changes sign under one -- so a reflection here would silently flip the
    # weak-symmetry constraint.
    if np.linalg.det(frame.T @ aligned) < 0:
        aligned[:, -1] *= -1.0
    return aligned


def _affine_frame(geometry: Geometry, cell_id: int, d: int) -> np.ndarray:
    """Orthonormal ``(3, d)`` basis of the cell's affine hull."""
    if d == 3:
        return np.eye(3)
    verts = sorted(geometry.complex.cell_vertices(cell_id, d))
    p = geometry.points[verts]
    u, s, vt = np.linalg.svd(p - p.mean(0))
    # same canonicalisation as mesh_frame, so a single-cell mesh gets the same
    # basis whichever route builds it
    return _align_to_axes(vt[:d].T)  # (3, d)


def _facet_normal(
    geometry: Geometry, fid: int, d: int, frame: np.ndarray
) -> np.ndarray:
    """A unit normal of a ``(d-1)``-facet, in local coordinates (sign fixed later)."""
    if d == 1:  # facets are vertices; the local space is a line
        return np.array([1.0])
    if d == 2:  # facets are edges; rotate the edge direction by 90 degrees
        a, b = geometry.points[geometry.complex.edge_vertices[fid]]
        t = (b - a) @ frame
        t /= np.linalg.norm(t)
        return np.array([t[1], -t[0]])
    n = geometry.facet_normals()[fid]  # d == 3: polygon normal, already ambient
    return n @ frame


def _facet_tangents(
    geometry: Geometry, fid: int, d: int, frame: np.ndarray
) -> np.ndarray:
    """``(d-1, d)`` in-facet orthonormal frame, in local coordinates.

    Derived from *globally* determined data (the canonical facet normal in 3D,
    the canonical edge direction in 2D) rather than from this cell's outward
    normal, so the two cells sharing a facet agree on its DOF basis.
    """
    if d == 1:
        return np.zeros((0, 1))
    if d == 2:
        a, b = geometry.points[geometry.complex.edge_vertices[fid]]
        t = (b - a) @ frame
        return (t / np.linalg.norm(t))[None, :]
    t1, t2 = geometry.facet_tangents(fid)
    return np.array([t1 @ frame, t2 @ frame])
