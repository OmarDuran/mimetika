r"""A lumped (diagonal) stress inner product -- the TPFA analogue for elasticity.

:class:`~mimetika.operators.elasticity.ElasticityInnerProduct` produces a dense
per-cell ``M``: eliminating ``sigma`` from the saddle-point system costs as much
as solving it, so the stress stays in the system.  This module builds the
mimetic counterpart of the two-point flux approximation instead -- an ``M`` that
is **diagonal**, so each facet traction can be eliminated on its own and what
remains is a cell-centred system in ``(u, s)``.

Three things have to give way for that, and every one of them is structural
rather than incidental.

**1.  The volumetric energy cannot be lumped.**  Split the complementary energy
(the separation is due to Cockburn):

    ``(1/2) int C^{-1} sigma : sigma
      = (1/4mu) int sigma_dev : sigma_dev  +  (a term in tr sigma alone)``

The deviatoric half is facet-localisable.  The volumetric half is not: ``tr
sigma`` is an average over *all* the facets of a cell -- a cell quantity -- and
no choice of geometry makes it separate.  Hence the name of the class.  Written
as a map on ``d x d`` matrices the piece that is left out is exactly rank one,

    ``C^{-1} = (1/2mu) I_{d^2} - (a/2mu) vec(I) vec(I)^T`` ,   ``a = nu/(1-2nu+d nu)``

so it is kept **out** of ``M`` and handed downstream as one rank-one update per
cell (:meth:`volumetric_operator`).  Two consumers exist.  The three-field
assembly folds it back into ``M`` (Woodbury in matrix form), keeping the field
structure ``(sigma, u, s)`` untouched at the price of the diagonality.  The
**four-field** assembly (:mod:`mimetika.assembly.four_field`) instead promotes
the auxiliary ``n_cells``-sized variable -- which *is* the solid pressure
``p_s = tr_h(sigma)/d`` -- to an explicit unknown, so ``M`` stays diagonal and
``D`` and ``A`` still keep their usual meaning.

**2.  The stress space is reduced: ``d`` DOFs per facet, not ``d^2``.**  The
DOFs here are the *traction vector* ``int_e sigma n_e`` alone; the ``d-1`` first
moments of the traction over each facet -- the rest of the AFW facet block --
are dropped.  This is a genuinely **different discrete space**, not a
re-conditioning of the AFW one: it is those moments that make the full operator
consistent for *linear* stresses, and without them the reconstruction space
shrinks to the constants, ``m = d^2`` (see :meth:`n_modes`).  Keeping them would
defeat the purpose, because the two-point condition below constrains only the
constant-traction rows -- the moment rows would still couple facets.  The
practical consequence is a lower formal order; the practical gain is that
``sigma`` is eliminable.

**3.  The operator is consistent on orthogonal cells and nowhere else.**  With
``d_i = x_i - x_c`` the offset from the cell **collocation point** to the facet
centroid, strong consistency ``M N = R`` on constant stresses forces, facet by
facet,

    ``M_i (sigma n_i) = (1/(2 mu |e_i|)) sigma d_i`` ,

whose unique solution in the ``(n, t)`` basis is
``M_i = (1/(2 mu |e_i|)) [[d_n, d_t], [-d_t, d_n]]``.  So ``M_i`` is diagonal
**iff ``d_t = 0``**, i.e. iff ``d_i`` is parallel to ``n_i``, and then

    ``M_i = (d_n / (2 mu |e_i|)) I`` ,   positive definite iff ``d_n > 0``.

Note what is *not* there: no condition on ``nu``.  Splitting the trace off
removed the material condition and left only the geometric one -- which is why
:meth:`local_matrices` and :meth:`assemble` are independent of the Poisson
ratio, and all of the ``nu`` dependence sits in the rank-one term.

Off orthogonal cells there is **no rescue**, and this was checked rather than
assumed: the consistent block is then non-symmetric, and its antisymmetric part
is not absorbed by the rotation multiplier -- some 96% of it lies outside
``range([D^T A^T])``, and the rank of that pair jumps from 3 to 5 on a skewed
quadrilateral.  Weakening the symmetry constraint therefore cannot recover
consistency.  The class consequently **guards**: the orthogonality defect
``max_i |d_t| / |d|`` is measured on every cell at construction and a
non-orthogonal cell is an error, not a warning (see :meth:`check_orthogonality`).

Because the collocation point ``x_c`` is a free parameter of the consistency
derivation -- ``u(x_i) - u(x_c) = eps (x_i - x_c)`` holds for any ``x_c`` -- the
condition ``d || n`` can be *arranged* rather than merely hoped for.

What it asks for is an **orthogonal complex**: a polytopal mesh carrying one point
per cell such that every facet is orthogonal to the segment joining the two cell
points it separates.  That is the object, and it is a property of the (mesh, points)
pair -- not of any cell shape.  A Voronoi/PEBI complex is the general construction,
its generators being the points; a Cartesian grid is the degenerate case where the
centroids already work.  Simplices with circumcentres are a *special case* of the
same idea via Delaunay duality, useful but not the definition -- and in 3D they do
not even deliver it (see :func:`circumcentres`).

Supply the points as ``collocation``.  Because any route to them is acceptable, the
**check** rather than the construction is the interface that matters here: whatever
produced the points, :meth:`check_orthogonality` is what decides whether the operator
is legitimate on this mesh.

A caveat that bites in 3D.  ``d || n`` constrains the cell point *and* the facet
point, and this class takes the facet point to be the facet centroid, as the rest of
the library does.  Orthogonality really wants it at the orthogonal projection of
``x_c`` onto the facet plane.  On a polygon facet in 2D -- an edge -- those coincide,
which is why 2D works exactly; on a polygonal facet in 3D they differ in general.
Full 3D polytopal support therefore needs facet points to become mesh data too, so
that an orthogonal complex is a pairing of cell points *and* facet points.

.. todo:: ask the user for the precise Cockburn reference for the
   deviatoric/volumetric separation before this docstring cites one.

Degrees of freedom
------------------
``dof[f, k] = int_f (sigma n_f)_k`` with ``n_f`` the **canonical** facet normal
(out of the ``+1``-incidence cell) and ``k`` a component in the mesh frame
``self.frame``; the global index is ``d * facet_id + k``.  This is the constant
block of the AFW layout with the moment rows removed -- so a traction vector
enters as ``dof = |f| t``, exactly as it does there, but the stride is ``d``
rather than ``d^2``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from mimetika.assembly.local import skew_generators

from mimetika.geometry.local_cell import LocalCell, mesh_frame
from mimetika.mesh.mesh import Mesh
from mimetika.operators.elasticity import cell_groups
from mimetika.operators.inner_product import stabilization_dim


class LumpedDeviatoricStress:
    """Diagonal stress inner product on orthogonal cells (``d`` DOFs per facet).

    Parameters
    ----------
    mesh
        Any mesh of dimension ``1 <= d <= 3``.
    mu, lam
        Isotropic Lame parameters, used only when ``material`` is not given.
    material
        Per-cell :class:`~mimetika.materials.Material`; takes precedence.
    collocation
        ``(n_cells, 3)`` collocation points ``x_c``, one per cell -- the cell
        points of an **orthogonal complex**.  Defaults to the cell centroids,
        which already satisfy the condition on a Cartesian grid.  On a general
        polytopal mesh supply the Voronoi/PEBI generators; :func:`circumcentres`
        covers simplicial cells in 2D.  The operator is consistent exactly when
        ``x_i - x_c`` is parallel to ``n_i``; how the points were obtained is
        irrelevant, which is why the guard checks the property rather than the
        provenance.
    orthogonality_tol
        Largest tolerated orthogonality defect ``max_i |d_t|/|d|``, checked at
        construction.  The defect is a *sine*, so it never exceeds 1: passing
        ``1.0`` disables the guard, which is only ever useful for studying the
        failure it exists to prevent.
    """

    #: :meth:`assemble` deliberately **excludes** the volumetric compliance --
    #: that is what keeps it diagonal.  ``M_full = assemble() + W^T diag(c) W``
    #: with :meth:`volumetric_operator`.  See the same flag on
    #: :class:`~mimetika.operators.elasticity.ElasticityInnerProduct`.
    volumetric_included = False

    def __init__(
        self,
        mesh: Mesh,
        mu: float = 1.0,
        lam: float = 1.0,
        material: "Material | None" = None,
        collocation: np.ndarray | None = None,
        orthogonality_tol: float = 1e-9,
    ) -> None:
        from mimetika.materials import Material, poisson_from_lame

        self.mesh = mesh
        d = mesh.dim
        if d < 1:
            raise ValueError("0-dimensional cells carry no stress")
        n_cells = mesh.num_cells(d)
        if material is None:
            material = Material(
                shear_modulus=mu, poisson=poisson_from_lame(mu, lam, d)
            )
        self.material = material.expand(n_cells)
        # ``_mu`` alone enters M; ``_a`` enters only the rank-one volumetric
        # term.  That separation *is* the design -- see the module docstring.
        self._mu = np.broadcast_to(self.material.shear_modulus, (n_cells,))
        self._a = np.broadcast_to(
            self.material.compliance_coefficient(d), (n_cells,)
        )
        self.mu = float(np.mean(self._mu))
        # mirrors ElasticityInnerProduct: ``lam`` is kept as passed and is *not*
        # re-derived from ``material``; only assembly/local.py reads it.
        self.lam = lam
        self.frame = mesh_frame(mesh.geometry)

        if collocation is None:
            self.collocation = np.array(mesh.geometry.centroids(d), dtype=float)
        else:
            self.collocation = np.asarray(collocation, dtype=float)
            if self.collocation.shape != (n_cells, 3):
                raise ValueError(
                    f"collocation must be ({n_cells}, 3), got "
                    f"{self.collocation.shape}"
                )
        self.orthogonality_tol = float(orthogonality_tol)
        self._normals: np.ndarray | None = None
        self._ortho: tuple[np.ndarray, ...] | None = None
        self.check_orthogonality()

    # -- sizes ----------------------------------------------------------------

    def facet_basis_size(self, d: int) -> int:
        """Number of basis functions per facet; ``ndf = d * facet_basis_size``.

        The lumped space carries the constant traction only.
        """
        return 1

    def constant_moment_offsets(self, d: int) -> np.ndarray:
        """Offsets within a facet block holding the constant traction moment.

        The lumped space carries *only* the constant traction, one DOF per
        component (``ndf = d``), so component ``k`` is simply at ``k`` -- there are
        no higher moments to skip over.  Contrast AFW, where the same quantity
        lives at ``k * d``.
        """
        return np.arange(d)

    def dofs_per_facet(self, d: int) -> int:
        """``d`` -- one traction vector, no facet moments (module docstring, 2)."""
        return d

    def n_modes(self, d: int) -> int:
        """``d^2`` -- the reconstruction space is the *constant* stresses only.

        Dropping the facet moments removes the DOFs that a linear stress would
        need in order to be reproduced, so admitting linear modes here would
        only manufacture an inconsistency.
        """
        return d * d

    # -- geometry: offsets, normals and the orthogonality guard ----------------

    def cell_groups(self):
        """Group cell ids by facet count; see :func:`.elasticity.cell_groups`."""
        return cell_groups(self.mesh)

    def _canonical_normals(self) -> np.ndarray:
        """``(n_facets, 3)`` unit normals, each pointing out of its ``+1`` cell.

        Same convention as :meth:`Geometry.facet_frame`, and for the same
        reason: multiplying by the incidence sign then gives the *outward*
        normal of either adjacent cell, without a per-cell star-shapedness test.
        That matters here because the test used elsewhere -- ``(x_e - x_c).n >
        0`` -- is a theorem about the *centroid*, and this class deliberately
        allows ``x_c`` to be a circumcentre, which can lie outside the cell.
        """
        if self._normals is not None:
            return self._normals
        g, d = self.mesh.geometry, self.mesh.dim
        if d == 3:
            # the canonical loop is the one the +1 cell traverses outward, so
            # the polygon normal already has the right orientation
            self._normals = g.facet_normals()
            return self._normals
        if d == 2:  # facets are edges: rotate the edge direction in the plane
            ev = g.complex.edge_vertices
            t = g.points[ev[:, 1]] - g.points[ev[:, 0]]
            t /= np.linalg.norm(t, axis=1, keepdims=True)
            n = np.cross(t, np.cross(self.frame[:, 0], self.frame[:, 1]))
        else:  # d == 1: facets are vertices, the "normal" is the line direction
            n = np.broadcast_to(self.frame[:, 0], (g.complex.num_cells(0), 3))
        self._normals = n * self._plus_cell_sign(n)[:, None]
        return self._normals

    def _plus_cell_sign(self, normals: np.ndarray) -> np.ndarray:
        """``+1`` where ``normals`` already points out of the ``+1``-incidence cell."""
        g, d = self.mesh.geometry, self.mesh.dim
        bm = self.mesh.complex.boundary_matrix(d).tocsr()  # (n_facets, n_cells)
        if int(np.diff(bm.indptr).min()) < 1:
            raise ValueError("every facet must bound at least one cell")
        first = bm.indptr[:-1]
        cell, value = bm.indices[first], bm.data[first]
        away = g.centroids(d - 1) - g.centroids(d)[cell]
        out = np.where(np.einsum("fi,fi->f", away, normals) >= 0.0, 1.0, -1.0)
        return out * np.sign(value)  # a -1-incidence cell reverses the test

    def _facet_geometry(self, facet_ids, signs, cell_ids):
        """``(d_vec, n_out, area)`` in ambient ``R^3`` for a group of cells.

        ``d_vec[b, f] = x_f - x_c`` is the only place the collocation point
        enters, and ``n_out`` is outward for the cell, not canonical.
        """
        g, d = self.mesh.geometry, self.mesh.dim
        d_vec = g.centroids(d - 1)[facet_ids] - self.collocation[cell_ids][:, None, :]
        n_out = signs[..., None] * self._canonical_normals()[facet_ids]
        return d_vec, n_out, g.measure(d - 1)[facet_ids]

    def _orthogonality_data(self):
        """``(defect, defect_facet, dn_min, dn_facet)``, one entry per cell."""
        if self._ortho is not None:
            return self._ortho
        d = self.mesh.dim
        n_cells = self.mesh.num_cells(d)
        defect = np.zeros(n_cells)
        defect_facet = np.zeros(n_cells, dtype=np.int64)
        dn_min = np.zeros(n_cells)
        dn_facet = np.zeros(n_cells, dtype=np.int64)
        for facet_ids, signs, cells in self.cell_groups():
            d_vec, n_out, _ = self._facet_geometry(facet_ids, signs, cells)
            dn = np.einsum("bfi,bfi->bf", d_vec, n_out)
            length = np.linalg.norm(d_vec, axis=2)
            tangential = np.linalg.norm(d_vec - dn[..., None] * n_out, axis=2)
            # the defect is the sine of the angle between d and n, so it is
            # dimensionless and bounded by 1 -- a degenerate d = 0 scores 0 here
            # and is caught instead by the d_n > 0 test
            sine = tangential / np.where(length > 0.0, length, 1.0)
            rows = np.arange(len(cells))
            j = sine.argmax(axis=1)
            defect[cells] = sine[rows, j]
            defect_facet[cells] = facet_ids[rows, j]
            k = dn.argmin(axis=1)
            dn_min[cells] = dn[rows, k]
            dn_facet[cells] = facet_ids[rows, k]
        self._ortho = (defect, defect_facet, dn_min, dn_facet)
        return self._ortho

    def orthogonality_defects(self) -> np.ndarray:
        """``(n_cells,)`` with ``max_i |d_t| / |d|`` -- zero on an orthogonal cell.

        The quantity the operator lives or dies by: it is the sine of the angle
        between the collocation offset and the facet normal, and the lumped
        operator is consistent exactly where it vanishes.
        """
        return self._orthogonality_data()[0]

    def check_orthogonality(self, tol: float | None = None) -> None:
        """Raise unless every cell is orthogonal and every ``d_n`` is positive.

        Run at construction.  It is an error and not a warning because there is
        nothing a caller could do with a warning: off orthogonal cells the
        consistent facet block is non-symmetric and its antisymmetric part is
        *not* absorbed by the rotation multiplier, so no downstream choice
        recovers consistency (module docstring, 3).
        """
        tol = self.orthogonality_tol if tol is None else float(tol)
        defect, defect_facet, dn, dn_facet = self._orthogonality_data()
        worst = int(np.argmax(defect))
        if defect[worst] > tol:
            raise ValueError(
                f"cell {worst} is not orthogonal: the collocation offset "
                f"d = x_f - x_c makes an angle with the facet normal on facet "
                f"{int(defect_facet[worst])}, |d_t|/|d| = {defect[worst]:.3e} > "
                f"{tol:.1e}.  A lumped deviatoric stress is consistent only when "
                f"d is parallel to n; supply cell points forming an orthogonal "
                f"complex (Voronoi/PEBI generators; circumcentres for 2D "
                f"simplices) or use "
                f"ElasticityInnerProduct."
            )
        thin = int(np.argmin(dn))
        if dn[thin] <= 0.0:
            raise ValueError(
                f"cell {thin} has a non-positive collocation distance: "
                f"d.n = {dn[thin]:.3e} <= 0 on facet {int(dn_facet[thin])}.  The "
                f"collocation point is on or outside that facet's plane, so the "
                f"lumped block d_n/(2 mu |e|) I is not positive definite."
            )

    # -- local matrices --------------------------------------------------------

    def _scale(self, lc: LocalCell) -> float:
        """``|E|^(1/d)``.

        Present for API parity only: the reduced space carries no non-constant
        modes, so there is nothing to non-dimensionalise.
        """
        return float(lc.volume ** (1.0 / lc.dim))

    def _local_offset(self, lc: LocalCell, cell_id: int) -> np.ndarray:
        """The collocation point of a cell in that cell's local frame."""
        return lc.to_local(self.collocation[cell_id])[0]

    def facet_data(self, lc: LocalCell, scale: float, offset=None):
        """``(moments, expansions)``, the reduced analogue of the AFW facet data.

        ``moments[i]`` is ``(1, 1)`` holding ``|e_i| = int_{e_i} 1`` and
        ``expansions[i]`` is ``(1, d)`` holding ``d_i = x_i - x_c``: with only
        the constant facet basis function and only constant modes left, the
        ``(nb, d+1)`` and ``(nb, d)`` blocks of
        :meth:`.ElasticityInnerProduct.facet_data` each collapse to a single row.

        ``scale`` is ignored (see :meth:`_scale`).  ``offset`` is the collocation
        point in local coordinates; ``None`` means the cell centroid, which is
        what ``lc`` is built around -- a non-default collocation point must be
        passed explicitly, because a :class:`LocalCell` does not know its own id.
        """
        nf = lc.n_facets
        moments = lc.facet_measures.reshape(nf, 1, 1)
        centred = lc.facet_centroids if offset is None else lc.facet_centroids - offset
        return moments, centred[:, None, :]

    def local_matrices(self, cell_id: int, with_facet_data: bool = False):
        """``(N, R, Kbar, volume, lc)`` for one cell, in the local frame.

        With the constant modes ``T_{(r,c)} = E_{rc}`` and the *identity* half of
        the compliance, ``T -> T/2mu``:

            ``N[(i,k),(r,c)]    = delta_kr n_i[c] |e_i|``
            ``R[(i,k),(r,c)]    = delta_kr d_i[c] / 2mu``
            ``Kbar              = I_{d^2} / 2mu``

        Every column of ``R`` is canonical -- a constant stress has a linear
        potential ``v(x) = T (x - x_c) / 2mu``, and ``R`` is its value at the
        facet centroids -- so there is nothing to complete by minimum norm.

        The Gram identity ``N^T R = |E| Kbar`` holds on **any** cell, because it
        is the divergence theorem ``sum_i |e_i| n_i (x) d_i = |E| I``; it is
        strong consistency ``M N = R`` that needs orthogonality.  The missing
        ``- a vec(I) vec(I)^T / 2mu`` in ``Kbar`` is the rank-one volumetric term
        that :meth:`volumetric_operator` carries -- which is why nothing here
        depends on the Poisson ratio.
        """
        lc = LocalCell.build(self.mesh.geometry, cell_id, self.frame)
        d, nf = lc.dim, lc.n_facets
        mu = float(self._mu[cell_id])
        eye = np.eye(d)
        moments, X = self.facet_data(
            lc, self._scale(lc), self._local_offset(lc, cell_id)
        )
        N = np.einsum(
            "kr,ic,ibs->ikbsrc", eye, lc.facet_normals, moments
        ).reshape(nf * d, d * d)
        R = np.einsum("kr,ibc->ikbrc", eye, X).reshape(nf * d, d * d) / (2.0 * mu)
        Kbar = np.eye(d * d) / (2.0 * mu)
        if with_facet_data:
            return N, R, Kbar, lc.volume, lc, X
        return N, R, Kbar, lc.volume, lc

    def facet_compliances(self, cell_id: int) -> tuple[np.ndarray, list[int]]:
        """``(values, facet_ids)`` with ``values[i] = d_n / (2 mu |e_i|)``.

        The scalar multiplying the identity on facet ``i`` -- the whole content
        of the local inner product, and the half-compliance that a facet-wise
        elimination puts in series with its neighbour's.
        """
        lc = LocalCell.build(self.mesh.geometry, cell_id, self.frame)
        offset = self._local_offset(lc, cell_id)
        dn = np.einsum(
            "ic,ic->i", lc.facet_centroids - offset, lc.facet_normals
        )
        return dn / (2.0 * float(self._mu[cell_id]) * lc.facet_measures), lc.facet_ids

    def local(self, cell_id: int) -> tuple[np.ndarray, list[int]]:
        """``(M_E, facet_ids)`` in the *global* (canonical-orientation) DOF basis.

        No sign conversion appears, unlike the AFW operator: each facet block is
        a multiple of the identity, and ``s M s`` with ``s = +-1`` leaves it
        alone.  ``d_n`` is measured against the outward normal, so the block is
        orientation-independent too.
        """
        values, facet_ids = self.facet_compliances(cell_id)
        return np.diag(np.repeat(values, self.mesh.dim)), facet_ids

    def stabilization_dim(self, cell_id: int) -> int:
        """``dim ker(N^T) = d (n_facets - d)``.

        Non-zero even on a simplex, in contrast with the AFW operator: the
        reduced space has ``d`` DOFs per facet against only ``d^2`` modes, and
        the *lumping* -- not the usual ``s (I - Q Q^T)`` -- is what fixes the
        remaining directions.
        """
        N, _, _, _, _ = self.local_matrices(cell_id)
        return stabilization_dim(N)

    # -- global assembly -------------------------------------------------------

    def diagonal(self) -> np.ndarray:
        """``(d * n_facets,)`` diagonal of the global inner product.

        Interior facets receive ``d_n/(2 mu |e|)`` from *both* sides.  That sum
        of half-compliances is the elasticity counterpart of TPFA's two
        half-transmissibilities in series -- and it is the reason the eliminated
        system is cell-centred.
        """
        d = self.mesh.dim
        idx, vals = [], []
        for facet_ids, signs, cells in self.cell_groups():
            d_vec, n_out, area = self._facet_geometry(facet_ids, signs, cells)
            dn = np.einsum("bfi,bfi->bf", d_vec, n_out)
            block = dn / (2.0 * self._mu[cells][:, None] * area)  # (nB, nf)
            idx.append((facet_ids[..., None] * d + np.arange(d)).ravel())
            vals.append(np.repeat(block, d, axis=1).ravel())
        return np.bincount(
            np.concatenate(idx),
            weights=np.concatenate(vals),
            minlength=d * self.mesh.num_cells(d - 1),
        )

    def assemble(self) -> sp.csr_matrix:
        """Assemble the global stress inner product -- diagonal, by construction."""
        return sp.diags(self.diagonal(), format="csr")

    # -- the rank-one volumetric coupling --------------------------------------

    def volumetric_coupling(self, cell_id: int):
        """``(w, facet_ids, c)``: the volumetric term of one cell is ``c w w^T``.

        ``w`` is ``R vec(I)`` -- the moment column of the *hydrostatic* mode --
        laid out in the canonical DOF basis, i.e. ``w[i*d + k] = s_i d_i[k]/2mu``
        with ``s_i`` the incidence sign.  Geometrically ``w . g = |E| tr_h(sigma)
        / 2mu``: it is the discrete trace, which is exactly the functional the
        solid pressure pairs with.

        With it, the *full* compliance of the cell is

            ``M_E + c w w^T`` ,   ``c = -2 mu a / |E|`` ,

        an SPD matrix minus a rank-one term -- Woodbury territory, with an
        auxiliary system of size ``n_cells``.  Both factors are smooth in the
        material: ``c`` is bounded and ``w`` is material-independent up to the
        ``1/2mu``, and at ``nu = 1/2`` (``a = 1/d``) nothing degenerates.

        Deliberately **not** added to ``M``: baking it in would destroy the
        diagonality that is the entire point.  The four-field formulation
        (:mod:`mimetika.assembly.four_field`) is the consumer that keeps the
        diagonality: it carries the auxiliary variable explicitly as the solid
        pressure, leaving ``D`` and ``A`` -- which act on the *total* stress
        DOFs -- untouched.
        """
        lc = LocalCell.build(self.mesh.geometry, cell_id, self.frame)
        offset = self._local_offset(lc, cell_id)
        w = (lc.facet_centroids - offset) * lc.signs[:, None]
        w = w.ravel() / (2.0 * float(self._mu[cell_id]))
        c = -2.0 * float(self._mu[cell_id]) * float(self._a[cell_id]) / lc.volume
        return w, lc.facet_ids, c

    def rotation_stabilization(self, gamma: float = 1.0) -> sp.csr_matrix:
        """``S`` for the ``(s, s)`` block: ``gamma`` times the cell-graph Laplacian.

        ``s^T S s = gamma sum_f w_f (s_L - s_R)^2`` over interior facets, one copy
        per skew component, with ``w_f = |e_f| h_f / 2mu``.

        Placed in the rotation block, not in ``M``, so ``M`` stays diagonal.
        Consistent for any ``gamma``: a linear displacement has constant rotation,
        so every jump vanishes and the patch test is unaffected.
        """
        d = self.mesh.dim
        nsk = len(skew_generators(d))
        n_cells = self.mesh.num_cells(d)
        inc = self.mesh.complex.boundary_matrix(d).tocsr()
        area = self.mesh.geometry.measure(d - 1)
        vol = self.mesh.geometry.measure(d)
        rows, cols, vals = [], [], []
        for f in range(self.mesh.num_cells(d - 1)):
            cells = inc.indices[inc.indptr[f] : inc.indptr[f + 1]]
            if len(cells) != 2:
                continue
            a, b = int(cells[0]), int(cells[1])
            h = 0.5 * (vol[a] + vol[b]) ** (1.0 / d)
            w = gamma * area[f] * h / (2.0 * 0.5 * (self._mu[a] + self._mu[b]))
            for k in range(nsk):
                ia, ib = a * nsk + k, b * nsk + k
                rows += [ia, ib, ia, ib]
                cols += [ia, ib, ib, ia]
                vals += [w, w, -w, -w]
        return sp.csr_matrix(
            (vals, (rows, cols)), shape=(nsk * n_cells, nsk * n_cells)
        )

    def volumetric_operator(self) -> tuple[sp.csr_matrix, np.ndarray]:
        """``(W, c)`` with the full compliance ``M + W^T diag(c) W``.

        ``W`` is ``(n_cells, n_dofs)``, one row per cell holding that cell's
        :meth:`volumetric_coupling` vector, and ``c`` is ``(n_cells,)``.  This is
        the form a Woodbury update wants: the correction has rank ``n_cells``
        globally but rank **one per cell**, so the auxiliary system is
        cell-centred and sparse.
        """
        d = self.mesh.dim
        n_cells, n_facets = self.mesh.num_cells(d), self.mesh.num_cells(d - 1)
        rows, cols, vals = [], [], []
        coefficients = np.zeros(n_cells)
        volume = self.mesh.geometry.measure(d)
        for facet_ids, signs, cells in self.cell_groups():
            d_vec, _, _ = self._facet_geometry(facet_ids, signs, cells)
            nf = facet_ids.shape[1]
            w = (d_vec @ self.frame) * signs[..., None]
            w = w / (2.0 * self._mu[cells][:, None, None])
            rows.append(np.repeat(cells, nf * d))
            cols.append((facet_ids[..., None] * d + np.arange(d)).ravel())
            vals.append(w.ravel())
            coefficients[cells] = (
                -2.0 * self._mu[cells] * self._a[cells] / volume[cells]
            )
        W = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_cells, d * n_facets),
        )
        return W, coefficients

    # -- batched construction (many cells at once) -----------------------------

    def local_matrices_batched(self, facet_ids, signs, cell_ids):
        """``(N, R, Kbar, vol, X)`` for a whole group of equal-shaped cells.

        Mathematically identical to :meth:`local_matrices`.  Unlike the AFW
        version this works in every dimension, because the reduced space needs
        no facet second moments -- only areas, centroids and normals.
        """
        d = self.mesh.dim
        nB, nf = facet_ids.shape
        mu = self._mu[cell_ids]
        eye = np.eye(d)
        d_vec, n_out, area = self._facet_geometry(facet_ids, signs, cell_ids)
        X = (d_vec @ self.frame)[:, :, None, :]  # (nB, nf, 1, d)
        moments = area[:, :, None, None]  # (nB, nf, 1, 1)
        N = np.einsum(
            "kr,bfc,bfes->bfkesrc", eye, n_out @ self.frame, moments
        ).reshape(nB, nf * d, d * d)
        R = np.einsum("kr,bfec->bfkerc", eye, X).reshape(nB, nf * d, d * d) / (
            2.0 * mu
        )[:, None, None]
        Kbar = np.eye(d * d)[None] / (2.0 * mu)[:, None, None]
        return N, R, Kbar, self.mesh.geometry.measure(d)[cell_ids], X

    def local_inner_products_batched(self, N, R, Kbar, vol):
        """Batched diagonal ``M_E``, read straight off the consistency equations.

        Row by row, the only diagonal ``M`` that can satisfy ``M N = R`` in a
        least-squares sense is ``m = (N.R)/(N.N)``, which here evaluates to
        ``d_n/(2 mu |e_i|)`` -- the same value for all ``d`` rows of a facet, so
        the block comes out a multiple of the identity on its own.  On an
        orthogonal cell that least-squares solution is *exact*; that equivalence
        is the cleanest statement of what the guard enforces.

        ``deficient`` is always false: nothing here can be rank-deficient, and
        the scalar fallback a caller would run on a flagged cell would rebuild
        this very same diagonal.
        """
        m = np.einsum("bij,bij->bi", N, R) / np.einsum("bij,bij->bi", N, N)
        M = np.zeros((N.shape[0], N.shape[1], N.shape[1]))
        idx = np.arange(N.shape[1])
        M[:, idx, idx] = m
        return M, np.zeros(N.shape[0], dtype=bool)


def circumcentres(mesh) -> np.ndarray:
    """``(n_cells, 3)`` circumcentres of a **simplicial** mesh.

    A convenience for one cell type, **not** the definition of a collocation point.
    The concept :class:`LumpedDeviatoricStress` needs is an orthogonal complex (see
    that class); circumcentres are simply the construction that yields one when the
    cells happen to be simplices, by Delaunay duality.  On a general polytopal mesh
    the primitive is the Voronoi/PEBI generator, and this function does not apply.

    Where it does work, it works because the circumcentre is equidistant from every
    vertex, so the perpendicular from it to a facet lands on that facet's own
    circumcentre.  The centroid has no such property.

    Solved as the linear system ``2 (v_i - v_0) . x = |v_i|^2 - |v_0|^2``, which is
    the definition (equidistance) rather than a formula to be trusted on sight.

    Two warnings, both real rather than theoretical:

    * **2D only, in practice.** The perpendicular foot is the *facet's* circumcentre.
      For an edge that is its midpoint, which is also its centroid, so 2D is exact
      (defect ~1e-16). For a polygonal facet in 3D the two differ, the condition
      fails, and on a distorted tetrahedron the circumcentre is measurably *worse*
      than the centroid. Do not reach for this in 3D; supply generators instead.
    * On an **obtuse** simplex the circumcentre falls *outside* the cell and the
      offset ``d_n`` to the far facet goes non-positive, which destroys positive
      definiteness.  This is the direct analogue of TPFA's negative transmissibility
      on a non-Delaunay grid.  It is not caught here -- build the operator and call
      :meth:`LumpedDeviatoricStress.check_orthogonality`, which tests for it.
    * With ``x_c`` a circumcentre the displacement unknown is ``u(x_c)``, not the cell
      average.  Nothing downstream may assume the latter.
    """
    d = mesh.dim
    points = mesh.geometry.points
    out = np.zeros((mesh.num_cells(d), 3))
    for cell in range(mesh.num_cells(d)):
        verts = sorted(mesh.complex.cell_vertices(cell, d))
        if len(verts) != d + 1:
            raise ValueError(
                f"circumcentres need simplices: cell {cell} has {len(verts)} "
                f"vertices, expected {d + 1}"
            )
        P = points[verts][:, :d]
        A = 2.0 * (P[1:] - P[0])
        rhs = (P[1:] ** 2).sum(axis=1) - (P[0] ** 2).sum()
        out[cell, :d] = np.linalg.solve(A, rhs)
    return out
