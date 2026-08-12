r"""De Rham (consistency-only) mimetic inner products.

The default, fully supported operators of the library: the scalar flux
product and its elasticity descendants (three- and four-field), all free of
stabilization by construction.  The names are structural: what makes the
construction work is the de Rham complex -- each field (a flux, a stress
row) is an H(div) unknown, and the divergence-free fields that complete the
local spaces to unisolvence are supplied by the exactness of the complex
(images of the curl).  The operators written in ``M_consistency + M_stab``
form (:mod:`mimetika.operators.diffusion`,
:mod:`mimetika.operators.elasticity`) are retained as examples of the
stabilized family.

Scalar member (the BDM equivalent):

The name is structural: the enrichment that makes the construction possible
is supplied by the de Rham complex -- the fields invisible to the divergence
are exactly the images of the curl.  **This operator is the BDM equivalent**:
on simplices it coincides, degree of freedom for degree of freedom, with the
classical mixed ``BDM_1``--``P_0`` finite element, and on polytopal cells it
is that element's extension.

Implements the *enriched scalar product*: facet ``P_1`` moments of the normal
flux as degrees of freedom (``d`` per facet), and a reconstruction space
``[P_1(E)]^d (+) W_E`` enlarged with divergence-free curl fields until the
degrees of freedom are unisolvent.  The local matrix is then the exact
Galerkin (mass) matrix of that space,

    ``M_E = |E| N^{-T} Kbar N^{-1}``    (``ker N^T = {0}``),

which is pure consistency: no stabilization matrix exists, by construction
rather than by cell type.  There is deliberately **no stabilized fallback**:
a cell whose enrichment cannot reach unisolvence by ``max_degree`` raises.

The enrichment is dictated by the exterior-calculus structure of the problem:
the fields invisible to the divergence are exactly the images of the curl
(``rot`` of scalar potentials in 2D, ``curl`` of vector potentials in 3D), so
enriching never perturbs the discrete equilibrium structure, and consistency
for constant fluxes survives verbatim -- the boundary pairing of a linear
potential is captured exactly by the facet ``P_1`` moments, and the volume
term dies at the centroid because every mode has cell-wise constant
divergence.

On a simplex the count ``d n_f = d (d+1)`` forbids enrichment, the local
space is ``BDM_1(E)`` and ``M_E`` its mass matrix: the method coincides,
degree of freedom for degree of freedom, with the classical mixed
``BDM_1``--``P_0`` finite element.  Off simplices, strong consistency holds
for constants (the patch test), not for linears -- the price of keeping the
degrees of freedom lowest order.

The built-in mesh quadrature is exact only for quadratics, while the Gram of
degree-``k`` enrichment needs degree ``2(k-1)``; the module therefore carries
its own high-order rules on the same signed fan/tet decompositions the
geometry layer uses.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import scipy.linalg
import scipy.sparse as sp

from mimetika.geometry.local_cell import LocalCell, mesh_frame
from mimetika.mesh.mesh import Mesh
from mimetika.operators.elasticity import ElasticityInnerProduct, cell_groups

_RANK_RTOL = 1e-10


# -- high-order reference rules (Gauss / collapsed Duffy) ---------------------


def _gauss01(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def _triangle_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Duffy rule on the reference triangle, exact for degree ``2n - 2``."""
    u, wu = _gauss01(n)
    v, wv = _gauss01(n)
    U, V = np.meshgrid(u, v, indexing="ij")
    W = np.outer(wu, wv) * (1.0 - U)  # Jacobian of y = v (1 - u)
    pts = np.column_stack([U.ravel(), (V * (1.0 - U)).ravel()])
    return pts, W.ravel()


def _tet_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Duffy rule on the reference tetrahedron, exact for degree ``2n - 3``."""
    u, wu = _gauss01(n)
    U, V, W = np.meshgrid(u, u, u, indexing="ij")
    y = V * (1.0 - U)
    z = W * (1.0 - U) * (1.0 - V)
    wt = (
        np.einsum("i,j,k->ijk", wu, wu, wu)
        * (1.0 - U) ** 2
        * (1.0 - V)
    )
    pts = np.column_stack([U.ravel(), y.ravel(), z.ravel()])
    return pts, wt.ravel()


# -- high-order quadrature on a LocalCell (signed decompositions) -------------


def _positively_oriented(w: np.ndarray) -> np.ndarray:
    """Fix the global sign of a signed decomposition: the measure is positive.

    The stored loop orientation and the cell's local frame need not agree on
    handedness; the signed fan is exact either way, up to this overall sign.
    """
    return -w if w.sum() < 0 else w


def _cell_quadrature_ho(
    geometry, cell_id: int, lc: LocalCell, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cell rule in local coordinates, apex at the centroid (the origin)."""
    cx = geometry.complex
    if lc.dim == 2:
        loop = cx.polygon_loops[cell_id]
        p = lc.to_local(geometry.points[list(loop)])
        ref, wref = _triangle_rule(n)
        pts, wts = [], []
        for i in range(len(loop)):
            a, b = p[i], p[(i + 1) % len(loop)]
            det = a[0] * b[1] - a[1] * b[0]  # 2 x signed area of (0, a, b)
            pts.append(ref @ np.vstack([a, b]))
            wts.append(wref * det)
        return np.vstack(pts), _positively_oriented(np.concatenate(wts))
    if lc.dim == 3:
        ref, wref = _tet_rule(n)
        pts, wts = [], []
        for fid, sgn in cx.facets_of(3, cell_id):
            loop = cx.polygon_loops[fid]
            if sgn < 0:
                loop = tuple(reversed(loop))
            fp = lc.to_local(geometry.points[list(loop)])
            fc = fp.mean(0)
            for i in range(len(loop)):
                a, b = fp[i], fp[(i + 1) % len(loop)]
                det = float(np.dot(fc, np.cross(a, b)))  # 6 x signed volume
                pts.append(ref @ np.vstack([fc, a, b]))
                wts.append(wref * det)
        return np.vstack(pts), _positively_oriented(np.concatenate(wts))
    raise ValueError("consistency-only product needs d in {2, 3}")


def _facet_quadrature_ho(
    geometry, fid: int, lc: LocalCell, i: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Facet rule in local coordinates (edge Gauss / polygon fan)."""
    cx = geometry.complex
    if lc.dim == 2:
        a, b = lc.to_local(geometry.points[cx.edge_vertices[fid]])
        u, w = _gauss01(n)
        return a + u[:, None] * (b - a), w * np.linalg.norm(b - a)
    loop = cx.polygon_loops[fid]
    fp = lc.to_local(geometry.points[list(loop)])
    fc = fp.mean(0)
    nrm = lc.facet_normals[i]
    ref, wref = _triangle_rule(n)
    pts, wts = [], []
    for k in range(len(loop)):
        a, b = fp[k], fp[(k + 1) % len(loop)]
        det = float(np.cross(a - fc, b - fc) @ nrm)  # 2 x signed area
        pts.append(fc + ref @ np.vstack([a - fc, b - fc]))
        wts.append(wref * det)
    p, w = np.vstack(pts), np.concatenate(wts)
    if w.sum() < 0:  # orientation of the fan vs. the outward normal
        w = -w
    return p, w


# -- reconstruction modes -----------------------------------------------------


def _p1_modes(eta: np.ndarray, d: int) -> np.ndarray:
    """``(nq, d(d+1), d)``: the ``d`` constants ``e_c`` first, then ``eta_l e_c``."""
    nq = len(eta)
    out = np.zeros((nq, d * (d + 1), d))
    for c in range(d):
        out[:, c, c] = 1.0
    for l in range(d):
        for c in range(d):
            out[:, d + l * d + c, c] = eta[:, l]
    return out


def _curl_modes(eta: np.ndarray, d: int, degree: int) -> np.ndarray:
    """``(nq, nc, d)`` divergence-free fields from degree-``degree`` potentials.

    2D: ``rot(phi) = (d2 phi, -d1 phi)`` for the homogeneous monomials ``phi``;
    3D: ``grad(phi) x e_a`` -- redundant generators (gradients have zero curl)
    are harmless, the pivoted selection discards dependent columns.
    """
    alphas = [
        tuple(np.bincount(c, minlength=d))
        for c in combinations_with_replacement(range(d), degree)
    ]

    def grad(alpha):
        g = np.zeros((len(eta), d))
        for l in range(d):
            if alpha[l] == 0:
                continue
            mono = np.ones(len(eta)) * alpha[l]
            for k in range(d):
                p = alpha[k] - (1 if k == l else 0)
                if p:
                    mono = mono * eta[:, k] ** p
            g[:, l] = mono
        return g

    fields = []
    for alpha in alphas:
        g = grad(alpha)
        if d == 2:
            fields.append(np.stack([g[:, 1], -g[:, 0]], axis=1))
        else:
            gx, gy, gz = g[:, 0], g[:, 1], g[:, 2]
            zero = np.zeros_like(gx)
            fields.append(np.stack([zero, gz, -gy], axis=1))   # grad x e_x
            fields.append(np.stack([-gz, zero, gx], axis=1))   # grad x e_y
            fields.append(np.stack([gy, -gx, zero], axis=1))   # grad x e_z
    return np.stack(fields, axis=1)


# -- the inner product --------------------------------------------------------


class DeRhamDiffusionInnerProduct:
    """Consistency-only flux inner product on facet ``P_1`` moments.

    The BDM equivalent: ``BDM_1``--``P_0`` on simplices, its polytopal
    extension elsewhere; the enrichment comes from the de Rham complex.

    ``d`` DOFs per facet, ordered ``facet * d + b`` with ``b = 0`` the constant
    moment (the average normal flux, which the discrete divergence reads) and
    ``b >= 1`` the moments against the centred, diameter-scaled in-facet
    coordinates -- the same facet basis as the elasticity product, so a stress
    row and a flux carry identical degrees of freedom.
    """

    def __init__(
        self,
        mesh: Mesh,
        K: np.ndarray | None = None,
        max_degree: int = 6,
    ) -> None:
        self.mesh = mesh
        K = np.eye(3) if K is None else np.asarray(K, dtype=float)
        if K.ndim == 3 and len(K) != mesh.num_cells(mesh.dim):
            raise ValueError("per-cell K must have one tensor per cell")
        self.K = K
        self.max_degree = int(max_degree)
        self.frame = mesh_frame(mesh.geometry)

    # -- sizes (mirrors the elasticity product's API) -------------------------

    def facet_basis_size(self, d: int) -> int:
        return d

    def dofs_per_facet(self, d: int) -> int:
        return d

    def constant_moment_offsets(self, d: int) -> np.ndarray:
        """The constant (average-flux) moment sits first in each facet block."""
        return np.array([0])

    # -- local construction ---------------------------------------------------

    def _cell_tensor(self, cell_id: int) -> np.ndarray:
        return self.K[cell_id] if self.K.ndim == 3 else self.K

    def _facet_chi(self, lc: LocalCell, i: int, pts: np.ndarray) -> np.ndarray:
        """Facet ``P_1`` basis at arbitrary local points: ``(nq, d)``."""
        rel = pts - lc.facet_centroids[i]
        scale = np.sqrt(lc.facet_measures[i])
        cols = [np.ones(len(pts))]
        for t in lc.facet_tangents[i]:
            cols.append(rel @ t / scale)
        return np.column_stack(cols)

    def _candidates(self, eta: np.ndarray, d: int, degree: int) -> np.ndarray:
        """All modes up to potential degree ``degree``: P1 then curls."""
        blocks = [_p1_modes(eta, d)]
        for deg in range(3, degree + 1):
            blocks.append(_curl_modes(eta, d, deg))
        return np.concatenate(blocks, axis=1)

    def local_matrices(self, cell_id: int):
        """``(N, G, lc, degree)``: unisolvent basis, exact Gram, geometry.

        ``N`` is square ``(d nf, d nf)`` and invertible -- selected from the
        ``P_1`` modes plus the smallest curl enrichment that reaches full
        rank.  Raises when ``max_degree`` is not enough: there is no
        stabilized fallback, by design.
        """
        N, G, lc, degree, _, _ = self._local(cell_id)
        return N, G, lc, degree

    def _local(self, cell_id: int):
        """Cached ``(N, G, lc, degree, idx, scale)``: also the selected basis.

        ``idx`` indexes the selected columns of the candidate modes, so
        ``self._candidates(pts / scale, d, degree)[:, idx]`` evaluates the
        unisolvent basis at arbitrary local points -- the trace-moment
        operator of the de Rham elasticity space needs exactly that.
        """
        hit = self._cache.get(cell_id) if hasattr(self, "_cache") else None
        if hit is not None:
            return hit

        g = self.mesh.geometry
        lc = LocalCell.build(g, cell_id, self.frame)

        if lc.dim == 2 and lc.n_facets == 3:
            out = self._local_simplex_2d(cell_id, lc)
            if not hasattr(self, "_cache"):
                self._cache = {}
            self._cache[cell_id] = out
            return out
        g = self.mesh.geometry
        lc = LocalCell.build(g, cell_id, self.frame)
        d, nf = lc.dim, lc.n_facets
        n_dof = d * nf
        scale = float(lc.volume ** (1.0 / d))
        quad_cache: dict = {}

        def facet_rule(degree: int):
            # integrand degree <= degree + 1; the rules are exact at 2n - 2
            n = degree // 2 + 2
            if n not in quad_cache:
                fq = [
                    _facet_quadrature_ho(g, lc.facet_ids[i], lc, i, n)
                    for i in range(nf)
                ]
                quad_cache[n] = (fq, [self._facet_chi(lc, i, fq[i][0])
                                      for i in range(nf)])
            return quad_cache[n]

        def dof_matrix(degree: int) -> np.ndarray:
            fquad, chi = facet_rule(degree)
            N = np.empty((n_dof, 0))
            cols = None
            for i in range(nf):
                pts, w = fquad[i]
                evals = self._candidates(pts / scale, d, degree)
                vn = np.einsum("qmc,c->qm", evals, lc.facet_normals[i])
                block = np.einsum("q,qb,qm->bm", w, chi[i], vn)
                cols = block.shape[1] if cols is None else cols
                N = np.vstack([N, block]) if N.size else block
            return N.reshape(n_dof, cols)

        m1 = d * (d + 1)
        degree, N_all = 2, dof_matrix(2)
        while np.linalg.matrix_rank(N_all, tol=_RANK_RTOL) < n_dof:
            degree += 1
            if degree > self.max_degree:
                raise RuntimeError(
                    f"cell {cell_id}: facet P1 moments not unisolvent for any "
                    f"curl enrichment up to degree {self.max_degree}; the "
                    "consistency-only product has no stabilization to fall "
                    "back on -- raise max_degree"
                )
            N_all = dof_matrix(degree)

        # keep the whole P1 block, complete it with pivoted-QR-selected curls
        N1 = N_all[:, :m1]
        Q1, _ = np.linalg.qr(N1)
        if np.linalg.matrix_rank(N1, tol=_RANK_RTOL) < m1:
            raise RuntimeError(
                f"cell {cell_id}: the P1 modes themselves are rank deficient "
                "on this cell"
            )
        proj = N_all[:, m1:] - Q1 @ (Q1.T @ N_all[:, m1:])
        _, _, piv = scipy.linalg.qr(proj, mode="economic", pivoting=True)
        # Sorted, so congruent cells produce identical N and G.  LAPACK breaks
        # equal-magnitude pivots by encounter order, which round-off flips from
        # cell to cell: on a structured quad mesh the same two curl modes were
        # selected as (7, 8) or (8, 7) at random.  M is invariant under the
        # permutation, but N and G are not, which defeats caching on congruent
        # cells and makes the operator reproducible only up to that permutation.
        # Sorting fixes the order; it does not fix which columns are chosen when
        # two candidates are near-tied in magnitude.
        sel = m1 + np.sort(piv[: n_dof - m1])
        idx = np.concatenate([np.arange(m1), sel])
        N = N_all[:, idx]

        # exact Gram of the selected basis: cell integrand degree 2(degree - 1)
        qp, qw = _cell_quadrature_ho(g, cell_id, lc, degree + 1)
        evals = self._candidates(qp / scale, d, degree)[:, idx]
        Kinv = np.linalg.inv(lc.project_tensor(self._cell_tensor(cell_id)))
        Kw = (evals @ Kinv.T) * qw[:, None, None]
        G = np.einsum("qic,qjc->ij", evals, Kw, optimize=True)
        out = (N, G, lc, degree, idx, scale)
        if not hasattr(self, "_cache"):
            self._cache = {}
        self._cache[cell_id] = out
        return out

    def _local_simplex_2d(self, cell_id: int, lc: LocalCell):
        """Closed-form ``(N, G, lc, degree, idx, scale)`` on triangles.

        No enrichment exists on a simplex, and every integral of the P1
        modes against the edge basis is polynomial: the degree-of-freedom
        matrix follows from the edge moments and the Gram from the cell
        second moments -- no high-order quadrature, no rank machinery.
        """
        d, nf = 2, 3
        scale = float(lc.volume ** 0.5)
        L = lc.facet_measures
        xe = lc.facet_centroids
        t = np.stack([tt[0] for tt in lc.facet_tangents])
        # edge moments of the mode scalars {1, eta_1, eta_2}
        mom = np.zeros((nf, 2, 3))
        mom[:, 0, 0] = L
        mom[:, 0, 1:] = L[:, None] * xe / scale
        mom[:, 1, 1:] = t * (L**2.5 / 12.0)[:, None] / scale
        # N[(i, b), s*d + c] = n_c * mom[i, b, s]
        N = np.einsum("ibs,ic->ibsc", mom, lc.facet_normals).reshape(
            2 * nf, 3 * d
        )
        Kinv = np.linalg.inv(lc.project_tensor(self._cell_tensor(cell_id)))
        S2 = np.einsum(
            "q,qi,qj->ij", lc.quad_weights, lc.quad_points, lc.quad_points
        )
        Mphi = np.zeros((3, 3))
        Mphi[0, 0] = lc.volume
        Mphi[1:, 1:] = S2 / scale**2
        G = np.kron(Mphi, Kinv)
        return N, G, lc, 2, np.arange(3 * d), scale

    def local(self, cell_id: int) -> tuple[np.ndarray, list[int]]:
        """``(M_E, facet_ids)`` in the global (canonical-orientation) basis."""
        N, G, lc, _ = self.local_matrices(cell_id)
        Y = np.linalg.solve(N.T, G)
        M = np.linalg.solve(N.T, Y.T).T  # N^{-T} G N^{-1}: pure consistency
        M = 0.5 * (M + M.T)
        s = np.repeat(lc.signs, lc.dim)
        return M * s[:, None] * s[None, :], lc.facet_ids

    def enrichment_degree(self, cell_id: int) -> int:
        """Smallest potential degree whose curls make the cell unisolvent."""
        return self.local_matrices(cell_id)[3]

    def canonical_constant_moments(self, cell_id: int) -> np.ndarray:
        """``R`` columns of the ``d`` constant modes, in the local basis.

        ``R[(i, b), c]`` is the coefficient of the linear potential
        ``psi_c = (K^{-1} e_c) . xi`` in the facet basis: the identity
        ``M N e_c = R e_c`` is the (S2) consistency the construction
        guarantees, and the tests verify.
        """
        g = self.mesh.geometry
        lc = LocalCell.build(g, cell_id, self.frame)
        d = lc.dim
        Kinv = np.linalg.inv(lc.project_tensor(self._cell_tensor(cell_id)))
        R = np.empty((d * lc.n_facets, d))
        for i in range(lc.n_facets):
            qp, _ = lc.facet_quadrature[i]
            psi = qp @ Kinv  # column c: (K^{-1} e_c) . xi at the quad points
            R[i * d : (i + 1) * d] = lc.expand_on_facet(i, psi)
        return R

    # -- global assembly ------------------------------------------------------

    def assemble(self) -> sp.csr_matrix:
        """Assemble over all facet DOFs (``d`` per facet, facet-major)."""
        d = self.mesh.dim
        n = d * self.mesh.num_cells(d - 1)
        rows, cols, vals = [], [], []
        for cid in range(self.mesh.num_cells(d)):
            M, fids = self.local(cid)
            gd = (d * np.asarray(fids)[:, None] + np.arange(d)).ravel()
            rows.append(np.repeat(gd, len(gd)))
            cols.append(np.tile(gd, len(gd)))
            vals.append(M.ravel())
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        )


class DeRhamDeviatoricStress:
    """Deviatoric stress inner product: ``d`` copies of the enriched scalar."""

    #: the volumetric part is carried by :meth:`volumetric_operator`, exactly
    #: as for the lumped space; assembly code keys off this flag.
    volumetric_included = False

    def __init__(
        self,
        mesh,
        mu: float = 1.0,
        lam: float = 1.0,
        material=None,
        max_degree: int = 6,
    ) -> None:
        # the AFW sibling supplies everything that is metric-independent of the
        # inner product: facet data (X), the volumetric pair (W, c), materials
        self._afw = ElasticityInnerProduct(mesh, mu=mu, lam=lam, material=material)
        self.mesh = mesh
        self.frame = self._afw.frame
        self.material = self._afw.material
        self._mu, self._a = self._afw._mu, self._afw._a
        self.mu, self.lam = self._afw.mu, self._afw.lam
        # one scalar consistency-only product with per-cell K = 2 mu I: its
        # Gram is the identity-part pairing (1/2mu) v . w of one stress row
        K = 2.0 * np.asarray(self._mu)[:, None, None] * np.eye(3)[None]
        self._scalar = DeRhamDiffusionInnerProduct(mesh, K=K, max_degree=max_degree)
        self._cache: dict[int, tuple] = {}

    # -- sizes (AFW layout) ----------------------------------------------------

    def facet_basis_size(self, d: int) -> int:
        return d

    def dofs_per_facet(self, d: int) -> int:
        return d * d

    def constant_moment_offsets(self, d: int) -> np.ndarray:
        return np.arange(d) * d

    # -- delegation to the AFW sibling ----------------------------------------

    def facet_data(self, lc, scale):
        return self._afw.facet_data(lc, scale)

    def _scale(self, lc) -> float:
        return self._afw._scale(lc)

    def volumetric_operator(self):
        """``(W, c)``: identical to the AFW pair -- it depends only on ``R``."""
        return self._afw.volumetric_operator()

    def cell_groups(self):
        return cell_groups(self.mesh)

    # -- local construction ----------------------------------------------------

    def _scalar_local(self, cell_id: int):
        """Cached scalar ``(N_s, G_s, lc, degree)`` -- the costly part."""
        return self._scalar._local(cell_id)[:4]

    def _row_indices(self, d: int, nf: int) -> list[np.ndarray]:
        """Full-layout DOF indices of row ``k``: ``facet d^2 + k d + basis``."""
        base = np.arange(nf)[:, None] * d * d + np.arange(d)[None, :]
        return [(base + k * d).ravel() for k in range(d)]

    def local_matrices(self, cell_id: int, with_facet_data: bool = False):
        """``(N, R, Kbar, vol, lc[, X])`` in the full ``d^2``-per-facet layout.

        ``N`` is square and invertible (the row blocks are), ``R`` is the
        unique solution of ``N^T R = |E| Kbar`` -- no completion step exists --
        and :func:`assemble_local_inner_product` on these returns pure
        consistency: the stabilization branch is unreachable.
        """
        Ns, Gs, lc, _ = self._scalar_local(cell_id)
        d, nf, vol = lc.dim, lc.n_facets, lc.volume
        n = d * d * nf
        m = Ns.shape[0]  # d nf scalar DOFs = scalar modes per row
        N = np.zeros((n, n))
        R = np.zeros((n, n))
        Kbar = np.zeros((n, n))
        Rs = np.linalg.solve(Ns.T, Gs)  # scalar R: N_s^{-T} G_s
        for k, idx in enumerate(self._row_indices(d, nf)):
            cols = slice(k * m, (k + 1) * m)
            N[np.ix_(idx, range(k * m, (k + 1) * m))] = Ns
            R[np.ix_(idx, range(k * m, (k + 1) * m))] = Rs
            Kbar[cols, cols] = Gs / vol
        if with_facet_data:
            _, X = self._afw.facet_data(lc, self._afw._scale(lc))
            return N, R, Kbar, vol, lc, X
        return N, R, Kbar, vol, lc

    def local(self, cell_id: int) -> tuple[np.ndarray, list[int]]:
        """``(M_E, facet_ids)`` in the global (canonical-orientation) basis."""
        Ns, Gs, lc, _ = self._scalar_local(cell_id)
        d, nf = lc.dim, lc.n_facets
        Ms = np.linalg.solve(Ns.T, np.linalg.solve(Ns.T, Gs).T).T
        Ms = 0.5 * (Ms + Ms.T)
        M = np.zeros((d * d * nf, d * d * nf))
        for idx in self._row_indices(d, nf):
            M[np.ix_(idx, idx)] = Ms
        s = np.repeat(lc.signs, d * d)
        return M * s[:, None] * s[None, :], lc.facet_ids

    def stabilization_dim(self, cell_id: int) -> int:
        """Zero on every cell: the scalar layer raises where it cannot be."""
        self._scalar_local(cell_id)
        return 0

    def enrichment_degree(self, cell_id: int) -> int:
        return self._scalar_local(cell_id)[3]

    def trace_moment_operator(self):
        """``(P, Gvol)``: the ``P_1`` trace moments and their Gram blocks.

        ``P`` maps canonical stress DOFs to the coefficients of the ``P_1``
        projection of ``tr R`` in the cell basis ``{1, xi / h}``; ``Gvol`` is
        block diagonal with the exact ``P_1`` Grams.  The folded member
        ``M_row - (a/2mu) P^T Gvol P`` equals the AFW product on simplices,
        where ``tr R`` *is* linear -- the linear solid pressure of
        :class:`~mimetika.assembly.four_field.FourFieldElasticity` carries
        exactly these moments.
        """
        d = self.mesh.dim
        nb = d + 1
        n_cells = self.mesh.num_cells(d)
        n_stress = d * d * self.mesh.num_cells(d - 1)
        rP, cP, vP = [], [], []
        rG, cG, vG = [], [], []
        for cid in range(n_cells):
            N, _, lc, degree, idx, scale = self._scalar._local(cid)
            qp, qw = _cell_quadrature_ho(self.mesh.geometry, cid, lc, degree + 1)
            evals = self._scalar._candidates(qp / scale, d, degree)[:, idx]
            phi = np.column_stack([np.ones(len(qp)), qp / scale])
            Gv = np.einsum("q,qa,qb->ab", qw, phi, phi)
            Ninv = np.linalg.inv(N)
            tr_map = np.zeros((len(qp), d * d * lc.n_facets))
            for k, dofidx in enumerate(self._row_indices(d, lc.n_facets)):
                tr_map[:, dofidx] = evals[:, :, k] @ Ninv
            Pc = np.linalg.solve(Gv, np.einsum("q,qa,qn->an", qw, phi, tr_map))
            Pc = Pc * np.repeat(lc.signs, d * d)[None, :]
            gd = (
                d * d * np.asarray(lc.facet_ids)[:, None] + np.arange(d * d)
            ).ravel()
            rP.append(np.repeat(nb * cid + np.arange(nb), len(gd)))
            cP.append(np.tile(gd, nb))
            vP.append(Pc.ravel())
            block = nb * cid + np.arange(nb)
            rG.append(np.repeat(block, nb))
            cG.append(np.tile(block, nb))
            vG.append(Gv.ravel())
        shapeP = (nb * n_cells, n_stress)
        P = sp.csr_matrix(
            (np.concatenate(vP), (np.concatenate(rP), np.concatenate(cP))),
            shape=shapeP,
        )
        Gvol = sp.csr_matrix(
            (np.concatenate(vG), (np.concatenate(rG), np.concatenate(cG))),
            shape=(nb * n_cells, nb * n_cells),
        )
        return P, Gvol

    # -- batched interface (the 3D assembly path loops through here) ----------

    def facet_data_batched(self, facet_ids, cells):
        return self._afw.facet_data_batched(facet_ids, cells)

    def _triangle_ms_batched(self, facet_ids, cells):
        """Stacked scalar products ``Ms`` on a group of 2D triangles.

        The closed forms of ``_local_simplex_2d`` vectorised over the group:
        canonical edge tangents, star-shapedness outward normals, edge
        moments, and the exact Gram from the triangle second moments
        ``S2 = (A/12) sum_k v_k v_k^T`` (centroid origin).  With the
        isotropic scalar tensor ``K = 2 mu I`` the frame projection is
        ``2 mu I_2``, so no per-cell inversion appears anywhere; the whole
        group reduces to one stacked ``solve``.
        """
        g = self.mesh.geometry
        Q = self.frame
        pts = g.points
        ev = self.mesh.complex.edge_vertices
        nB = len(cells)
        area = g.measure(2)[cells]
        scale = np.sqrt(area)
        cent = g.centroids(2)[cells]

        a = pts[ev[facet_ids, 0]]
        b = pts[ev[facet_ids, 1]]
        e = (b - a) @ Q  # (nB, 3, 2), canonical direction
        L = np.linalg.norm(e, axis=2)
        t = e / L[..., None]
        n = np.stack([t[..., 1], -t[..., 0]], axis=-1)
        xe = (0.5 * (a + b) - cent[:, None, :]) @ Q
        flip = np.sign(np.einsum("bic,bic->bi", xe, n))
        flip[flip == 0.0] = 1.0
        n = n * flip[..., None]  # outward

        # edge moments of the mode scalars {1, eta_1, eta_2}
        mom = np.zeros((nB, 3, 2, 3))
        mom[:, :, 0, 0] = L
        mom[:, :, 0, 1:] = L[..., None] * xe / scale[:, None, None]
        mom[:, :, 1, 1:] = (t * (L ** 2.5 / 12.0)[..., None]
                            / scale[:, None, None])
        Ns = np.einsum("birs,bic->birsc", mom, n).reshape(nB, 6, 6)

        loops = np.array([self.mesh.complex.polygon_loops[int(c)]
                          for c in cells])
        v = (pts[loops] - cent[:, None, :]) @ Q  # centred vertices (nB, 3, 2)
        S2 = np.einsum("bkc,bkd->bcd", v, v) * (area / 12.0)[:, None, None]
        Mphi = np.zeros((nB, 3, 3))
        Mphi[:, 0, 0] = area
        Mphi[:, 1:, 1:] = S2 / area[:, None, None]  # scale^2 = area
        Kinv = np.eye(2)[None] / (2.0 * np.asarray(self._mu)[cells,
                                                             None, None])
        Gs = np.einsum("bpq,bcd->bpcqd", Mphi, Kinv).reshape(nB, 6, 6)

        NsT = np.swapaxes(Ns, 1, 2)
        Ms = np.linalg.solve(NsT, np.swapaxes(np.linalg.solve(NsT, Gs), 1, 2))
        return 0.5 * (Ms + np.swapaxes(Ms, 1, 2))

    def assemble(self) -> sp.csr_matrix:
        """Assemble over all facet DOFs (``d^2`` per facet, AFW layout).

        On 2D meshes the triangle groups go through the batched closed
        forms; anything else (polygons, 3D) takes the per-cell path.
        """
        d = self.mesh.dim
        ndf = d * d
        n = ndf * self.mesh.num_cells(d - 1)
        rows, cols, vals = [], [], []
        for facet_ids, signs_g, cells in self.cell_groups():
            nf = facet_ids.shape[1]
            if d == 2 and nf == 3:
                Ms = self._triangle_ms_batched(facet_ids, cells)
                s6 = np.repeat(signs_g, 2, axis=1)
                Ms = Ms * s6[:, :, None] * s6[:, None, :]
                for k in range(d):
                    # global dofs of row k: facet d^2 + k d + basis
                    gdk = (ndf * facet_ids[:, :, None]
                           + k * d + np.arange(d)).reshape(len(cells), 6)
                    rows.append(np.repeat(gdk, 6, axis=1).ravel())
                    cols.append(np.tile(gdk, (1, 6)).ravel())
                    vals.append(Ms.ravel())
                continue
            for cid in cells:
                M, fids = self.local(int(cid))
                gd = (ndf * np.asarray(fids)[:, None] + np.arange(ndf)).ravel()
                rows.append(np.repeat(gd, len(gd)))
                cols.append(np.tile(gd, len(gd)))
                vals.append(M.ravel())
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        )

    def local_matrices_batched(self, facet_ids, signs, cells):
        """Stacked ``(N, R, Kbar, vol, X)`` for a group of equal-``nf`` cells.

        Enrichment selection is cell-by-cell (degrees may differ inside a
        group), so this stacks the scalar path; the shapes agree because the
        selected basis always has exactly ``d nf`` modes per row.
        """
        outs = [self.local_matrices(int(c), with_facet_data=True) for c in cells]
        N = np.stack([o[0] for o in outs])
        R = np.stack([o[1] for o in outs])
        Kbar = np.stack([o[2] for o in outs])
        vol = np.array([o[3] for o in outs])
        X = np.stack([o[5] for o in outs])
        return N, R, Kbar, vol, X

    def local_inner_products_batched(self, N, R, Kbar, vol):
        """``M = N^{-T} (|E| Kbar) N^{-1}`` per cell; nothing is ever deficient."""
        Mloc = np.empty_like(N)
        for b in range(len(N)):
            G = vol[b] * Kbar[b]
            Mloc[b] = np.linalg.solve(
                N[b].T, np.linalg.solve(N[b].T, G).T
            ).T
            Mloc[b] = 0.5 * (Mloc[b] + Mloc[b].T)
        return Mloc, np.zeros(len(N), dtype=bool)

class DeRhamElasticityInnerProduct(DeRhamDeviatoricStress):
    """The full AFW compliance built as ``d`` copies of the BDM product.

    ``assemble()`` returns ``M_row - (a/2mu) P^T Gvol P``: the block-diagonal
    row-wise (BDM) part plus the exact ``P_1`` trace-energy term.  On a
    simplicial mesh this **is** the Arnold--Falk--Winther product, entry by
    entry -- the implementation reflects the structural fact that AFW is
    ``d`` copies of the ``BDM_1`` inner product coupled only through the
    trace of the compliance; off simplices it is the row-wise member with
    the ``P_1``-projected trace energy.  Drop-in three-field usage:
    ``MixedElasticity(mesh, inner=DeRhamElasticityInnerProduct(mesh, ...))``.
    """

    #: full compliance: the three-field assembly folds nothing back in, and
    #: the four-field assembly strips the constant volumetric part as for AFW.
    volumetric_included = True

    def _trace_scaling(self) -> sp.dia_matrix:
        return sp.diags(
            np.repeat(np.asarray(self._a) / (2.0 * np.asarray(self._mu)),
                      self.mesh.dim + 1)
        )

    def assemble(self) -> sp.csr_matrix:
        M = super().assemble()
        P, Gvol = self.trace_moment_operator()
        return (M - P.T @ (self._trace_scaling() @ Gvol) @ P).tocsr()
