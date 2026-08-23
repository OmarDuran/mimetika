r"""The linear pressure patch test under the Riesz map, against a direct factorization.

The pair is a d-cochain p and a (d-1)-cochain q on the mesh complex,

    star_K q - d^T p = 0 ,        d q = f ,

with d the exterior derivative -- signed incidence, integer, metric-free -- and
star_K the discrete Hodge star on (d-1)-cochains.  K enters star_K and nothing
else, so a jump in K is a jump in the metric and not in the complex.

The datum is affine.  Then dp is constant, q = -K dp is constant, and the field
lies in the lowest-order space exactly, so the direct answer is its interpolant
wherever the star reproduces it.  Both solvers see the SAME discrete system, so a
disagreement between them is the iterative method and never the discretization --
which is what makes the direct solve usable as the reference.

The Riesz map is the Gram matrix of the norm in which the operator is an
isomorphism,

    ||q||^2 = (star_K^-1 q, q) + ||d q||^2 ,     ||p||^2 = ||p||_{L2}^2 ,

so its iteration count is bounded by the inf-sup constant of that pairing rather
than by h.  Both terms carry K: star_K^-1 by construction, and ||d q||^2 through
the multiplier weight W, which stands for the Schur complement B star_K^-1 B^T
and so scales with K as well.

The five products differ in the space star_K is built on, not in d:

    derham_bdm      d moments per facet, [P_1]^d
    derham_rt       one flux per facet; RT_0 pinned, div-free enrichment
    stabilized_rt   one flux per facet; span{K e_i} with the kernel penalized
    diagonal_tpfa   one flux per facet; the diagonal primal-dual star
    adaptive_rt     the per-cell selection between the last two, eta in {0, 1}
"""

import numpy as np
import pytest

import mimetika_cxx as mk

DIRECT = mk.SolverOptions()
RIESZ = mk.SolverOptions(
    method="gmres", preconditioner="riesz", rtol=1e-10, max_iterations=2000
)

PRODUCTS = ("derham_bdm", "derham_rt", "stabilized_rt", "diagonal_tpfa", "adaptive_rt")

# The families differ in what they ask of star_K, not of d: a cartesian cell is
# K-orthogonal, a simplex generally is not, and the annulus is curved and graded.
LADDERS = {
    "cartesian_2d": (2, [mk.box([n, n, 1], 2, mk.Family.cartesian, [1.0, 1.0, 1.0])
                         for n in (4, 8, 16)]),
    "simplex_2d": (2, [mk.box([n, n, 1], 2, mk.Family.simplex, [1.0, 1.0, 1.0])
                       for n in (4, 8, 16)]),
    "cartesian_3d": (3, [mk.box([n, n, n], 3, mk.Family.cartesian, [1.0, 2.0, 3.0])
                         for n in (3, 4, 6)]),
    "simplex_3d": (3, [mk.box([n, n, n], 3, mk.Family.simplex, [1.0, 2.0, 3.0])
                       for n in (2, 3, 4)]),
    "annulus_2d": (2, [mk.annulus(n, 2 * n, 2, mk.Family.simplex, 1.0, 3.0, 1.0)
                       for n in (3, 5, 8)]),
}

# Where each product reproduces the affine field, measured rather than assumed.
#
#   derham_rt, stabilized_rt   everywhere: both are unisolvent on the constant
#                              flux the datum induces, on every cell here
#   derham_bdm                 the datum is ONE scalar per facet and loads the
#                              constant moment alone, so the d-1 higher moments
#                              take zero. Harmless where the cell is a box,
#                              first order on a simplex
#   diagonal_tpfa              only where the mesh is K-orthogonal. The
#                              Freudenthal cut of a square IS, hence simplex_2d;
#                              its 3D counterpart is not
#   adaptive_rt                the scan flags nothing on these meshes, so eta is
#                              one everywhere and the product is stabilized_rt.
#                              Its other selections are exercised below
EXACT_ON = {
    "derham_bdm": {"cartesian_2d", "cartesian_3d"},
    "derham_rt": set(LADDERS),
    "stabilized_rt": set(LADDERS),
    "diagonal_tpfa": {"cartesian_2d", "cartesian_3d", "simplex_2d"},
    "adaptive_rt": set(LADDERS),
}

CASES = [(p, f) for p in PRODUCTS for f in sorted(LADDERS)]


def checkerboard(n_cells, contrast, seed=0):
    """K in {1, 1/contrast}, scattered: a jump on roughly half the interior facets.

    Scattered rather than layered on purpose -- a layered K leaves most facets
    with equal K on both sides, so the star is only locally perturbed.
    """
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n_cells) < 0.5, 1.0, 1.0 / contrast)


def linear_patch(mesh, dim, product, contrast=1.0, degeneracy=None):
    """p = ((x - x_min) . n)/L on every boundary facet: pure Dirichlet, no nullspace."""
    model = mk.FlowModel(mesh, dim, 1.0, getattr(mk.FluxRealization, product))
    if contrast != 1.0:
        model.set_permeability(list(checkerboard(mesh.count(dim), contrast)))
    if degeneracy is not None:
        model.set_degeneracy_percent(degeneracy)
    pts = np.array([mesh.point(v) for v in range(mesh.count(0))])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    length = float(np.linalg.norm((hi - lo)[:dim]))
    direction = (hi - lo)[:dim] / length
    for f in mk.boundary_facets(mesh, dim):
        x = mk.centroid(mesh, dim - 1, int(f))
        value = float(np.dot(np.asarray(x)[:dim] - lo[:dim], direction) / length)
        model.add_pressure([int(f)], value)
    return model, lo, direction, length


def cells(model):
    return np.array([model.cell_pressure(e) for e in range(model.n_cells)])


def interpolant(mesh, dim, lo, direction, length):
    """Pi_0 p: the affine field at the centroids, which is its cell average."""
    return np.array([
        float(np.dot(np.asarray(mk.centroid(mesh, dim, e))[:dim] - lo[:dim], direction) / length)
        for e in range(mesh.count(dim))
    ])


def solved(mesh, dim, product, contrast=1.0, degeneracy=None):
    """(iterations, riesz-vs-direct, direct-vs-interpolant), the last two relative."""
    ref, lo, direction, length = linear_patch(mesh, dim, product, contrast, degeneracy)
    ref.solve(options=DIRECT)
    exact = cells(ref)

    itr, *_ = linear_patch(mesh, dim, product, contrast, degeneracy)
    report = itr.solve(options=RIESZ)
    assert report.reason == "CONVERGED_RTOL", (product, report.reason)
    got = cells(itr)

    scale = max(float(np.abs(exact).max()), 1e-300)
    patch = interpolant(mesh, dim, lo, direction, length)
    return (report.iterations,
            float(np.abs(got - exact).max() / scale),
            float(np.abs(exact - patch).max() / scale))


@pytest.mark.parametrize("product,family", CASES)
def test_the_riesz_answer_is_the_direct_answer(product, family):
    """Same discrete system, two solvers: the gap is the tolerance and nothing else."""
    dim, meshes = LADDERS[family]
    for mesh in meshes:
        _, departure, _ = solved(mesh, dim, product)
        assert departure < 1e-7, (product, family, mesh.count(dim), departure)


@pytest.mark.parametrize("product,family", CASES)
def test_the_count_does_not_grow_under_refinement(product, family):
    """h-independence: the bound is the inf-sup constant, which does not see h.

    The ladders span a factor of 16 in cell count, so growth in h would show as
    hundreds of iterations and a small absolute cap is the sharp statement. The
    diagonal star starts lower than the rest, hence the drift bound beside it.
    """
    dim, meshes = LADDERS[family]
    counts = [solved(mesh, dim, product)[0] for mesh in meshes]
    print(f"  {product:<14} {family:<13} {[m.count(dim) for m in meshes]} -> {counts} its")
    assert max(counts) <= 15, (product, family, counts)
    assert counts[-1] - counts[0] <= 6, (product, family, counts)


@pytest.mark.parametrize("product,family", CASES)
def test_the_patch_is_reproduced_where_the_product_claims_it(product, family):
    """Both directions, so a claim cannot pass by being vacuous.

    Where the product reproduces the affine field the discrete answer IS its
    interpolant; where it does not, the departure is O(1) in h and must show.
    """
    dim, meshes = LADDERS[family]
    departures = [solved(mesh, dim, product)[2] for mesh in meshes]
    if family in EXACT_ON[product]:
        assert max(departures) < 1e-12, (product, family, departures)
    else:
        assert min(departures) > 1e-4, (product, family, departures)


# star_K carries K, so a jump in K is a jump in the metric. The norm carries it
# too: W stands for B star_K^-1 B^T, which scales with K, so the multiplier
# weight is K_E |E| and not |E|. Without that factor the map is built on a
# different pairing from the operator's, and the mismatch is paid in iterations
# -- 267 against 22 at a contrast of 1e6 on cartesian_2d, measured.
CONTRASTS = (1.0, 1e2, 1e4, 1e6)
CONTRAST_CASES = [(p, f) for p in ("stabilized_rt", "diagonal_tpfa")
                  for f in sorted(LADDERS)]


@pytest.mark.parametrize("product,family", CONTRAST_CASES)
def test_the_contrast_is_carried_by_the_norm(product, family):
    """K over six orders of magnitude, on the dense-mass and lumped-mass products.

    Face-orthogonal cells stay flat. The simplex families still grow: there
    K_E |E| is a poorer stand-in for the facet transmissibility the Schur
    complement sums, harmonic across the jump. The accuracy assertion is the one
    that must hold everywhere -- a degraded preconditioner may cost iterations
    and must not cost correctness.
    """
    dim, meshes = LADDERS[family]
    mesh = meshes[len(meshes) // 2]
    counts = []
    for contrast in CONTRASTS:
        its, departure, _ = solved(mesh, dim, product, contrast)
        counts.append(its)
        assert departure < 1e-5, (product, family, contrast, departure)
    print(f"  {product:<14} {family:<13} K {list(CONTRASTS)} -> {counts} its")
    assert counts[0] <= 15, (product, family, counts)
    limit = 60 if family.startswith("cartesian") else 600
    assert max(counts) <= limit, (product, family, counts)


# eta is DERIVED, never given: the degeneracy scan flags cells at a threshold and
# those take the diagonal star. On a uniform box every cell equals its node-star
# mean, so the scan is all-or-nothing there; the graded annulus is what produces
# a spread, and these five thresholds land on five distinct selections of it.
ETA_MESH = mk.annulus(5, 10, 2, mk.Family.simplex, 1.0, 3.0, 1.0)
ETA_SELECTIONS = ((50.0, 0), (88.0, 25), (92.0, 50), (99.0, 74), (150.0, 100))


@pytest.mark.parametrize("threshold,on_the_star", ETA_SELECTIONS)
def test_adaptive_rt_over_five_eta_distributions(threshold, on_the_star):
    """Five selections on 100 cells: 0, 25, 50, 74 and 100 of them on the star.

    The fraction is asserted first, so the five cases cannot silently collapse
    onto one eta and report five passes for a single distribution.
    """
    model, *_ = linear_patch(ETA_MESH, 2, "adaptive_rt", degeneracy=threshold)
    model.build()
    eta = np.asarray(model.eta)
    assert eta.size == ETA_MESH.count(2)
    assert set(np.unique(eta)) <= {0.0, 1.0}, np.unique(eta)  # a selection, not a blend
    assert int((eta == 0.0).sum()) == on_the_star, (threshold, int((eta == 0.0).sum()))

    its, departure, patch = solved(ETA_MESH, 2, "adaptive_rt", degeneracy=threshold)
    print(f"  eta=0 on {on_the_star:3d}/100 cells -> {its:2d} its, patch {patch:.2e}")
    assert departure < 1e-7, (threshold, departure)
    assert its <= 15, (threshold, its)


def test_the_eta_endpoints_are_the_two_products_they_select_between():
    """eta == 1 IS stabilized_rt and eta == 0 IS diagonal_tpfa, cell for cell.

    A selection, so at either end it must not merely resemble its member but
    equal it: the same star is being built on every cell.
    """
    ends = {}
    for name in ("stabilized_rt", "diagonal_tpfa"):
        model, *_ = linear_patch(ETA_MESH, 2, name)
        model.solve(options=DIRECT)
        ends[name] = cells(model)

    for threshold, expect in ((50.0, "stabilized_rt"), (150.0, "diagonal_tpfa")):
        model, *_ = linear_patch(ETA_MESH, 2, "adaptive_rt", degeneracy=threshold)
        model.solve(options=DIRECT)
        assert np.allclose(cells(model), ends[expect], rtol=0, atol=1e-12), threshold


def test_the_patch_error_follows_the_selection():
    """The stabilized star is exact on this mesh and the diagonal one is not, so
    the error is bounded by the two ends and reached at the diagonal one.

    Measured: 4.2e-16, 1.88e-02, 1.87e-02, 2.88e-02, 3.50e-02 over the five
    selections. Not monotone -- 25 and 50 cells tie and cross by 0.3% -- because
    the error depends on WHICH cells were handed over and not only how many: a
    cell the scan flags contributes according to how far from K-orthogonal it is.
    So the claim here is the envelope, not an ordering.
    """
    errors = [solved(ETA_MESH, 2, "adaptive_rt", degeneracy=t)[2]
              for t, _ in ETA_SELECTIONS]
    print("  patch error by selection: " + ", ".join(f"{e:.2e}" for e in errors))
    assert errors[0] < 1e-12, errors   # nothing flagged: the stabilized product
    assert errors[-1] > 1e-3, errors   # everything flagged: the diagonal star
    assert errors[-1] == max(errors), errors
    assert all(errors[0] <= e <= errors[-1] for e in errors), errors
