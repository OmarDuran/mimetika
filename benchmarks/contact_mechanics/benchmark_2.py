r"""Benchmark 2 -- inclined displaced fault, constant friction (paper 4.1).

The reservoir of benchmark 1, but the fault dips at ``theta = 70`` degrees
from horizontal and carries Coulomb friction with a constant coefficient
``mu = 0.52``.  Depletion loads the fault in shear; where the shear traction
reaches the slip threshold ``-mu`` times the effective normal traction, the
fault slips.  At ``p = -25`` MPa the slip occurs in two separate patches
around the reservoir edges at ``y = +-75`` m; the patches grow with depletion
and merge at ``p ~= -26.9`` MPa (paper Fig. 12).  The paper's own numerical
results require the wide ``W = 18,000`` m domain to match the semi-analytical
ones (its Sect. 4.1), which is the domain used here.

The mesh is generated with **gmsh**: triangles conforming to the inclined
fault line (embedded curve) and to the reservoir boundaries, graded from the
fault outwards.  The mechanics is the de Rham (mimetic-AFW-BDM) four-field
system; on triangles it coincides with the AFW element.

The semi-analytical post-slip solution (Cauchy singular integral equations,
Jansen & Meulenbroek 2022) is not closed-form; the pointwise comparison is
against the paper's published dataset (4TU, doi 10.4121/d77f1a2c-29ea-4572-
ad72-e33ed8dc8d22): the pre-slip stresses of Fig. 8 and the post-slip
Coulomb stress function of Figs. 9-10.  The dataset's *slip* rows are
unusable -- the ``delta`` rows of Figs. 6, 9/10 and 14 are byte-identical
copies of the Fig. 14 pre-nucleation profile (peak 3.74 mm) -- so the slip
profiles are compared against the paper's reported observables instead: the
two-patch structure, patch extents, and the merging pressure (Fig. 12).

Reference (APA): Novikov, A., Shokrollahzadeh Behbahani, S., Voskov, D.,
Hajibeygi, H., & Jansen, J.-D. (2024). Benchmarking numerical simulation of
induced fault slip with semi-analytical solutions. *Geomechanics and
Geophysics for Geo-Energy and Geo-Resources, 10*, 182.
https://doi.org/10.1007/s40948-024-00896-1

Run with ``python -m benchmarks.contact_mechanics.benchmark_2``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import scipy.sparse as sp

from mimetika.assembly.four_field import FourFieldElasticity
from mimetika.assembly.mixed import boundary_facets
from mimetika.assembly.poromechanics import PoroMechanics
from mimetika.contact import ContactDriver, SignoriniCoulomb
from mimetika.materials import Material
from mimetika.mesh.fracture import facets_on_plane
from mimetika.mesh.mesh import Mesh

from mimetika.simulation import (
    MechanicsBC, PoromechanicsIC, PoromechanicsSolver,
)

from benchmarks.contact_mechanics.common import Parameters
from benchmarks.contact_mechanics.reference_data import DATA as REFERENCE_DATA
from benchmarks.contact_mechanics.reference_data import load as load_reference


def reference_coulomb(level_pa: float):
    """Semi-analytical post-slip Coulomb function ``(y, Sigma_C)`` (Figs. 9-10).

    ``Sigma_C = |Sigma_shear| - mu |Sigma_n'|`` evaluated on the slipped
    state: zero on the slipping patches, negative elsewhere.
    """
    tag = {-25e6: "25", -27e6: "27"}.get(round(level_pa, 3))
    if tag is None or not REFERENCE_DATA.exists():
        return None
    data = load_reference("9 & 10 left")
    return data[f"y_{tag}"], data[f"Sigma_C_post_{tag}"]


def reference_pre_slip():
    """Pre-slip fault stresses at ``p = -25`` MPa (Fig. 8): ``y``,
    ``Sigma_shear``, ``Sigma_slip`` and ``Sigma_C_pre = Sigma_shear -
    Sigma_slip`` (the identity holds exactly in the dataset)."""
    if not REFERENCE_DATA.exists():
        return None
    data = load_reference("8 left")
    return (data["y"], data["Sigma_shear"], data["Sigma_slip"],
            data["Sigma_shear"] - data["Sigma_slip"])


_slip_cache: dict = {}


def reference_slip_exact(level_pa: float):
    """Semi-analytical slip for Figs. 9-10, reconstructed from genuine rows.

    The dataset's ``delta`` rows are corrupted (all four hold the Fig. 14
    curve), but the slip is recoverable from what is genuine: on the patches
    the slip-induced shear must cancel the pre-slip Coulomb excess,

        ``Sigma_C_pre(y) = A pv-int delta'(xi) / (y - xi) dxi`` ,

    with ``A = G / (2 pi (1 - nu))`` (paper Eq. 21) and the patch ends taken
    from the zeros of the genuine ``Sigma_C_post`` rows.  The loading at any
    level follows from the ``p = -25`` MPa data because each stress component
    is affine in ``p`` (fixed in-situ part plus a depletion part linear in
    ``p``).  Piecewise-linear ``delta``, element-midpoint collocation, dense
    least squares; verified peaks 7.5/8.7 mm at -25 and 12.8 mm at -27
    against the paper's analytic curves.
    """
    key = round(level_pa, 3)
    tag = {-25e6: "25", -27e6: "27"}.get(key)
    if tag is None or not REFERENCE_DATA.exists():
        return None
    if key in _slip_cache:
        return _slip_cache[key]

    p = wide_parameters()
    A = p.shear_modulus / (2.0 * np.pi * (1.0 - p.poisson))
    th = np.radians(p.dip)
    n = np.array([np.sin(th), -np.cos(th)])
    t = np.array([np.cos(th), np.sin(th)])

    d8 = load_reference("8 left")
    y8 = d8["y"]
    sig0 = np.array([p.stress_tensor(yy)[0] for yy in y8])
    shear0 = np.einsum("i,kij,j->k", t, sig0, n)
    slip0 = -p.friction * (
        np.einsum("i,kij,j->k", n, sig0, n)
        + p.biot * np.array([p.pressure(yy) for yy in y8])
    )
    s = level_pa / -25e6
    excess = ((shear0 + s * (d8["Sigma_shear"] - shear0))
              - (slip0 + s * (d8["Sigma_slip"] - slip0)))

    d9 = load_reference("9 & 10 left")
    yc, cc = d9[f"y_{tag}"], d9[f"Sigma_C_post_{tag}"]
    at0 = np.abs(cc) < 1e3
    runs = np.split(np.where(at0)[0],
                    np.where(np.diff(np.where(at0)[0]) > 1)[0] + 1)
    patches = [(yc[r].min(), yc[r].max()) for r in runs if len(r) > 3]

    nodes, elems = [], []
    for (a, b) in patches:
        xs = np.linspace(a, b, 401)
        base = len(nodes)
        nodes.extend(xs)
        elems.extend([(base + i, base + i + 1) for i in range(400)])
    nodes = np.asarray(nodes)
    free = [i for i in range(len(nodes))
            if not any(np.isclose(nodes[i], e) for pp in patches for e in pp)]
    mids = np.array([(nodes[i] + nodes[j]) / 2 for i, j in elems])
    M = np.zeros((len(mids), len(nodes)))
    for e, (i, j) in enumerate(elems):
        xl, xr = nodes[i], nodes[j]
        with np.errstate(divide="ignore"):
            k = np.log(np.abs((mids - xl) / (mids - xr)))
        k[e] = 0.0  # PV over the element's own midpoint vanishes by symmetry
        M[:, j] += k / (xr - xl)
        M[:, i] -= k / (xr - xl)
    sol = np.linalg.lstsq(A * M[:, free], np.interp(mids, y8, excess),
                          rcond=None)[0]
    delta = np.zeros(len(nodes))
    delta[free] = sol
    _slip_cache[key] = (nodes, np.maximum(delta, 0.0))
    return _slip_cache[key]

# the paper widens the domain to W = 18,000 m (its Fig. 11) but keeps
# H = 4500 m: the linear in-situ profiles extrapolate to a *tensile* fault
# above ~3.5 km, so a taller domain opens the fault top unphysically
WIDE = dict(width=18000.0, height=4500.0, dip=70.0)


def wide_parameters(**overrides) -> Parameters:
    return Parameters(**{**WIDE, **overrides})


# -- gmsh mesh conforming to the inclined fault -------------------------------


def build(parameters: Parameters, spacing: float = 2.0,
          boundary_spacing: float = 100.0):
    """Mesh, fault facets and depletion pressure of the inclined reservoir.

    The fault line (dip ``theta``, through the origin, full height) and the
    reservoir boundaries ``y = +-a, +-b`` are embedded curves, so the mesh
    conforms to both.  The sizes follow the paper's DARTS grid (its Table 3
    and Fig. 16e): ``spacing`` (2 m) on the fault where it crosses the
    reservoir band ``|y| <= b``, coarser along the outer fault, growing to
    ``boundary_spacing`` (100 m) at the domain edge.
    """
    import gmsh

    W, H = parameters.width, parameters.height
    a, b = parameters.fault_a, parameters.fault_b
    cot = 1.0 / np.tan(np.radians(parameters.dip))

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    occ = gmsh.model.occ
    rect = occ.addRectangle(-W / 2, -H / 2, 0, W, H)
    fault = occ.addLine(
        occ.addPoint(-H / 2 * cot, -H / 2, 0), occ.addPoint(H / 2 * cot, H / 2, 0)
    )
    bands = [
        occ.addLine(occ.addPoint(-W / 2, y, 0), occ.addPoint(W / 2, y, 0))
        for y in (-b, -a, a, b)
    ]
    occ.fragment([(2, rect)], [(1, fault)] + [(1, l) for l in bands])
    occ.synchronize()

    fld = gmsh.model.mesh.field
    dist = fld.add("Distance")
    # after fragmentation the fault is split into several curves; take all
    # curves lying on the fault line
    fault_curves = []
    for dim, tag in gmsh.model.getEntities(1):
        x0, y0, z0, x1, y1, z1 = (
            *gmsh.model.getBoundingBox(dim, tag)[:3],
            *gmsh.model.getBoundingBox(dim, tag)[3:],
        )
        mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        if abs(mx - my * cot) < 1e-6 * H and abs(y1 - y0) > 1e-9:
            fault_curves.append(tag)
    # fragmentation split the fault at y = +-a, +-b: the curves inside the
    # reservoir band carry the refined size, the outer ones a moderate one
    center_curves = []
    for tag in fault_curves:
        box = gmsh.model.getBoundingBox(1, tag)
        if -b - 1.0 <= box[1] and box[4] <= b + 1.0:
            center_curves.append(tag)
    fld.setNumbers(dist, "CurvesList", center_curves)
    fld.setNumber(dist, "Sampling", 400)
    thr = fld.add("Threshold")
    fld.setNumber(thr, "InField", dist)
    fld.setNumber(thr, "SizeMin", spacing)
    fld.setNumber(thr, "SizeMax", boundary_spacing)
    fld.setNumber(thr, "DistMin", 15.0 * spacing)
    fld.setNumber(thr, "DistMax", 6.0 * (a + b))
    outer = fld.add("Distance")
    fld.setNumbers(outer, "CurvesList",
                   [t for t in fault_curves if t not in center_curves])
    fld.setNumber(outer, "Sampling", 400)
    thr_outer = fld.add("Threshold")
    fld.setNumber(thr_outer, "InField", outer)
    fld.setNumber(thr_outer, "SizeMin", min(15.0 * spacing, boundary_spacing))
    fld.setNumber(thr_outer, "SizeMax", boundary_spacing)
    fld.setNumber(thr_outer, "DistMin", 30.0 * spacing)
    fld.setNumber(thr_outer, "DistMax", 6.0 * (a + b))
    combined = fld.add("Min")
    fld.setNumbers(combined, "FieldsList", [thr, thr_outer])
    fld.setAsBackgroundMesh(combined)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.model.mesh.generate(2)

    tags, coords, _ = gmsh.model.mesh.getNodes()
    pts = coords.reshape(-1, 3)
    remap = {int(t): i for i, t in enumerate(tags)}
    tris = []
    for etype, _, conn in zip(*gmsh.model.mesh.getElements(2)):
        if etype == 2:
            tris += [
                [remap[int(n)] for n in tri]
                for tri in np.asarray(conn).reshape(-1, 3)
            ]
    gmsh.finalize()
    mesh = Mesh.from_polygons(pts, tris)

    normal, _ = parameters.fault_basis()
    fault_facets = facets_on_plane(
        mesh, [0.0, 0.0, 0.0], [normal[0], normal[1], 0.0], atol=1e-6 * H
    )

    return mesh, fault_facets, pressure_field(mesh, parameters)


def pressure_field(mesh, parameters: Parameters) -> np.ndarray:
    """Depletion of the offset reservoir, split by the inclined fault."""
    cot = 1.0 / np.tan(np.radians(parameters.dip))
    a, b = parameters.fault_a, parameters.fault_b
    centroids = mesh.geometry.centroids(2)
    side = centroids[:, 0] - centroids[:, 1] * cot  # < 0: left of the fault
    lower = np.where(side < 0, -b, -a)
    upper = np.where(side < 0, a, b)
    inside = (centroids[:, 1] > lower) & (centroids[:, 1] < upper)
    return np.where(inside, parameters.depletion, 0.0)


# -- mechanics ----------------------------------------------------------------


def material_of(parameters: Parameters) -> Material:
    return Material(
        shear_modulus=parameters.shear_modulus,
        poisson=parameters.poisson,
        biot=parameters.biot,
    )


def boundary_conditions(mesh, parameters: Parameters) -> MechanicsBC:
    """Incremental-problem frame: zero overburden increment on the top,
    rollers everywhere else, zero incremental boundary displacement."""
    centroids = mesh.geometry.centroids(1)
    boundary = boundary_facets(mesh)
    top = [f for f in boundary
           if abs(centroids[f][1] - parameters.height / 2) < 1e-6]
    rollers = [f for f in boundary if f not in set(top)]
    zero = lambda x: np.zeros((len(np.atleast_2d(x)), 3))  # noqa: E731
    free = lambda x: np.zeros((len(np.atleast_2d(x)), 3, 3))  # noqa: E731
    return MechanicsBC(traction=free, traction_facets=top,
                       roller_facets=rollers, dirichlet=zero)


def poromechanics_solver(mesh, parameters: Parameters, cache=None,
                         driver=None) -> PoromechanicsSolver:
    """The benchmark's :class:`PoromechanicsSolver`: quasi-static, direct.

    A fracture tag implies contact, so the fault enters as
    ``{"fault": driver}``; without a driver the solver is the plain
    unfractured poromechanics one (assembly, locked solves).
    """
    return PoromechanicsSolver(
        mesh, material_of(parameters),
        fractures=None if driver is None else {"fault": driver},
        bc=boundary_conditions(mesh, parameters),
        ic=PoromechanicsIC(prestress=None),
        dt=0.0, linear_solver="direct",
        cache=cache,
    )


def mechanics_factory(mesh, parameters: Parameters, pressure, cache=None):
    """The mechanics of one depletion level, as the driver expects it.

    A thin wrapper over :meth:`PoromechanicsSolver.mechanics`, kept for the
    callers that only need the assembled system.
    """
    solver = poromechanics_solver(mesh, parameters, cache=cache)

    def factory(contact=None):
        return solver.mechanics(pressure, contact=contact)

    return factory


def insitu_prestress(mesh, fault, parameters: Parameters,
                     pressure=None) -> np.ndarray:
    """In-situ traction resolved in each fault facet's own frame.

    On the inclined fault both components are nonzero; the frame is the
    driver's (``facet_frame``: normal first, tangent second), so signs are
    consistent with the tractions the driver reads back.
    """
    bm = mesh.complex.boundary_matrix(2).tocsr()  # (facets x cells)
    out = np.zeros((len(fault), 2))
    for i, f in enumerate(fault):
        frame = mesh.geometry.facet_frame(int(f))
        n, t = frame[0][:2], frame[1][:2]
        y = mesh.geometry.centroids(1)[int(f)][1]
        sigma = parameters.stress_tensor(y)[0]
        # the Coulomb threshold acts on the *effective* traction (paper
        # Eq. 29): add the Biot pore-pressure term to the normal.  Where only
        # one side of the fault is depleted (75 < |y| < 150) the fault takes
        # the *depleted* side's pressure -- min(), not the two-side mean: the
        # 4TU Sigma_slip curve (Fig. 8) matches min() to 0.55 MPa rms and is
        # off by mu alpha |dp|/2 ~ 5.9 MPa under the mean
        cells = bm[int(f)].indices
        p_fault = parameters.pressure(y) + float(pressure[cells].min())
        out[i] = (n @ sigma @ n + parameters.biot * p_fault, t @ sigma @ n)
    return out


def simulate(parameters: Parameters, spacing: float = 2.0,
             boundary_spacing: float = 100.0, built=None, substeps: int = 1,
             warm_from_locked: bool = False):
    """Solve at ``parameters.depletion``; return slip and patch structure.

    ``warm_from_locked`` starts the contact Newton at the **locked**
    (unfractured) solution's fault tractions instead of zero.  On the
    confined ``W = 4500`` m domain the cold start diverges: the first trial
    has the fault carrying nothing, the projection clips metre-scale
    excursions, and the iteration exits the physical basin into a spurious
    whole-fault runaway -- even at levels where the fault is fully stuck.
    The locked tractions are one direct solve away and sit next to the
    contact solution (their map residual is just ``r`` times the
    enforcement-truncation gap), so Newton converges in a handful of steps.
    ``substeps > 1`` additionally ramps the depletion with warm restarts.
    """
    if built is None:
        mesh, fault, pressure = build(parameters, spacing, boundary_spacing)
        cache = {}
    else:  # reuse the mesh; the pressure belongs to *this* depletion level
        mesh, fault, _, cache = built
        pressure = pressure_field(mesh, parameters)
    lam = (2.0 * parameters.shear_modulus * parameters.poisson
           / (1.0 - 2.0 * parameters.poisson))
    driver = ContactDriver(
        mesh, fault, SignoriniCoulomb(friction=parameters.friction),
        prestress=None, mu=parameters.shear_modulus, lam=lam,
        tolerance=1e-10, max_iterations=400,
    )
    solver = poromechanics_solver(mesh, parameters, cache=cache,
                                  driver=driver)
    state = None
    for k in range(substeps):
        stage = replace(parameters,
                        depletion=parameters.depletion * (k + 1) / substeps)
        pressure = pressure_field(mesh, stage)
        state = solver.step(
            pressure,
            state=state,
            prestress=insitu_prestress(mesh, fault, stage, pressure),
            warm_from_locked=(state is None and warm_from_locked),
        )
    y = mesh.geometry.centroids(1)[np.asarray(fault, dtype=int)][:, 1]
    order = np.argsort(y)
    slip = driver.per_facet(state.jump)[:, 1]

    # post-slip Coulomb function: total effective traction is the in-situ
    # prestress (which already carries the Biot pore term on the normal) plus
    # the solved incremental traction, read on the fault facets themselves
    total = solver.fault_tractions(state)
    coulomb = np.abs(total[:, 1]) + parameters.friction * total[:, 0]

    yy, ss = y[order], slip[order]
    patches = slip_patches(yy, ss)

    # resolved fault slip: the BDM facet moments carry a *linear* profile
    # per facet -- read it at the two facet Gauss points (see benchmark_3)
    pw = ContactDriver(mesh, fault, SignoriniCoulomb(friction=parameters.friction),
                       prestress=None, mu=parameters.shear_modulus, lam=lam,
                       enforcement="pointwise")
    _, _, rhs_full = solver.mechanics(pressure)  # cached
    gap_pts = pw.gap(state.problem, state.solution, rhs=rhs_full)
    y_pts = np.concatenate([mesh.geometry.quadrature(1, int(f))[0][:, 1]
                            for f in fault])
    op = np.argsort(y_pts)
    slip_pts = np.abs(gap_pts[:, 1])
    return {
        "y": yy, "slip": ss, "coulomb": coulomb[order], "patches": patches,
        "y_pts": y_pts[op], "slip_pts": slip_pts[op],
        "state": state, "cells": mesh.num_cells(2), "mesh": mesh,
        "fault": fault, "built": (mesh, fault, pressure, cache),
    }


def pre_slip_stress(parameters: Parameters, spacing: float = 2.0,
                    built=None):
    """Fault stresses on the **locked** fault (Fig. 8): the plain continuum
    under the depletion load, no contact block, one direct solve."""
    from mimetika.contact import FrictionlessBilateral

    if built is None:
        mesh, fault, pressure = build(parameters, spacing)
        cache = {}
    else:
        mesh, fault, _, cache = built
        pressure = pressure_field(mesh, parameters)
    solver = poromechanics_solver(mesh, parameters, cache=cache)
    problem, solution, rhs = solver.locked_solution(pressure)
    lam = (2.0 * parameters.shear_modulus * parameters.poisson
           / (1.0 - 2.0 * parameters.poisson))
    driver = ContactDriver(mesh, fault, FrictionlessBilateral(),
                           mu=parameters.shear_modulus, lam=lam)
    total = driver.per_facet(
        driver.expand_to_points(insitu_prestress(mesh, fault, parameters,
                                                 pressure))
        + driver.tractions(solution["stress"])
    )
    y = mesh.geometry.centroids(1)[np.asarray(fault, dtype=int)][:, 1]
    order = np.argsort(y)
    return {
        "y": y[order],
        "shear": np.abs(total[order, 1]),
        "threshold": -parameters.friction * total[order, 0],
        "coulomb": (np.abs(total[order, 1])
                    + parameters.friction * total[order, 0]),
        "built": (mesh, fault, pressure, cache),
    }


def slip_patches(yy, ss, tol: float = 1e-6, interpolate: bool = False):
    """Contiguous slipping runs ``(y0, y1, peak)`` above ``tol``.

    The default suits the paper's mm-scale observables; near the slip
    *onset* the physical slip is sub-micron, so onset-sensitive uses (the
    Fig. 12 boundary curves) pass a smaller ``tol``.  ``interpolate``
    refines each boundary to the linear zero crossing between the last
    slipping sample and its locked neighbour -- sub-sample positions
    instead of sample-quantized ones.
    """
    aa = np.abs(ss)
    slipping = aa > tol
    patches, start = [], None

    def cross(inside: int, outside: int) -> float:
        if not interpolate or outside < 0 or outside >= len(yy):
            return float(yy[inside])
        da = aa[inside] - aa[outside]
        if da <= 0:
            return float(yy[inside])
        f = (aa[inside] - tol) / da
        return float(yy[inside] + f * (yy[outside] - yy[inside]))

    for i, flag in enumerate(slipping):
        if flag and start is None:
            start = i
        if (not flag or i == len(ss) - 1) and start is not None:
            end = i if flag else i - 1
            patches.append((cross(start, start - 1), cross(end, end + 1),
                            float(aa[start:end + 1].max())))
            start = None
    return patches


def pair_average(y, v):
    """Two-facet moving average, on facet-midpoint positions.

    The exact set-valued law resolves a near-threshold plateau with an
    alternating stick/at-threshold active set, so the reaction traction
    carries a facet-scale, mean-zero oscillation; averaging adjacent facets
    removes exactly that mode and nothing else.
    """
    return 0.5 * (y[:-1] + y[1:]), 0.5 * (v[:-1] + v[1:])


def figure(parameters, results, path, paper_figure: str = "9-10"):
    """Post-slip Coulomb stress (left) and slip (right) -- one paper figure.

    Mirrors the layout of the paper's Figs. 9 and 10: ``Sigma_C`` clipped to
    ``[-2, 1]`` MPa, ``|y| <= 100`` m; the left references are the genuine
    ``Sigma_C_post`` rows, the right ones the reconstructed exact slip
    (:func:`reference_slip_exact`).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 6.4), sharey=True)
    for level, res in results.items():
        inside = np.abs(res["y"]) <= 100.0
        yc, cc = pair_average(res["y"], res["coulomb"])
        inc = np.abs(yc) <= 100.0
        (line,) = left.plot(cc[inc] / 1e6, yc[inc],
                            lw=1.5, label=f"p = {level / 1e6:g} MPa")
        ref = reference_coulomb(level)
        if ref is not None:
            yr, cr = ref
            sel = np.abs(yr) <= 100.0
            left.plot(cr[sel][::20] / 1e6, yr[sel][::20], "o", ms=3.2,
                      mfc="white", mew=0.9, lw=0, color=line.get_color())
        inp = np.abs(res["y_pts"]) <= 100.0
        right.plot(res["slip_pts"][inp] * 1e3, res["y_pts"][inp],
                   lw=1.5, color=line.get_color())
        refs = reference_slip_exact(level)
        if refs is not None:
            yn, dn = refs
            right.plot(dn[::16] * 1e3, yn[::16], "o", ms=3.2, mfc="white",
                       mew=0.9, lw=0, color=line.get_color())
    for ax in (left, right):
        for edge in (parameters.fault_a, -parameters.fault_a):
            ax.axhline(edge, color="k", lw=0.6, ls=":")
        ax.grid(alpha=0.25)
    left.axvline(0.0, color="k", lw=0.6)
    left.set_xlim(-2.0, 1.0)
    left.set_xlabel(r"$\Sigma_C$   (MPa)")
    left.set_ylabel("y   (m)")
    left.set_ylim(-100, 100)
    right.set_xlabel(r"$\delta$   (mm)")
    from matplotlib.lines import Line2D

    handles, labels = left.get_legend_handles_labels()
    handles += [Line2D([], [], color="k", lw=1.5, label="mimetic-AFW-BDM"),
                Line2D([], [], color="k", ls="", marker="o", ms=3.2,
                       mfc="white", mew=0.9, label="semi-analytical (4TU)")]
    left.legend(handles=handles, fancybox=True, framealpha=0.9,
                edgecolor="0.8", fontsize=8.5)
    fig.suptitle("Benchmark 2 -- inclined displaced fault, "
                 f"constant friction $\\mu$ = {parameters.friction:g}, "
                 f"W = {parameters.width:g} m   "
                 f"(Novikov et al. 2024, Fig. {paper_figure})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def initial_state(parameters: Parameters, spacing: float = 2.0, built=None):
    """The pre-depletion stress state (paper Sect. 2, Fig. 3).

    Gravity body force plus the paper's depth-varying boundary loads (its
    Eqs. 6-7), entering as the analytic initial stress tensor prescribed on
    the top and side boundaries, vertical rollers at the bottom; no
    depletion, no contact.  The initial field (Eq. 8) is linear in ``y``,
    which the mimetic stress space contains exactly, so the facet tractions
    read along the embedded 70-degree line reproduce Eq. 12 to solver
    precision -- and they are primary unknowns, not a cell-to-line
    interpolation.
    """
    from mimetika.contact import FrictionlessBilateral
    from mimetika.operators.derham import DeRhamDeviatoricStress
    from mimetika.solver.saddle import solve_saddle

    if built is None:
        mesh, fault, pressure = build(parameters, spacing)
        built = (mesh, fault, pressure, {})
    mesh, fault, _, cache = built

    material = Material(shear_modulus=parameters.shear_modulus,
                        poisson=parameters.poisson, biot=parameters.biot)
    centroids = mesh.geometry.centroids(1)
    boundary = boundary_facets(mesh)
    top = [f for f in boundary
           if abs(centroids[f][1] - parameters.height / 2) < 1e-6]
    bottom = [f for f in boundary
              if abs(centroids[f][1] + parameters.height / 2) < 1e-6]
    sides = [f for f in boundary
             if f not in set(top) and f not in set(bottom)]

    def sigma0(x):
        x = np.atleast_2d(x)
        out = np.zeros((len(x), 3, 3))  # ambient 3x3; only the xy block acts
        for i, yy in enumerate(x[:, 1]):
            out[i, :2, :2] = parameters.stress_tensor(yy)[0]
        return out

    # bulk density (1 - phi) rho_s + phi rho_fl of Table 2; equilibrium
    # div sigma = rho g e_y -- compression grows downward
    rho_g = (0.85 * 2650.0 + 0.15 * 1020.0) * 9.81

    def gravity(x):
        out = np.zeros((len(np.atleast_2d(x)), 3))
        out[:, 1] = rho_g
        return out

    # the initial state is *poroelastic*: the total-stress field of Eq. 8 is
    # compatible only together with the Biot term of the hydrostatic initial
    # pressure -- without it the solve pivots to wrong stress gradients
    if "coupling" in cache:
        coupling = cache["coupling"]
    else:
        poro = PoroMechanics(mesh, material)
        coupling = (sp.diags(poro.material.pressure_coupling(2))
                    @ poro.trace_operator())
        cache["coupling"] = coupling
    p0 = np.array([parameters.pressure(yy)
                   for yy in mesh.geometry.centroids(2)[:, 1]])

    # the paper's Fig. 1 configuration: normal loads on ALL four boundaries
    # (zero boundary shear), rigid modes removed by three discrete pins --
    # u_x at the bottom centre, u_y at one mid-height point per side.  A
    # roller along the whole bottom is a different problem and develops the
    # spurious boundary shear this configuration exists to avoid.
    problem = FourFieldElasticity(
        mesh, inner=DeRhamDeviatoricStress(mesh, material=material))
    matrix, rhs = problem.assemble_constrained(
        body_force=gravity,
        extra_rhs=-(coupling.T @ p0),
        traction=sigma0,
        traction_facets=list(top) + list(sides) + list(bottom),
    )
    from mimetika.assembly.mixed import _constrain

    c2 = mesh.geometry.centroids(2)
    W2, H2 = parameters.width / 2, parameters.height / 2
    bot_center = int(np.argmin(np.hypot(c2[:, 0], c2[:, 1] + H2)))
    left_mid = int(np.argmin(np.hypot(c2[:, 0] + W2, c2[:, 1])))
    right_mid = int(np.argmin(np.hypot(c2[:, 0] - W2, c2[:, 1])))
    offset_u = problem.n_stress + problem.n_pressure
    pins = np.array([offset_u + 2 * bot_center,
                     offset_u + 2 * left_mid + 1,
                     offset_u + 2 * right_mid + 1])
    matrix, rhs = _constrain(matrix.tocsr(), rhs, pins, np.zeros(3))
    solution = problem.split(solve_saddle(matrix, rhs, problem.block_sizes,
                                          method="direct"))
    lam = (2.0 * parameters.shear_modulus * parameters.poisson
           / (1.0 - 2.0 * parameters.poisson))
    driver = ContactDriver(mesh, fault, FrictionlessBilateral(),
                           mu=parameters.shear_modulus, lam=lam)
    tot = driver.per_facet(driver.tractions(solution["stress"]))
    y = centroids[np.asarray(fault, dtype=int)][:, 1]
    # per-facet frame signs -> the paper's global convention: normal
    # (sin, -cos), tangent (cos, sin) of the 70-degree line
    th = np.radians(parameters.dip)
    t_glob = np.array([np.cos(th), np.sin(th), 0.0])
    n_glob = np.array([np.sin(th), -np.cos(th), 0.0])
    s = np.empty(len(fault))
    for i, f in enumerate(fault):
        frame = mesh.geometry.facet_frame(int(f))
        s[i] = np.sign(frame[0] @ n_glob) * np.sign(frame[1] @ t_glob)
    order = np.argsort(y)
    return {"y": y[order], "normal": tot[order, 0],
            "shear": (s * tot[:, 1])[order], "built": built}


def figure_3(parameters, init, path):
    """Initial stresses along the 70-degree line (paper Fig. 3)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 6.4), sharey=True)
    left.plot(init["normal"] / 1e6, init["y"], lw=1.5,
              label="mimetic-AFW-BDM")
    right.plot(init["shear"] / 1e6, init["y"], lw=1.5)
    if REFERENCE_DATA.exists():
        dl = load_reference("3 left")
        dr = load_reference("3 right")
        left.plot(dl["sigma_normal"][::25] / 1e6, dl["y"][::25], "ro",
                  ms=3.4, mfc="white", mew=0.9, lw=0,
                  label="semi-analytical (4TU)")
        right.plot(dr["sigma_shear"][::25] / 1e6, dr["y"][::25], "ro",
                   ms=3.4, mfc="white", mew=0.9, lw=0)
    for ax in (left, right):
        ax.grid(alpha=0.25)
    left.set_xlabel(r"$\sigma_\perp^0$   (MPa)")
    left.set_ylabel("y   (m)")
    left.set_ylim(-parameters.height / 2, parameters.height / 2)
    right.set_xlabel(r"$\sigma_\parallel^0$   (MPa)")
    left.legend(fancybox=True, framealpha=0.9, edgecolor="0.8", fontsize=8.5)
    fig.suptitle("Benchmark 2 -- initial stresses on the 70-degree line, "
                 f"W = {parameters.width:g} m   "
                 "(Novikov et al. 2024, Fig. 3)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_8(parameters, pre, path):
    """Pre-slip stresses on the locked fault at ``p = -25`` MPa (paper Fig. 8).

    Left: shear ``Sigma_par`` and slip threshold ``Sigma_sl``; right: the
    pre-slip Coulomb stress.  Both compared through ``|shear|`` -- our facet
    tangent sign is per-facet, the dataset's is a global convention.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = reference_pre_slip()
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 6.4), sharey=True)
    left.plot(pre["shear"] / 1e6, pre["y"], lw=1.5, color="tab:blue",
              label=r"$|\Sigma_\parallel|$  mimetic-AFW-BDM")
    left.plot(pre["threshold"] / 1e6, pre["y"], lw=1.5, color="tab:orange",
              label=r"$\Sigma_{sl}$  mimetic-AFW-BDM")
    right.plot(pre["coulomb"] / 1e6, pre["y"], lw=1.5,
               label="mimetic-AFW-BDM")
    if ref is not None:
        yr, shear_r, slip_r, _ = ref
        left.plot(np.abs(shear_r[::40]) / 1e6, yr[::40], "o", ms=3.2,
                  mfc="white", mew=0.9, lw=0, color="tab:blue",
                  label="semi-analytical (4TU)")
        left.plot(slip_r[::40] / 1e6, yr[::40], "^", ms=3.4, mfc="white",
                  mew=0.9, lw=0, color="tab:orange")
        right.plot((np.abs(shear_r) - slip_r)[::40] / 1e6, yr[::40], "o",
                   ms=3.2, mfc="white", mew=0.9, lw=0, color="tab:blue",
                   label="semi-analytical (4TU)")
    for ax in (left, right):
        for edge in (parameters.fault_a, -parameters.fault_a,
                     parameters.fault_b, -parameters.fault_b):
            ax.axhline(edge, color="k", lw=0.5, ls=":")
        ax.grid(alpha=0.25)
    right.axvline(0.0, color="k", lw=0.6)
    left.set_xlabel(r"$\Sigma_\parallel$ and $\Sigma_{sl}$   (MPa)")
    left.set_ylabel("y   (m)")
    left.set_ylim(-250, 250)
    right.set_xlabel(r"$\Sigma_C$   (MPa)")
    left.legend(fancybox=True, framealpha=0.9, edgecolor="0.8", fontsize=8.5)
    right.legend(fancybox=True, framealpha=0.9, edgecolor="0.8", fontsize=8.5)
    fig.suptitle("Benchmark 2 -- pre-slip stresses at p = -25 MPa, "
                 f"W = {parameters.width:g} m   "
                 "(Novikov et al. 2024, Fig. 8)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_12(parameters, sweep, merged_at, path):
    """Slip-patch boundaries against depletion (paper Fig. 12).

    Only two of the four reference boundary curves survive in the 4TU file
    -- the lower patch's outer boundary (stored with names and values
    swapped) and its inner one; the upper pair was overwritten (see
    :mod:`.reference_data`).  The merging pressure is unaffected: the inner
    reference curve ends exactly at the merge.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    series = {"outer lower": [], "inner lower": [],
              "inner upper": [], "outer upper": []}
    for level, patches in sweep:
        if not patches:
            continue
        x = -level
        if len(patches) == 1:
            y0, y1, _ = patches[0]
            if y0 < 0.0 < y1:  # merged: only the outer boundaries remain
                series["outer lower"].append((x, y0))
                series["outer upper"].append((x, y1))
            elif y0 >= 0.0:  # one-sided onset: only the upper patch exists
                series["inner upper"].append((x, y0))
                series["outer upper"].append((x, y1))
            else:
                series["outer lower"].append((x, y0))
                series["inner lower"].append((x, y1))
        else:
            series["outer lower"].append((x, patches[0][0]))
            series["inner lower"].append((x, patches[0][1]))
            series["inner upper"].append((x, patches[-1][0]))
            series["outer upper"].append((x, patches[-1][1]))
    first = True
    for pts in series.values():
        if not pts:
            continue
        arr = np.array(pts)
        ax.plot(arr[:, 0], arr[:, 1], "-", color="tab:blue", lw=1.5,
                label="mimetic-AFW-BDM" if first else None)
        first = False
    if REFERENCE_DATA.exists():
        d = load_reference("12")
        # the two genuine rows, under the file's scrambled names
        ax.plot(-d["y_tilde_1"] / 1e6, d["-p_1"], "ro", ms=2.4, mfc="white",
                mew=0.7, lw=0, markevery=40, label="semi-analytical (4TU)")
        ax.plot(-d["-p_2"] / 1e6, d["y_tilde_2"], "ro", ms=2.4, mfc="white",
                mew=0.7, lw=0, markevery=40)
        # the upper-boundary rows were overwritten in the published file;
        # the problem is near-symmetric, so mirror the genuine lower pair
        ax.plot(-d["y_tilde_1"] / 1e6, -d["-p_1"], "r^", ms=2.6, mfc="white",
                mew=0.6, lw=0, markevery=40, alpha=0.7,
                label="4TU mirrored (upper rows lost)")
        ax.plot(-d["-p_2"] / 1e6, -d["y_tilde_2"], "r^", ms=2.6, mfc="white",
                mew=0.6, lw=0, markevery=40, alpha=0.7)
        ax.axvline(26.87, color="r", lw=0.8, ls=":",
                   label="merging (4TU: -26.87 MPa)")
    if merged_at is not None:
        ax.axvline(-merged_at, color="tab:blue", lw=0.8, ls="--",
                   label=f"merging (ours: {merged_at:g} MPa)")
    for edge in (parameters.fault_a, -parameters.fault_a):
        ax.axhline(edge, color="k", lw=0.5, ls=":")
    ax.set_xlabel(r"$-p$   (MPa)")
    ax.set_ylabel(r"$\tilde{y}_i$   (m)")
    ax.set_xlim(12, 30)
    ax.set_ylim(-100, 100)
    ax.grid(alpha=0.25)
    ax.legend(fancybox=True, framealpha=0.9, edgecolor="0.8", fontsize=8.5)
    ax.set_title("Benchmark 2 -- slip patch boundaries, "
                 f"W = {parameters.width:g} m   (Novikov et al. 2024, "
                 "Fig. 12)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=2.0,
                        help="mesh size on the fault at the reservoir, in "
                             "metres (2 m = the paper's DARTS grid, Table 3)")
    parser.add_argument("--levels", type=float, nargs="+", default=[-25.0, -27.0],
                        help="depletion levels in MPa")
    parser.add_argument("--figure",
                        default="benchmarks/contact_mechanics/benchmark_2.png",
                        help="path of the slip-profile figure")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--width", choices=["4500", "18000", "both"],
                        default="both",
                        help="domain width: 4500 m (paper Fig. 9), 18000 m "
                             "(Fig. 10), or both")
    parser.add_argument("--vtk", nargs="?", const="out/benchmark_2",
                        default=None,
                        help="write the final level's state as .pvd/.vtu")
    arguments = parser.parse_args()

    print("Benchmark 2 -- inclined displaced fault, constant friction\n")
    print("  paper: two patches at -25 MPa, merging at ~ -26.9 MPa "
          "(Figs. 9, 10, 12)\n")

    domains = {"4500": [(4500.0, "9")], "18000": [(18000.0, "10")],
               "both": [(4500.0, "9"), (18000.0, "10")]}[arguments.width]
    for width, fig_tag in domains:
        parameters = wide_parameters(width=width)
        print(f"  -- W = {width:g} m (paper Fig. {fig_tag}), "
              f"dip = {parameters.dip:g} deg, mu = {parameters.friction:g} --")
        built, results = None, {}

        stem, dot, ext = arguments.figure.rpartition(".")
        base = stem if dot else arguments.figure
        suffix = f"{dot}{ext}" if dot else ".png"

        # Fig. 3: the initial (pre-depletion) state, on the paper's Sect. 2
        # domain (W = 4500 m only -- the reference is that domain's state)
        if width == 4500.0:
            init = initial_state(parameters, spacing=arguments.spacing,
                                 built=built)
            built = init["built"]
            if REFERENCE_DATA.exists():
                dl = load_reference("3 left")
                dr = load_reference("3 right")
                rn = np.sqrt(np.mean((np.interp(dl["y"], init["y"],
                                                init["normal"])
                                      - dl["sigma_normal"]) ** 2))
                rs = np.sqrt(np.mean((np.interp(dr["y"], init["y"],
                                                init["shear"])
                                      - dr["sigma_shear"]) ** 2))
                print("  initial state vs 4TU Fig. 3: rms sigma_perp = "
                      f"{rn / 1e6:.2e} MPa, sigma_par = {rs / 1e6:.2e} MPa")
            if arguments.figure and not arguments.no_plots:
                print("  wrote", figure_3(parameters, init,
                                          f"{base}_fig3{suffix}"))
            print()

        for level in arguments.levels:
            stage = replace(parameters, depletion=level * 1e6)
            res = simulate(stage, spacing=arguments.spacing, built=built,
                           warm_from_locked=width <= 4500.0)
            built = res["built"]
            results[level * 1e6] = res
            state = res["state"]
            print(f"  p = {level:6.1f} MPa   ({res['cells']} cells, "
                  f"converged={state.converged} in {state.iterations} "
                  "iterations)")
            if not res["patches"]:
                print("    no slip")
            for y0, y1, peak in res["patches"]:
                print(f"    patch  y in [{y0:8.2f}, {y1:8.2f}] m   "
                      f"peak |slip| = {peak * 1e3:6.2f} mm")
            n_p = len(res["patches"])
            status = "merged" if n_p == 1 else "separate" if n_p > 1 else "--"
            print(f"    ({status})")
            ref = reference_coulomb(level * 1e6)
            if ref is not None:
                yr, cr = ref
                yc, cc = pair_average(res["y"], res["coulomb"])
                ours = np.interp(yr, yc, cc)
                sel = np.abs(yr) <= 100.0
                rms = np.sqrt(np.mean((ours[sel] - cr[sel]) ** 2))
                print("    vs semi-analytical (4TU), Sigma_C over "
                      f"|y| <= 100 m: rms = {rms / 1e6:.3f} MPa "
                      f"(scale {np.abs(cr[sel]).max() / 1e6:.1f} MPa)")
            refs = reference_slip_exact(level * 1e6)
            if refs is not None:
                yn, dn = refs
                ours = np.interp(yn, res["y"], np.abs(res["slip"]))
                rms = np.sqrt(np.mean((ours - dn) ** 2))
                print("    vs semi-analytical (reconstructed), slip: "
                      f"rms = {rms * 1e3:.3f} mm "
                      f"(peaks {np.abs(res['slip']).max() * 1e3:.2f} / "
                      f"{dn.max() * 1e3:.2f} mm)")
            print()

        pre_ref = reference_pre_slip()
        if (pre_ref is not None
                and -25.0 in [round(lv, 6) for lv in arguments.levels]):
            stage = replace(parameters, depletion=-25e6)
            pre = pre_slip_stress(stage, spacing=arguments.spacing,
                                  built=built)
            yr, shear_r, slip_r, coulomb_r = pre_ref
            sel = np.abs(yr) <= 100.0
            print("  pre-slip (locked fault) at p = -25 MPa vs 4TU Fig. 8, "
                  "|y| <= 100 m:")
            # our tangent sign is per-facet (facet_frame), the dataset's is a
            # global convention: compare through |shear|, and rebuild the
            # Coulomb function from it so both sides use the same definition
            for name, ours_c, ref_c in (
                ("|Sigma_shear|", pre["shear"], np.abs(shear_r)),
                ("Sigma_slip   ", pre["threshold"], slip_r),
                ("Sigma_C_pre  ", pre["coulomb"], np.abs(shear_r) - slip_r),
            ):
                ours = np.interp(yr, pre["y"], ours_c)
                rms = np.sqrt(np.mean((ours[sel] - ref_c[sel]) ** 2))
                print(f"    {name}: rms = {rms / 1e6:.3f} MPa "
                      f"(scale {np.abs(ref_c[sel]).max() / 1e6:.1f} MPa)")
            print()
            # the reference is the (wide-domain) semi-analytics, so the
            # figure comes from the W = 18000 m pass whenever that runs
            want_fig8 = (width == 18000.0
                         or all(w != 18000.0 for w, _ in domains))
            if want_fig8 and arguments.figure and not arguments.no_plots:
                print("  wrote", figure_8(parameters, pre,
                                          f"{base}_fig8{suffix}"))

        if arguments.figure and not arguments.no_plots:
            print("  wrote", figure(parameters, results,
                                    f"{base}_fig{fig_tag}{suffix}",
                                    paper_figure=fig_tag))
            print()

        if width == 18000.0 and arguments.figure and not arguments.no_plots:
            # Fig. 12: track the patch boundaries down the depletion sweep,
            # reusing the mesh and the cached factorization level by level.
            # The inner boundaries collapse onto each other within a fraction
            # of an MPa at merging, so that interval is re-swept at 0.05 MPa
            # -- otherwise the fork of the figure never closes.
            def probe(level):
                stage = replace(parameters, depletion=level * 1e6)
                res = simulate(stage, spacing=arguments.spacing, built=built)
                # boundaries from the facet-MEAN slip: on stuck facets it is
                # clean to 1e-12 m, so a 1e-8 threshold resolves the micron
                # onset.  (The pointwise readback is unusable here: it
                # carries the ~1e-5 m enforcement-truncation residual.)
                patches = slip_patches(res["y"], res["slip"], tol=1e-8)
                return level, patches, res["built"]

            sweep = []
            for level in np.arange(-12.0, -30.0 - 1e-9, -0.25):
                level, patches, built = probe(level)
                sweep.append((level, patches))
            counts = [(lv, len(p)) for lv, p in sweep]
            merged_at = None
            for (l1, c1), (l2, c2) in zip(counts, counts[1:]):
                if c1 >= 2 and c2 == 1:
                    for fine in np.arange(l1 - 0.05, l2, -0.05):
                        fine, patches, built = probe(fine)
                        sweep.append((fine, patches))
                        if merged_at is None and len(patches) == 1:
                            merged_at = fine
                    break
            sweep.sort(key=lambda t: -t[0])
            print(f"  patch boundaries swept p = -12..-30 MPa; merging at "
                  f"p ~ {merged_at:.2f} MPa (4TU: -26.87, paper: ~ -26.9)"
                  if merged_at is not None else
                  "  patch boundaries swept p = -12..-30 MPa; no merge seen")
            print("  wrote", figure_12(parameters, sweep, merged_at,
                                       f"{base}_fig12{suffix}"))
            print()

    if arguments.vtk:
        from mimetika.postprocess import (
            MixedDimensionalSeries, contact_fields, mechanics_fields,
        )

        mesh, fault, pressure, _ = built
        series = MixedDimensionalSeries(arguments.vtk, mesh, fault)
        for level, res in sorted(results.items()):
            state = res["state"]
            lam = (2.0 * parameters.shear_modulus * parameters.poisson
                   / (1.0 - 2.0 * parameters.poisson))
            driver = ContactDriver(mesh, fault, SignoriniCoulomb(
                friction=parameters.friction),
                mu=parameters.shear_modulus, lam=lam)
            series.write(
                abs(level) / 1e6,
                bulk=mechanics_fields(state.problem, state.solution, pressure),
                fracture=contact_fields(driver, state),
            )
        print(f"  wrote {series.collection}")


if __name__ == "__main__":
    main()
