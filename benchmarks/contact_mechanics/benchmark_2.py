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


def mechanics_factory(mesh, parameters: Parameters, pressure):
    """Depletion load and boundary frame; the de Rham space is the default."""
    material = Material(
        shear_modulus=parameters.shear_modulus,
        poisson=parameters.poisson,
        biot=parameters.biot,
    )
    poro = PoroMechanics(mesh, material)
    coupling = sp.diags(poro.material.pressure_coupling(2)) @ poro.trace_operator()
    extra = -(coupling.T @ pressure)

    centroids = mesh.geometry.centroids(1)
    boundary = boundary_facets(mesh)
    top = [f for f in boundary
           if abs(centroids[f][1] - parameters.height / 2) < 1e-6]
    rollers = [f for f in boundary if f not in set(top)]
    zero = lambda x: np.zeros((len(np.atleast_2d(x)), 3))  # noqa: E731
    free = lambda x: np.zeros((len(np.atleast_2d(x)), 3, 3))  # noqa: E731

    from mimetika.operators.derham import DeRhamDeviatoricStress

    def factory(contact=None):
        problem = FourFieldElasticity(
            mesh, contact=contact,
            inner=DeRhamDeviatoricStress(mesh, material=material),
        )
        matrix, rhs = problem.assemble_constrained(
            dirichlet=zero,
            extra_rhs=extra,
            traction=free,
            traction_facets=top,  # zero overburden increment
            roller_facets=rollers,
        )
        return problem, matrix, rhs

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
             boundary_spacing: float = 100.0, built=None):
    """Solve at ``parameters.depletion``; return slip and patch structure."""
    if built is None:
        mesh, fault, pressure = build(parameters, spacing, boundary_spacing)
    else:  # reuse the mesh; the pressure belongs to *this* depletion level
        mesh, fault, _ = built
        pressure = pressure_field(mesh, parameters)
    lam = (2.0 * parameters.shear_modulus * parameters.poisson
           / (1.0 - 2.0 * parameters.poisson))
    driver = ContactDriver(
        mesh, fault, SignoriniCoulomb(friction=parameters.friction),
        prestress=None, mu=parameters.shear_modulus, lam=lam,
        tolerance=1e-10, max_iterations=400,
    )
    driver.prestress = driver.expand_to_points(
        insitu_prestress(mesh, fault, parameters, pressure)
    )
    state = driver.solve_step(
        mechanics_factory(mesh, parameters, pressure), solver="newton"
    )
    y = mesh.geometry.centroids(1)[np.asarray(fault, dtype=int)][:, 1]
    order = np.argsort(y)
    slip = driver.per_facet(state.jump)[:, 1]

    # post-slip Coulomb function: total effective traction is the in-situ
    # prestress (which already carries the Biot pore term on the normal) plus
    # the solved incremental traction, read on the fault facets themselves
    total = driver.per_facet(
        driver.prestress + driver.tractions(state.solution["stress"])
    )
    coulomb = np.abs(total[:, 1]) + parameters.friction * total[:, 0]

    yy, ss = y[order], slip[order]
    slipping = np.abs(ss) > 1e-6
    patches = []
    start = None
    for i, flag in enumerate(slipping):
        if flag and start is None:
            start = i
        if (not flag or i == len(ss) - 1) and start is not None:
            end = i if flag else i - 1
            patches.append((yy[start], yy[end],
                            float(np.abs(ss[start:end + 1]).max())))
            start = None
    return {
        "y": yy, "slip": ss, "coulomb": coulomb[order], "patches": patches,
        "state": state, "cells": mesh.num_cells(2), "mesh": mesh,
        "fault": fault, "built": (mesh, fault, pressure),
    }


def pre_slip_stress(parameters: Parameters, spacing: float = 2.0,
                    built=None):
    """Fault stresses on the **locked** fault (Fig. 8): the plain continuum
    under the depletion load, no contact block, one direct solve."""
    from mimetika.contact import FrictionlessBilateral
    from mimetika.solver.saddle import solve_saddle

    if built is None:
        mesh, fault, pressure = build(parameters, spacing)
    else:
        mesh, fault, _ = built
        pressure = pressure_field(mesh, parameters)
    problem, matrix, rhs = mechanics_factory(mesh, parameters, pressure)(None)
    solution = problem.split(
        solve_saddle(matrix, rhs, problem.block_sizes, method="direct")
    )
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
        "built": (mesh, fault, pressure),
    }


def pair_average(y, v):
    """Two-facet moving average, on facet-midpoint positions.

    The exact set-valued law resolves a near-threshold plateau with an
    alternating stick/at-threshold active set, so the reaction traction
    carries a facet-scale, mean-zero oscillation; averaging adjacent facets
    removes exactly that mode and nothing else.
    """
    return 0.5 * (y[:-1] + y[1:]), 0.5 * (v[:-1] + v[1:])


def figure(parameters, results, path):
    """Slip profiles at the requested depletion levels (paper Figs. 9-10)."""
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
        right.plot(np.abs(res["slip"][inside]) * 1e3, res["y"][inside],
                   lw=1.5, color=line.get_color())
    for ax in (left, right):
        for edge in (parameters.fault_a, -parameters.fault_a):
            ax.axhline(edge, color="k", lw=0.6, ls=":")
        ax.grid(alpha=0.25)
    left.axvline(0.0, color="k", lw=0.6)
    left.set_xlabel(r"$\Sigma_C$   (MPa, two-facet average)")
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
                 f"constant friction $\\mu$ = {parameters.friction:g}   "
                 "(Novikov et al. 2024, Sect. 4.1, Figs. 9-10)",
                 fontsize=10)
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
    parser.add_argument("--vtk", nargs="?", const="out/benchmark_2",
                        default=None,
                        help="write the final level's state as .pvd/.vtu")
    arguments = parser.parse_args()

    parameters = wide_parameters()
    print("Benchmark 2 -- inclined displaced fault, constant friction\n")
    print(f"  dip = {parameters.dip:g} deg, mu = {parameters.friction:g}, "
          f"W = {parameters.width:g} m, H = {parameters.height:g} m")
    print("  paper: two patches at -25 MPa, merging at ~ -26.9 MPa "
          "(Figs. 9, 10, 12)\n")

    built, results = None, {}
    for level in arguments.levels:
        stage = replace(parameters, depletion=level * 1e6)
        res = simulate(stage, spacing=arguments.spacing, built=built)
        built = res["built"]
        results[level * 1e6] = res
        state = res["state"]
        print(f"  p = {level:6.1f} MPa   ({res['cells']} cells, "
              f"converged={state.converged} in {state.iterations} iterations)")
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
            print("    vs semi-analytical (4TU), Sigma_C (two-facet average) "
                  f"over |y| <= 100 m: rms = {rms / 1e6:.3f} MPa "
                  f"(scale {np.abs(cr[sel]).max() / 1e6:.1f} MPa)")
        print()

    pre_ref = reference_pre_slip()
    if pre_ref is not None and -25.0 in [round(lv, 6) for lv in arguments.levels]:
        stage = replace(parameters, depletion=-25e6)
        pre = pre_slip_stress(stage, spacing=arguments.spacing, built=built)
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

    if arguments.figure and not arguments.no_plots:
        print("  wrote", figure(parameters, results, arguments.figure))

    if arguments.vtk:
        from mimetika.postprocess import (
            MixedDimensionalSeries, contact_fields, mechanics_fields,
        )

        mesh, fault, pressure = built
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
