"""Convergence and timing benchmark for the de Rham (consistency-only) family.

Methods
-------
* mimetic-BDM (Darcy):    DeRhamDiffusionInnerProduct, mixed solve.
* mimetic-AFW 3F:         MixedElasticity(inner=DeRhamElasticityInnerProduct).
* mimetic-AFW 4F:         FourFieldElasticity(inner=DeRhamDeviatoricStress).
* AFW (extended product): MixedElasticity default -- the reference member.

Meshes: structured triangles (the coincidence regime: BDM_1 / AFW exactly)
and structured quads (the enriched regime: constants-only strong
consistency).  Manufactured solutions are smooth trigonometric fields;
errors are relative l2 norms of the DOF vectors against the interpolants,
so on triangles the 3F/4F methods must reproduce the AFW errors exactly.

Run:  python benchmarks/derham/convergence.py [--sizes 4 8 16]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from mimetika.assembly.four_field import FourFieldElasticity
from mimetika.assembly.mixed import MixedElasticity
from mimetika.geometry.local_cell import LocalCell
from mimetika.mesh import structured_quads, structured_triangles
from mimetika.operators.elasticity import ElasticityInnerProduct
from mimetika.operators.derham import DeRhamDiffusionInnerProduct
from mimetika.operators.derham import (
    DeRhamDeviatoricStress,
    DeRhamElasticityInnerProduct,
)

PI = np.pi
MU = LAM = 1.0


# -- manufactured solutions ---------------------------------------------------

def p_exact(x):
    return np.sin(PI * x[:, 0]) * np.sin(PI * x[:, 1])


def flux_exact(x):
    out = np.zeros((len(x), 3))
    out[:, 0] = -PI * np.cos(PI * x[:, 0]) * np.sin(PI * x[:, 1])
    out[:, 1] = -PI * np.sin(PI * x[:, 0]) * np.cos(PI * x[:, 1])
    return out


def darcy_source(x):
    return 2 * PI**2 * p_exact(x)


def disp_exact(x):
    out = np.zeros((len(x), 3))
    out[:, 0] = np.sin(PI * x[:, 0]) * np.sin(PI * x[:, 1])
    out[:, 1] = 0.5 * np.cos(PI * x[:, 0]) * np.cos(PI * x[:, 1])
    return out


def stress_exact(x):
    """sigma = 2 mu eps + lam tr(eps) I for the displacement above."""
    c, s = np.cos(PI * x[:, 0]), np.sin(PI * x[:, 0])
    C, S = np.cos(PI * x[:, 1]), np.sin(PI * x[:, 1])
    out = np.zeros((len(x), 3, 3))
    out[:, 0, 0] = (2 * MU + 0.5 * LAM) * PI * c * S
    out[:, 1, 1] = (-MU + 0.5 * LAM) * PI * c * S
    out[:, 0, 1] = out[:, 1, 0] = 0.5 * MU * PI * s * C
    return out


def body_force(x):
    """f = div sigma (library convention)."""
    out = np.zeros((len(x), 3))
    out[:, 0] = -(2.5 * MU + 0.5 * LAM) * PI**2 * np.sin(PI * x[:, 0]) * np.sin(
        PI * x[:, 1]
    )
    out[:, 1] = (0.5 * LAM - 0.5 * MU) * PI**2 * np.cos(PI * x[:, 0]) * np.cos(
        PI * x[:, 1]
    )
    return out


# -- mixed Darcy driver on the BDM space --------------------------------------

def solve_darcy(mesh):
    d = mesh.dim
    ip = DeRhamDiffusionInnerProduct(mesh)
    t0 = time.perf_counter()
    M = ip.assemble()
    n_cells, n_facets = mesh.num_cells(d), mesh.num_cells(d - 1)
    ndofs = d * n_facets

    rows, cols, vals = [], [], []
    F = np.zeros(n_cells)
    u_I = np.zeros(ndofs)
    n_sides = np.zeros(n_facets)
    for cid in range(n_cells):
        lc = LocalCell.build(mesh.geometry, cid, ip.frame)
        qp, qw = mesh.geometry.quadrature(d, cid)
        F[cid] = qw @ darcy_source(qp)
        for i, fid in enumerate(lc.facet_ids):
            n_sides[fid] += 1
            rows.append(cid)
            cols.append(d * fid)
            vals.append(lc.signs[i])
            if n_sides[fid] == 1:  # interpolate the exact flux once per facet
                fq, fw = lc.facet_quadrature[i]
                B, _ = lc.facet_scalar_basis(i)
                un = (flux_exact(lc.to_ambient(fq)) @ lc.frame) @ lc.facet_normals[i]
                u_I[d * fid : d * (fid + 1)] = lc.signs[i] * np.einsum(
                    "q,qb,q->b", fw, B, un
                )
    D = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, ndofs))
    A = sp.bmat([[M, -D.T], [D, None]], format="csr")
    t_asm = time.perf_counter() - t0

    t0 = time.perf_counter()
    sol = spla.spsolve(A, np.concatenate([np.zeros(ndofs), F]))  # p = 0 on bnd
    t_slv = time.perf_counter() - t0
    u_h, p_h = sol[:ndofs], sol[ndofs:]

    p_I = np.array([p_exact(x[None, :3])[0] for x in mesh.geometry.centroids(d)])
    eu = np.linalg.norm(u_h - u_I) / np.linalg.norm(u_I)
    ep = np.linalg.norm(p_h - p_I) / np.linalg.norm(p_I)
    return eu, ep, t_asm, t_slv


# -- elasticity drivers --------------------------------------------------------

def solve_elasticity(problem):
    t0 = time.perf_counter()
    sol = problem.solve(dirichlet=disp_exact, body_force=body_force)
    t_tot = time.perf_counter() - t0
    sig_I = problem.interpolate_stress(stress_exact)
    u_I = problem.interpolate_displacement(disp_exact)
    es = np.linalg.norm(sol["stress"] - sig_I) / np.linalg.norm(sig_I)
    eu = np.linalg.norm(sol["displacement"] - u_I) / np.linalg.norm(u_I)
    return es, eu, t_tot


ELASTIC = {
    "AFW (extended)": lambda m: MixedElasticity(
        m, inner=ElasticityInnerProduct(m, mu=MU, lam=LAM)
    ),
    "mimetic-AFW 3F": lambda m: MixedElasticity(
        m, inner=DeRhamElasticityInnerProduct(m, mu=MU, lam=LAM)
    ),
    "mimetic-AFW 4F": lambda m: FourFieldElasticity(
        m, inner=DeRhamDeviatoricStress(m, mu=MU, lam=LAM)
    ),
}


def rate(errs):
    return [""] + [
        f"{np.log2(errs[i - 1] / errs[i]):5.2f}" for i in range(1, len(errs))
    ]


def run(sizes):
    for mesh_name, gen in (
        ("triangles", structured_triangles),
        ("quads", structured_quads),
    ):
        print(f"\n=== {mesh_name} ===")
        errs_u, errs_p = [], []
        print("mimetic-BDM (Darcy)")
        for n in sizes:
            eu, ep, ta, ts = solve_darcy(gen(n, n))
            errs_u.append(eu)
            errs_p.append(ep)
            print(
                f"  n={n:3d}  e_flux={eu:9.3e}  e_p={ep:9.3e}"
                f"  asm={ta:6.2f}s  solve={ts:6.2f}s"
            )
        print(f"  rates flux: {rate(errs_u)}   p: {rate(errs_p)}")

        for name, make in ELASTIC.items():
            errs_s, errs_d = [], []
            print(name)
            for n in sizes:
                es, eu, tt = solve_elasticity(make(gen(n, n)))
                errs_s.append(es)
                errs_d.append(eu)
                print(
                    f"  n={n:3d}  e_sigma={es:9.3e}  e_u={eu:9.3e}"
                    f"  total={tt:6.2f}s"
                )
            print(f"  rates sigma: {rate(errs_s)}   u: {rate(errs_d)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16])
    run(ap.parse_args().sizes)
