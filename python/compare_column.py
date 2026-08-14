"""The three-cell column's first step: the C++ operator against the Python's.

Run:  python python/compare_column.py /tmp/col.bin

WHAT HAS TO BE RECONCILED, and why none of it is a discrepancy:

1. DEGREE-OF-FREEDOM ORDER.  Within a facet exokal's ProductSpace runs the
   component fastest (``b*d + k``) while the Python runs the moment fastest
   (``k*d + b``).  Facets and cells are numbered by their own meshes, so both
   are matched on centroids.

2. FACET ORIENTATION.  Each mesh assigns its own canonical normal from its
   stored vertex cycle.  A facet whose normals disagree has every moment
   signed, and ``t2 = n x t1`` signed a second time, so the two operators are
   related by ``P = (+) diag(s, s, 1)`` per facet and component.  ``P`` is an
   involution: it leaves the spectrum and the solution alone and changes
   every entry.

3. THE SADDLE-POINT CONVENTION.  exokal writes ``[M, -B^T; +B, 0]``; the
   Python writes it symmetrically.  That is a sign on a whole field's columns,
   so it is searched over rather than assumed -- five fields, thirty-two
   cases, and the answer has to be one of them or the two really do disagree.

4. CONSTRAINTS.  The C++ substitutes (an identity row, the column kept, the
   value in the state); the Python eliminates symmetrically (row and column
   dropped, a scaled diagonal put back, the lifting moved to the right-hand
   side).  Both impose the same condition and neither can be compared to the
   other on a pinned row.  The operator lives on the free-free block, and the
   right-hand sides are compared as what they both are: the first Newton step.
"""

from __future__ import annotations

import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, "src")
sys.path.insert(0, "python")

from read_system import read_system  # noqa: E402

from mimetika.assembly.mixed import boundary_facets  # noqa: E402
from mimetika.assembly.poromechanics import PoroMechanics  # noqa: E402
from mimetika.materials import Material  # noqa: E402
from mimetika.mesh import structured_box  # noqa: E402
from mimetika.operators.derham import DeRhamDeviatoricStress  # noqa: E402

import os
HEIGHT, WIDTH, NZ = float(os.environ.get('COLH', 10.0)), 1.0, 3
MU = float(os.environ.get('COLMU', 6.0e8))
ALPHA = float(os.environ.get('COLALPHA', 0.9))
INV_M = float(os.environ.get('COLINVM', 1.0e-10))
NU = float(os.environ.get('COLNU', 0.2))
PERM = float(os.environ.get('COLPERM', 1.0e-13))
VISC = float(os.environ.get('COLVISC', 1.0e-3))
LOAD = float(os.environ.get('COLLOAD', 1.0e7))
D = 3


def python_system(dt: float):
    mesh = structured_box(1, 1, NZ, (WIDTH, WIDTH, HEIGHT))
    material = Material(shear_modulus=MU, poisson=NU, biot=ALPHA,
                        inverse_biot_modulus=INV_M,
                        permeability=PERM, viscosity=VISC)
    poro = PoroMechanics(mesh, material,
                         stress_inner=DeRhamDeviatoricStress(mesh, material=material))

    g = mesh.geometry
    cen = g.centroids(2)
    bnd = np.asarray(sorted(boundary_facets(mesh)))
    top = [int(f) for f in bnd if abs(cen[f][2] - HEIGHT) < 1e-9]
    confined = [int(f) for f in bnd
                if abs(cen[f][2]) < 1e-9 or abs(cen[f][0]) < 1e-9
                or abs(cen[f][0] - WIDTH) < 1e-9 or abs(cen[f][1]) < 1e-9
                or abs(cen[f][1] - WIDTH) < 1e-9]

    load = np.zeros((3, 3))
    load[2, 2] = -LOAD
    S, rhs, _ = poro.assemble(
        dt=dt,
        dirichlet=lambda x: np.zeros((len(np.atleast_2d(x)), 3)),
        traction=lambda x: np.broadcast_to(load, (len(np.atleast_2d(x)), 3, 3)),
        traction_facets=top, roller_facets=confined, no_flow=confined,
    )
    # the pinned set, from the same routines that pinned it -- a structural
    # test ("rows with one entry") also catches rows the operator happens to
    # leave nearly empty, which is a different thing entirely
    pinned = np.zeros(S.shape[0], dtype=bool)
    dofs, _ = poro.mechanics.traction_moments(top, lambda x: np.broadcast_to(
        load, (len(np.atleast_2d(x)), 3, 3)))
    pinned[np.asarray(dofs, dtype=np.int64)] = True
    pinned[poro.mechanics.roller_dofs(confined)] = True
    ndf = poro._ndf_q
    fac = np.asarray(sorted(confined))
    pinned[poro._flux_offset + (ndf * fac[:, None] + np.arange(ndf)).ravel()] = True

    return mesh, poro, sp.csr_matrix(S), np.asarray(rhs), pinned


def cpp_geometry(path: str):
    rows = [l.split() for l in open(path + ".geo") if l.strip()]
    nf, nc = int(rows[0][0]), int(rows[0][1])
    fac = np.array([[float(x) for x in r] for r in rows[1:1 + nf]])
    cel = np.array([[float(x) for x in r] for r in rows[1 + nf:1 + nf + nc]])
    return fac[:, :3], fac[:, 3:6], cel


def match(a, b):
    """``i -> j`` matching two point sets, refusing anything but a bijection."""
    out = np.array([int(np.argmin(np.linalg.norm(b - p, axis=1))) for p in a])
    if len(set(out.tolist())) != len(a):
        raise SystemExit("centroid matching is not a bijection: different meshes")
    worst = max(np.linalg.norm(b[out[i]] - a[i]) for i in range(len(a)))
    if worst > 1e-9:
        raise SystemExit(f"centroid matching is loose ({worst:.2e}): different meshes")
    return out


S_B2 = False


def main(path: str, dt: float) -> int:
    s = read_system(path)
    fx, fn, cx = cpp_geometry(path)
    mesh, poro, A_py, rhs_py, py_pinned = python_system(dt)
    g = mesh.geometry

    f_map = match(fx, g.centroids(2))          # C++ facet -> Python facet
    c_map = match(cx, g.centroids(D))          # C++ cell  -> Python cell
    sign = np.array([np.sign(fn[i] @ g.facet_normals()[f_map[i]]) for i in range(len(fx))])
    print(f"{len(fx)} facets, {len(cx)} cells; "
          f"{int((sign < 0).sum())} with opposite canonical normals")

    nf, nc = len(fx), len(cx)
    n_stress, n_flux = 9 * nf, 3 * nf
    off_py = {"s_0": 0, "u_0": n_stress, "g_0": n_stress + D * nc,
              "q_0": n_stress + 2 * D * nc, "p_0": n_stress + 2 * D * nc + n_flux}
    if A_py.shape[0] != off_py["p_0"] + nc:
        raise SystemExit(f"python system is {A_py.shape[0]} dofs, expected "
                         f"{off_py['p_0'] + nc}")
    if s.n != A_py.shape[0]:
        raise SystemExit(f"{s.n} C++ dofs vs {A_py.shape[0]} Python dofs")

    # the C++ dof -> Python dof map, and the orientation sign it carries
    perm = np.empty(s.n, dtype=np.int64)
    sgn = np.ones(s.n)
    for i in range(nf):
        j, si = f_map[i], sign[i]
        for b in range(D):
            for k in range(D):
                # exokal: (facet*3 + b)*3 + k     python: facet*9 + k*3 + b
                perm[s.fields["s_0"][0] + (i * D + b) * D + k] = off_py["s_0"] + j * 9 + k * D + b
                sgn[s.fields["s_0"][0] + (i * D + b) * D + k] = (
                    si if (b < 2 or S_B2) else 1.0)
            perm[s.fields["q_0"][0] + i * D + b] = off_py["q_0"] + j * D + b
            sgn[s.fields["q_0"][0] + i * D + b] = si if b < 2 else 1.0
    for i in range(nc):
        j = c_map[i]
        for k in range(D):
            perm[s.fields["u_0"][0] + i * D + k] = off_py["u_0"] + j * D + k
            perm[s.fields["g_0"][0] + i * D + k] = off_py["g_0"] + j * D + k
        perm[s.fields["p_0"][0] + i] = off_py["p_0"] + j
    if len(set(perm.tolist())) != s.n:
        raise SystemExit("the dof map is not a bijection")

    # push the C++ system into the Python's numbering and orientation
    Q = sp.csr_matrix((sgn, (perm, np.arange(s.n))), shape=(s.n, s.n))
    A_cpp = Q @ s.A @ Q.T
    r_cpp = Q @ s.residual
    pinned = np.zeros(s.n, dtype=bool)
    pinned[perm[s.pinned]] = True

    # THE CONSTRAINTS THEMSELVES, before any operator: the two codes must have
    # pinned the same degrees of freedom, or nothing downstream means anything.
    print(f"pinned: {int(pinned.sum())} C++, {int(py_pinned.sum())} Python, "
          f"{'identical sets' if np.array_equal(pinned, py_pinned) else 'DIFFERENT'}")
    if not np.array_equal(pinned, py_pinned):
        raise SystemExit("the two codes pinned different degrees of freedom")
    free = np.flatnonzero(~pinned)

    B_cpp = A_cpp[free][:, free]
    B_py = A_py[free][:, free]
    scale = abs(B_py).max()

    # THE SADDLE-POINT CONVENTION is a choice of the sign each balance equation
    # is written with -- a scaling of ROWS.  It is not a congruence, and it must
    # not be searched as one: exokal's [M, -B^T; +B, 0] differs from the
    # symmetric form in the second row only, which no A -> D A D can produce.
    # A COLUMN sign would be a redefinition of an unknown, which is a different
    # claim, so the two are searched separately and both are reported.
    names = ["s_0", "u_0", "g_0", "q_0", "p_0"]
    sizes = [n_stress, D * nc, D * nc, n_flux, nc]
    field_of = np.concatenate([np.full(n, i) for i, n in enumerate(sizes)])[free]
    best = None
    for rb in range(1 << len(names)):
        rv = np.array([1.0 if (rb >> i) & 1 == 0 else -1.0 for i in range(len(names))])
        L = sp.diags(rv[field_of]) @ B_cpp
        for cb in range(1 << len(names)):
            cv = np.array([1.0 if (cb >> i) & 1 == 0 else -1.0 for i in range(len(names))])
            e = abs(B_py - L @ sp.diags(cv[field_of])).max() / scale
            if best is None or e < best[0]:
                best = (e, rv, cv)
    err, rv, cv = best
    row, col = rv[field_of], cv[field_of]

    # THE CELL SCALING. exokal writes the momentum balance as the cell AVERAGE
    # of div sigma and the Python as its INTEGRAL, so the C++ blocks carry a
    # factor 1/|E| and its displacement unknown carries |E|.  That is a
    # positive diagonal congruence -- it moves no solution and it is not a sign,
    # so it is MEASURED here rather than searched: the factor is read off the
    # (u,s) block and then checked against the cell volumes, which is the
    # difference between establishing it and assuming it.
    vol = g.measure(D)
    meas = np.ones(len(free))
    for i, name in enumerate(["u_0", "g_0"]):
        rows = np.flatnonzero(field_of == names.index(name))
        gi = free[rows] - off_py[name]
        rat = []
        for r, cell in zip(rows, gi // D):
            a = B_py[r]; b = (sp.diags(row) @ B_cpp @ sp.diags(col))[r]
            k = np.flatnonzero(abs(a.toarray().ravel()) > 1e-30)
            if len(k):
                rat.append((abs(b.toarray().ravel()[k]).max()
                            / abs(a.toarray().ravel()[k]).max(), vol[cell]))
        if rat:
            f_meas = np.array([x for x, _ in rat])
            v = np.array([y for _, y in rat])
            print(f"  {name} row scaling: measured {f_meas.min():.6f}..{f_meas.max():.6f}, "
                  f"1/|E| = {(1.0/v).min():.6f}..{(1.0/v).max():.6f}, "
                  f"max |measured - 1/|E|| = {abs(f_meas - 1.0/v).max():.2e}")
        meas[rows] = 1.0 / vol[gi // D]
    # apply it on both sides: the row (the equation) and the column (the unknown)
    row = row / meas
    col = col / meas
    print(f"row signs (equation convention): {dict(zip(names, [int(x) for x in rv]))}")
    print(f"col signs (unknown convention):  {dict(zip(names, [int(x) for x in cv]))}")
    print(f"  operator   rel |A_py - A_cpp| = {err:.3e}   (free block, {len(free)} dofs)")

    # BLOCK BY BLOCK, each against ITS OWN scale.  This operator spans twenty
    # orders of magnitude -- a compliance of 1/G next to a divergence of |f| --
    # so a single global norm would report every block as matching the instant
    # the stiffest one did.
    Bc = sp.diags(row) @ B_cpp @ sp.diags(col)
    print(f"  operator, block by block (each against its own scale):")
    for i, ni in enumerate(names):
        ri = field_of == i
        if not ri.any():
            continue
        for j, nj in enumerate(names):
            cj = field_of == j
            P_, C_ = B_py[ri][:, cj], Bc[ri][:, cj]
            if P_.nnz == 0 and C_.nnz == 0:
                continue
            sc = max(abs(P_).max() if P_.nnz else 0.0, abs(C_).max() if C_.nnz else 0.0)
            df = abs(P_ - C_).max() if (P_ - C_).nnz else 0.0
            print(f"    ({ni[0]},{nj[0]})  |.| = {sc:10.3e}   max |diff| = {df:10.3e}"
                  f"   rel = {df / sc:.3e}")

    # THE RIGHT-HAND SIDE, as what the two of them both are: the first Newton
    # step from the unloaded state.  The C++ solves A dx = -r with the pins in
    # the state; the Python solves S x = rhs with the lifting on the right.
    # The free state is zero at the first step, so dx_f is x_f and -r_f is rhs_f.
    #
    # The lifting is the only thing on either side here, so a mismatch is a
    # disagreement about the PINNED VALUES rather than about the operator --
    # checked first, since it is the cheaper explanation.
    v_cpp = (sgn * s.pin_value)[s.pinned][np.argsort(np.argsort(perm[s.pinned]))]
    v_py = np.zeros(s.n)
    v_py[pinned] = 0.0
    # the Python keeps its pin values only in the lifting; recover them as the
    # C++ does, from the same traction routine, to compare like with like
    print("  pinned values (traction moments on the loaded facet):")
    top_dofs = np.flatnonzero(pinned)[abs(v_cpp) > 0] if (abs(v_cpp) > 0).any() else []
    print(f"    C++ nonzero pins: {len(top_dofs)}  values "
          f"{np.unique(np.round(v_cpp[abs(v_cpp) > 0], 6))}")
    print(f"    python nonzero  : {len(np.unique(rhs_py[rhs_py != 0]))} distinct rhs values")

    if len(sys.argv) > 3 and sys.argv[3] == "dump":
        np.set_printoptions(precision=5, suppress=False, linewidth=200)
        f0 = int(np.argmin(np.linalg.norm(fx - np.array([0.5, 0.5, HEIGHT / NZ]), axis=1)))
        j0 = f_map[f0]
        rows = off_py["s_0"] + j0 * 9 + np.arange(9)
        keep = np.array([r in set(free.tolist()) for r in rows])
        ri = np.array([np.flatnonzero(free == r)[0] for r in rows[keep]])
        cu = np.flatnonzero(field_of == 1)
        print("\n(s,u) rows of the TOP facet, python then c++:")
        print(np.asarray(B_py[ri][:, cu].todense()))
        print(np.asarray(Bc[ri][:, cu].todense()))
        cs = np.flatnonzero(field_of == 0)[:9]
        print("\n(s,s) top-left 9x9, python then c++:")
        print(np.asarray(B_py[:9][:, cs].todense()))
        print(np.asarray(Bc[:9][:, cs].todense()))

    if len(sys.argv) > 3 and sys.argv[3] == "where":
        Bc2 = sp.diags(row) @ B_cpp @ sp.diags(col)
        ns = int((field_of == 0).sum())
        Dd = (B_py - Bc2)[:ns][:, :ns].tocoo()
        big = np.argsort(-abs(Dd.data))[:8]
        print("\nlargest (s,s) differences:")
        inv = {v: k for k, v in enumerate(free)}
        for t_ in big:
            r_, c_, v_ = int(Dd.row[t_]), int(Dd.col[t_]), Dd.data[t_]
            gr, gc = free[r_], free[c_]
            fr, kr, br = gr // 9, (gr % 9) // 3, gr % 3
            fc, kc, bc_ = gc // 9, (gc % 9) // 3, gc % 3
            print(f"  facet {fr:2d} k={kr} b={br}  x  facet {fc:2d} k={kc} b={bc_}"
                  f"   py {B_py[r_, c_]: .6e}  cpp {Bc2[r_, c_]: .6e}  diff {v_: .3e}")
        # and the pinned values themselves
        pv = np.zeros(s.n)
        pv[perm] = sgn * s.pin_value
        tf = [i for i in range(nf) if abs(fx[i][2] - HEIGHT) < 1e-9][0]
        jd = off_py["s_0"] + f_map[tf] * 9 + np.arange(9)
        print(f"\n  C++ traction pins on the loaded facet: {pv[jd]}")
        dofs, vals = poro.mechanics.traction_moments(
            [int(f_map[tf])], lambda x: np.broadcast_to(
                np.diag([0.0, 0.0, -LOAD]), (len(np.atleast_2d(x)), 3, 3)))
        print(f"  python traction pins:                  {np.asarray(vals)}")

    lhs = -(row * r_cpp[free])
    rhs = rhs_py[free]
    print("  residual, block by block:")
    for i, name in enumerate(names):
        rows = field_of == i
        if not rows.any():
            continue
        sc = max(abs(rhs[rows]).max(), abs(lhs[rows]).max(), 1e-300)
        df = abs(rhs[rows] - lhs[rows]).max()
        print(f"    {name:5s} |.| = {sc:10.3e}   max |diff| = {df:10.3e}   rel = {df / sc:.3e}")
    return 0


if __name__ == "__main__":
    S_B2 = "b2" in sys.argv
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/col.bin",
                          float(sys.argv[2]) if len(sys.argv) > 2 else 1.0))
